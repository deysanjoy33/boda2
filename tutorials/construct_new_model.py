"""
train_boda2.py
==============
Cluster-ready training script converted from construct_new_model.ipynb.
Trains a BassetBranched model on pre-split MPRA data using boda2.

Usage
-----
python train_boda2.py \
    --train_path  /path/to/train.csv \
    --val_path    /path/to/dev.csv \
    --test_path   /path/to/test.csv \
    --output_dir  ./outputs \
    --activity_columns K562_log2FC \
    --batch_size  1024 \
    --max_epochs  200 \
    --min_epochs  60 \
    --accelerator gpu \
    --devices     1
"""

import os
import re
import sys
import argparse
import tempfile
from functools import partial

import json
import numpy as np
import torch
import pandas as pd
import lightning.pytorch as ptl
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, Dataset

import boda
from boda.common import constants, utils
from boda.data.mpra_datamodule import DNAActivityDataset
from boda.graph.cnn_prediction import CNNBasicTraining
from boda.graph.utils import spearman_correlation, shannon_entropy


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Lightning v2 patch  (fixes removed validation_epoch_end hook)
# ──────────────────────────────────────────────────────────────────────────────

def apply_lightning_v2_patch():
    """
    Patches CNNBasicTraining to be compatible with Lightning >= 2.0.
    validation_epoch_end was removed; outputs must be accumulated manually
    and the hook renamed to on_validation_epoch_end.
    """
    _orig_validation_step = CNNBasicTraining.validation_step

    def _patched_validation_step(self, batch, batch_idx):
        out = _orig_validation_step(self, batch, batch_idx)
        if not hasattr(self, '_val_outputs'):
            self._val_outputs = []
        self._val_outputs.append(out)
        return out

    def _patched_on_validation_epoch_end(self):
        val_step_outputs = getattr(self, '_val_outputs', [])
        if not val_step_outputs:
            return

        arit_mean    = torch.stack([b['loss']   for b in val_step_outputs], dim=0).mean()
        harm_mean    = torch.stack([b['metric'] for b in val_step_outputs], dim=0) \
                           .mean(dim=0).pow(-1).mean().pow(-1)
        epoch_preds  = torch.cat([b['preds']  for b in val_step_outputs], dim=0)
        epoch_labels = torch.cat([b['labels'] for b in val_step_outputs], dim=0)

        spearman, mean_spearman              = spearman_correlation(epoch_preds, epoch_labels)
        shannon_pred                          = shannon_entropy(epoch_preds)
        shannon_label                         = shannon_entropy(epoch_labels)
        _, specificity_mean_spearman          = spearman_correlation(shannon_pred, shannon_label)

        self.aug_log(external_metrics={
            'current_epoch':            self.current_epoch,
            'arithmetic_mean_loss':     arit_mean,
            'harmonic_mean_loss':       harm_mean,
            'prediction_mean_spearman': mean_spearman.item(),
            'entropy_spearman':         specificity_mean_spearman.item(),
        })
        self._val_outputs.clear()

    CNNBasicTraining.validation_step = _patched_validation_step
    if hasattr(CNNBasicTraining, 'validation_epoch_end'):
        del CNNBasicTraining.validation_epoch_end
    CNNBasicTraining.on_validation_epoch_end = _patched_on_validation_epoch_end
    print("Lightning v2 patch applied.", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  SimpleMPRA_DataModule  (accepts pre-split files)
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMPRA_DataModule(ptl.LightningDataModule):
    """
    Simplified MPRA DataModule that accepts pre-split train / val / test files.

    val_chrs / test_chrs can be:
      - str  → treated as a file path and loaded directly
      - list → chromosome list used to filter rows from the train file
    """

    def __init__(self,
                 datafile_path,
                 val_chrs=None,
                 test_chrs=None,
                 sequence_column='sequence',
                 activity_columns=None,
                 chr_column='chr',
                 sep='\t',
                 batch_size=32,
                 padded_seq_len=600,
                 left_flank=constants.MPRA_UPSTREAM,
                 right_flank=constants.MPRA_DOWNSTREAM,
                 num_workers=8,
                 duplication_cutoff=None,
                 use_reverse_complements=False,
                 **kwargs):
        super().__init__()
        if activity_columns is None:
            activity_columns = ['K562_log2FC']

        self.datafile_path           = datafile_path
        self.val_chrs                = val_chrs
        self.test_chrs               = test_chrs
        self.sequence_column         = sequence_column
        self.activity_columns        = activity_columns
        self.chr_column              = chr_column
        self.sep                     = sep
        self.batch_size              = batch_size
        self.padded_seq_len          = padded_seq_len
        self.left_flank              = left_flank
        self.right_flank             = right_flank
        self.num_workers             = num_workers
        self.duplication_cutoff      = duplication_cutoff
        self.use_reverse_complements = use_reverse_complements

        self.pad_column_name = 'padded_seq'
        self.padding_fn = partial(
            utils.row_pad_sequence,
            in_column_name=self.sequence_column,
            padded_seq_len=self.padded_seq_len,
            upStreamSeq=self.left_flank,
            downStreamSeq=self.right_flank,
        )
        self.dataset_train = None
        self.dataset_val   = None
        self.dataset_test  = None

    def _load_columns(self):
        cols = [self.sequence_column, *self.activity_columns]
        need_chr = (isinstance(self.val_chrs,  list) and self.val_chrs) or \
                   (isinstance(self.test_chrs, list) and self.test_chrs)
        if need_chr:
            cols.append(self.chr_column)
        return cols

    def _df_to_dataset(self, df, is_train=False):
        df = df.copy()
        print(f'  Padding {len(df):,} sequences...', flush=True)
        df[self.pad_column_name] = df.apply(self.padding_fn, axis=1)

        list_tensor_seq = []
        for _, row in df.iterrows():
            list_tensor_seq.append(utils.row_dna2tensor(row, in_column_name=self.pad_column_name))

        sequences  = torch.stack(list_tensor_seq)
        activities = torch.Tensor(df[self.activity_columns].to_numpy())

        if is_train:
            return DNAActivityDataset(
                sequences, activities,
                sort_tensor=torch.max(activities, dim=-1).values,
                duplication_cutoff=self.duplication_cutoff,
                use_reverse_complements=self.use_reverse_complements,
            )
        return TensorDataset(sequences, activities)

    def _load_split(self, source, train_df=None, label=''):
        if isinstance(source, str):
            print(f'  Loading {label} from file: {source}', flush=True)
            cols = [self.sequence_column, *self.activity_columns]
            return utils.parse_file(file_path=source, columns=cols, sep=self.sep)
        elif isinstance(source, list) and source:
            print(f'  Filtering {label} by chromosomes: {source}', flush=True)
            assert train_df is not None and self.chr_column in train_df.columns, \
                "chr_column must be present in train_df when val/test_chrs is a list"
            return train_df[train_df[self.chr_column].isin(set(source))].reset_index(drop=True)
        return None

    def setup(self, stage='train'):
        print('-' * 60, flush=True)
        print('Setting up SimpleMPRA_DataModule...\n', flush=True)

        cols = self._load_columns()
        print(f'Loading train file: {self.datafile_path}', flush=True)
        train_df = utils.parse_file(file_path=self.datafile_path, columns=cols, sep=self.sep)
        print(f'  {len(train_df):,} rows loaded.\n', flush=True)

        print('Building TRAIN dataset:', flush=True)
        self.dataset_train = self._df_to_dataset(train_df, is_train=True)
        print(f'  → {len(self.dataset_train):,} examples (incl. augmentation)\n', flush=True)

        val_df = self._load_split(self.val_chrs, train_df=train_df, label='VAL')
        if val_df is not None:
            print('Building VAL dataset:', flush=True)
            self.dataset_val = self._df_to_dataset(val_df, is_train=False)
            print(f'  → {len(self.dataset_val):,} examples\n', flush=True)

        test_df = self._load_split(self.test_chrs, train_df=train_df, label='TEST')
        if test_df is not None:
            print('Building TEST dataset:', flush=True)
            self.dataset_test = self._df_to_dataset(test_df, is_train=False)
            print(f'  → {len(self.dataset_test):,} examples\n', flush=True)

        print('-' * 60, flush=True)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True,  num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,   batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,  batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_best(my_model, checkpoint_callback):
    """Load the best checkpoint weights back into the model."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            best_path = checkpoint_callback.best_model_path
            get_epoch = re.search(r'epoch=(\d*)', best_path).group(1)
            print(f'Best model path : {best_path}', file=sys.stderr)
            print(f'File exists     : {os.path.isfile(best_path)}', file=sys.stderr)
            ckpt = torch.load(best_path)
            my_model.load_state_dict(ckpt['state_dict'])
            print(f'Loaded weights from epoch {get_epoch}', file=sys.stderr)
        except (KeyError, AttributeError):
            print('Could not load best checkpoint; keeping most recent model.', file=sys.stderr)
    return my_model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train a BassetBranched model with boda2')

    # Data
    p.add_argument('--train_path',        required=True,  help='Path to train CSV/TSV')
    p.add_argument('--val_path',          required=True,  help='Path to val CSV/TSV (or comma-sep chr list)')
    p.add_argument('--test_path',         required=True,  help='Path to test CSV/TSV (or comma-sep chr list)')
    p.add_argument('--sep',               default=',',    help='File delimiter (default: comma)')
    p.add_argument('--sequence_column',   default='sequence')
    p.add_argument('--activity_columns',  nargs='+',      default=['K562_log2FC'])
    p.add_argument('--batch_size',        type=int,       default=1024)
    p.add_argument('--padded_seq_len',    type=int,       default=600)
    p.add_argument('--num_workers',       type=int,       default=8)
    p.add_argument('--duplication_cutoff',type=float,     default=2.0)
    p.add_argument('--use_reverse_complements', action='store_true', default=False)

    # Model
    p.add_argument('--n_outputs',         type=int,       default=1)
    p.add_argument('--n_linear_layers',   type=int,       default=1)
    p.add_argument('--linear_channels',   type=int,       default=1000)
    p.add_argument('--linear_activation', default='ReLU')
    p.add_argument('--linear_dropout_p',  type=float,     default=0.12)
    p.add_argument('--n_branched_layers', type=int,       default=3)
    p.add_argument('--branched_channels', type=int,       default=140)
    p.add_argument('--branched_activation', default='ReLU')
    p.add_argument('--branched_dropout_p',type=float,     default=0.56)
    p.add_argument('--loss_criterion',    default='L1KLmixed')
    p.add_argument('--loss_beta',         type=float,     default=5.0)

    # Optimizer / scheduler
    p.add_argument('--lr',                type=float,     default=0.0033)
    p.add_argument('--weight_decay',      type=float,     default=3.43e-4)
    p.add_argument('--T_0',              type=int,        default=4096)

    # Trainer
    p.add_argument('--min_epochs',        type=int,       default=2)
    p.add_argument('--max_epochs',        type=int,       default=6)
    p.add_argument('--accelerator',       default='gpu')
    p.add_argument('--devices',           type=int,       default=1)
    p.add_argument('--precision',         default='16-mixed')
    p.add_argument('--patience',          type=int,       default=5)

    # Output
    p.add_argument('--output_dir',        default='./outputs')
    p.add_argument('--model_filename',    default='best_model.pt')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation on test set
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_test(model, test_loader, activity_columns, output_dir, device='cpu'):
    """
    Run inference on the test set and report per-target and overall metrics.

    Metrics reported per activity column:
      - Pearson r
      - Spearman rho
      - MSE
      - MAE
      - R²

    Results are printed to stdout and saved as:
      <output_dir>/test_metrics.json        ← numbers
      <output_dir>/test_predictions.csv     ← per-example preds vs labels
    """
    model.eval()
    model.to(device)

    all_preds  = []
    all_labels = []

    print('\nRunning inference on test set...', flush=True)
    for batch in test_loader:
        sequences, labels = batch
        sequences = sequences.to(device)
        preds = model(sequences)               # (B, n_outputs)
        all_preds.append(preds.cpu().float())
        all_labels.append(labels.cpu().float())

    all_preds  = torch.cat(all_preds,  dim=0).numpy()  # (N, n_outputs)
    all_labels = torch.cat(all_labels, dim=0).numpy()  # (N, n_outputs)

    # ── Per-column metrics ────────────────────────────────────────────────────
    metrics = {}
    print('\n' + '=' * 60)
    print('TEST SET RESULTS')
    print('=' * 60)

    for i, col in enumerate(activity_columns):
        preds_i  = all_preds[:, i]
        labels_i = all_labels[:, i]

        pearson_r,  _ = pearsonr(preds_i, labels_i)
        spearman_rho, _ = spearmanr(preds_i, labels_i)
        mse           = mean_squared_error(labels_i, preds_i)
        mae           = mean_absolute_error(labels_i, preds_i)
        ss_res        = np.sum((labels_i - preds_i) ** 2)
        ss_tot        = np.sum((labels_i - labels_i.mean()) ** 2)
        r2            = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        metrics[col] = {
            'pearson_r':   round(float(pearson_r),   4),
            'spearman_rho':round(float(spearman_rho),4),
            'mse':         round(float(mse),          6),
            'mae':         round(float(mae),          6),
            'r2':          round(float(r2),           4),
            'n_examples':  int(len(preds_i)),
        }

        print(f'\n  [{col}]')
        print(f'    Pearson  r   : {pearson_r:.4f}')
        print(f'    Spearman rho : {spearman_rho:.4f}')
        print(f'    R²           : {r2:.4f}')
        print(f'    MSE          : {mse:.6f}')
        print(f'    MAE          : {mae:.6f}')
        print(f'    N examples   : {len(preds_i):,}')

    # ── Overall mean across columns (if multi-output) ─────────────────────────
    if len(activity_columns) > 1:
        mean_pearson  = np.mean([metrics[c]['pearson_r']    for c in activity_columns])
        mean_spearman = np.mean([metrics[c]['spearman_rho'] for c in activity_columns])
        metrics['mean_across_targets'] = {
            'mean_pearson_r':    round(float(mean_pearson),  4),
            'mean_spearman_rho': round(float(mean_spearman), 4),
        }
        print(f'\n  [Mean across all targets]')
        print(f'    Mean Pearson  r   : {mean_pearson:.4f}')
        print(f'    Mean Spearman rho : {mean_spearman:.4f}')

    print('\n' + '=' * 60, flush=True)

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved to : {metrics_path}', flush=True)

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pred_cols  = {f'pred_{c}':  all_preds[:, i]  for i, c in enumerate(activity_columns)}
    label_cols = {f'label_{c}': all_labels[:, i] for i, c in enumerate(activity_columns)}
    results_df = pd.DataFrame({**label_cols, **pred_cols})
    preds_path = os.path.join(output_dir, 'test_predictions.csv')
    results_df.to_csv(preds_path, index=False)
    print(f'Predictions saved to : {preds_path}', flush=True)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Apply Lightning v2 compatibility patch
    apply_lightning_v2_patch()

    # ── Data ──────────────────────────────────────────────────────────────────
    data = SimpleMPRA_DataModule(
        datafile_path          = args.train_path,
        val_chrs               = args.val_path,
        test_chrs              = args.test_path,
        sep                    = args.sep,
        sequence_column        = args.sequence_column,
        activity_columns       = args.activity_columns,
        batch_size             = args.batch_size,
        padded_seq_len         = args.padded_seq_len,
        num_workers            = args.num_workers,
        duplication_cutoff     = args.duplication_cutoff,
        use_reverse_complements= args.use_reverse_complements,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = boda.model.BassetBranched(
        n_outputs            = args.n_outputs,
        n_linear_layers      = args.n_linear_layers,
        linear_channels      = args.linear_channels,
        linear_activation    = args.linear_activation,
        linear_dropout_p     = args.linear_dropout_p,
        n_branched_layers    = args.n_branched_layers,
        branched_channels    = args.branched_channels,
        branched_activation  = args.branched_activation,
        branched_dropout_p   = args.branched_dropout_p,
        loss_criterion       = args.loss_criterion,
        loss_args            = {'beta': args.loss_beta},
    )

    # ── Graph (training loop) ─────────────────────────────────────────────────
    graph = boda.graph.CNNBasicTraining(
        model          = model,
        optimizer      = 'Adam',
        optimizer_args = {
            'lr':           args.lr,
            'betas':        [0.9, 0.999],
            'weight_decay': args.weight_decay,
            'amsgrad':      True,
        },
        scheduler          = 'CosineAnnealingWarmRestarts',
        scheduler_monitor  = None,
        scheduler_interval = 'step',
        scheduler_args     = {'T_0': args.T_0},
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath    = os.path.join(args.output_dir, 'checkpoints'),
        filename   = 'best-{epoch}-{prediction_mean_spearman:.4f}',
        monitor    = 'prediction_mean_spearman',
        mode       = 'max',
        save_top_k = 1,
        save_last  = True,
        verbose    = False,
    )

    stopping_callback = EarlyStopping(
        monitor   = 'prediction_mean_spearman',
        patience  = args.patience,
        mode      = 'max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = ptl.Trainer(
        accelerator = args.accelerator,
        devices     = args.devices,
        min_epochs  = args.min_epochs,
        max_epochs  = args.max_epochs,
        precision   = args.precision,
        callbacks   = [checkpoint_callback, stopping_callback, lr_monitor],
        default_root_dir = args.output_dir,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(graph, data)

    # ── Save best model ───────────────────────────────────────────────────────
    graph = set_best(graph, checkpoint_callback)
    save_path = os.path.join(args.output_dir, args.model_filename)
    torch.save(graph.model.state_dict(), save_path)
    print(f'\nBest model weights saved to: {save_path}', flush=True)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    device = 'cuda' if (args.accelerator == 'gpu' and torch.cuda.is_available()) else 'cpu'
    evaluate_on_test(
        model            = graph.model,
        test_loader      = data.test_dataloader(),
        activity_columns = args.activity_columns,
        output_dir       = args.output_dir,
        device           = device,
    )


if __name__ == '__main__':
    main()