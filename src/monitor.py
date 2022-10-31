"""From timm's monitor. It implements TensorBoard usage
"""

import csv
import logging
import os
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Union

import torch

_logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError as e:
    HAS_TB = False
    SummaryWriter = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def summary_row_dict(results, index=None, index_name='epoch'):
    assert isinstance(results, dict)
    row_dict = OrderedDict()
    if index is not None:
        row_dict[index_name] = index
    if not results:
        return row_dict
    if isinstance(next(iter(results.values())), dict):
        # each key in results is a per-phase results dict, flatten by prefixing with phase name
        for p, pr in results.items():
            assert isinstance(pr, dict)
            row_dict.update([('_'.join([p, k]), v) for k, v in pr.items()])
    else:
        row_dict.update(results)
    return row_dict


class SummaryCsv:

    def __init__(self, output_dir, filename='summary.csv'):
        self.output_dir = output_dir
        self.filename = os.path.join(output_dir, filename)
        self.needs_header = not os.path.exists(self.filename)

    def update(self, row_dict):
        with open(self.filename, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=row_dict.keys())
            if self.needs_header:  # first iteration (epoch == 1 can't be used)
                dw.writeheader()
                self.needs_header = False
            dw.writerow(row_dict)


_sci_keys = {'lr'}


def _add_kwargs(text_update, name_map=None, **kwargs):

    def _to_str(key, val):
        if isinstance(val, float):
            if key.lower() in _sci_keys:
                return f'{key}: {val:.3e} '
            else:
                return f'{key}: {val:.4f}'
        else:
            return f'{key}: {val}'

    def _map_name(key, name_map, capitalize=True):
        if name_map is None:
            if capitalize:
                return key.capitalize() if not key.isupper() else key
            else:
                return key
        return name_map.get(key, None)

    for k, v in kwargs.items():
        if isinstance(v, dict):
            # log each k, v of a dict kwarg as separate items
            for kk, vv in v.items():
                name = _map_name(kk, name_map)
                if not name:
                    continue
                text_update += [_to_str(kk, vv)]
        else:
            name = _map_name(k, name_map, capitalize=True)
            if not name:
                continue
            text_update += [_to_str(name, v)]


class Monitor:

    def __init__(self,
                 experiment_name=None,
                 output_dir=None,
                 logger=None,
                 hparams=None,
                 log_wandb=False,
                 output_enabled=True,
                 log_tensorboard=False,
                 wandb_project: Optional[str] = None,
                 wandb_id: Optional[str] = None):
        self.output_dir = output_dir  # for tensorboard, csv, text file (TODO) logging
        self.logger = logger or logging.getLogger('log')
        hparams = hparams or {}

        # Setup CSV writer(s)
        if output_dir is not None:
            self.csv_writer = SummaryCsv(output_dir=output_dir)
        else:
            self.csv_writer = None

        # Setup Tensorboard
        if log_tensorboard and SummaryWriter is not None:
            self.summary_writer = SummaryWriter(log_dir=output_dir)
        else:
            self.summary_writer = None  # FIXME tensorboard

        # Setup W&B
        self.wandb_run = None
        if log_wandb:
            wandb_project = wandb_project or experiment_name
            if HAS_WANDB:
                self.wandb_run = wandb.init(project=wandb_project, config=hparams, id=wandb_id, resume="allow")
            else:
                _logger.warning("You've requested to log metrics to wandb but package not found. "
                                "Metrics not being logged to wandb, try `pip install wandb`")

        self.output_enabled = output_enabled
        # FIXME image save

    def log_step(
        self,
        phase: str,
        step_idx: int,
        step_end_idx: Optional[int] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        rate: Optional[Union[float, Tuple[float, float]]] = None,
        phase_suffix: str = '',
        **kwargs,
    ):
        """ log train/eval step
        """
        if not self.output_enabled:
            return
        if 'num_steps' in kwargs:
            step_end_idx = max(0, kwargs.pop('num_steps') - 1)
        phase_title = f'{phase.capitalize()} ({phase_suffix})' if phase_suffix else f'{phase.capitalize()}:'
        progress = 100. * step_idx / step_end_idx if step_end_idx else 0.
        rate_str = ''
        if isinstance(rate, (tuple, list)):
            rate_str = f'Rate: {rate[0]:.2f}/s ({rate[1]:.2f}/s)'
        elif rate is not None:
            rate_str = f'Rate: {rate:.2f}/s'
        text_update = [
            phase_title,
            f'{epoch}' if epoch is not None else None,
            f'[{step_idx}]' if step_end_idx is None else None,
            f'[{step_idx}/{step_end_idx} ({progress:>3.0f}%)]' if step_end_idx is not None else None,
            rate_str,
            f'Loss: {loss:.5f}' if loss is not None else None,
        ]
        _add_kwargs(text_update, **kwargs)
        log_str = ' '.join(item for item in text_update if item)
        self.logger.info(log_str)
        if self.summary_writer is not None:
            ...

    def log_phase(self,
                  phase: str = 'eval',
                  epoch: Optional[int] = None,
                  name_map: Optional[dict] = None,
                  **kwargs):
        """log completion of evaluation or training phase
        """
        if not self.output_enabled:
            return

        title = [
            f'{phase.capitalize()}',
            f'epoch: {epoch}' if epoch is not None else None,
            'completed. ',
        ]
        title_str = ' '.join(i for i in title if i)
        results = []
        _add_kwargs(results, name_map=name_map, **kwargs)
        log_str = title_str + ', '.join(item for item in results if item)
        self.logger.info(log_str)

    def write_summary(
        self,
        results: Dict,  # Dict or Dict of Dict where first level keys are treated as per-phase results
        index: Optional[Union[int, str]] = None,
        index_name: str = 'epoch',
    ):
        """ Log complete results for all phases (typically called at end of epoch)

        Args:
            results (dict or dict[dict]): dict of results to write, or multiple dicts where first level
                key is the name of results dict for each phase
            index: value for row index (typically epoch #)
            index_name:  name for row index header (typically 'epoch')
        """
        if not self.output_enabled:
            return

        row_dict = summary_row_dict(index=index, index_name=index_name, results=results)
        if self.csv_writer:
            self.csv_writer.update(row_dict)
        if self.wandb_run is not None:
            wandb.log(row_dict)
        if self.summary_writer is not None:
            self.summary_writer.add_scalars("", tag_scalar_dict=row_dict, global_step=row_dict["epoch"])
