"""Core training and eval functions.

The functions in this module are adapted from PyTorch Image Models by Ross Wightman

The original ones can be found at https://github.com/rwightman/pytorch-image-models/

The original license can be found at this link:
https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE
"""

from collections import OrderedDict
from dataclasses import replace
from typing import Optional, Tuple

import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from deepspeed.profiling.flops_profiler import FlopsProfiler
from timm.bits import (AccuracyTopK, AvgTensor, DeviceEnv, Monitor, Tracker, TrainServices, TrainState)

from src import utils
from src.attacks import AttackFn


def train_one_epoch(
    state: utils.AdvTrainState,
    services: utils.TrainServices,
    loader,
    dev_env: DeviceEnv,
):

    f = open("stats.txt", "a")
    
    if services.co2_tracker is not None:
        services.co2_tracker.start()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    tracker = Tracker()
    # FIXME move loss meter into task specific TaskMetric
    loss_meter = AvgTensor()
    accuracy_meter = AccuracyTopK(topk=(1, ))
    robust_accuracy_meter = AccuracyTopK(topk=(1, ))

    prof = FlopsProfiler(state.model)
    prof.start_profile()

    state.model.train()
    state.updater.reset()  # zero-grad

    step_end_idx = len(loader) - 1
    tracker.mark_iter()
    for step_idx, (sample, target) in enumerate(loader):
        tracker.mark_iter_data_end()

        # FIXME move forward + loss into model 'task' wrapper
        with dev_env.autocast():
            loss, output, adv_output = state.compute_loss_fn(state.model, sample, target, state.epoch)

        state.updater.apply(loss)

        tracker.mark_iter_step_end()

        state.updater.after_step(
            after_train_step,
            state,
            services,
            dev_env,
            step_idx,
            step_end_idx,
            tracker,
            loss_meter,
            accuracy_meter,
            robust_accuracy_meter,
            (output, adv_output, target, loss),
        )

        tracker.mark_iter()
        # end for

    if hasattr(state.updater.optimizer, 'sync_lookahead'):
        state.updater.optimizer.sync_lookahead()

    top1, = accuracy_meter.compute().values()
    robust_top1, = robust_accuracy_meter.compute().values()
    end.record()
    torch.cuda.synchronize()

    if services.co2_tracker is not None:
        emissions: float = services.co2_tracker.stop()  # type: ignore
    else:
        emissions = float("NaN")

    prof.stop_profile()
    flops = prof.get_total_flops()
    training_flops = flops * 3 * 1e-12
    prof.end_profile()

    gpu_hours = start.elapsed_time(end) * 1e-3 / 3600
    cloud_dollar_cost_lower_bound = 0.43 * 4 * gpu_hours # use 4 GPUs
    cloud_dollar_cost_upper_bound = 1.46 * 4 * gpu_hours # use 4 GPUs
    electricity_dollar_cost = 0.12 * (300 * 4 * gpu_hours / 1000) * 1.58 # use 4 GPUs

    #f.write(f"Epoch: {state.epoch}; Latency (ms): {start.elapsed_time(end)}; Emissions (kg): {emissions}\n")
    f.write(f"Epoch: {state.epoch}; Latency (ms): {start.elapsed_time(end)}; Emissions (kg): {emissions}; FLOPS (T): {training_flops}; gpu_hour (h): {gpu_hours}; cloud_computing_cost ($): {cloud_dollar_cost_lower_bound, cloud_dollar_cost_upper_bound}; electricity_cost ($): {electricity_dollar_cost}\n")
    f.close()
    #co2tracker.stop()

    # TODO(@zishenwan, @kshitij11): add here time and energy used
    return OrderedDict([('loss', loss_meter.compute().item()), ('top1', top1.item()),
                        ('robust_top1', robust_top1.item()), ('eps', state.eps_schedule(state.epoch)),
                        ('lr', state.updater.get_average_lr()), ('latency', start.elapsed_time(end)),
                        ('emissions', emissions), ('flops', training_flops),
                        ('cloud_computing_cost_lb', cloud_dollar_cost_lower_bound), ('cloud_computing_cost_ub', cloud_dollar_cost_upper_bound),
                        ('electricity_cost', electricity_dollar_cost)])


def after_train_step(
    state: TrainState,
    services: TrainServices,
    dev_env: DeviceEnv,
    step_idx: int,
    step_end_idx: int,
    tracker: Tracker,
    loss_meter: AvgTensor,
    accuracy_meter: AccuracyTopK,
    robust_accuracy_meter: AccuracyTopK,
    tensors: Tuple[torch.Tensor, ...],
):
    """
    After the core loss / backward / gradient apply step, we perform all non-gradient related
    activities here including updating meters, metrics, performing logging, and writing checkpoints.

    Many / most of these operations require tensors to be moved to CPU, they shoud not be done
    every step and for XLA use they should be done via the optimizer step_closure. This function includes
    everything that should be executed within the step closure.

    Args:
        state:
        services:
        dev_env:
        step_idx:
        step_end_idx:
        tracker:
        loss_meter:
        accuracy_meter:
        robust_accuracy_meter:
        tensors:

    Returns:

    """
    end_step = step_idx == step_end_idx

    with torch.no_grad():
        output, adv_output, target, loss = tensors
        loss_meter.update(loss, output.shape[0])

        if len(target.size()) > 1:
            target = target.argmax(dim=-1)

        accuracy_meter.update(output, target)
        if adv_output is not None:
            robust_accuracy_meter.update(adv_output, target)

        if state.model_ema is not None:
            # FIXME should ema update be included here or in train / updater step? does it matter?
            state.model_ema.update(state.model)

        state = replace(state, step_count_global=state.step_count_global + 1)
        cfg = state.train_cfg

        if services.monitor is not None and end_step or step_idx % cfg.log_interval == 0:
            global_batch_size = dev_env.world_size * output.shape[0]
            loss_avg = loss_meter.compute()
            top1, = accuracy_meter.compute().values()
            robust_top1, = robust_accuracy_meter.compute().values()

            if services.monitor is not None:
                lr_avg = state.updater.get_average_lr()
                services.monitor.log_step('Train',
                                          step_idx=step_idx,
                                          step_end_idx=step_end_idx,
                                          epoch=state.epoch,
                                          loss=loss_avg.item(),
                                          top1=top1.item(),
                                          robust_top1=robust_top1.item(),
                                          rate=tracker.get_avg_iter_rate(global_batch_size),
                                          lr=lr_avg)

        if services.checkpoint is not None and cfg.recovery_interval and (end_step or (step_idx + 1) %
                                                                          cfg.recovery_interval == 0):
            services.checkpoint.save_recovery(state)

        if state.lr_scheduler is not None:
            # FIXME perform scheduler update here or via updater after_step call?
            state.lr_scheduler.step_update(num_updates=state.step_count_global)


def after_eval_step(logger: Monitor, step_idx: int, step_end_idx: int, loss_meter: AvgTensor,
                    accuracy_meter: AccuracyTopK, robust_accuracy_meter: AccuracyTopK,
                    tensors: Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor,
                                   torch.Tensor], phase_suffix: str, log_interval: int):
    with torch.no_grad():
        last_step = step_idx == step_end_idx
        output, adv_output, target, loss = tensors

        loss_meter.update(loss, output.size(0))
        accuracy_meter.update(output, target)

        if adv_output is not None:
            robust_accuracy_meter.update(adv_output, target)

        if last_step or step_idx % log_interval == 0:
            top1, top5 = accuracy_meter.compute().values()
            if adv_output is not None:
                robust_top1, _ = robust_accuracy_meter.compute().values()
            else:
                robust_top1 = None

            loss_avg = loss_meter.compute()
            logger.log_step(
                'Eval',
                step_idx=step_idx,
                step_end_idx=step_end_idx,
                loss=loss_avg.item(),
                top1=top1.item(),
                top5=top5.item(),
                robust_top1=robust_top1.item() if robust_top1 is not None else None,
                phase_suffix=phase_suffix,
            )


def evaluate(model: nn.Module,
             loss_fn: nn.Module,
             loader,
             state: TrainState,
             logger: Monitor,
             dev_env: DeviceEnv,
             phase_suffix: str = '',
             log_interval: int = 10,
             attack: Optional[AttackFn] = None,
             use_mp_loader: bool = False,
             n_samples: Optional[int] = None,):
    tracker = Tracker()
    losses_m = AvgTensor()
    # FIXME move loss and accuracy modules into task specific TaskMetric obj
    accuracy_m = AccuracyTopK()
    robust_accuracy_m = AccuracyTopK()

    model.eval()

    end_idx = len(loader) - 1
    tracker.mark_iter()
    tot_samples = 0
    with torch.no_grad():
        for step_idx, (sample, target) in enumerate(loader):
            tracker.mark_iter_data_end()
            tot_samples += sample.size(0)
            with dev_env.autocast():
                output = model(sample)
                loss = loss_fn(output, target)

                if attack is not None:
                    with torch.enable_grad():
                        if dev_env.type_xla:
                            model.train()
                        adv_sample = attack(model, sample, target)
                        model.eval()
                        adv_output = model(adv_sample)
                else:
                    adv_output = None

                # FIXME, explictly marking step for XLA use since I'm not using the parallel xm loader
                # need to investigate whether parallel loader wrapper is helpful on tpu-vm or
                # only use for 2-vm setup.
                if dev_env.type_xla and not use_mp_loader:
                    dev_env.mark_step()
                elif dev_env.type_cuda:
                    dev_env.synchronize()

            # FIXME uncommenting this fixes race btw model `output`/`loss` and loss_m/accuracy_m meter input
            # for PyTorch XLA GPU use.
            # This issue does not exist for normal PyTorch w/ GPU (CUDA) or PyTorch XLA w/ TPU.
            # loss.item()

            tracker.mark_iter_step_end()
            state.updater.after_step(after_eval_step, logger, step_idx, end_idx, losses_m, accuracy_m,
                                     robust_accuracy_m, (output, adv_output, target, loss), phase_suffix,
                                     log_interval)

            tracker.mark_iter()
            if tot_samples >= n_samples:
                break

    top1, top5 = accuracy_m.compute().values()
    robust_top1, _ = robust_accuracy_m.compute().values()

    # TODO(@zishenwan, @kshitij11): add here time and energy used
    results = OrderedDict([
        ('loss', losses_m.compute().item()),
        ('top1', top1.item()),
        ('robust_top1', robust_top1.item()),
    ])
    return results
