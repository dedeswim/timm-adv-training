## Pre-requisites

This repo works with:

- Python `3.8.10`, and will probably work with newer versions.
- `torch==1.8.1` and `1.10.1`, and it will probably work with PyTorch `1.9.x`.
- `torchvision==0.9.1` and `0.11.2`, and it will probably work with torchvision `0.10.x`.
- The other requirements are in `requirements.txt`, hence they can be installed with `pip install -r requirements.txt`

In case you want to use Weights and Biases, after installing the requisites, install `wandb` with `pip install wandb`, and run `wandb login`.

In case you want to read or write your results to a Google Cloud Storage bucket (which is supported by this repo), install the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud), and login. Then you are ready to use GCS for both storing data and experiments results, as well as download checkpoints by using paths in the form of `gs://bucket-name/path-to-dir-or-file`.

## Training

All these commands are meant to be run on TPU VMs with 8 TPU cores. They can be easily adapted to work on GPUs by using `torch.distributed.launch` (and by removing the `launch_xla.py --num-devices 8` part). The LR can be scaled as explained in the appendix of our paper (which follows DeiT's convention). More info about how to run the training script on TPUs and GPUs can be found in `timm.bits`'s [README](https://github.com/rwightman/pytorch-image-models/tree/bits_and_tpu/timm/bits#timm-bits).

To log the results to W&B it is enough to add the flag `--log-wandb`. The W&B experiment will have the name passed to the `--experiment` flag.

### Training an XCiT-S12 on ImageNet

<details>

```bash
python launch_xla.py --num-devices 8 train.py $DATA_DIR --dataset tfds/imagenet2012 --experiment $EXPERIMENT --output $OUTPUT --model xcit_small_12_p16_224 --config configs/xcit-adv-training.yaml
```

Where `$OUTPUT` should be the dir where the checkpoints are saved, `$EXPERIMENT` is the name of the experiment for W&B logging and to use as a subdirectory of `$OUTPUT`, and `$DATA_DIR` is the directory where the data are saved. For instance, TFDS saved the data in `~/tensorflow_data` by default.

## Validation

For validating using full AA models trained on ImageNet, CIFAR-10 and CIFAR-100 it is recommended to use [this](#validating-using-robustbench) command. To evaluate using APGD-CE only, or to evaluate other datasets than those above (e.g., Caltech101 and Oxford Flowers), then use [this](#validating-using-the-validation-script) script instead.

### Validating using RobustBench

<details>

This script will run the full AutoAttack using RobustBench's interface.

```bash
python3 validate_robustbench.py --data-dir $DATA_DIR --dataset $DATASET --model $MODEL --batch-size 1024 --checkpoint $CHECKPOINT --eps $EPS
```

If the model has been trained using a specific mean and std, then they should be specified with the `--mean` and `--std` flags, similarly to training.

</details>

### Validating using the validation script

Do not use this script to run APGD-CE or AutoAttack on TPU (and XLA in general), as the compilation will take an unreasonable amount of time.

<details>

```bash
python3 validate.py $DATA_DIR --dataset $DATASET --log-freq 1 --model $MODEL --checkpoint $CHECKPOINT --mean <mean> --std <std> --attack $ATTACK --attack-eps $EPS
```

If the model has been trained using a specific mean and std, then they should be specified with the `--mean` and `--std` flags, and the `--normalize-model` flag should be specified, similarly to training. Otherwise the `--no-normalize` flag sould be specified. For both Caltech101 and Oxford Flowers, you should specify `--num-classes 102`, and for Caltech101 only `--split test`. If you just want to run PGD, then you can specify the number of steps with `--attack-steps 200`.

</details>

## Code

A large amount of the code is adapted from [`timm`](https://github.com/rwightman/pytorch-image-models), in particular from the `bits_and_tpu` branch. The code by Ross Wightman is originally released under Apache-2.0 License, which can be found [here](https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE).

The entry point for training is [train.py](train.py). While in [src](src/) there is a bunch of utility modules, as well as model definitions (which are found in [src/models](src/models/)).

### Tests

In order to run the unit tests, install pytest via `pip install pytest`, and run

```bash
python -m pytest .
```
