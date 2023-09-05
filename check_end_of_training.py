import yaml
import sys
from pathlib import Path

import torch

def check_end_of_training(checkpoint_path, config_path):
    checkpoint = torch.load(checkpoint_path)
    trained_epochs = checkpoint['epoch']
    with open(config_path) as f:
        config_dict = yaml.load(f)
    epochs_to_train = config_dict['epochs'] + config_dict['warmup_epochs']
    return trained_epochs >= epochs_to_train - 1
    

if __name__ == '__main__':
    exp_path = Path(sys.argv[1])
    checkpoint_path = exp_path / 'last.pth.tar'
    config_path = exp_path / 'args.yaml'
    training_ended = check_end_of_training(checkpoint_path, config_path)
    sys.exit(0 if training_ended else 1)