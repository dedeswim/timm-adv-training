import csv
import functools
import json
import sys
import timm
import tqdm
import yaml
from pathlib import Path

from src import models

LAST_CHECKPOINT = 'last.pth.tar'
CONFIG_PATH = 'args.yaml'
SUMMARY_PATH = 'summary.csv'
AA_RESULTS = 'aa_results-best-8.0.json'


SUMMARY_FIELDS_TO_KEEP = {
    'train_flops',
    'train_loss',
    'train_top1',
    'train_robust_top1',
    'train_flops',
    'train_electricity_cost',
    'eval_loss',
    'eval_robust_top1',
    'train_emissions'
}

SUMMARY_FIELDS_TO_RENAME = {
    "train_top1": "train_acc",
    'train_robust_top1': "train_robust_acc",
    "eval_robust_top1": "pgd40_robust_acc",
}

SUMMARY_FIELDS_TO_AVERAGE = {
    'train_flops',
    'train_emissions',
    'train_electricity_cost',
}

def check_end_of_training(trained_epochs: int, config_dict: dict) -> bool:
    epochs_to_train = config_dict['epochs'] + config_dict['warmup_epochs']
    return trained_epochs >= epochs_to_train - 1


@functools.lru_cache
def count_model_parameters(model_name: str):
    model = timm.create_model(model_name, pretrained=False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def float_or_none(x):
    try:
       return float(x)
    except:
       return None


def build_info_from_experiment(exp_path: Path) -> dict:
    with open(exp_path / SUMMARY_PATH) as f:
        summary = [{str(k): float_or_none(v) for k, v in row.items() if k in SUMMARY_FIELDS_TO_KEEP} for row in csv.DictReader(f, skipinitialspace=True)]
 
    results = summary[-1]
    
    for field in SUMMARY_FIELDS_TO_AVERAGE:
        results[field] = sum(row[field] for row in summary if row[field] is not None) / len(summary)
    
    for old_name, new_name in SUMMARY_FIELDS_TO_RENAME.items():
        results[new_name] = results.pop(old_name)
     
    with open(exp_path / CONFIG_PATH) as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)
    if not check_end_of_training(len(summary), config_dict):
        raise ValueError(f'Experiment {exp_path} is not finished yet.')
    
    results['epochs'] = config_dict['epochs'] + config_dict['warmup_epochs']
    results['model'] = config_dict['model']
    results['model_parameters'] = count_model_parameters(config_dict['model'])
    results['synthetic_data'] = config_dict['combine_dataset'] is not None
    results['at'] = config_dict['adv_training']
    results['at_steps'] = config_dict['attack_steps']
    
    if not (exp_path / AA_RESULTS).exists():
        raise ValueError(f'AA not run for {exp_path}.')
    with open(exp_path / AA_RESULTS) as f:
        aa_results = json.load(f)
    
    results['aa_robust_acc'] = aa_results['robust_top1']
    results['eval_acc'] = aa_results['top1']
    
    return results
    


if __name__ == '__main__':
    exps_path = Path(sys.argv[1])
    results = []
    for exp_path in tqdm.tqdm(exps_path.iterdir()):
        if "robust-hw" in exp_path.name:
            try:
                results.append(build_info_from_experiment(exp_path))
            except ValueError as e:
                print(e, exp_path)

    with open('results.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
