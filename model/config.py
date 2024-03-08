import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, List
from pydantic_settings import BaseSettings as _BaseSettings
from pydantic import root_validator, validator, BaseModel
import pprint as pp
from dataclasses import dataclass, field
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import torch.nn as nn
import torch 
import yaml

_T = TypeVar("_T")

PathLike = Union[str, Path]

def _resolve_path_exists(value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    p = value.resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def path_validator(field: str) -> classmethod:
    decorator = validator(field, allow_reuse=True)
    _validator = decorator(_resolve_path_exists)
    return _validator


def from_yaml(filename):
    with open(filename) as fp:
        raw_data = yaml.safe_load(fp)
    return raw_data

class BaseConfig(BaseModel):
  @classmethod 
  def read_yaml(cls, path_name:str) -> Dict: 
      with open(path_name, 'r') as file: 
         cfg = yaml.safe_load(file)
         return cls(**cfg)
      
  @classmethod
  def read_yaml_test(cls, path_name:str) -> Dict: 
      with open(path_name, 'r') as file: 
         cfg = yaml.safe_load(file)
         return cls(**cfg)
    
  @staticmethod
  def dump_yaml(instance, path_name: str):
      with open(path_name, 'w') as file:
          data_to_dump = instance.dict()
          dump = yaml.safe_dump(data_to_dump, default_flow_style=False)
          file.write(dump)
         
class DataConfig(BaseModel):
    data_path:str='./'
    num_samples:int=10000
    batch_size:int=64
    train_ratio:float=0.8
    val_ratio:float=0.15

class ModelConfig(BaseModel): 
    num_blocks: int=5
    input_size: int=768
    output_size: int=768
    block_pattern: list[int]=[512,256,512]
    batch_norm:bool=True
    activation: str='ReLU'
    lr: float=1e-4
    optimizer_name: str='AdamW'
    optimizer: Dict[str, Any]={'lr':1e-3, 'weight_decay':1e-3}
    scheduler_name: str='CosineAnnealing'
    scheduler: Dict[str, Any]={'T_max':5}

class MetricConfig(BaseModel):
    margin:float=0.2 
    loss_name:str='Contrastive'
    loss:Dict[str, Any]={'Contrastive':{
                                        'pos_margin':0, 
                                        'neg_margin':1
                                        }
                        }
    mining_name:str='None'
    accuracy_name:str='Standard'
    accuracy:Dict[str, Any]={'include':('precision_at_1', 
                                        'mean_average_precision'),
                             'k':1 
                             }

class Config(BaseConfig): 
    # data configurations 
    data_path:str='./'
    num_samples:int=10000
    batch_size:int=64
    train_ratio:float=0.8
    val_ratio:float=0.15

    # metric model configs 
    num_blocks: int=5
    input_size: int=768
    output_size: int=768
    block_pattern: list[int]=[512,256,512]
    batch_norm:bool=True
    activation: str='ReLU'
    lr: float=1e-4
    optimizer_name: str='AdamW'
    optimizer: Dict[str, Any]={'lr':1e-3, 'weight_decay':1e-3}
    scheduler_name: str='CosineAnnealing'
    scheduler: Dict[str, Any]={'T_max':5}

    # metric learning utils function configurations
    margin:float=0.2 
    loss_name:str='Contrastive'
    loss:Dict[str, Any]={'Contrastive':{
                                        'pos_margin':0, 
                                        'neg_margin':1
                                        }
                        }
    mining_name:str='None'
    accuracy_name:str='Standard'
    accuracy:Dict[str, Any]={'include':('precision_at_1', 
                                        'mean_average_precision'),
                             'k':1 
                             }
    

def dump_config_to_yaml(config: Config, path_name: str):
    with open(path_name, 'w') as file:
        data_to_dump = config.dict()
        dump = yaml.safe_dump(data_to_dump, default_flow_style=False, sort_keys=False)
        file.write(dump)


if __name__=="__main__": 
  settings = Config()
  dump_config_to_yaml(settings, 
                      'path_to_output_file.yaml'
                      )
  


  settings = Config.read_yaml_test('path_to_output_file.yaml')
  print('Done')

    # model_settings = ModelConfig()
    # metric_settings = MetricConfig()
    # data_settings = DataConfig()
  



#   OPTIMIZERS = {
#     "AdamW": torch.optim.AdamW,
#     "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR
# }
#   LOSSES = {
#      'TripleMargin': losses.TripletMarginLoss,
#      'AngularMargin': losses.ArcFaceLoss,
#      'CrossEntropy': nn.CrossEntropyLoss, 
#      'NCA': losses.NCALoss, 
#      'NTXent': losses.NTXentLoss, 
#      'Angular': losses.AngularLoss
#   }

  # opt_config = OptConfig()
  # loss_config = LossConfig()

  # optimizer = OPTIMIZERS[opt_config.optimizer_name](**opt_config.optimizer_hparams, params=model.parameters())
  # scheduler = OPTIMIZERS[opt_config.scheduler_name](**opt_config.scheduler_hparams, optimizer=optimizer)

  # class_config = ClassConfig.read_yaml('/homes/bhsu/gb_2024/my_gb_files/metric_learn/metric_learning/training/angular_linear.yaml', 
  #                                      cfg_type='classifier')
  
  # # classifier = Classifier(**class_config)

  # classifier = Classifier(**{'input_size':100, 
  #                            'output_size': 50})
  






  print('Done')