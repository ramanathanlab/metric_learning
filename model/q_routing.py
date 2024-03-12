from __future__ import annotations
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch, lightning as L, torch.nn as nn, torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import pandas as pd, umap, numpy as np, matplotlib.pyplot as plt, wandb, argparse
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from pytorch_metric_learning.distances import LpDistance
from pydantic import BaseModel
from typing import Dict, List, Any
from tqdm import tqdm
from config import BaseConfig, Config
from utils import plot_embeddings, SVD, get_alignment, get_uniformity
from dataset import ProxyDataset, Metric_Dataset
from models import MetricNet
from sklearn.decomposition import TruncatedSVD


# General Plan: 
# 1.) Dataset Structure: 
# Columns: q_1 embeddings, q_2 embeddings, indicator: q_1==q_2 (1 if yes, 0 if no)?

# pytorch metric learning works on this: 
# assign labels q_1 = class 1, q_2 = class_2 
# pass this into the miner, which expects embeddings + "class labels"
# hard negative mining 

# loss function will be NTXent or Contrastive 

