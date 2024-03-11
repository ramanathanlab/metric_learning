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
from utils import plot_embeddings
from dataset import ProxyDataset
from models import MetricNet
from sklearn.decomposition import TruncatedSVD
from pyarrow.parquet import ParquetFile
import pyarrow as pa

if __name__=="__main__": 
    data_path='/lambda_stor/homes/bhsu/gb_2024/my_gb_files/metric_learn/data/arxiv_emb_processed_multi.parquet'
    pf = ParquetFile(data_path) 
    first_rows = next(pf.iter_batches(batch_size = 1000)) 
    data = pa.Table.from_batches([first_rows]).to_pandas() 
    data["embedding"] = data["embedding"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
    data["one_label"] = data["one_label"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
    features_tensor = torch.stack(tuple(data["embedding"].values))
    target_tensor = torch.stack(tuple(data["one_label"].values))

    # Step 2: Compute Singular Value Decomposition (SVD)
    U, S, V = torch.svd(features_tensor)

    S_normalized = S / S.max()
    # Step 3: Plot the singular values
    plt.plot(S_normalized.numpy())
    plt.title('Singular Value Distribution')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.show()


    print('Done')