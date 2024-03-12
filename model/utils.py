import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from cycler import cycler
from lightning.pytorch.loggers import WandbLogger
import umap
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import lightning as L, torch, torch.nn as nn, torch.nn.functional as F, torchmetrics
from torchmetrics.classification import Accuracy, F1Score 
from torchmetrics import Metric
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from pytorch_metric_learning import distances, losses, miners, reducers, testers, samplers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, umap, pyarrow as pa, argparse
from pyarrow.parquet import ParquetFile
from tqdm import tqdm 
from rich.console import Console
from rich.table import Table
import pytorch_metric_learning.utils.logging_presets as LP
import wandb
from torchmetrics.classification import Accuracy, ConfusionMatrix
from transformers import (AdamW,AutoConfig,
                          AutoModelForCausalLM,AutoTokenizer,
                          get_linear_schedule_with_warmup)
from torch.optim.lr_scheduler import CosineAnnealingLR

# original implementation: https://github.com/SsnL/align_uniform
def get_alignment(x, y, alpha=2):
    """
    bsz : batch size (number of positive pairs)
    d   : latent dim
    x   : Tensor, shape=[bsz, d]
      latents for one side of positive pairs
    y   : Tensor, shape=[bsz, d]
      latents for the other side of positive pairs
    """
    alignment=(x - y).norm(p=2, dim=1).pow(alpha).mean()
    return alignment

def get_uniformity(x:torch.tensor, t=2):
    '''Given batched embeddings of input of [batch, embed_dim], calculate the average alignment of features'''
    sq_pdist = torch.pdist(x, p=2).pow(2) 
    uniformity = sq_pdist.mul(-t).exp().mean().log()
    return uniformity

def plot_embeddings(umap_embeddings:np.array, 
                    labels:np.array,
                    epoch:int) -> wandb.Image:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='Spectral', s=5)
    # cbar = plt.colorbar(boundaries=np.arange(len(np.unique(test_labels))+1)-0.5)
    # cbar.set_ticks(np.array(np.arange(len(np.unique(test_labels)))))
    # cbar.ax.set_yticklabels([v for k, v in label_map.items() if k in test_labels]) # [k for k, v in label_map.items() if v in test_labels]
    plt.title(f'UMAP Projection of the Embeddings, Epoch: {epoch}', fontsize=24)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Anchor', 'Positive', 'Negative'])
    plt.show()
    wandb_image = wandb.Image(scatter)
    return wandb_image


def SVD(features_tensor:torch.tensor):
    # Step 2: Compute Singular Value Decomposition (SVD)
    U, S, V = torch.svd(features_tensor)
    S_normalized = (S / S.max()).mean()
    # Step 3: Plot the singular values
    plt.plot(S_normalized.cpu().numpy())
    plt.title('Singular Value Distribution')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    wandb_image_svd=wandb.Image(plt)
    return S_normalized, wandb_image_svd

# def plot_alignment()



#################################### Old functions used in old model; might be useful in the future ######################

def get_most_labels(train_labels:list, 
                    test_labels:list
                    ) -> list: 
    '''Returns the largest set of class_labels as a list. This is because sometimes we
    get rare labels only in one of the train/test sets'''
    if len(train_labels.unique()) <= len(test_labels.unique()): 
        class_labels = train_labels.unique().int().tolist()
    elif len(train_labels.unique()) > len(test_labels.unique()):
        class_labels = test_labels.unique().int().tolist()
    return class_labels

def map_class_accuracies(class_labels:list, 
                         accuracies:dict, 
                         label_map:dict
                         ) -> dict:
    '''Unpacks accuracies object into dictionary of accuracies by matching with class_labels and label_map'''
    # TODO: check if this is right
    accuracies_dict = {class_labels[i]: accuracies['precision_at_1'][i] 
                       for i in range(len(class_labels)) 
                       if i in range(len(accuracies['precision_at_1']))}
    return accuracies_dict

def plot_confusion_matrix(confusion_matrix:torch.tensor, 
                          label_map:dict, 
                          epoch:int
                          )-> wandb.Image:
    '''Takes in a confusion matrix of numpy ints and label_map dict
    and returns a wandb image for logging'''
    confusion_matrix = confusion_matrix.detach().cpu().numpy().astype('int')
    plt.figure(figsize=(10, 8))
    im = plt.imshow(confusion_matrix, interpolation='nearest', cmap='turbo')
    plt.title(f'Confusion Matrix for Epoch: {epoch}')
    tick_marks = np.arange(len(label_map))
    plt.xticks(tick_marks, [label_map[k] for k in sorted(label_map.keys())], rotation=45)
    plt.yticks(tick_marks, [label_map[k] for k in sorted(label_map.keys())])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > im.norm(confusion_matrix).max() / 2. else "white")
    cbr = plt.colorbar()
    cbr.set_label('Count')
    wandb_image_cm = wandb.Image(plt)
    return wandb_image_cm

def get_classification_accuracies(confusion_matrix:torch.Tensor, 
                                  class_labels:list, 
                                  label_map:dict) -> dict: 
    '''Takes in '''
    accuracy_per_class = (confusion_matrix.diag()/confusion_matrix.sum(1)*100).detach().cpu().numpy().tolist()
    class_accuracies_dict = {class_labels[i]: accuracy_per_class[i] for i in range(len(class_labels))}
    mapped_class_accuracies_dict = {}
    for label_name, label_id in label_map.items():
        if label_id in class_accuracies_dict:
            mapped_class_accuracies_dict[label_name] = class_accuracies_dict[label_id]
    return mapped_class_accuracies_dict

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)