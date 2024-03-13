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
from utils import plot_embeddings, SVD, get_alignment, get_uniformity, get_pos_anchor
from dataset import ProxyDataset, Metric_Dataset
from models import MetricNet
from sklearn.decomposition import TruncatedSVD

class SimCSE_Loss:
    def __init__(self, **config): 
        self.temp=config['temp']

    def __call__(self, h, h_plus): 
        embeddings = torch.cat([h, h_plus], dim=0)
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2) / self.temp
        labels = torch.arange(h.size(0)).repeat(2).type_as(h).long()
        loss = F.cross_entropy(cosine_sim, labels)
        return loss
    
def nt_xent_loss(x, temperature):
    assert len(x.size()) == 2
    
    xcs = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1)
    xcs[torch.eye(x.size(0)).bool()] = float("-inf")

    target = torch.arange(8)
    target[0::2] += 1
    target[1::2] -= 1
    
    return F.cross_entropy(xcs / temperature, target, reduction="mean")
    
    
class MetricModel(L.LightningModule):
    def __init__(self,
                 cfg:Config,
                 train_dataset:Dataset,
                 val_dataset:Dataset
                 ): 
        super(MetricModel, self).__init__()
        self.num_blocks=cfg.num_blocks
        self.input_size=cfg.input_size
        self.output_size=cfg.output_size
        self.block_pattern=cfg.block_pattern
        self.batch_norm=cfg.batch_norm
        self.activation=cfg.activation
        self.lr=cfg.lr
        self.batch_size=cfg.batch_size
        self.model=MetricNet(self.num_blocks,self.input_size,self.output_size,
                             self.block_pattern,self.batch_norm,self.activation,lr=self.lr)
        self.optimizer=OPTIMIZERS[cfg.optimizer_name](**cfg.optimizer, params=self.model.parameters())
        self.scheduler=OPTIMIZERS[cfg.scheduler_name](**cfg.scheduler, optimizer=self.optimizer)
        self.automatic_optimization = False # manual optimize in case we wanna do two models

        self.margin=cfg.margin
        self.miner=MINERS[cfg.mining_name]
        self.loss_fn=LOSS[cfg.loss_name](**cfg.loss)
        self.accuracy=ACCURACY[cfg.accuracy_name](**cfg.accuracy)
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset

        self.alignment_list = []
        self.uniformity_list = []

    def training_step(self, batch, batch_idx):
        metric_opt = self.optimizers()
        metric_opt.zero_grad()
        X, y = batch  
        h = self.model(X)
        h_plus = self.model(X)
        loss=self.loss_fn(h, h_plus) # 
        self.manual_backward(loss) 
        metric_opt.step()
        self.log_dict({"Metric Loss:": loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        '''Calculate the accuracies so everything in the enumerate loop
        Get total accuracy and class''' 
        X, y = batch

        # calculate validation loss 
        self.model.train() # need this to activate dropout
        h = self.model(X)
        h_plus = self.model(X)
        loss = self.loss_fn(h, h_plus)
        self.log_dict({'Validation Loss:':loss}, prog_bar=True)
        self.model.eval() # reset to eval mode for other validation metrics

        # I do this to get pos/anchor pairs based on index matching
        pos, anchor = get_pos_anchor(X, y)
        pos_embed = self.model(pos)
        anchor_embed = self.model(anchor)

        self.alignment_list.append(get_alignment(pos_embed, anchor_embed)) # how well to features of pos pairs align?
        self.uniformity_list.append(get_uniformity(h)) # h or h_plus okay, we are testing for feature scattering
    

    def on_validation_epoch_end(self):
        half_size = int(len(self.val_dataset)*0.5)
        X, y = self.val_dataset[:half_size] # grab half of the dataset
        X = X.type_as(next(self.model.parameters()))
        y = y.type_as(next(self.model.parameters()))

        # plotting UMAP
        # umap_labels = np.array([0]*len(self.val_dataset) + [1]*len(self.val_dataset) + [2]*len(self.val_dataset))
        umap_labels = np.array(y.detach().cpu())
        h = self.model(X)
        umap_embeddings = umapper.fit_transform(h.detach().cpu().numpy())
        wandb_image_umap = plot_embeddings(umap_embeddings, umap_labels, self.current_epoch)
        self.logger.experiment.log({f'UMAP-Plot':wandb_image_umap})

        # SVD plot 
        svd, wandb_image_svd = SVD(h)
        self.logger.experiment.log({f'SVD-Plot':wandb_image_svd})
        self.log_dict({"SVD Embeddings:": svd}, prog_bar=True)

        # alignment plot 
        alignment_avg = torch.mean(torch.stack(self.alignment_list))
        self.log_dict({'Average Alignment:': alignment_avg})

        # uniformity plot 
        uniformity_avg = torch.mean(torch.stack(self.uniformity_list))
        self.log_dict({'Average Uniformity:': uniformity_avg})

        self.alignment_list=[]
        self.uniformity_list=[]

        # pearsons need gold labels for ranking what 0, 1, 2, 3, 4, 5 in terms of cosine similarity 
        # we need to see if cosing similarity of projected embeddings correlate with ranking of similarity

        # IDEA: since we are just evaluating on mean average precision
        # for "classifying" if nearest neighbor is correct, we can create 3 synthetic classes
        # class 1 is anchor, class 2 is positive, class 3 is negative
        # originally, the cloud of embeddings are all near one another. Over time, they 
        # should start moving farther and farther from one another aka we get 3 distinct clusters
        # so MAP should increase

        # accuracies = self.accuracy_calculator.get_accuracy(
        #     val_embeddings, val_labels, train_embeddings, train_labels, False
        # )
        # avg_correct=self.dist_correct/self.total_samples*100
        # self.log('val_avg_correct', avg_correct)
        # print(f"Acc of negative pair identification: {avg_correct} %")

    def configure_optimizers(self):
        metric_optim=self.optimizer
        metric_sched=self.scheduler
        return {
                "optimizer":metric_optim, 
                "lr_scheduler":{
                    "scheduler":metric_sched
                }
            }
    
if __name__=="__main__": 

    parser = argparse.ArgumentParser(description="pytorch run")
    parser.add_argument('--data_path', default='/lambda_stor/homes/bhsu/gb_2024/my_gb_files/metric_learn/data/arxiv_emb_processed_multi.parquet', 
                        type=str)
    parser.add_argument('--num_epochs', default=15, type=int)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--log_offline', default=True, type=bool)
    parser.add_argument('--config_path', default='/homes/bhsu/gb_2024/my_gb_files/cerebras/metric_learning/config.yaml', 
                        type=str)
    args = parser.parse_args()

    # constants
    ACTIVATION={'ReLU':nn.ReLU()}
    OPTIMIZERS={'AdamW':AdamW, 
                'CosineAnnealing':CosineAnnealingLR}
    MINERS={'None':miners.EmbeddingsAlreadyPackagedAsTriplets()}
    LOSS={'Contrastive':losses.ContrastiveLoss,
          'NTXent':losses.NTXentLoss,
          'NCA':losses.NCALoss, 
          'SimCSE':SimCSE_Loss}
    ACCURACY={'Standard':AccuracyCalculator}

    cfg = Config.read_yaml(args.config_path)

    # proxydata = ProxyDataset(cfg)
    dataset = Metric_Dataset(cfg)
    dataset.setup()
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()
    val_loader = dataset.val_dataloader()
    
    umapper = umap.UMAP()  

    model=MetricModel(cfg, 
                    dataset.train_dataset, 
                    dataset.validation_dataset
                    )

    wandb_logger = WandbLogger(project="metric_lightning", offline=args.log_offline)
    trainer = L.Trainer(max_epochs=args.num_epochs, devices=[3], logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



