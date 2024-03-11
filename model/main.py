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
from dataset import ProxyDataset
from models import MetricNet
from sklearn.decomposition import TruncatedSVD

cfg = Config.read_yaml('/homes/bhsu/gb_2024/my_gb_files/cerebras/metric_learning/path_to_output_file.yaml')

dataset = ProxyDataset(cfg)
dataset.setup()
train_loader = dataset.train_dataloader()
test_loader = dataset.test_dataloader()
val_loader = dataset.val_dataloader()

    
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
        self.loss_fn=LOSS[cfg.loss_name](**cfg.loss[cfg.loss_name])
        self.accuracy=ACCURACY[cfg.accuracy_name](**cfg.accuracy)
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.dist_correct=0 # counter that will be used in validation 
        self.total_samples=0 # counter for num_samples for validation
        self.batch_counter=0

    def training_step(self, batch, batch_idx):
        metric_opt = self.optimizers()
        metric_opt.zero_grad()

        q, a, p, n = batch # question label, anchor/question, pos/sim_doc, neg/diff_doc
        a_embed = self.model(a)
        p_embed = self.model(p)
        n_embed = self.model(n)

        # cat all embeddings by batch, and generate their ref indices
        # from: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/686
        embeddings=torch.cat([a_embed, p_embed, n_embed], dim=0)
        a_idx = torch.arange(0, self.batch_size).type_as(a).long()
        p_idx = torch.arange(self.batch_size, self.batch_size*2).type_as(p).long()
        n_idx = torch.arange(self.batch_size*2, self.batch_size*3).type_as(n).long()
        loss = self.loss_fn(embeddings, indices_tuple=(a_idx, p_idx, n_idx)) # loss calculated via embed + corr. idx

        self.manual_backward(loss) 
        metric_opt.step()
        self.log_dict({"Metric Loss:": loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        '''Calculate the accuracies so everything in the enumerate loop
        Get total accuracy and class''' 
        # TODO: get the number of correct triplets in each batch 
        # aka does the model identify d(a, n) > d(a, p) + margin?
        q, a, p, n = batch
        a_embed = self.model(a)
        p_embed = self.model(p)
        n_embed = self.model(n)
        # accuracies = self.accuracy.get_accuracy()
        dmat_ap = F.pairwise_distance(a_embed, p_embed, p=2) # 2nd norm
        dmat_an = F.pairwise_distance(a_embed, n_embed, p=2)
        dmat_diff = dmat_an - dmat_ap - self.margin
        self.dist_correct += torch.where(dmat_diff>0, 1, 0).sum()
        self.total_samples += a.size(0) 
        self.batch_counter += 1
        print("done")

    def on_validation_epoch_end(self):
        # get all a, p, n
        q, a, p, n = self.val_dataset[:]
        umap_labels = np.array([0]*len(self.val_dataset) + [1]*len(self.val_dataset) + [2]*len(self.val_dataset))

        embeddings = torch.cat([a,p,n])
        embeddings = embeddings.type_as(next(self.model.parameters()))
        proj_embeddings = self.model(embeddings)
        proj_embeddings = proj_embeddings.detach().cpu().numpy()
        umap_embeddings = umapper.fit_transform(proj_embeddings)
        wandb_image_umap = plot_embeddings(umap_embeddings, umap_labels, self.current_epoch)
        self.logger.experiment.log({f'UMAP-Plot':wandb_image_umap})

        # evaluation metrics
        # SVD plot 
        svd, wandb_image_svd = SVD(proj_embeddings)
        self.logger.experiment.log({f'SVD-Plot':wandb_image_svd})
        self.log_dict({"SVD Embeddings:": svd}, prog_bar=True)

        # calculate alignment and uniformity 
        # for alignment, we need two embeddings that we know are similar aka x and y are positive pairs that we know exist
        if 
        alignment = get_alignment(embeddings)
        self.log_dict({'Alignment:': alignment}, prog_bar=True)
        uniformity = get_uniformity(embeddings)
        self.log_dict({'Uniformity:': uniformity}, prog_bar=True)

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
        avg_correct=self.dist_correct/self.total_samples*100
        self.log('val_avg_correct', avg_correct)
        print(f"Acc of negative pair identification: {avg_correct} %")
        self.dist_correct=0
        self.total_samples=0
        self.batch_counter=0
        # run 

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
    args = parser.parse_args()

    # constants
    ACTIVATION={'ReLU':nn.ReLU()}
    OPTIMIZERS={'AdamW':AdamW, 
                'CosineAnnealing':CosineAnnealingLR}
    MINERS={'None':miners.EmbeddingsAlreadyPackagedAsTriplets()}
    LOSS={'Contrastive':losses.ContrastiveLoss,
          'NTXent':losses.NTXentLoss,
          'NCA':losses.NCALoss}
    ACCURACY={'Standard':AccuracyCalculator}

    cfg = Config.read_yaml('/homes/bhsu/gb_2024/my_gb_files/cerebras/metric_learning/path_to_output_file.yaml')

    proxydata = ProxyDataset(cfg)
    proxydata.setup()
    train_loader = proxydata.train_dataloader()
    test_loader = proxydata.test_dataloader()
    val_loader = proxydata.val_dataloader()
    
    umapper = umap.UMAP()  

    model=MetricModel(cfg, 
                      proxydata.train_dataset, 
                      proxydata.val_dataset
                      )

    wandb_logger = WandbLogger(project="metric_lightning", offline=args.log_offline)
    trainer = L.Trainer(max_epochs=args.num_epochs, devices=[1], logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



