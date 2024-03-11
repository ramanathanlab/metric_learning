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

# import CrossEntropyLoss

def SimCSE_Loss(temp:int): 
    temp = 10
    h = -2*torch.randn(batch_size, 3, feature_dim)+1  # (batch dim, contrastive triplet, features dim)
    num = torch.exp(F.cosine_similarity(h[:, 0, :], h[:, 1, :], dim=1) / temp)

    norm_hi = torch.sqrt(torch.sum(torch.square(h[:, 0, :]), dim=1))
    norm_hj_plus = torch.sqrt(torch.sum(torch.square(h[:, 1, :]), dim=1))
    norm_hj_minus = torch.sqrt(torch.sum(torch.square(h[:, 2, :]), dim=1))
    sim_denom1 = torch.outer(norm_hi, norm_hj_plus) * temp
    sim_denom2 = torch.outer(norm_hi, norm_hj_minus) * temp
    v1 = h[:, 0, :] @ h[:, 1, :].t() / sim_denom1
    v2 = h[:, 0, :] @ h[:, 2, :].t() / sim_denom2
    vec_denom = torch.sum(torch.exp(v1) + torch.exp(v2), dim=1)
    return 


    
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

        self.alignment_tot = torch.tensor(0)
        self.alignment = []
        # self.uniformity_tot = torch.tensor(0)
        # self.alignment_tot=[]

    def training_step(self, batch, batch_idx):
        metric_opt = self.optimizers()
        metric_opt.zero_grad()

        X, y = batch  

        X = model(X)

        # cat all embeddings by batch, and generate their ref indices
        # from: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/686
        # embeddings=torch.cat([a_embed, p_embed, n_embed], dim=0)



        loss = self.loss_fn(embeddings, indices_tuple=(a_idx, p_idx, n_idx)) # loss calculated via embed + corr. idx

        self.manual_backward(loss) 
        metric_opt.step()
        self.log_dict({"Metric Loss:": loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        '''Calculate the accuracies so everything in the enumerate loop
        Get total accuracy and class''' 
        # TODO: get the number of correct triplets in each batch 
        X, y = batch
        match_y = y.unsqueeze(1)==y.unsqueeze(0)
        pos_pairs_idx = match_y.fill_diagonal_(0).nonzero()
        pos, anchor = X[pos_pairs_idx[0]].unsqueeze(1)

        pos_embed = self.model(pos)
        anchor_embed = self.model(anchor)

        self.alignment_tot=self.alignment_tot.type_as(pos_embed)
        # self.uniformity_tot=self.uniformity_tot.type_as(anchor_embed)


        self.alignment.append(get_alignment(pos_embed, anchor_embed))
    
        self.alignment_tot += get_alignment(pos_embed, anchor_embed)
        # self.uniformity_tot += get_uniformity(anchor_embed)


    def on_validation_epoch_end(self):
        # get all a, p, n
        half_size = int(len(self.val_dataset)*0.5)
        X, y = self.val_dataset[:half_size] # grab half of the dataset
        X = X.type_as(next(self.model.parameters()))
        y = y.type_as(next(self.model.parameters()))
        # umap_labels = np.array([0]*len(self.val_dataset) + [1]*len(self.val_dataset) + [2]*len(self.val_dataset))
        umap_labels = np.array(y.detach().cpu())
        # embeddings = self.model(X)
        # embeddings = embeddings.type_as(next(self.model.parameters()))
        proj_embeddings = self.model(X)
        # proj_embeddings = proj_embeddings.detach().cpu().numpy()
        umap_embeddings = umapper.fit_transform(proj_embeddings.detach().cpu().numpy())
        wandb_image_umap = plot_embeddings(umap_embeddings, umap_labels, self.current_epoch)
        self.logger.experiment.log({f'UMAP-Plot':wandb_image_umap})

        # evaluation metrics
        # SVD plot 
        svd, wandb_image_svd = SVD(proj_embeddings)
        self.logger.experiment.log({f'SVD-Plot':wandb_image_svd})
        self.log_dict({"SVD Embeddings:": svd}, prog_bar=True)

        # calculate alignment and uniformity 
        # for alignment, we need two embeddings that we know are similar aka x and y are positive pairs that we know exist
        # match_y = y.unsqueeze(1)==y.unsqueeze(0)
        # pos_pairs_idx = match_y.fill_diagonal_(0).nonzero()
        # pos, anchor = X[pos_pairs_idx[0]]
        # pos_embed = self.model(pos)
        # anchor_embed = self.model(anchor)

        # alignment = get_alignment(pos_embed, anchor_embed)
        self.log_dict({'Alignment:': self.alignment_tot/len(self.val_dataset)}, prog_bar=True)
        # uniformity = get_uniformity(anchor_embed)
        # self.log_dict({'Uniformity:': self.uniformity_tot/len(self.val_dataset)}, prog_bar=True)

        self.alignment_tot *= 0
        self.alignment=[]
        # self.alignment_tot = 0
        # self.uniformit_tot = 0

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
          'NCA':losses.NCALoss}
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
    trainer = L.Trainer(max_epochs=args.num_epochs, devices=[2], logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



