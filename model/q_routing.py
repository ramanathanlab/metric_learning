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
from utils import plot_embeddings, SVD, get_alignment, get_uniformity, get_pos_anchor, plot_embeddings_nolabels
from dataset import ProxyDataset, Metric_Dataset
from models import MetricNet
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN


# General Plan: 
# 1.) Dataset Structure: 
# Columns: q_1 embeddings, q_2 embeddings, indicator: q_1==q_2 (1 if yes, 0 if no)?

# pytorch metric learning works on this: 
# assign labels q_1 = class 1, q_2 = class_2 
# pass this into the miner, which expects embeddings + "class labels"
# hard negative mining 

# loss function will be NTXent or Contrastive 
    
    
class MetricModel(L.LightningModule):
    def __init__(self,
                 cfg:Config,
                 train_dataset:Dataset,
                 validation_dataset:Dataset
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
        self.validation_dataset=validation_dataset
        self.cluster=DBSCAN(eps=0.5, min_samples=10)

        self.alignment_list = []
        self.uniformity_list = []
        self.SVD_list = []

    def training_step(self, batch, batch_idx):
        metric_opt = self.optimizers()
        metric_opt.zero_grad()
        # assumptions: 
        # - we only compare x1[i] == x2[i] 
        # - 1 is pos, 0 is neg
        # - we do not compare anchors with each other aka no x1[i] with x1[j], i!=j
        q, x1, x2, labels = batch
        pos_idx = torch.where(labels==1)[0]
        neg_idx = torch.where(labels==0)[0]
        idx_tuple = (pos_idx, pos_idx, neg_idx, neg_idx) # I match by row of tensors for pos, neg pairs
        e1 = self.model(x1)
        e2 = self.model(x2)

        loss = self.loss_fn(e1, labels, idx_tuple, ref_emb=e2, ref_labels=labels)
        self.manual_backward(loss)
        metric_opt.step()
        self.log_dict({"Training Loss:": loss}, prog_bar=True)
    

    def validation_step(self, batch, batch_idx):
        q, x1, x2, labels = batch
        pos_idx = torch.where(labels==1)[0]
        neg_idx = torch.where(labels==0)[0]
        idx_tuple = (pos_idx, pos_idx, neg_idx, neg_idx)
        e1 = self.model(x1)
        e2 = self.model(x2)
        loss = self.loss_fn(e1, labels, idx_tuple, ref_emb=e2, ref_labels=labels)
        self.log_dict({"Validation Loss:": loss}, prog_bar=True)

        e_anchor = e1[pos_idx]
        e_pos = e2[pos_idx]
        self.alignment_list.append(get_alignment(e_anchor, e_pos))
        self.uniformity_list.append(get_uniformity(torch.cat((e_anchor, e_pos), dim=0)))
        # SVD_batch, SVD_image = SVD()
        # self.SVD_list.append()
        # '''Calculate the accuracies so everything in the enumerate loop
        # Get total accuracy and class''' 
        # # X, y = batch
        # q, a, p_n, l = batch
        # # calculate validation loss 
        # self.model.train() # need this to activate dropout
        # h = self.model(X)
        # h_plus = self.model(X)
        # loss = self.loss_fn(h, h_plus)
        # self.log_dict({'Validation Loss:':loss}, prog_bar=True)
        # self.model.eval() # reset to eval mode for other validation metrics

        # # I do this to get pos/anchor pairs based on index matching
        # pos, anchor = get_pos_anchor(X, y)
        # pos_embed = self.model(pos)
        # anchor_embed = self.model(anchor)

        # self.alignment_list.append(get_alignment(pos_embed, anchor_embed)) # how well to features of pos pairs align?
        # self.uniformity_list.append(get_uniformity(h)) # h or h_plus okay, we are testing for feature scattering
    

    def on_validation_epoch_end(self):

        half_size = int(len(self.validation_dataset)*0.5)
        q, x_1, x_2, labels = self.validation_dataset[:half_size] # grab half of the dataset
        x_1 = x_1.type_as(next(self.model.parameters()))
        x_2 = x_2.type_as(next(self.model.parameters()))

        cat_embed = self.model(torch.cat((x_1, x_2), dim=0))
        umap_embed = umapper.fit_transform(cat_embed.detach().cpu().numpy())
        wandb_image_umap = plot_embeddings_nolabels(umap_embed, self.current_epoch) # plot just embeddings
        self.logger.experiment.log({f'UMAP-Plot-Epoch{self.current_epoch}':wandb_image_umap})
        print("Done")

        # perform kmeans on the metric embeddings and use those soft labels for UMAP
        cluster_labels = self.cluster.fit_predict(cat_embed.detach().cpu().numpy())
        wandb_image_cluster = plot_embeddings(umap_embed, cluster_labels, self.current_epoch)
        self.logger.experiment.log({f'UMAP-Cluster-Plot-Epoch{self.current_epoch}':wandb_image_cluster})

        # plotting alignment and uniformity metrics
        alignment_avg = torch.mean(torch.stack(self.alignment_list))
        self.log_dict({'Average Alignment:': alignment_avg})

        uniformity_avg = torch.mean(torch.stack(self.uniformity_list))
        self.log_dict({'Average Uniformity:': uniformity_avg})





        # # plotting UMAP
        # umap_labels = np.array([0]*len(self.validation_dataset) + [1]*len(self.validation_dataset) + [2]*len(self.val_dataset))
        # umap_labels = np.array(y.detach().cpu())
        # h = self.model(X)
        # umap_embeddings = umapper.fit_transform(h.detach().cpu().numpy())
        # wandb_image_umap = plot_embeddings(umap_embeddings, umap_labels, self.current_epoch)
        # self.logger.experiment.log({f'UMAP-Plot':wandb_image_umap})

        # # SVD plot 
        # svd, wandb_image_svd = SVD(h)
        # self.logger.experiment.log({f'SVD-Plot':wandb_image_svd})
        # self.log_dict({"SVD Embeddings:": svd}, prog_bar=True)

        # # alignment plot 
        # alignment_avg = torch.mean(torch.stack(self.alignment_list))
        # self.log_dict({'Average Alignment:': alignment_avg})

        # # uniformity plot 
        # uniformity_avg = torch.mean(torch.stack(self.uniformity_list))
        # self.log_dict({'Average Uniformity:': uniformity_avg})

        # self.alignment_list=[]
        # self.uniformity_list=[]

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
          'SimCSE':losses.NTXentLoss}
    ACCURACY={'Standard':AccuracyCalculator}

    cfg = Config.read_yaml(args.config_path)

    # proxydata = ProxyDataset(cfg)
    dataset = ProxyDataset(cfg)
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
