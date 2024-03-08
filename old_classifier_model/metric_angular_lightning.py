
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


########## helper functions ##########
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

def plot_embeddings(umap_embeddings:np.array, 
                    test_labels:list, 
                    label_map:dict, 
                    epoch:int) -> wandb.Image:
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=test_labels, cmap='Spectral', s=5)
    cbar = plt.colorbar(boundaries=np.arange(len(np.unique(test_labels))+1)-0.5)
    cbar.set_ticks(np.array(np.arange(len(np.unique(test_labels)))))
    cbar.ax.set_yticklabels([v for k, v in label_map.items() if k in test_labels]) # [k for k, v in label_map.items() if v in test_labels]
    plt.title(f'UMAP Projection of the Embeddings, Epoch: {epoch}', fontsize=24)
    wandb_image = wandb.Image(plt)
    return wandb_image


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

#################### models and classes #####################


class Metric_Dataset(L.LightningDataModule): 
    def __init__(self, 
                 data_path: str, 
                 num_samples:int, 
                 train_ratio:float, 
                 val_ratio:float, 
                 batch_size:int
                 ): 
        super(Metric_Dataset, self).__init__()
        self.data_path = data_path 
        self.num_samples = num_samples 
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = int(os.cpu_count() * 0.7)

    def setup(self, stage=None): 
        pf = ParquetFile(self.data_path) 
        first_rows = next(pf.iter_batches(batch_size = self.num_samples)) 
        self.data = pa.Table.from_batches([first_rows]).to_pandas() 
        self.data["embedding"] = self.data["embedding"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        self.data["one_label"] = self.data["one_label"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        self.features_tensor = torch.stack(tuple(self.data["embedding"].values))
        self.target_tensor = torch.stack(tuple(self.data["one_label"].values))
        self.dataset = TensorDataset(self.features_tensor, self.target_tensor)
        # split into train_val_test ratio sizes
        total_size = len(self.data)
        train_size = int(self.train_ratio * total_size)
        validation_size = int(train_size * self.val_ratio)
        test_size = total_size - train_size - validation_size
        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset, 
                                                                                      [train_size, 
                                                                                       validation_size, 
                                                                                       test_size])
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, pin_memory=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, num_workers=self.num_workers, drop_last=True)
    
    def input_size(self):
        return self.features_tensor.shape[1]
    
    def num_classes(self): 
        return len(torch.unique(self.target_tensor).tolist())
    
    def return_labels(self):
        return (torch.unique(self.target_tensor)).long().tolist()
    
    

class Metric_Model(nn.Module): 
    def __init__(self, 
                 input_size:int, 
                 output_size:int, 
                 num_classes:int
                 ): 
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, self.output_size)

    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        out = F.relu(self.fc5(x))
        return out
    

########################################## Olmo Encoder Module for Token embeddings #################
class OlmoEncoder(nn.Module): 
    def __init__(self, 
                 model_name="allenai/OLMo-7B"
                 ): 
        super(OlmoEncoder, self).__init__()
        # model 
        self.model_name = model_name
        # transformer setup 
        self.model_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, self.model_config)

    def forward(self, x): 
        enc_inputs = self.tokenizer(x, return_tensors='pt', return_token_type_ids=False)
        output = self.model(**enc_inputs)
        return output


class Encoder_Classifier(L.LightningDataModule): 
    def __init__(self):
        super(Encoder_Classifier, self).__init__()
    
###################################################################


class Classifier(nn.Module): 
    def __init__(self, 
                 input_size:int, 
                 output_size:int
                 ): 
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.output_size)

    def forward(self, x): 
        out = F.relu(self.fc1(x))
        return out

class Metric_Classifier(L.LightningModule): 
    def __init__(self,
                # model input
                metric_input_size:int, 
                metric_output_size:int,
                classifier_input_size:int, 
                classifier_output_size:int,  
                num_classes:int, 
                # settings for 
                mining_fn:miners.TripletMarginMiner,
                metric_loss_fn:losses.TripletMarginLoss, 
                class_loss_fn:nn.CrossEntropyLoss, 
                # validation metrics
                train_dataset:Dataset, 
                val_dataset:Dataset, 
                accuracy_calculator:AccuracyCalculator,
                label_map:dict,
                umap_embed,
                # hyperparams 
                lr: float, 
                alpha:float
                 ):
        super().__init__()

        # model settings
        self.metric_input_size = metric_input_size
        self.metric_output_size = metric_output_size
        self.classifier_input_size = classifier_input_size
        self.classifier_output_size = classifier_output_size
        self.num_classes = num_classes
        # training materials
        self.metric_model = Metric_Model(self.metric_input_size, 
                                self.metric_output_size, 
                                self.num_classes)
        self.classifier = Classifier(self.classifier_input_size, 
                                    self.classifier_output_size)
        self.mining_fn = mining_fn
        self.metric_loss_fn = metric_loss_fn 
        self.class_loss_fn = class_loss_fn
        self.automatic_optimization = False # since we are doing two optimizers 
        self.accuracy_calculator = accuracy_calculator

        # validation materials 
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # self.accuracy_calc = accuracy_calc
        self.label_map = label_map
        self.umap_embed = umap_embed
        self.class_labels = list(label_map.values())

        # metrics
        self.multi_acc_avg = Accuracy(task="multiclass", 
                                        num_classes=self.num_classes)
        self.multi_acc_class = Accuracy(task="multiclass", 
                                        num_classes=self.num_classes, 
                                        average="none")
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.confmat_tot = torch.zeros((self.num_classes, self.num_classes))
        
        # hyperparams 
        self.lr = lr
        self.alpha = alpha
        
    def forward(self, x): 
        embedding = self.metric_model(x)
        logits = self.classifier(embedding)
        return embedding, logits
    
    def training_step(self, batch, batch_idx): 
        data, labels = batch

        metric_opt, class_opt, loss_optim = self.optimizers()
        metric_scheduler, class_scheduler, loss_scheduler = self.lr_schedulers() 
        metric_opt.zero_grad()
        class_opt.zero_grad()

        embeddings = self.metric_model(data)
        indices_tuple = self.mining_fn(embeddings, labels) 
        metric_loss = self.metric_loss_fn(embeddings, labels.long(), indices_tuple)
        logits = self.classifier(embeddings)
        class_loss = self.class_loss_fn(logits, labels.long())
        combined_loss = metric_loss + self.alpha*class_loss

        self.manual_backward(combined_loss)
        metric_opt.step()
        class_opt.step()
        loss_optim.step()

        metric_scheduler.step()
        class_scheduler.step()
        loss_scheduler.step()

        # log metrics for the whole batch 
        self.log_dict({"Metric Loss:": metric_loss, 
                       "Classifier Loss": class_loss,
                       "Combined Loss": combined_loss, 
                       "Number Triplets Mined": self.mining_fn.num_triplets}, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        '''Calculate the accuracies so everything in the enumerate loop
        Get total accuracy and class''' 
        # get total classifier accuracy 
        data, labels = batch
        embeddings = self.metric_model(data)
        logits = self.classifier(embeddings)
        multi_acc_avg = self.multi_acc_avg(logits, labels)
        self.log_dict({"Multi-Class Accuracy Average":multi_acc_avg}, 
                on_epoch=True, prog_bar=True)

        multi_acc_per = self.multi_acc_class(logits, labels)
        for idx, metric_value in enumerate(multi_acc_per):  # Iterate over tensor elements.
            self.log(f"Acc-by-Class: {self.class_labels[idx]}", metric_value, on_epoch=True)

        _, preds = torch.max(logits, 1)
        self.confmat_tot = self.confmat_tot.type_as(data)
        cm = self.confmat(preds, labels)
        self.confmat_tot += cm
        
    def on_validation_epoch_end(self):
        # plotting accumulated confusion matrix 
        wanb_image_cm = plot_confusion_matrix(self.confmat_tot, self.label_map, self.current_epoch)
        self.logger.experiment.log({'Confusion-Matrix:': wanb_image_cm})
        self.confmat_tot *= 0
        # grabbing all embeddings for KNN retrieval acc
        train_embeddings, train_labels = get_all_embeddings(self.train_dataset, self.metric_model)
        val_embeddings, val_labels = get_all_embeddings(self.val_dataset, self.metric_model)
        train_labels = train_labels.squeeze(1)
        val_labels = val_labels.squeeze(1)
        accuracies = self.accuracy_calculator.get_accuracy(
            val_embeddings, val_labels, train_embeddings, train_labels, False
        )
        accuracies_dict = map_class_accuracies(self.class_labels, accuracies, self.label_map)
        self.logger.experiment.log({'KNN-Accuracies':accuracies_dict})

        # UMAP embeddings
        train_labels = train_labels.detach().cpu().numpy()
        val_labels = val_labels.detach().cpu().numpy()
        train_embeddings = train_embeddings.detach().cpu().numpy()
        val_embeddings = val_embeddings.detach().cpu().numpy()
        umap_embeddings = self.umap_embed.fit_transform(val_embeddings)
        wandb_image_umap = plot_embeddings(umap_embeddings, val_labels, self.label_map, self.current_epoch)
        self.logger.experiment.log({f'UMAP-Plot':wandb_image_umap})
     
    def configure_optimizers(self): 
        '''1 each opt/sched for metric_model, class_model, and cosine_loss func'''
        metric_optim = torch.optim.AdamW(self.metric_model.parameters(), lr=self.lr)
        class_optim = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)
        loss_optim = torch.optim.AdamW(self.metric_loss_fn.parameters(), lr=self.lr)

        metric_scheduler = CosineAnnealingLR(optimizer=metric_optim, T_max=10) # restart annealing every 10 steps
        class_scheduler = CosineAnnealingLR(optimizer=class_optim, T_max=10)
        loss_scheduler = CosineAnnealingLR(optimizer=loss_optim, T_max=10)
        return (
            {
                "optimizer":metric_optim, 
                "lr_scheduler":{
                    "scheduler":metric_scheduler,
                    "monitor":"metric_to_track"
                }, 
            },
            {
                "optimizer":class_optim, 
                "lr_scheduler":{
                    "scheduler":class_scheduler, 
                    "monitor":"metric_to_track"
                }
            },
            {
                "optimizer":loss_optim, 
                "lr_scheduler":{
                    "scheduler":loss_scheduler, 
                    "monitor":"metric_to_track"
                }
            }
        )


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="pytorch run")
    parser.add_argument('--num_epochs', default=15, type=int)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--num_samples', default=100000, type=int)
    parser.add_argument('--data_path', 
                        #"/lus/eagle/projects/RL-fold/bhsu/gb_files/metric_learn/arxiv_emb_processed_multi.parquet"
                        default="/lambda_stor/homes/bhsu/gb_2024/my_gb_files/metric_learn/data/arxiv_emb_processed_multi.parquet", 
                        type=str)
    parser.add_argument('--val_ratio', default=0.15, type=float)
    parser.add_argument('--train_ratio', default=0.7, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--embed_ratio', default=0.75, type=float)
    parser.add_argument('--num_devices', default=1, type=int)
    parser.add_argument('--log_offline', default=False, type=bool)
    args = parser.parse_args()

    datamodule = Metric_Dataset(data_path=args.data_path, 
                         num_samples=args.num_samples, 
                         train_ratio=0.7,
                         val_ratio=0.15, 
                         batch_size=args.batch_size)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # metric + classifier model setup 
    reducer = reducers.ThresholdReducer(low=0)
    metric_loss_fn = losses.ArcFaceLoss(num_classes=datamodule.num_classes(), embedding_size=128, margin=28.6, scale=64)
    mining_fn = miners.AngularMiner(angle=20)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",
                                                    "mean_average_precision"), 
                                                    k=1, 
                                                    return_per_class=True)

    class_loss_fn = nn.CrossEntropyLoss()
    umapper = umap.UMAP()   
    label_map = dict(zip(datamodule.data["one_label"].apply(int), datamodule.data["one_cat"]))

    metric_classifier = Metric_Classifier(metric_input_size=768, 
                                          metric_output_size=128, 
                                          classifier_input_size=128,
                                          classifier_output_size=len(datamodule.return_labels()),
                                          num_classes=len(datamodule.return_labels()),
                                          mining_fn=mining_fn,
                                          metric_loss_fn=metric_loss_fn,
                                          class_loss_fn=class_loss_fn,
                                          train_dataset=datamodule.train_dataset, 
                                          val_dataset=datamodule.validation_dataset,
                                          accuracy_calculator=accuracy_calculator,
                                          label_map=label_map,
                                          umap_embed=umapper,
                                          lr=args.lr,
                                          alpha=0.4
                                          )
    
    wandb_logger = WandbLogger(project="metric_lightning", offline=args.log_offline)

    # trainer setup 
    # list(range(args.num_devices))
    trainer = L.Trainer(max_epochs=args.num_epochs, devices=[1, 2, 3, 4, 5, 6], logger=wandb_logger)
    trainer.fit(model=metric_classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)

    