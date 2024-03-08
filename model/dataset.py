import pyarrow
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from config import Config, BaseConfig
import pandas as pd, numpy as np

class ProxyDataset(L.LightningDataModule): 
    def __init__(self, 
                #  data_path:str, 
                #  num_samples:int,
                #  batch_size:int,
                #  train_ratio:float,
                #  val_ratio:float
                 cfg:Config
                 ) -> L.LightningDataModule: 
        super().__init__()
        self.data_path=cfg.data_path
        self.num_samples=cfg.num_samples
        self.batch_size=cfg.batch_size
        self.train_ratio=cfg.train_ratio
        self.val_ratio=cfg.val_ratio

    def setup(self, stage=None):
        questions=torch.rand(self.num_samples)
        q_docs = [torch.rand(768) for _ in range(len(questions))]
        sim_docs = [torch.rand(768) for _ in range(len(questions))]
        diff_docs = [torch.rand(768) for _ in range(len(questions))]
        data = pd.DataFrame({
        'questions': questions.numpy(), 
        'q_docs': q_docs, # Convert tensor to numpy for DataFrame
        'sim_docs': sim_docs,
        'diff_docs': diff_docs
          })
        questions_tensor = torch.tensor(data['questions'].to_list())
        q_docs_tensor = torch.stack(data['sim_docs'].to_list())
        sim_docs_tensor = torch.stack(data['sim_docs'].to_list()) 
        diff_docs_tensor = torch.stack(data['diff_docs'].to_list())
        self.dataset = TensorDataset(questions_tensor, q_docs_tensor, sim_docs_tensor, diff_docs_tensor)

        total_size = len(self.dataset)
        train_size = int(self.train_ratio*total_size)
        valid_size = int(self.val_ratio*train_size)
        test_size = total_size - train_size - valid_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, 
                                                                                                valid_size, 
                                                                                                test_size])
        
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True)
    
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True)