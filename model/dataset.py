import pyarrow
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from config import Config, BaseConfig
import pandas as pd, numpy as np
from pyarrow.parquet import ParquetFile
from config import MetricConfig
import pyarrow as pa, os
from datasets import load_dataset
from pathlib import Path

# helper function for loading QQP dataset
def load_qqp(p = Path('/eagle/projects/argonne_tpc/siebenschuh/metric_data/QQP/QQP_questions_400k.csv')):
    """Load QQP dataframe from path p
    """
    show_limited_example = True
    qqp_path = Path(p)

    df_qqp = pd.read_csv(qqp_path)
    assert qqp_path.is_file(), "File path `qqp_path` invalid"
    return df_qqp

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
        # q_docs = [torch.rand(768) for _ in range(len(questions))]
        q1_embed = [torch.rand(768) for _ in range(len(questions))]
        q2_embed = [torch.rand(768) for _ in range(len(questions))]
        label = np.random.choice([0, 1], size=(len(questions))).tolist() # 1 if similar, 0 if no

        data = pd.DataFrame({
        'questions': questions.numpy(), 
        'q1_embed': q1_embed, 
        'q2_embed': q2_embed, 
        'label': label
          })
        questions_tensor = torch.tensor(data['questions'].to_list())
        q1_embed_tensor = torch.stack(data['q1_embed'].to_list())
        q2_embed_tensor = torch.stack(data['q2_embed'].to_list()) 
        label_tensor = torch.tensor(data['label'])
        self.dataset = TensorDataset(questions_tensor, q1_embed_tensor, q2_embed_tensor, label_tensor)

        total_size = len(self.dataset)
        train_size = int(self.train_ratio*total_size)
        valid_size = int(self.val_ratio*train_size)
        test_size = total_size - train_size - valid_size

        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset, [train_size, 
                                                                                                valid_size, 
                                                                                                test_size])
        
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True)
    
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, drop_last=True)
    

class Question_Dataset(L.LightningDataModule): 
    def __init__(self, 
                 cfg:MetricConfig
                 ): 
        super(Question_Dataset, self).__init__()
        self.data_path=cfg.data_path
        self.num_samples=cfg.num_samples
        self.train_ratio=cfg.train_ratio
        self.val_ratio=cfg.val_ratio

    def setup(self, stage=None): 
        data=load_dataset('code_x_glue_cc_clone_detection_big_clone_bench')


        questions=torch.rand(self.num_samples)
        q_1 = [torch.rand(768) for _ in range(len(questions))]
        q_2 = [torch.rand(768) for _ in range(len(questions))]
        data = pd.DataFrame({
        'questions': questions.numpy(), 
        'question_1':q_1, 
        'question_2':q_2
          })
        
        q1_tensor = torch.stack(data['question_1'].to_list()) 
        q2_tensor = torch.stack(data['question_2'].to_list())
        self.dataset = TensorDataset(questions, q1_tensor, q2_tensor)

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


class Metric_Dataset(L.LightningDataModule): 
    def __init__(self, 
                 cfg:MetricConfig
                 ): 
        super(Metric_Dataset, self).__init__()
        self.data_path = cfg.data_path 
        self.num_samples = cfg.num_samples 
        self.train_ratio = cfg.train_ratio
        self.val_ratio = cfg.val_ratio
        self.batch_size = cfg.batch_size
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
    

if __name__=="__main__": 
    # data_path = Path('QQP_questions_400k.csv')
    # data=pd.read_csv(data_path)
    data=load_dataset('code_x_glue_cc_clone_detection_big_clone_bench')





    print('Done')