import pyarrow
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import pandas as pd, numpy as np
from pyarrow.parquet import ParquetFile
import pyarrow as pa, os
from datasets import load_dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer

if __name__=="__main__":
    data=pd.read_csv('QQP_questions_400k.csv')
    data=data[['question1', 'question2', 'is_duplicate']]
    data['question1'] = data['question1'].astype(str)
    data['question2'] = data['question2'].astype(str)
    # data[['q1_embed', 'q2_embed']]=data[['question1', 'question2']]
    data['is_duplicate']=data['is_duplicate'].apply(lambda x: np.array(x, dtype=np.float32))
    # dataset=TensorDataset(data)
    # dataloader=DataLoader(dataset, batch_size=10000)

    model=SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

    # data['q1_embed']=data['q1_embed'].apply(lambda x: model.encode(x))
    # data['q2_embed']=data['q2_embed'].apply(lambda x: model.encode(x))
    for i, row in data.iterrows(): 
        row['q1_embed']=model.encode(row['question1'])
        row['q2_embed']=model.encode(row['question2'])
    # for i, row in data.iterrows(): 
    #     row['q1_embed']=model.encode(row['q1_embed'])
    #     row['q2_embed']=model.encode(row['q2_embed'])
    
    # print("test done")

    data.to_parquet('./QQP_questions_400k.parquet')

    




    # for i in enumerate(dataloader): 
        


