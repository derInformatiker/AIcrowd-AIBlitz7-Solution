import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class ImgDataset(Dataset):
    def __init__(self,df,mode,transforms = None):
        self.imageID = df['ImageID']
        self.labels = df['label']
        self.transforms = transforms
        self.mode = mode
        
    def __getitem__(self,x):
        path = self.imageID.iloc[x]
        label = float(self.labels.iloc[x])
        if self.mode == 'train':
            i = cv2.imread(f'D:/rover/ch3/data/'+str(path)+'.jpg')
        else:
            i = cv2.imread(f'D:/rover/ch3/data/{self.mode}/'+str(path)+'.jpg')
        
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        if self.transforms:
            i = self.transforms(image = i)['image']
        i = cv2.resize(i,(128,128))
        i = torch.tensor(i) / 255.0
        i = i.permute(2,0,1)
        if self.mode != 'test':
            return i, label-1
        else:
            return i
    
    def __len__(self):
        return len(self.imageID)
    
def getTrainDs(train_tr = None):
    train_df = pd.read_csv('D:/rover/ch3/data/trainAndVal.csv')
    return ImgDataset(train_df,'train',train_tr)

def getValDs(val_tr):
    val_df = pd.read_csv('D:/rover/ch3/data/val.csv')
    return ImgDataset(val_df,'val',val_tr)

def getTestDs(test_tr):
    val_df = pd.read_csv('D:/rover/ch3/data/sample_submission.csv')
    return ImgDataset(val_df,'test',test_tr)

def writeSub(p):
    test_df = pd.read_csv('D:/rover/ch3/data/sample_submission.csv')
    