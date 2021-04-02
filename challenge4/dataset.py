import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class ImgDataset(Dataset):
    def __init__(self,df,mode,transforms = None,test=False):
        self.imageID = df['ImageID']
        self.labels = df['label']
        self.transforms = transforms
        self.mode = mode
        self.test = test
        
    def __getitem__(self,x):
        path = self.imageID.iloc[x]
        label = float(self.labels.iloc[x])
        i = cv2.imread(f'data/{self.mode}/'+str(path)+'.jpg')
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            i = self.transforms(image = i)['image']
            
        i = torch.tensor(i) / 255.0
        #.unsqueeze_(-1)
        i = i.permute(2,0,1)
        if not self.test:
            return i, label
        else:
            return i
    
    def __len__(self):
        return len(self.imageID)
    
def getTrainDs(train_tr = None):
    train_df = pd.read_csv('data/labels.csv')
    return ImgDataset(train_df,'train',train_tr)

def getTestDs(train_tr = None):
    train_df = pd.read_csv('data/labels.csv')
    return ImgDataset(train_df,'train',train_tr,True)

def writeSub(p):
    test_df = pd.read_csv('data/sample_submission.csv')
    test_df['label']=p
    test_df.to_csv('submission.csv',index = False)
    