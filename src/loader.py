import numpy as np
from preprocessing import preprocess_image, get_mask_and_regr, add_number_of_cars, remove_out_image_cars
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 
import pandas as pd
import torch
import cv2
import os

def load_data(PATH='../../data/pku-autonomous-driving/'):
    train = pd.read_csv(PATH + 'train.csv')
    test = pd.read_csv(PATH + 'sample_submission.csv')
    return train, test

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

def imread_id(img_id, PATH='../../data/pku-autonomous-driving/', fast_mode=False):
    path = PATH + 'train_images/img_id' + '.jpg'
    return imread(path, fast_mode=fast_mode)


class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, 
                 dataframe, 
                 root_dir,
                 root_dir_dropmasks='../../data/pku-autonomous-driving/test_masks/{}.jpg', 
                 training=True, 
                 transform=None,
                 img_w=2048,
                 img_h=512):
        
        self.df = dataframe
        self.root_dir = root_dir
        self.root_dir_dropmasks = root_dir_dropmasks
        self.transform = transform
        self.training = training
        self.img_w = img_w
        self.img_h = img_h

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels, _ = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1
        
        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, img_w=self.img_w, img_h=self.img_h, flip=flip)
        img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        
        if self.training:
            mask, regr, mask_gaus = get_mask_and_regr(img0, 
                                                      labels, 
                                                      img_w=self.img_w, 
                                                      img_h=self.img_h,
                                                      flip=flip)
            regr = np.rollaxis(regr, 2, 0)
            dropmask = 0
        else:
            mask, regr, mask_gaus = 0, 0 ,0
            dropmask_name = self.root_dir_dropmasks.format(idx)
            if os.path.isfile(dropmask_name):
                dropmask = imread(dropmask_name, True)
                dropmask = preprocess_image(dropmask, self.img_w, self.img_h)
            else:
                dropmask = np.zeros((self.img_h, self.img_w, 3))
                
        img = torch.as_tensor(img, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        mask_gaus = torch.as_tensor(mask_gaus, dtype=torch.float32)
        regr = torch.as_tensor(regr, dtype=torch.float32)
        dropmask = torch.as_tensor(dropmask, dtype=torch.float32)
        
        return [img, mask, regr, mask_gaus, dropmask]
    
def train_dev_test(PATH = '../../data/pku-autonomous-driving/', test_size = 0.2, 
                   random_state=47, remove_out_image=True):
    
    train, test = load_data(PATH)
    
    if remove_out_image:
        train = remove_out_image_cars(train)
    else:
        train = add_number_of_cars(train)
    
    df_test = add_number_of_cars(test)
    
    train_images_dir = PATH + 'train_images/{}.jpg'
    test_images_dir = PATH + 'test_images/{}.jpg'
    df_train, df_dev = train_test_split(train, test_size=test_size, random_state=random_state)

    # Create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, training=True)
    dev_dataset = CarDataset(df_dev, train_images_dir, training=True)
    test_dataset = CarDataset(df_test, test_images_dir, training=False)
    
    return train, test, train_dataset, dev_dataset, test_dataset

if __name__=="__main__":
    train, test, train_dataset, dev_dataset, test_dataset=train_dev_test(PATH = '../data/pku-autonomous-driving/')
    print(train_dataset[0])