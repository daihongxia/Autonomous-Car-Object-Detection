#!pip install efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils
from tqdm import tqdm
import gc
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

def get_mesh(batch_size, shape_x, shape_y, device=torch.device("cuda")):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh

def set_dropout(model, drop_rate):
    # source: https://discuss.pytorch.org/t/how-to-increase-dropout-rate-during-training/58107/4
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
            print("name:", name)
            print("children:\n", child)


def effnet_dropout(drop_rate, ver='b2'):
    base_model0 = EfficientNet.from_pretrained(f"efficientnet-{ver}")
    set_dropout(base_model0, drop_rate)
    return base_model0


class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    
    def __init__(self, n_classes, 
                 dropout_rate=0.02, 
                 batch_size=8, 
                 img_w=2048, 
                 device=torch.device("cuda"),
                 ver='b2'):
        super(MyUNet, self).__init__()
        
        self.drop_rate = dropout_rate
        self.base_model = effnet_dropout(drop_rate = self.drop_rate, ver=ver)
        
        #self.base_model = EfficientNet.from_pretrained('efficientnet-'+ver)
        
        self.batch_size = batch_size
        self.img_w = img_w
        self.device=device
        
        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        
        self.mp = nn.MaxPool2d(2)
        if ver=='b0':
            self.up1 = up(1282 + 1024, 512)
        elif ver=='b2':
            self.up1 = up(1410 + 1024, 512)

        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        
        x_center = x[:, :, :, self.img_w // 8: -self.img_w // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(self.device)
        feats = torch.cat([bg, feats, bg], 3)
        
        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
        
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x
    

def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss

def criter_objective(prediction, mask, regr, mask_gaus, 
                     w_mask = 0.1, w_regr = 0.9, gamma=2, 
                     size_average=True, 
                     device=torch.device("cuda")
                    ):
    
    w_mask = torch.tensor(w_mask, device=device)
    w_regr = torch.tensor(w_regr, device=device)
    
    pred_mask = torch.sigmoid(prediction[:, 0])
    pos_mask = (mask_gaus == 1)
    neg_mask = (mask_gaus < 1)
    pos_loss = -torch.log(pred_mask + 1e-12) * pos_mask * torch.pow(1 - pred_mask, gamma)
    neg_loss = -torch.log(1-pred_mask + 1e-12) * torch.pow(pred_mask, gamma) * neg_mask * torch.pow(1-mask_gaus, 4)
    mask_loss = pos_loss+neg_loss    
    mask_loss = mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = w_mask * mask_loss + w_regr * regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss

class car_detector():
    
    def __init__(self, 
                 model, 
                 train_dataset,
                 dev_dataset,
                 device = torch.device("cuda"), 
                 optimizer=optim.AdamW,
                 lr=0.01,
                 n_epoch = 12, 
                 batch_size = 4,
                 w_mask = 0.1,
                 history=None,
                 PATH='../model/'
                ):
        self.PATH = PATH
        self.model = model
        self.device = device
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.n_epoch = n_epoch
        
        self.batch_size = batch_size
        self.w_mask = w_mask

        # Create data generators - they will produce batches
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        #self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, 
        #                                            step_size=max(n_epoch, 10) * len(self.train_loader) // 3,
        #                                            gamma=0.1)
        
        self.exp_lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[7,10,12,14,17,20], gamma=0.5)
        
        if history is None:
            self.history = pd.DataFrame()
        else:
            self.history = history
        
    def train_model(self, epoch, history=None):
        
        model = self.model
        train_loader = self.train_loader
        device= self.device
        optimizer = self.optimizer
        exp_lr_scheduler = self.exp_lr_scheduler
        batch_size = self.batch_size
        w_mask = self.w_mask
        w_regr = 1.-w_mask
        
        model.train()
        
        for batch_idx, (img_batch, mask_batch, regr_batch, mask_gaus_batch, _) in enumerate(tqdm(train_loader)):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            mask_gaus_batch = mask_gaus_batch.to(device)
        
            optimizer.zero_grad()
            output = model(img_batch)
            
            loss = criter_objective(output, 
                                    mask_batch, 
                                    regr_batch, 
                                    mask_gaus_batch, 
                                    w_mask = w_mask, 
                                    w_regr = w_regr,
                                    device=device)
            #if history is not None:
            #    history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        exp_lr_scheduler.step()
    
        print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
            epoch,
            optimizer.state_dict()['param_groups'][0]['lr'],
            loss.data))
        if history is not None:
            history.loc[epoch,'train_loss'] = loss.data.cpu().numpy()
        
    def evaluate_model(self, epoch, history=None):
        
        model = self.model
        dev_loader = self.dev_loader
        device= self.device
        batch_size = self.batch_size
        w_mask = self.w_mask
        w_regr = 1.-w_mask
        
        model.eval()
        total_loss = torch.tensor(0., requires_grad=False)
        total_mask_loss = torch.tensor(0., requires_grad=False)
        total_regr_loss = torch.tensor(0., requires_grad=False)
    
        with torch.no_grad():
            for img_batch, mask_batch, regr_batch, _, _ in tqdm(dev_loader):
                img_batch = img_batch.to(device)
                mask_batch = mask_batch.to(device)
                regr_batch = regr_batch.to(device)
            
                output = model(img_batch)
                val_loss, mask_loss, regr_loss = criterion(output, 
                                                           mask_batch, 
                                                           regr_batch, 
                                                           size_average=False)
                total_loss += val_loss.data
                total_mask_loss += mask_loss.data
                total_regr_loss += regr_loss.data
    
        total_loss /= len(dev_loader.dataset)
        total_mask_loss /= len(dev_loader.dataset)
        total_regr_loss /= len(dev_loader.dataset)
    
        if history is not None:
            history.loc[epoch, 'dev_loss'] = total_loss.cpu().numpy()
            history.loc[epoch, 'dev_mask_loss'] = total_mask_loss.cpu().numpy()
            history.loc[epoch, 'dev_regr_loss'] = total_regr_loss.cpu().numpy()
        
        best_val_loss = np.min(history['dev_loss'].dropna().values)
        
        history.loc[epoch, 'best_val_loss'] = best_val_loss
        
        #print('Dev loss: {:.4f}'.format(loss))
        
        print("\n=========================")
        print(f"epoch={epoch}")
        print(f"VAL LOSS: {total_loss:.2f}, \tBest Val Loss: {best_val_loss:.2f}")
        print(f"mask_val_loss: {total_mask_loss:.2f}, \tregr_val_loss: {total_regr_loss:.2f}")
        
        
        return total_loss, total_mask_loss, total_regr_loss
    
    def fit(self, start_epoch=0):
        
        if 'dev_loss' in self.history:
            prev_val_loss = np.min(self.history['dev_loss'].dropna().values)
        else:
            prev_val_loss = 9999.
        
        not_better_times = 0
        PATH = self.PATH
        for epoch in range(start_epoch, self.n_epoch+start_epoch):
            
            name='model_'+str(epoch)+'epoch.pth'

            if not_better_times >= 3:
                break
                
            torch.cuda.empty_cache()
            gc.collect()
            self.train_model(epoch, self.history)
            
            new_total_loss, new_mask_loss, new_regr_loss = self.evaluate_model(epoch, self.history)
            
            if new_total_loss < prev_val_loss:
                print('New best model obtained!')
                self.save_model(name=name, PATH='../model/')
                prev_val_loss = new_total_loss
            else:
                not_better_times += 1
               
            self.history.to_csv(PATH+name+'_history.csv')
                
    
    def save_model(self, name='model.pth', PATH='../model/'):
        torch.save(self.model.state_dict(), PATH+name)
        
    def view_train_loss(self):
        self.history['train_loss'].iloc[100:].plot()
        
    def view_dev_loss(self):
        series = self.history.dropna()['dev_loss']
        plt.scatter(series.index, series);
        
    def predict_image(self, img):
        output = self.model(torch.tensor(img[None]).to(device))
        logits = output[0,0].data.cpu().numpy()
        