import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-load_model', default='./models/rgb_charades.pt',type=str)
parser.add_argument('-root',default='/home/r/renpengzhen/Datasets/Charades_v1_rgb/Charades_v1_rgb', type=str)
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-save_dir', default='./save_model',type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset


def run(max_steps=64e3, mode='', root='', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)]) #对视频数据进行中心裁剪，大小为224

    #训练数据集的加载
    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #测试集的加载
    val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True) #测试集的视频个数1863

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model)) #加载模型参数
    i3d.cuda()

    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode,模型已经训练好了，直接调用其中的参数即可
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        i = 0
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name = data #inputs: torch.Size([1, 3, 348, 100, 100])
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                continue

            b,c,t,h,w = inputs.shape


            ts = 1600 #原版为1600
            if t > ts:
                features = []
                for start in range(1, t-56, ts):
                    end = min(t-1, start+ts+56)
                    start = max(1, start-48)
                    with torch.no_grad():
                        ip = Variable(torch.from_numpy(inputs.cpu().numpy()[:,:,start:end]).cuda())
                    #提取特征
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                with torch.no_grad():
                    inputs = Variable(inputs.cuda())
                features = i3d.extract_features(inputs) #使用i3d提取特征
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
            if i % 10 == 0:
                print('finished: {}'.format(i))
            i += 1





if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_dir=args.save_dir, load_model=args.load_model)
