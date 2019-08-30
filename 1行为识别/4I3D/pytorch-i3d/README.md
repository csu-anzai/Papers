https://github.com/piergiaj/pytorch-i3d

# I3D models trained on Kinetics

## Overview

This repository contains trained models reported in the paper "[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew Zisserman.

This code is based on Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d). Including PyTorch versions of their models.

## Note
This code was written for 

- PyTorch 0.3.
-  Version 0.4 and newer may cause issues.


# Fine-tuning and Feature Extraction
We provide code to extract I3D features and fine-tune I3D for charades. Our fine-tuned models on **charades** are also available in the models director (in addition to Deepmind's trained models). The deepmind pre-trained models were converted to PyTorch and give identical results (flow_imagenet.pt and rgb_imagenet.pt). These models were pretrained on imagenet and kinetics (see [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) for details). 

## Fine-tuning I3D
[train_i3d.py](train_i3d.py) contains the code to fine-tune I3D based on the details in the paper and obtained from the authors. Specifically, this version follows the settings to fine-tune on the [Charades](allenai.org/plato/charades/) dataset based on the author's implementation that won the Charades 2017 challenge. Our fine-tuned RGB and Flow I3D models are available in the model directory (rgb_charades.pt and flow_charades.pt).

This relied on having the optical flow and RGB frames extracted and saved as images on dist. [charades_dataset.py](charades_dataset.py) contains our code to load video segments for training.

## Feature Extraction
[extract_features.py](extract_features.py) contains the code to load a pre-trained I3D model and extract the features and save the features as numpy arrays. The [charades_dataset_full.py](charades_dataset_full.py) script loads an entire video to extract per-segment features.

# Problems

## 数据集：root='/ssd/Charades_v1_rgb' #23

https://allenai.org/plato/charades/
http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar

Charades-Ego is dataset composed of 7860 videos of daily indoors activities collected through Amazon Mechanical Turk recorded from both third and first person. The dataset contains 68,536 temporal annotations for 157 action classes.

charads - ego是由亚马逊土耳其机械公司(Amazon Mechanical Turk)收集的7860个日常室内活动视频组成的数据集，分别来自第三人称和第一人称。数据集包含了157个action类的68,536个时态注释。

```latex
###########################################################
Charades_v1_rgb.tar
###########################################################
These frames were extracted at 24fps using the following ffmpeg call for each video in the dataset:

line=pathToVideo
MAXW=320
MAXH=320
filename=$(basename $line)
ffmpeg -i "$line" -qscale:v 3 -filter:v "scale='if(gt(a,$MAXW/$MAXH),$MAXW,-1)':'if(gt(a,$MAXW/$MAXH),-1,$MAXH)',fps=fps=24" "/somepath/${filename%.*}/${filename%.*}_%0d.jpg";

The files are stored as Charades_v1_rgb/id/id-000000.jpg where id is the video id and 000000 is the number of the frame at 24fps.
```

Charades is dataset composed of **9848** videos of daily indoors activities collected through Amazon Mechanical Turk. 267 different users were presented with a sentence, that includes objects and actions from a fixed vocabulary, and they recorded a video acting out the sentence (like in a game of [Charades](https://en.wikipedia.org/wiki/Charades)). The dataset contains 66,500 temporal annotations for **157 action classes**, 41,104 labels for 46 object classes, and 27,847 textual descriptions of the videos. This work was presented at [ECCV2016](http://www.eccv2016.org/).

### extract_features.py

- 数据集加载：


```python
#测试集
mdata = make_dataset(split_file, split, root, mode, num_classes=157)
len(mdata)
Out[49]: 1863
#总的数据集的个数
data.__len__()
Out[50]: 9848
#训练集的个数
mdata_training = make_dataset(split_file, 'training', root, mode, num_classes=157)
len(mdata_training)
Out[52]: 7985
#验证
len(mdata)+len(mdata_training)
Out[53]: 9848
```
- 测试

```python
root = '/home/r/renpengzhen/Datasets/Charades_v1_rgb/Charades_v1_rgb'
mode = 'rgb'
save_dir = './save_model'
split = 'charades/charades.json'
batch_size = 1
load_model = './models/rgb_charades.pt'

import os
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
# setup dataset
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)]) #对视频数据进行中心裁剪，大小为224

#训练数据集的加载
dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

#测试集的加载
val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True) #测试集的视频个数1863

dataloaders = {'train': dataloader, 'val': val_dataloader}
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(157)
i3d.load_state_dict(torch.load(load_model)) #加载模型参数
i3d.cuda()
```

```python
inputs=torch.rand(1,3,1024,224,224)
t = 1024
ts = 100 #原版为1600
if t > ts:
    features = []
    for start in range(1, t-56, ts):
        end = min(t-1, start+ts+56)
        start = max(1, start-48)
        with torch.no_grad():
            ip = Variable(torch.from_numpy(inputs.cpu().numpy()[:,:,start:end]).cuda()) #提取特征
features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())   
np.concatenate(features, axis=0).shape
#Out[4]: (250, 7, 7, 1024)
```

```python
features = i3d.extract_features(Variable(torch.rand(1,3,20,224,224).cuda())) #使用i3d提取特征
print(features.shape)
#Out[5]: torch.Size([1, 1024, 3, 7, 7])

for data in dataloaders['val']:
# get the inputs
    inputs, labels, name = data
    break
with torch.no_grad():
    inputs = Variable(inputs.cuda())
features = i3d.extract_features(inputs) #使用i3d提取特征
#Out[11]: torch.Size([1, 1024, 43, 1, 1])
np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
```



### 将视频图片数据img转换为tensor数据

```
charades_dataset_full.py
```