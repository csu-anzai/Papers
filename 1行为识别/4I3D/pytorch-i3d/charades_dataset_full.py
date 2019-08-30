# 这个程序可以将视频转换为对应的tensor数据
import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    将数据转换为tensor数据类型(C x T x H x W)
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
    #对不符合大小的帧图片进行放大，与合乎要求的图片一起构成np.array类型的数据[帧数nf*h*w*3]即(T x H x W x C)
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]] #读取图片<class 'numpy.ndarray'> (180, 320, 3)
    w,h,c = img.shape

    #对小于226的图片进行放大
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h) #放大的比例
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc) #(226, 402, 3)
    img = (img/255.)*2 - 1 #将像素标准化到0-1之间
    frames.append(img)
  return np.asarray(frames, dtype=np.float32) #(296, 226, 402, 3)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(str(root), vid)):
            continue #如果当前目录不存在，则跳出当前循环，查找下一个视频数据
        num_frames = len(os.listdir(os.path.join(root, vid)))#取出帧的个数
        if mode == 'flow':
            num_frames = num_frames//2
            
        label = np.zeros((num_classes,num_frames), np.float32) #为每一帧图片进行标签初始化

        fps = num_frames/data[vid]['duration'] #计算帧频
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                #fr表示的是视频中帧数的序号
                #fr/fps：表示该帧在视频中的位置及时间点/时间戳
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification，类似于对视频中的每一帧进行one-hot编码,第几行有1表明该帧属于第几类，同一段视频可能同时属于多个类别
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        #(vid='FQ6OB'视频名称, label：每一帧的标签, data[vid]['duration']：持续的时间, num_frames：帧数)
        i += 1
    '''
    dataset其中的一个实例：
    ('D6DC1', array([[0., 0., 0., ..., 0., 0., 0.],
         [0., 0., 0., ..., 0., 0., 0.],
         [0., 0., 0., ..., 0., 0., 0.],
         ...,
         [0., 0., 0., ..., 0., 0., 0.],
         [0., 0., 0., ..., 0., 0., 0.],
         [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), 29.25, 703)
    '''
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        #dataset = Dataset(split_file='charades/charades.json', 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            vid: 视频的名称，例如：'FQ6OB'

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index] #视频名称，标签，持续时间，帧数
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            #'./save_model/FQ6OB.npy'
            return 0, 0, vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 1, nf) #变换数据类型为numpy.ndarray
        else:
            imgs = load_flow_frames(self.root, vid, 1, nf)

        imgs = self.transforms(imgs) #裁剪图片(296, 224, 224, 3)

        #video_to_tensor(imgs)将numpy.ndarray转换为tensor数据类型
        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
