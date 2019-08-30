import os
import cv2
import pickle as pkl
import natsort
import numpy as np
import imageio
import optparse
import random
import numpy.random
import math
from core import image_utils
from multiprocessing.dummy import Pool




class AsyncVideoReaderCharadesForI3DTorchModel():
    def __init__(self, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_video_frames_in_batch(self, frames_pathes):
        self.__is_busy = True

        n_pathes = len(frames_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, frames_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:\
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = resize_crop(img)
        img = img.astype(np.float32)
        # normalize such that values range from -1 to 1
        img /= float(127.5)
        img -= 1.0
        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

class AsyncImageReaderCharadesForResNetTorchModel():
    def __init__(self, img_mean, img_std, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224

        self.__img_mean = img_mean
        self.__img_std = img_std

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True

        n_pathes = len(image_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = image_utils.resize_crop(img)
        img = img.astype(np.float32)
        img /= float(255)
        img = img[:, :, (2, 1, 0)]
        img[:, :, 0] = (img[:, :, 0] - self.__img_mean[0]) / self.__img_std[0]
        img[:, :, 1] = (img[:, :, 1] - self.__img_mean[1]) / self.__img_std[1]
        img[:, :, 2] = (img[:, :, 2] - self.__img_mean[2]) / self.__img_std[2]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()