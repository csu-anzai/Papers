# Timeception for Complex Action Recognition

![Keras](./data/assets/badge-keras.png "Keras") ![Keras](./data/assets/badge-tensorflow.png "TensorFlow") ![Keras](./data/assets/badge-pytorch.png "PyTorch")

This code repository is the implementation for the paper [Timeception for Complex Action Recognition](https://arxiv.org/abs/1812.01289).
We provide the implementation for 3 different libraries: `keras`, `tensorflow` and `pytorch`.

![Timeception for Complex Action Recognition](./data/assets/timeception_layer.jpg "Timeception Block")

## 对长期时间依赖性上的解决方案：

| timeception结构                                   | 解决方案                                                     |
| ------------------------------------------------- | ------------------------------------------------------------ |
| ![1564644942094](README.assets/1564644942094.png) | 1、**长期时间依赖性的处理：**采用多<br/>层堆叠的方式进行实现，以期学得<br/>更复杂抽象的时空特征。<br/><br>2、**时间成本的处理：**采用通道分组<br/>并同时引入通道混洗和连接操作 |

> - **N-Group操作之后：**
>
>     Concat ==> Relu ==> Shuffle ==>对时域的T进行下采样MaxPool
>
> - **采用通道分组的原因：**
>   优点：可以降低参数量，减少时间成本。
>   缺点：每个分组都值包含通道之间的部分相关性，不利于通道之间的信息交互。
>   处理办法：增加通道混洗和连接操作。通道混洗可以增加编码跨通道之间的相关性。因为每组通道都是所有通道的一个随机子集，都只包含部分的可能性。而这可以通过通道混洗和组合得到缓解

## 为了能够应对复杂动作多变的时间范围，采用多尺度替换固定尺度

### 多尺度实现的两种方式：

1. Inception卷积结构
2. dilated convolutions空洞卷积结构

![1564643171762](README.assets/1564643171762.png)

### 具体的实习方法：

![1564670952238](README.assets/1564670952238.png)

> **Note:**注意在实际编码过程中是先使用$1*1*1$的卷积核对通道数进行缩减，然后再使用深度可分离卷积进行卷积，与图中画的方向相反。

----

**仅以第一次循环为例,展示每一个branch的输入与输出的过程：**

- **Conv3d：**`kernel_size=(1, 1, 1)`，是为降低通道数量

- **DepthwiseConv1DLayer：**采用的是深度可分离的一维卷积，让输入和输出的`channels`相同，同时只对`n_timesteps`进行卷积，参数量为：`in_channels*kernel_size*out_channels`

| branch | is_dilated                                                   | no_dilated                                                   |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1      | channels_in=128, channels_out=32, process:Conv3d==>BN,  kernel_size=(1,1,1) | channels_in=128, channels_out=32, process:Conv3d==>BN,  <br>kernel_size=(1,1,1) |
| 2      | channels_in=32,==>128==> channels_out=32, process: Conv3d==>DepthwiseConv1DLayer==>BN, kernel_size=3, dilation_rates=1 | channels_in=128,==>32==> channels_out=32, <br/>process: Conv3d==>DepthwiseConv1DLayer==<br/>>BN, kernel_size=3, dilation_rates=1 |
| 3      | channels_in=32,==>128==> channels_out=32, process: Conv3d==>DepthwiseConv1DLayer==>BN, kernel_size=3, dilation_rates=2 | channels_in=128,==>32==> channels_out=32, <br/>process: Conv3d==>DepthwiseConv1DLayer<br/>==>BN, kernel_size=5, dilation_rates=1 |
| 4      | channels_in=32,==>128==> channels_out=32, process: Conv3d==>DepthwiseConv1DLayer==>BN, kernel_size=3, dilation_rates=3 | channels_in=128,==>32==> channels_out=32, <br/>process: Conv3d==>DepthwiseConv1DLayer<br/>==>BN, kernel_size=7, dilation_rates=1 |
| 5      | channels_in=32,==>128==> channels_out=32, process: Conv3d==>DepthwiseConv1DLayer==>BN, kernel_size=3, dilation_rates=3 | channels_in=128, channels_out=32, process: Conv3d==>MaxPool3d====><br/>torch.nn.ReplicationPad3d==>BN |

## PyTorch

Using `pytorch`, we can define `timeception` as a module.
Then we use it along with another model definition.
For example, here we define 4 `timeception` layers followed by a `dense` layer for classification.

```python
#测试代码
import numpy as np
import torch as T
from nets import timeception_pytorch

# define input tensor: batch*channels*n_timesteps*h*w
input = T.tensor(np.zeros((1, 1024, 128, 7, 7)), dtype=T.float32)

# define 4 layers of timeception
module = timeception_pytorch.Timeception(input.size(), n_layers=4)

# feedforward the input to the timeception layers 
tensor = module(input)

# the output is (32, 2480, 8, 7, 7)
print (tensor.size())
```





















































































## Citation

Please consider citing this work using this BibTeX entry

```bibtex
@inproceedings{hussein2018timeception,
  title     = {Timeception for Complex Action Recognition},
  author    = {Hussein, Noureldien and Gavves, Efstratios and Smeulders, Arnold WM},
  booktitle = {CVPR},
  year      = {2019}
}
```

## How to Use?

### Keras

Using `keras`, we can define `timeception` as a sub-model.
Then we use it along with another model definition.
For example, here we define 4 `timeception` layers followed by a `dense` layer for classification.

```python
from keras import Model
from keras.layers import Input, Dense
from nets.layers_keras import MaxLayer
from nets.timeception import Timeception

# define the timeception layers
timeception = Timeception(1024, n_layers=4)

# define network for classification
input = Input(shape=(128, 7, 7, 1024))
tensor = timeception(input)
tensor = MaxLayer(axis=(1, 2, 3))(tensor)
output = Dense(100, activation='softmax')(tensor)
model = Model(inputs=input, outputs=output)
model.summary()
```

This results in the model defined as:

```
Layer (type)  Output Shape              Param #   
================================================
(InputLayer)  (None, 128, 7, 7, 1024)   0         
(Timeception) (None, 8, 7, 7, 2480)     1494304   
(MaxLayer)    (None, 2480)              0         
(Dense)       (None, 100)               248100    
================================================
Total params: 1,742,404
```

### Tensorflow

Using `tensorflow`, we can define `timeception` as a list of nodes in the computational graph.
Then we use it along with another model definition.
For example, here a functions defines 4 `timeception` layers.
It takes the input tensor, feedforward it to the `timeception` layers and return the output tensor `output`.

```python
import tensorflow as tf
from nets import timeception

# define input tensor
input = tf.placeholder(tf.float32, shape=(None, 128, 7, 7, 1024))

# feedforward the input to the timeception layers
tensor = timeception.timeception_layers(input, n_layers=4)

# the output is (?, 8, 7, 7, 2480)
print (tensor.get_shape())
```

### PyTorch

Using `pytorch`, we can define `timeception` as a module.
Then we use it along with another model definition.
For example, here we define 4 `timeception` layers followed by a `dense` layer for classification..

```python
import numpy as np
import torch as T
from nets import timeception_pytorch

# define input tensor
input = T.tensor(np.zeros((1, 1024, 128, 7, 7)), dtype=T.float32)

# define 4 layers of timeception
module = timeception_pytorch.Timeception(input.size(), n_layers=4)

# feedforward the input to the timeception layers 
tensor = module(input)

# the output is (32, 2480, 8, 7, 7)
print (tensor.size())
```

### Installation

We use python 2.7.15, provided by Anaconda 4.6.2, and we depend on the following python packages.
- Keras 2.2.4
- Tensorflow 1.10.1
- PyTorch 1.0.1

## Training

## Testing

## Fine-tuning

### Pretrained Models

### Charades

We will add all pretrained models for Charades by the end of April.
For testing, start with the script `./scripts/test_charades_timeception.sh`.
In order to change which baseline is uses for testing, set the `-- config-file` using on of the following options.

### 2D-ResNet-152

Timeception on top of 2D-ResNet-152 as backnone.

|  Config File | Backbone | TC Layers | Frames  | mAP (%)  | Model |
|---|:---:|:---:|:---:|:---:|:---:|
| [charades_r2d_tc3_f32.yaml](./configs/charades_r2d_tc3_f32.yaml)     | R2D   | 3 | 32  | 30.37  | [Link](./data/charades/charades_r2d_tc3_f32.pkl)   |
| [charades_r2d_tc3_f64.yaml](./configs/charades_r2d_tc3_f64.yaml)     | R2D   | 3 | 64  | 31.25  | [Link](./data/charades/charades_r2d_tc3_f64.pkl)   |
| [charades_r2d_tc4_f128.yaml](./configs/charades_r2d_tc4_f128.yaml)   | R2D   | 4 | 128 | 31.82  | [Link](./data/charades/charades_r2d_tc4_f128.pkl)  |

### I3D

Timeception on top of ResNet-152 as backnone.

|  Config File | Backbone | TC Layers | Frames  | mAP (%)  | Model |
|---|:---:|:---:|:---:|:---:|:---:|
| [charades_i3d_tc3_f256.yaml](./configs/charades_i3d_tc3_f256.yaml)    | I3D  | 3 | 256  | 33.89  | [Link](./data/charades/charades_i3d_tc3_f256.pkl)   |
| [charades_i3d_tc3_f512.yaml](./configs/charades_i3d_tc3_f512.yaml)    | I3D  | 3 | 512  | 35.46  | [Link](./data/charades/charades_i3d_tc3_f512.pkl)   |
| [charades_i3d_tc4_f1024.yaml](./configs/charades_i3d_tc4_f1024.yaml)  | I3D  | 4 | 1024 | 37.20  | [Link](./data/charades/charades_i3d_tc4_f1024.pkl)  |

### 3D-ResNet-100
Timeception on top of 3D-ResNet-100 as backnone.


|  Config File | Backbone | TC Layers | Frames  | mAP (%)  | Model |
|---|:---:|:---:|:---:|:---:|:---:|
| [charades_r3d_tc4_f1024.yaml](./configs/charades_r3d_tc4_f1024.yaml)  | R3D  | 4 | 1024 |  41.1  | [Link](./data/charades/charades_r3d_tc4_f1024.pkl)  |


### Kinetics 400

We will add all pretrained models for Kinetics 400 by the end of June.

## License

The code and the models in this repo are released under the GNU 3.0 [LICENSE](LICENSE).



