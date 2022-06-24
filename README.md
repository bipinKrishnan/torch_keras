# TorchKeras
**Recreating Keras with PyTorch**

[functional_api_v1](https://github.com/bipinKrishnan/torchkeras/blob/master/functional_api_v1.ipynb) - An easy to use framework inspired by Tensorflow's keras but for PyTorch users.

[![article](https://img.shields.io/badge/-article-blueviolet)](https://towardsdatascience.com/recreating-keras-functional-api-with-pytorch-cc2974f7143c?source=---------4----------------------------)  [![nbviewer](https://img.shields.io/badge/-nbviewer-blue)](https://nbviewer.jupyter.org/github/bipinKrishnan/torchkeras/blob/master/functional_api_v1.ipynb)

### Installing the library
      pip install git+https://github.com/bipinKrishnan/torchkeras
      
### Usage

* Training convolutional neural network on CIFAR-100 dataset

```python
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from torchkeras import Input, Model
from torchkeras.layers import Dense, Conv2d, Flatten


bs = 128
transform = transforms.ToTensor()

# prepare train and test sets
trainset = torchvision.datasets.CIFAR100(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
    )
testset = torchvision.datasets.CIFAR100(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
    )

trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)
testloader = DataLoader(testset, batch_size=bs)

# build the model architecture
input = Input((3, 32, 32))
x = Conv2d(3, 3, 1, 'same', 1, nn.ReLU())(input)
y = Conv2d(5, 3, 1, 0, 1, nn.ReLU())(x)
z = Conv2d(6, 3, 1, 'same', 1, nn.ReLU())(y)
a = Flatten()(z)
b = Dense(100, activation=nn.ReLU())(a)

model = Model(input, b, device)
model.compile(optim.Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss())

print(model.summary())

# train the model
model.fit_generator(trainloader, 3)

#### Output
# Epoch 1/3
# 391/391 [=========================] - 27s 69ms/step - train_loss: : 4.5869
# Epoch 2/3
# 391/391 [=========================] - 30s 76ms/step - train_loss: : 4.5388
# Epoch 3/3
# 391/391 [=========================] - 28s 72ms/step - train_loss: : 4.5173

# evaluate the model
model.evaluate_generator(testloader)

#### Output
# 79/79 [=========================] - 3s 42ms/step - eval_loss: 2.3131

# get the predictions
out = model.predict_generator(testloader)
 ```
