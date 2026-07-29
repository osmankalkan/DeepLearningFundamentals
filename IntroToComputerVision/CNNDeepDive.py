
import torch
from torch import nn
import torchvision
from torch.xpu import manual_seed
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

torch.manual_seed(42)
images = torch.randn(size=(32,3,32,32)) #batch size , color_channels,height , width
test_image = images[0]
#print(f"Image batch size : {images.shape} -> batch size , color_channels,height , width ")
#print(f"Single image  size : {test_image.shape} ->  color_channels,height , width ")

torch.manual_seed(42)

conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

#conv_layer(test_image).shape

x = torch.randn(1,3,64,64)
print("Input: " , x.shape)

conv1 = nn.Conv2d(in_channels=3,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1)

conv2 = nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1)

pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

x = conv1(x)
print("After conv 1 : " , x.shape)

x = conv2(x)
print("After conv 2 : " , x.shape)

x = pool1(x)
print("After pool 1 : " , x.shape)

conv3 = nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1)

conv4 = nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1)

pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

x = conv3(x)
print("After conv 3 : " , x.shape)

x = conv4(x)
print("After conv 4 : " , x.shape)

x = pool2(x)
print("After pool 2 : " , x.shape) #After pool 2 :  torch.Size([1, 10, 8, 8]) 10*8*8 = 640

