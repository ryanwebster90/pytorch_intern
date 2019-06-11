import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os

# Define function to load individual images
loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor
#device = torch.device("cuda")
device = torch.device("cpu")
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# load an image
x = image_loader('self-portrait.jpg')
print('Original size')
print(x.size())

# 3 channels in because of RGB, 4 channels out to demonstrate convoluton
layer_conv1 = nn.Conv2d(3,4,kernel_size=3,padding=1)
print(layer_conv1)
y = layer_conv1(x)

print('Size after Conv2d(3,4)')
print(y.size())
for i in range(y.size(1)):
    #save one channel at time, as y is no longer RGB
    torchvision.utils.save_image(y[:,i,:,:],'y_conv_channel_' + str(i) + '.jpg',normalize=True)
    

layer_pool = nn.AvgPool2d(2)
y = layer_pool(x)
torchvision.utils.save_image(y,'y_AvgPool2d(2).jpg',normalize=True)
print('Size after AvgPool2d(2)')
print(y.size())

# thislayer is very useful because we can determine output size!
layer_adaptive_pool = nn.AdaptiveAvgPool2d((64,64))
y = layer_adaptive_pool(x)
print('Size after AdaptiveAvgPool2d(64,64)')
print(y.size())

torchvision.utils.save_image(y,'y_AdaptiveAvgPool2d(64,64).jpg',normalize=True)

# upsampling layers increase the size of a layer. Typically these are used in generators.
layer_upsample = nn.Upsample(scale_factor=2)
y = layer_upsample(x)
print('Size after nn.Upsample(scale_factor=2)')
print(y.size())

torchvision.utils.save_image(y,'y_nn.Upsample(scale_factor=2).jpg',normalize=True)

# A few other layers which do not alter the size
layer_relu = nn.ReLU()
y = layer_relu(x)
# can you verify x is >=0 ?

layer_in = nn.InstanceNorm2d(3)
y = layer_in(x)
# can you verify that y's mean and standard deviation have changed?
torchvision.utils.save_image(y,'y_nn.InstanceNorm2d(3).jpg',normalize=True)

