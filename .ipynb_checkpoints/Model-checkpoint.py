from torchvision import models
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the model
class MODEL(nn.Module):    
    '''    
    Classical Model 
    The simple Model Th input (16,3,128,128)
    2x Conv+BN+MP(feature extraction) + 2xFCL(classifier)
    ----------
    Example:
    model = MODEL()
    '''       
    def __init__(self):
        super(MODEL, self).__init__()
        # 1st CNN layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd CNN layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 1st fully layer
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=64)
        # 2nd fully layer
        self.fc2 = nn.Linear(in_features=64, out_features=1)
        # activation layer
        self.relu = nn.ReLU()

    def forward(self, x): # Input shape : [16,3,128,128]
        # 1st layer -> relu 
        x = self.relu(self.conv1(x)) # [16,32,128,128]
        x = self.bn1(x)
        x = self.pool(x) # [16,32,64,64]
        # 2nd layer -> relu
        x = self.relu(self.conv2(x)) # [16,64,64,64]
        x = self.bn2(x)
        x = self.pool(x) # [16,64,32,32]

        # flattern layer
        x = x.view(-1, 64*32*32)# [16,128]
        # 1st fully layer + relu
        x = self.relu(self.fc1(x)) # [16,64]
        # 2nd fully connected model
        x = self.fc2(x) # [16,2] # 2 - number of labels
        # Check whether the last layer is neccesary
        return x

class pretrainedUsedMODEL(nn.Module):
    ''' 
    Using the Pretrained Model (Vgg16)
    Unfreeze the last layer ConV
    Replace classifer with 2 FCN 
    ----------
    Example:
    model = pretrainedUsedMODEL()
       
    '''
    def __init__(self):
        super(pretrainedUsedMODEL, self).__init__()
        # Pretrained model
        self.vgg16 = models.vgg16(pretrained=True) # call the vgg
        # 1st fully layer
        self.fc1 = nn.Linear(in_features=self.vgg16.classifier[0].in_features,out_features=4096)
        # 2nd fully layer
        self.fc2 = nn.Linear(in_features=4096, out_features=1)
        # activation layer
        self.relu = nn.ReLU()
        # freeze all model parameters
        for name, param in self.vgg16.named_parameters():
          param.requires_grad = False

        # Unfreezed the last layer
        unfreeze_layers = self.vgg16.features[28:]
        for layer in unfreeze_layers:
          for param in layer.parameters():
            param.requires_grad = True
        # Replace Classifier
        self.vgg16.classifier = nn.Sequential(
                         self.fc1,
                         self.relu,
                         self.fc2 ,
                         )
    def forward(self, x): # Input shape : [16,3,128,128] # 64 : batch size
        # VGG16 features
        x = self.vgg16(x)
        return x


def test_model():
  x = torch.randn((16,3,128,128))
  #model = MODEL()
  model = pretrainedUsedMODEL()
  preds = model(x)
  print('x.shape',x.shape)
  print(preds.shape)
  #assert preds.shape == x.shape
  
if __name__ == "__main__":
  test_model()