import torch
import torchvision

# Download dataset or set up dataset here
transform = torchvision.transforms.Compose([torch.vision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root ="~/torch_datasets",
    train = True,
    transform = transform,
    download = True,
)

train_dataset = torchvision.datasets.MNIST(
    root ="~/torch_datasets",
    train = False,
    transform = transform,
    download = True,
)

# Load dataset for Pytorch training
train_loader = torch.utils.DataLoader(
    train_dataset, 
    batch_size = 128,
    shuffle = True,
    num_workers = 4,
    pin_memory = True
    }

test_loader = torch.utils.DataLoader(
    train_dataset, 
    batch_size = 32,
    shuffle = False,
    num_workers = 4,
    
)