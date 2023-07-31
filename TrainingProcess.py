import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_fn(loader, model, optimizer, loss_fn, scaler,device):
    '''
    Training process
    model = MODEL().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
      train_fn(trainLoader, model, optimizer, loss_fn, scaler,DEVICE)
    '''
    loop = tqdm(loader)
    running_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
      # Get data to cuda if possible
      data = data.to(device=device)
      #targets = torch.tensor(targets) #
      #targets = targets.to(device=device)
      # Convert to the float and unsqueeze is reshape the target --> just 1 dimension
      targets = targets.float().unsqueeze(1).to(device=device)

      # forward
      with torch.cuda.amp.autocast():        
        scores = model(data)
        loss = loss_fn(scores, targets) #
        running_loss += loss.item()

      # backward
      optimizer.zero_grad()      
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      # update tqdm loop
      loop.set_postfix(loss=loss.item())

    return running_loss

def val_fn(loader, model, loss_fn,device):
    '''
    Training process
    model = MODEL().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
      train_fn(trainLoader, model, optimizer, loss_fn, scaler,DEVICE)
    '''
    model.eval()
    running_loss =0
    for data, targets in loader:
      # Get data to cuda if possible
      data = data.to(device=device)
      targets = targets.float().unsqueeze(1).to(device=device)

      # forward
      with torch.cuda.amp.autocast():
        scores = model(data)
        loss = loss_fn(scores, targets) #
        running_loss += loss.item()
    return running_loss