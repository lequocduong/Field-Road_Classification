from sklearn.model_selection import ParameterGrid
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def HyperparameterTuning(init_lr,max_lr,num):
  '''
  init_lr = 1e-5
  max_lr =  1e-3
  num=     100
  HyperparameterTuning(init_lr,max_lr,num)
  '''
  # Define hyperparameter space for grid search
  param_grid = {
      'learning_rate': list(np.linspace(init_lr,max_lr,num)),     
  }
 
  best_accuracy = 0.0
  best_hyperparameters = None
  for params in ParameterGrid(param_grid):
      model = pretrainedUsedMODEL().to(DEVICE)
      loss_fn = nn.BCEWithLogitsLoss() # binary cross entropy loss function
      optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
      scaler = torch.cuda.amp.GradScaler()
      train_loss = 0
      accuracy = 0
      for epoch in range(NUM_EPOCHS):
        # Loss
        train_loss = train_fn(trainLoader, model, optimizer, loss_fn, scaler, DEVICE)
        # Accuracy
        accuracy = check_accuracy(valLoader, model, DEVICE)

        if accuracy > best_accuracy:
          best_accuracy = accuracy
          best_hyperparameters = params

  print("Best Hyperparameters:", best_hyperparameters)
  print("Best Accuracy:", best_accuracy)
