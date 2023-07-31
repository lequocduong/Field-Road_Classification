import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
  print("=> Saving checkpoint")
  torch.save(state, filename)

def load_checkpoint(checkpoint, model):
  print("=> Loading checkpoint")
  model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
  '''
  check_accuracy(val_loader, model, DEVICE)
  '''
  num_correct = 0
  num_samples = 0
  acc = 0
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      # Last activation layer
      scores = torch.sigmoid(model(x))
      # Keep predictions = 1
      predictions = (scores>0.5).float()
      # Sum of correct
      num_correct += (predictions.reshape(-1) == y).sum()
      num_samples += predictions.shape[0] # 30

    acc = (float(num_correct)/float(num_samples))*100
    print(f'Got {num_correct} / {num_samples} with accuracy {acc:.2f}')

  model.train()
  return acc

def make_prediction(model, transform, root_dir,originalLabel, device,name_file,save=True):
  '''
  original_dataset_dir = '/content/drive/MyDrive/Example/Trimble_dataset/'
  root_dir = original_dataset_dir + 'test'
  originalLabel = np.array([1,1,1,1,1,1,0,0,0,0])
  name_file = 'Model'
  make_prediction(model, transform, root_dir, DEVICE,name_file,save=True)
  The last procedure - print out the csv containing the classification
  '''
  def threshold(x):
    if x>0.5:
      res = 1
    else:
      res = 0
    return res

  files = os.listdir(root_dir)
  preds = []
  model.eval()

  files = os.listdir(root_dir)
  for file in tqdm(files):      
      img = cv2.imread(os.path.join(root_dir, file))
      img = transform(image = img)['image'].unsqueeze(0).to(device)# add 1 dimension
      with torch.no_grad():
          pred = torch.sigmoid(model(img))
          preds.append(pred.item())

  data = {#'id': np.arange(1, len(preds)+1),
          'ID': files,
          'predictedVal': np.array(preds),
          'originalLabel': originalLabel,
          }
  df = pd.DataFrame(data)
  df['predictedLabel'] = df['predictedVal'].apply(threshold)
  acc = (df['originalLabel'] == df['predictedLabel']).sum()

  if save:
      if not os.path.exists(savePath):
          os.makedirs(savePath)
      df.to_csv(savePath+f'{name_file}.csv', index=False)
  model.train()
  print("\n Done with predictions")
  print(f'Test Accuracy: {acc/len(df):.2%}')
  return df

def ResultPlot(val_acc,train_acc,train_loss,val_loss,NUM_EPOCHS,name_fig,save=False):
  '''
  name_fig = 'Model'
  ResultPlot(val_acc,train_acc,train_loss,val_loss,NUM_EPOCHS,save=False)
  '''
  num_epochs = [i for i in range(NUM_EPOCHS)]
  title0= 'The Model Accuracy'
  title1= 'The Model Loss'
  fig,ax = plt.subplots(ncols=2,nrows=1, figsize=(15,5))
  ax[0].plot(num_epochs, val_acc, label='Valid Accuracy')
  ax[0].plot(num_epochs, train_acc, label='Train Accuracy')
  ax[0].legend()
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('Accuracy')
  ax[0].set_title(title0)
  ax[0].grid()

  ax[1].plot(num_epochs,train_loss,label ='Train_loss')
  ax[1].plot(num_epochs,val_loss,label ='Val_loss')
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Loss')
  ax[1].set_title(title1)
  ax[1].legend()
  ax[1].grid()
  plt.show()

  if save:
      if not os.path.exists(savePath):
          os.makedirs(savePath)
      fig.savefig(savePath+f'{name_fig}.jpg')