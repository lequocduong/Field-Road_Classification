import torch
import os
import cv2
from torch.utils.data import Dataset
class FieldRoadDataset(Dataset):
    '''    
    Generate DataLoader 
    ----------
    Attributes:
    root_dir : str
        root directory
    transform: default: None
        transform method
    class_names: list
        List of classes
    ----------
    Returns:    
    DataLoader contains image and label
    ----------
    Example:
    val_ds = FieldRoadDataset(val_root_dir,val_transform)
    train_ds = FieldRoadDataset(train_root_dir,train_transform)       
    '''
    def __init__(self, root_dir, transform=None):
        super(FieldRoadDataset, self).__init__
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        # return class name in root_dir ['fiels','roads'] # change the text_image location
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
          # Take the names of picture # Ex: '41.jpg'
          files = os.listdir(os.path.join(root_dir,name))
          # return the image link and label (0-fields and 1-roads)
          self.data += list(zip(files, [index]*len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
      img_file, label = self.data[index]
      # aggregate the root_dir and class names
      root_and_dir = os.path.join(self.root_dir, self.class_names[label])
      img_path = os.path.join(root_and_dir, img_file)
      image = cv2.imread(img_path)

      if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

      return image,label

def checkFieldRoadDataset(data):
  '''
  checkFieldRoadDataset(train_ds)
  '''
  for i in data:
    print(i)
    break
