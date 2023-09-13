import os, shutil
import numpy as np
import math

def checkAndCreateDir(dir):
    '''    
    Check whether the Dir exists
    If not, create one
    ----------
    Attributes:
    dir : str
        repository      
    ----------
    Example:
    checkAndCreateDir(dataset_dir)
    '''  
  if not os.path.exists(dir):
    os.mkdir(dir)

def DirInitialization(base_dir,datasetList):
    '''    
    Directory Initialization
    ----------
    Attributes:
    base_dir : str
        file repository
    datasetList: list
        List of dataset (train,valid,test)
    ----------
    Example:
    original_dataset_dir = 'dataset/'
    datasetList =['train','valid','test']
    DirInitialization(original_dataset_dir,datasetList)
    '''
  # Create the train - validation dir
  for dataset in datasetList:
    dataset_dir = os.path.join(base_dir, dataset)
    checkAndCreateDir(dataset_dir)

def CopyImage(base_dir,des_dir,Class,data_names):
    '''    
    Copy Image for base directory to the destination one 
    ----------
    Attributes:
    base_dir : str
        base repository
    des_dir: str
        destination repository
    Class: str
        class
    data_names: list
        List of files
    ----------
    Example:
    classList = ['fields','roads']
    base_dir = data_path
    dataList: list of data [fiels/1.jpg - roads/5.jpeg]
    '''
  for data in data_names:
    # Take the link image and copy to the new dir
    root = base_dir+'/'+Class
    checkAndCreateDir(root)
    des = des_dir+'/'+ Class
    checkAndCreateDir(des)
    src = os.path.join(root, data)
    dst = os.path.join(des, data)
    shutil.copyfile(src, dst)

def SplittingDataset(original_dataset_dir,data_path,datasetList,SplittingRatio,classList):
    '''    
    Copy Image for base directory to the destination one 
    ----------
    Attributes:
    original_dataset_dir : str
        original repository
    data_path: str
        data repository
    datasetList: str
        class
    SplittingRatio: tuple
        ratio of Train,validation,test
    classList: list
        list of class        
    ----------
    Example:
    original_dataset_dir = 'dataset/'
    data_path = os.path.join(original_dataset_dir,'dataset')
    datasetList =['train','valid','test']
    SplittingRatio = (0.8,0.2,0)
    classList = ['fields','roads']
    SplittingDataset(original_dataset_dir,data_path,datasetList,SplittingRatio,classList)
    '''
  # Dir Creation
  DirInitialization(original_dataset_dir,datasetList)
  # Train Ratio
  ratio_max = np.max(SplittingRatio)
  ratio_min = np.min(SplittingRatio)
  for dataset,ratio in zip(datasetList,SplittingRatio):
    des_dir = original_dataset_dir + dataset
    for Class in classList:
      # Split dataset
      data_names = os.listdir(os.path.join(data_path,Class))
      # Split index
      spliitingInd = math.ceil(len(data_names)*ratio)
      if ratio == ratio_max:
        data_names_Train = data_names[:spliitingInd]
      elif ratio == 0:
          break
      elif ratio == ratio_min:
        data_names_Train = data_names[-spliitingInd:len(data_names)]
      else:
        spliitingInd_Train =  math.ceil(len(data_names)*ratio_max)
        data_names_Train = data_names[spliitingInd_Train:spliitingInd_Train+spliitingInd]
      # Copy image
      CopyImage(data_path,des_dir,Class,data_names_Train)