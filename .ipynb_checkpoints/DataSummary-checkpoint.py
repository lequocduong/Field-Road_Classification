### Data Generator
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import math
def data_extractor(data_path,Class):
    """
    fields = data_extractor(data_path,classList[0])
    """
    all_names = os.listdir(data_path + Class + '/')
    image_names = [el for el in all_names ]
    file = []
    for img in image_names:
        file.append(data_path + Class+ '/' + img)
    return file
def Extract_Img_Info(Class):
    '''
    nameList, imgshapeList, imList = Extract_Img_Info(fields)
    '''
    imList=[]
    imgshapeList=[]
    nameList=[]
    for img_path in Class:
        img = cv2.imread(img_path)
        imList.append(img)
        imgshapeList.append(img.shape)
        nameList.append(img_path.split('/')[1] + '-' +img_path.split('/')[2].split('.')[0])
    return nameList , imgshapeList , imList
def LabelExtraction(name,numLabel=True):
    '''
    Apply in the DataFrame
    '''
    label = name.split('-')[0]
    if numLabel:
        match label:
            case 'fields':
                 label = 0
            case 'roads':
                 label = 1
            case _:
                raise Exception("Not having in the class list")
    return label

def DataGenerator(data_path, classList, savePath,save=True):
    '''
    Generate the .csv file for the dataset
    df = DataGenerator(data_path, classList, savePath)
    df_raw = pd.read_pickle(savePath+'dataset_raw.pkl')
    '''
    df = pd.DataFrame()
    for category in classList:
        category_data = data_extractor(data_path,category)
        nameList, imgshapeList, imList = Extract_Img_Info(category_data)
        data =[]
        for i in range(len(nameList)):
            data.append([nameList[i],imgshapeList[i],imList[i],category_data[i]])
        df_cat = pd.DataFrame(data, columns=['ID','Img_Shape','Img_Info','Img_Dir'])
        df = pd.concat([df, df_cat])
    df.reset_index(inplace=True)
    df.drop('index',axis=1,inplace=True)
    df['label'] = df['ID'].apply(LabelExtraction)
    if save:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        df.to_pickle(savePath+ 'dataset_raw.pkl') # For Reuse
        df.to_csv(savePath+'dataset_raw.csv') # For Read
    return df

### Data Preprocessing
def Normalize(x):
    return x/255.0
def Resize(x):
    dim = (128,128)
    return cv2.resize(x, dim, interpolation = cv2.INTER_AREA)
def ImgShape(x):
    return x.shape
def LabelExtraction(name,numLabel=True):
    '''
    Apply in the DataFrame
    '''
    label = name.split('-')[0]
    if numLabel:
        match label:
            case 'fields':
                 label = 0
            case 'roads':
                 label = 1
            case _:
                raise Exception("Not having in the class list")
    return label
def DataPreprocessing(df_raw,save=True):
    '''
    df = DataPreprocessing(df_raw)
    df = pd.read_pickle(savePath+'dataset_raw.pkl')
    '''
    # reshape
    df_raw['Img_Info_R'] = df_raw['Img_Info'].apply(Resize)
    # Normalize
    df_raw['Img_Info_R_N'] = df_raw['Img_Info_R'].apply(Normalize)
    # ImageShape
    df_raw['Img_Shape_Resize'] = df_raw['Img_Info_R_N'].apply(ImgShape)

    # Rearrange the dataset
    dropList = ['Img_Info','Img_Info_R','Img_Shape']
    df = df_raw.drop(dropList,axis=1)
    #df['label'] = df['ID'].apply(LabelExtraction)
    if save:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        df.to_pickle(savePath+ 'dataset.pkl') # For Reuse
        df.to_csv(savePath+'dataset.csv') # For Read
    return df

### Showing Image
def showImg(image):
    '''
    # Normal just use plt.imshow() - this with high resolution
    before process - showImg(df_raw['Img_Info'][0])
    after process - showImg(df['Img_Info_R_N'][0])
    '''
    plt.imshow(image)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)