# Field-Road_Classification

Classifies an image as containing either the road or the field (using  <a href="https://drive.google.com/file/d/1pOKhKzIs6-oXv3SlKrzs0ItHI34adJsT/view">field-road dataset</a>), but could easily be extended to other image classification problems.

### Dependencies:
- PyTorch / Torchvision
- Numpy
- Pandas
- CUDA

## Data

The data directory structure I used was:

* project
  * dataset
    * train
      * roads
      * fields
    * vali
      * roads
      * fields
    * test


## Performance
The result of the notebook in this repo produced a log loss score on Kaggle's hidden dataset of 0.04988 -- further gains can probably be achieved by creating an ensemble of classifiers using this approach. 
