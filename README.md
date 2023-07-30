# Field-Road_Classification

Classifies an image as containing either the road or the field (using  <a href="https://drive.google.com/file/d/1pOKhKzIs6-oXv3SlKrzs0ItHI34adJsT/view">field-road dataset</a>), but could easily be extended to other image classification problems.

### Dependencies:
- PyTorch / Torchvision
- Numpy
- Pandas
- CUDA
- etc... 
## Data

The data directory structure I used was:

* project
  * dataset
  * train
    * roads
    * fields
  * valil
    * roads
    * fields
  * test
  * arb


## Performance
The result of the notebook in this repo produced a binary category entropy loss score on the roads-field dataset with high performance. More information in ResultSummay.pdf
