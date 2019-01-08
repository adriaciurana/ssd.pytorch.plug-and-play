# ssd.pytorch.plug-and-play
## Philosophy
Implementation of the generalized SSD architecture in pytorch.
This implementation tries to simplify the learning process of an SSD in different ways:
- It is a single file + models folder, which allows a quick adaptation to the project where you want to use it.
- It is easy to use this scheme to implement other architectures such as SSD-512, Mobilenet, etc... (new architectures will be added in future updates)
  Which allows a correct generalization of models based on the original SSD.
- It is extremely documented what the code tries to do understandable to all and thus be able to improve it continuously.

Much of the code has been influenced by the following implementations:
- https://github.com/amdegroot/ssd.pytorch
- https://medium.com/@smallfishbigsea/understand-ssd-and-implement-your-own-caa3232cd6ad

¡That is why all the gratitude belongs to them!

## Train with own Dataset
You can directly use the train.py file to train with your dataset. This includes the monitoring of learning using tensorboardX.

To learn a new model in our dataset, only one nn.Dataset class must be generated, this must be returned in each element:
- image (size does not matter)
- arrray numpy from:

  - x1 (top-left x corner with respect to the coordinates of the image).
   
  - y1 (top-left y corner with respect to the coordinates of the image).
   
  - x2 (bottom-right x corner with respect to the coordinates of the image).
   
  - y2 (bottom-right y corner with respect to the coordinates of the image).
   
  - index of the class.

See the dataset / logos / LogoDataset class for more information.

## Pretrained models
TODO

## Data augmentation
The imgaug library is used to apply data augmentation, which makes the process very simple. https://imgaug.readthedocs.io/en/latest/

## Examples:
![1](doc/0.jpg)
![1](doc/1.jpg)
![1](doc/2.jpg)
![1](doc/3.jpg)
![1](doc/4.jpg)
![1](doc/5.jpg)
![1](doc/6.jpg)
![1](doc/7.jpg)
![1](doc/8.jpg)
