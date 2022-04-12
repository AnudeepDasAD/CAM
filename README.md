# CS 497: Winter 2022 Project - Analyzing Biases in CNNs for Faces through Conditional Activation Maps

## Introduction
TODO: Short overview of project purpose and scope


## Other sections
* Conditional activation maps
* Model training/development
* Webcam demo on own face?
* Bunch of data analysis of various types
* Results on gender neutral people


## Conditional Activation Maps
We propose a simple technique to expose the implicit attention of Convolutional Neural Networks on the image. It highlights the most informative image regions relevant to the predicted class. You could get attention-based model instantly by tweaking your own CNN a little bit more. The paper is published at [CVPR'16](http://arxiv.org/pdf/1512.04150.pdf).

The popular networks such as ResNet, DenseNet, SqueezeNet, Inception already have global average pooling at the end, so you could generate the heatmap directly without even modifying the network architecture.

The framework of the Class Activation Mapping is as below:
![Framework](http://cnnlocalization.csail.mit.edu/framework.jpg)

Some predicted class activation maps are:
![Results](http://cnnlocalization.csail.mit.edu/example.jpg)


## References
The work on conditional activation maps is taken from here:
```
@inproceedings{zhou2016cvpr,
    author    = {Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Oliva, Aude and Torralba, Antonio},
    title     = {Learning Deep Features for Discriminative Localization},
    booktitle = {Computer Vision and Pattern Recognition},
    year      = {2016}
}
```
