# CS 497: Winter 2022 Project - Analyzing Biases in CNNs on Human Faces through Conditional Activation Maps

## Introduction
TODO: Short overview of project purpose and scope


## Conditional Activation Maps
We propose a simple technique to expose the implicit attention of Convolutional Neural Networks on the image. It highlights the most informative image regions relevant to the predicted class. You could get attention-based model instantly by tweaking your own CNN a little bit more. The paper is published at [CVPR'16](http://arxiv.org/pdf/1512.04150.pdf).

The popular networks such as ResNet, DenseNet, SqueezeNet, Inception already have global average pooling at the end, so you could generate the heatmap directly without even modifying the network architecture.

The framework of the Class Activation Mapping is as below:
![Framework](http://cnnlocalization.csail.mit.edu/framework.jpg)

Some predicted class activation maps are:
![Results](http://cnnlocalization.csail.mit.edu/example.jpg)


## Model Training and Development


## Demo using Webcam

We can apply the CAM technique to our own faces in real time through the user's webcam. You can see a demo with this mp4 file: [CAM_webcam.mp4](/demo/CAM_webcam.mp4), or [here](https://user-images.githubusercontent.com/55476249/163021617-5f20ea2d-6683-4447-8c03-0743a89b1beb.mp4).

You can also try it yourself on your own system by running [pytorch_CAM.py](/pytorch_CAM.py).


## Sample CAM Results

Here are some CAM results on test celebrity images from the CelebA dataset.

![182638](https://user-images.githubusercontent.com/55476249/163027226-80131b10-6e70-47eb-9376-1b74617db6af.jpg)
![182661](https://user-images.githubusercontent.com/55476249/163027353-a8d1155a-488d-4077-9703-1a5229720ce9.jpg)
![182671](https://user-images.githubusercontent.com/55476249/163027392-fe62b09d-2e72-4e14-a278-1e8f4dc165a6.jpg)
![182643](https://user-images.githubusercontent.com/55476249/163027726-3aca1712-3bc6-4ba8-82fd-dd02145b8919.jpg)

This indicates that for females, there is significant weight put on the chin and glabella, and for males, there is indication of less weight on the chin and glabella, more weight on the right cheek, and much less weight on the left cheek.

Since the weightage of the entire image changes across test examples, there is also indication that not only specific areas of a person's face impact the algorithm's prediction, but the background itself is also taken into account. We expected that the colours for the weights would change only a little on the person's face and the background would be left alone, which is not the case. 


## Bunch of data analysis of various types 

You can find the analyses for the balanced and unbalanced data [here](/model_bias_analysis/celeba_dataset_analysis_balance.ipynb) and [here](/model_bias_analysis/celeba_dataset_analysis_unbalance.ipynb) respectively. These files are essentially the same, except the flag on cell 15 (use_balanced) is changed. Analysis starts from cell 26.

We find that the accuracy for males is greater (97.3% vs 96.6%) despite having fewer males in the training dataset (38.6% vs 61.4%). Since the CAM paper indicates that a net with a CNN "looks" at an entire image, the fact that ImageNet (which was used to train ResNet) has images with human faces in them, the imbalance of male and female humans in ImageNet may have led to an imbalance in the performance for our downstream facial recognition task. Balancing the number of males and females made the accuracies a little closer together, though, curiously, it decreased both of the accuracies (96.9% vs 96.5% now). 

Also, interestingly, the accuracy is higher for older, darker males, and younger, paler females. Though the accuracy dropped for paler younger females when the 
training set was balanced. Darker, younger females performed better when sex alone was balanced.


## Results on gender neutral people


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
