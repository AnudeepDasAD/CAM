# CS 497: Winter 2022 Project - Analyzing Biases in CNNs on Human Faces through Conditional Activation Maps

Anudeep Das, Raymond Zhou

## Introduction
In this project, we explore the bias in convolutional neural networks (CNN) when trained on and applied to people's faces. We train a ResNet model on the CelebA dataset, which consists of images of celebrities and 40 corresponding labels such as male/female, pale/dark, and young/old. All labels are binary, and our model is a binary classifier to predict an indivisual's gender. In addition to analyzing accuracy across various intersectional demographics, we also make use of class activation maps (CAM). These are essentially heat maps which reveal which parts of an image the CNN found most valuable during classification. 

In [Conditional Activation Maps](#conditional-activation-maps), we describe CAMs further. Then, we expand upon the model delopment process in [Model Training and Development](#model-training-and-development). Some sample CAM results on our test dataset are provided in [Sample CAM Results](#sample-cam-results). A user can use their own webcam and face, and a (short) demo is provided in [CAM Demo with Webcam](#cam-demo-with-webcam). In [Model Bias Analysis](#model-bias-analysis), we thoroughly analyze model accuracy on different groups, as well as comparing the effects of a training dataset balanced for gender vs unbalanced. We apply the model to non-binary individuals and anlyze the results in [Non-Binary Analysis](#non-binary-analysis), and references are in [References](#references). Each section makes note of the relevant files.

Originally, we had wanted to focus on a facial recognition system specifically, but ran into issues surrounding the architecture and layer shapes, which were unable to produce CAM images properly. So, we instead work with a CNN which predicts an individual's gender, which still allows the same type of analysis. Additionally, there were some hurdles with the model training that are expanded upon later, which caused us to concentrate on a binary classifier. 


## Conditional Activation Maps
We using conditional activation maps (CAM) to expose the implicit attention of Convolutional Neural Networks on each image. It highlights the most informative image regions relevant to the predicted class. The work is published at [CVPR'16](http://arxiv.org/pdf/1512.04150.pdf). This techniques rely on a global average pooling layer near the end, which popular networks such as ResNet, DenseNet, SqueezeNet, and Inception already have.

The framework of the Class Activation Mapping is as below:
![Framework](http://cnnlocalization.csail.mit.edu/framework.jpg)


## Model Training and Development

We take an existing ResNet model and retrain it using the CelebA dataset. The code and model for this section can be found in [/model_training](/model_training/). We load the pretrained model on ImageNet, then retrain for 5 further epochs on CelebA. We chose 5 as the number of epochs as the validation accuracy was already quite high and plateauing (unlikely to improve further) and due to how incredibly long it takes for the model to train. 

The original training dataset is unbalanced in terms of gender. For comparison, we create a balanced version of the model by removing images from the class with a greater number of examples so that the dataset is balanced. 

Previously, attempts at training a model which predicts all 40 categories of the CelebA labels caused convergence issues, so we focused on predicting gender only. As well, we tried to train a version using SqueezeNet for performance benefits, but it performed poorly on validation. Thus, our ResNet model is a binary classifier at predicting gender.


## Sample CAM Results

Here are some CAM results on test celebrity images from the CelebA dataset.

![182638](https://user-images.githubusercontent.com/55476249/163027226-80131b10-6e70-47eb-9376-1b74617db6af.jpg)
![182661](https://user-images.githubusercontent.com/55476249/163027353-a8d1155a-488d-4077-9703-1a5229720ce9.jpg)
![182671](https://user-images.githubusercontent.com/55476249/163027392-fe62b09d-2e72-4e14-a278-1e8f4dc165a6.jpg)
![182643](https://user-images.githubusercontent.com/55476249/163027726-3aca1712-3bc6-4ba8-82fd-dd02145b8919.jpg)

This indicates that for females, there is significant weight put on the chin and glabella, and for males, there is indication of less weight on the chin and glabella, more weight on the right cheek, and much less weight on the left cheek.

Since the weightage of the entire image changes across test examples, there is also indication that not only specific areas of a person's face impact the algorithm's prediction, but the background itself is also taken into account. We expected that the colours for the weights would change only a little on the person's face and the background would be left alone. This is not the case, however, as the CAM images reveal that the model is finding the entire image (including the background) to be relevant for prediction. 


## CAM Demo with Webcam

We can apply the CAM technique to our own faces in real time through the user's webcam. You can see a short demo with this mp4 file: [CAM_webcam.mp4](/demo/CAM_webcam.mp4) or [here](https://user-images.githubusercontent.com/55476249/163021617-5f20ea2d-6683-4447-8c03-0743a89b1beb.mp4).

You can also try it yourself on your own system by running [pytorch_CAM.py](/pytorch_CAM.py).


## Model Bias Analysis

We analyze model performance across different groups, as well as the balanced and unbalanced models. You can find the analyses for the balanced and unbalanced data [here](/model_bias_analysis/celeba_dataset_analysis_balance.ipynb) and [here](/model_bias_analysis/celeba_dataset_analysis_unbalance.ipynb), respectively. These files are essentially the same, except the flag on cell 15 (use_balanced) is changed. Analysis starts from cell 26.

We find that the accuracy for males is greater than females (97.3% vs 96.6%) despite having fewer males in the training dataset (38.6% vs 61.4%). Our ResNet model was originally trained on ImageNet, which has images with human faces in [them](/demo/imagenet_collage.png). Since the CAM paper indicates that a network with a CNN "looks" at an entire image, imbalances in ImageNet may have led to an imbalance in the performance for our downstream facial recognition task. This shows that fundamental machine learning tools, like ImageNet, can cause biases in any future work built on them.

Balancing the number of males and females made the accuracies a little closer together, though curiously, it decreased both of the accuracies (96.9% vs 96.5% now). Also, interestingly, the accuracy is higher for older, darker males, and younger, paler females. However, the accuracy dropped for paler, younger females when the training set was balanced. Darker, younger females performed better when sex alone was balanced. Overall, it seems that balancing the training data for gender can cause unpredictable changes in accuracy for different intersectional groups (does not always improve accuracy), which shows that balance in dataset composition is a difficult problem to solve.


## Non-Binary Analysis
Since the model was trained on data which was labelled male or female (binary), we also evaluate the model on celebrities who identify as non-binary and analyze the results, both in terms of the prediction accuracy and the output CAM images. We also have a well-known public figure for comparison. The model predicts every single example as male, and the CAM images produced seem to focus on irrelevant parts of each image. Further results are at [non_binary_analysis](/non_binary_analysis/).

![Non-binary CAM result 1. Model focuses on irrelevant parts of image](/non_binary_analysis/results/1.jpg)


## References
The work on conditional activation maps is taken from [here](http://cnnlocalization.csail.mit.edu):
```
@inproceedings{zhou2016cvpr,
    author    = {Zhou, Bolei and Khosla, Aditya and Lapedriza, Agata and Oliva, Aude and Torralba, Antonio},
    title     = {Learning Deep Features for Discriminative Localization},
    booktitle = {Computer Vision and Pattern Recognition},
    year      = {2016}
}
```
