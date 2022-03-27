# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io
from PIL import Image
from torchvision import models, transforms, datasets
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import os
import torch
import json

# input image
LABELS_file = 'imagenet-simple-labels.json'
image_file = 'test.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 1
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# This allows us to get the output blobs?
net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight- the second last param
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    # batch size? num_channels, height, width
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        # weight_softmax[idx] holds the probability that the ground truth classification
        #   if the image is idx
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


# load the imagenet category list
with open(LABELS_file) as f:
    classes = json.load(f)



# Attempting to access webcam
cap = cv2.VideoCapture(0)

while True:
    #reads the code frame-by-frame
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    gray_pil = Image.fromarray(gray)

    img_tensor = preprocess(gray_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    # img = cv2.imread('test.jpg')
    height, width, _ = gray.shape
    resized = cv2.resize(CAMs[0],(width, height))
    heatmap = cv2.applyColorMap(resized, cv2.COLORMAP_JET)

    camresult = cv2.addWeighted(heatmap, 0.3, frame, 0.5,0)

    # Originally, the CAM.jpg worked with "result". Webcam does not look additions and multiplications
    # result = heatmap * 0.3 + frame * 0.5
    cv2.imwrite('CAM2.jpg', camresult)
    cv2.imshow('CAM', camresult)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# output the prediction
# for i in range(0, 5):
#     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))


