import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import torch
import pandas as pd
import os
import csv

# input image
LABELS_file = 'imagenet-simple-labels.json'
# image_file = 'testBiden.jpg'

data_dir = "D:\\Users\\Anude\\Documents\\CS 497\\CAM\\celebA\\img_align_celeba\\img_align_celeba"
test_df = pd.read_pickle('test.pickle')
test_file_endings = list(test_df.drop('labels',1).index)
test_file_names_full = [os.path.join(data_dir,name) for name in test_file_endings]

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    # net = torch.load('resnet_bsize8_epoch5_full_1.pt')
    net = torch.load('resnet_bsize8_epoch5_full_1_bal.pt')
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
net.eval()
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules["0"]._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-4].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
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
# with open(LABELS_file) as f:
#     classes = json.load(f)

classes = ['Female','Male']
predictions = []

for i, image_file in enumerate(test_file_names_full):
    # load test image
    img_pil = Image.open(image_file)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)
    index_pred = torch.argmax(logit).item()
    predictions.append(index_pred)


    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    

    # output the prediction
    # for i in range(0, 5):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    

    # render the CAM and output
    
    print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(image_file)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(f'./results_test_balance/{test_file_endings[i]}', result)

    # net = models.resnet18(pretrained=True)
    # net = torch.load('resnet_bsize8_epoch5_full_1_bal.pt')
    # finalconv_name = 'layer4'
    # feature_blobs = []
    # net._modules["0"]._modules.get(finalconv_name).register_forward_hook(hook_feature)
    # params = list(net.parameters())
    # weight_softmax = np.squeeze(params[-4].data.numpy())
    

# pred_df = pd.DataFrame(predictions, columns=['predictions'])
# pred_df.to_csv('./results/predictions.csv', index=False)
with open("predictions_test_balance.csv","w") as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(predictions)