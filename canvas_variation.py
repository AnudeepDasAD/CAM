import io
from PIL import Image
import numpy as np
import cv2
import torch
import pandas as pd
import os
import csv
import random

data_dir = "D:\\Users\\Anude\\Documents\\CS 497\\CAM\\celebA\\img_align_celeba\\img_align_celeba"
test_df = pd.read_pickle('test.pickle')
test_file_endings = list(test_df.drop('labels',1).index)
test_file_names_full = [os.path.join(data_dir,name) for name in test_file_endings]

height = 400
width = 400

for i, image_file in enumerate(test_file_names_full):
    background = np.zeros((height, width, 3), np.uint8)
    background[:, 0:width] = (255, 255, 255)

    img_file = cv2.imread(image_file)
    x_offset = random.randint(0,150)
    y_offset = random.randint(0,150)

    background[y_offset:y_offset+img_file.shape[0], x_offset:x_offset+img_file.shape[1]] = img_file

    cv2.imwrite(f'./test_images_resized/{test_file_endings[i]}', background)
