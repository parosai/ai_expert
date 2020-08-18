import os
import sys
import numpy as np 
import cv2
import tensorflow as tf 
#from tensorflow.keras import Model
#from keras.layers import Dense, Flatten, Conv2D, Flatten
#from keras.models import Sequential, Model


def ConvertImg(img):
    height, width, channel = img.shape

    LT_x = 0
    LT_y = 0
    for y in range(0, height):
        for x in range(0, width):
            b = img.item(y, x, 0)
            g = img.item(y, x, 1)
            r = img.item(y, x, 2)
            if r >= 50 and b >= 50 and g <= 10:
                LT_x = x
                LT_y = y
                break;
        if LT_x >= 1:
            break;

    RB_x = 0
    RB_y = 0
    for y in range(height-1, 0, -1):
        for x in range(width-1, 0, -1):
            b = img.item(y, x, 0)
            g = img.item(y, x, 1)
            r = img.item(y, x, 2)
            if r >= 50 and b >= 50 and g <= 10:
                RB_x = x
                RB_y = y
                break;
        if RB_x >= 1:
            break;

    # height = RB_y - LT_y + 1
    # width = RB_x - LT_x + 1
    # conved_img = np.zeros((height, width, 3), np.uint8)
    #
    # for y in range(LT_y, RB_y+1):
    #     for x in range(LT_x, RB_x+1):
    #         b = img.item(y, x, 0)
    #         g = img.item(y, x, 1)
    #         r = img.item(y, x, 2)
    #
    #         if r >= 150 and g >= 150:
    #             conved_img[y - LT_y, x - LT_x, 2] = 255
    #         elif g >= 100 and b >= 100:
    #             conved_img[y - LT_y, x - LT_x, 0] = 255
    #             conved_img[y - LT_y, x - LT_x, 1] = 255
    #             conved_img[y - LT_y, x - LT_x, 2] = 255

    return img[LT_y:RB_y, LT_x:RB_x]


path_src = 'C:/MyWorks/ai_expert/mini_project/dataset/training'
path_dst = 'C:/MyWorks/ai_expert/mini_project/dataset_crop/training'

folds = os.listdir(path_src)

for fold in folds:
    files = os.listdir(path_src + '/' + fold)

    if not (os.path.isdir(path_dst + '/' + fold)):
        os.makedirs(os.path.join(path_dst + '/' + fold))

    for file in files:
        path_img = path_src + '/' + fold + '/' + file
        img = cv2.imread(path_img)
        img = ConvertImg(img)

        path_save = path_dst + '/' + fold + '/' + file
        cv2.imwrite(path_save, img)


print('Finished!')


#encoder = Sequential()
#encoder.add(Dense(512, activation='relu', input_shape=(224*224, )))



