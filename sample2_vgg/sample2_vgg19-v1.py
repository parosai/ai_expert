
import cv2
import os
import numpy as np
import numpy.linalg as LA
#import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
#from skimage.feature import hog
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import shutil

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='0'




class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained = True) # vgg 19 model is imported
        self.vgg19.classifier = self.vgg19.classifier[0:4]

    def forward(self, x):
        out = self.vgg19(x)
        return out


vgg19 = VGG19().cuda()
vgg19.eval()




def get_latent_vectors(path_img_files):
    n = len(path_img_files)
    latent_matrix = np.zeros((n, 4096))

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   ## ???

    for index, img_path in enumerate(path_img_files):
        image_np = Image.open(img_path)
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 224), mode='constant')
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_np = transform(image_np)
        image_np = Variable(image_np.unsqueeze(0))  # batch size, channel, height, width
        image_np = image_np.cuda()

        feature = vgg19(image_np)
        feature = feature.squeeze().cpu().data.numpy()
        feature = feature.reshape((1, 4096))  # Feature Flatten
        feature = feature / LA.norm(feature)  # Feature Normalization
        latent_matrix[index] = feature

        print(index, '/', n)
        #if (index >= 1000) :
        #    break;

    return latent_matrix





###  Testset
PATH_TESTSET = '../dataset_crop/test/'

testset_files = []
folders = os.listdir(PATH_TESTSET)
for fold in folders:
    path = PATH_TESTSET + fold
    files = os.listdir(path)
    for f in files:
        testset_files.append(path + '/' + f)

latent_matrix_testset = get_latent_vectors(testset_files)



### Template
PATH_TEMPLATE = '../dataset_crop/template/Edge-Loc/397.png'

template_files = []
template_files.append(PATH_TEMPLATE)

latent_matrix_templates = get_latent_vectors(template_files)



### Calculate cos similarity
cos_similarity = np.dot(latent_matrix_templates, latent_matrix_testset.T) / (LA.norm(latent_matrix_templates)*LA.norm(latent_matrix_testset))
sorted_index = np.argsort(cos_similarity)[0][::-1]  # sort the scores
cos_similarity = cos_similarity[0, sorted_index]



### save result
THRESHOLD = 0.03134
PATH_RESULT = './result/'
for idx, value in enumerate(cos_similarity):
    if (value < THRESHOLD):
        break

    if not (os.path.isdir(PATH_RESULT)):
        os.makedirs(os.path.join(PATH_RESULT))

    src = testset_files[sorted_index[idx]]
    tmp = src.split('/')
    dst = PATH_RESULT + tmp[len(tmp)-1]
    shutil.copyfile(src, dst)



print('Finished !')