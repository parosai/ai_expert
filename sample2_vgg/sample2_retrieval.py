
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

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='0'



class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained = True) # vgg 19 model is imported
        self.vgg19.classifier = self.vgg19.classifier[0:4]

    def forward(self, x):
        # for layer in self.trucated_classifer:
        #     x = layer(x)
        out = self.vgg19(x)
        return out

# Set our model with pre-trained model
vgg19 = VGG19().cuda()




def extract_deep_features(path, feature_extractor, feature_size):
    #start_time = time.time()

    list_imgs_names = os.listdir(path)  # list_imgs_names
    N = len(list_imgs_names)
    feature_all = np.zeros((N, feature_size))  # create an array to store features
    image_all = []  # define empy array to store image names

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # extract features
    for index, img_name in enumerate(list_imgs_names):
        img_path = os.path.join(path, img_name)

        # Image Read & Resize
        image_np = Image.open(img_path)  # Read the images
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 224), mode='constant')  # Resize the images
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_np = transform(image_np)
        image_np = Variable(image_np.unsqueeze(0))  # bs, c, h, w
        image_np = image_np.cuda()

        # Extract Feature
        feature = feature_extractor(image_np)
        feature = feature.squeeze().cpu().data.numpy()
        feature = feature.reshape((1, feature_size))  # Feature Flatten
        feature = feature / LA.norm(feature)  # Feature Normalization
        feature_all[index] = feature
        image_all.append(img_name)

    #time_elapsed = time.time() - start_time

    #print('Feature extraction complete in {:.02f}s'.format(time_elapsed % 60))

    return feature_all, image_all




def test_deep_feature(feature_extractor, feature_size):
    # Extract features from the dataset
    # Extract features from data
    path = './db'
    feats, image_list = extract_deep_features(path, feature_extractor, feature_size=feature_size)

    # test image path
    # Extract features from query image
    test = './templates'
    feat_single, image = extract_deep_features(test, feature_extractor, feature_size=feature_size)

    # Calculate the scores
    scores = np.dot(feat_single, feats.T)
    sort_ind = np.argsort(scores)[0][::-1]  # sort the scores
    scores = scores[0, sort_ind]

    # Show the results
    maxres = 10
    imlist = [image_list[index] for i, index in enumerate(sort_ind[0:maxres])]
    print("top %d images in order are: " % maxres, imlist)

    fig = plt.figure(figsize=(16, 10))
    for i in range(len(imlist)):
        sample = imlist[i]
        img = mpimg.imread('./db' + '/' + sample)
        ax = fig.add_subplot(2, 5, i + 1)
        ax.autoscale()
        plt.tight_layout()
        plt.imshow(img, interpolation='nearest')
        ax.set_title('{:.3f}%'.format(scores[i]))
        ax.axis('off')
    plt.show()



test_deep_feature(vgg19, feature_size=4096)