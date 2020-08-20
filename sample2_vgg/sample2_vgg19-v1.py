
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

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE



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
#vgg19.training(False)
vgg19.eval()



loop = 0
def get_latent_vectors(path_img_files):
    global  loop

    n = len(path_img_files)
    latent_matrix = np.zeros((n, 4096))

    #transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   ## ???

    for index, img_path in enumerate(path_img_files):
        image_np = Image.open(img_path)
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 224), mode='constant')
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        #image_np = transform(image_np)
        image_np = Variable(image_np.unsqueeze(0))  # batch size, channel, height, width
        image_np = image_np.cuda()

        feature = vgg19(image_np)
        feature = feature.squeeze().cpu().data.numpy()
        feature = feature.reshape((1, 4096))  # Feature Flatten
        feature = feature / LA.norm(feature)  # Feature Normalization
        latent_matrix[index] = feature

        print(str(loop))
        loop += 1

        #if (index >= 1000) :
        #    break;

    return latent_matrix



### Template
PATH_TEMPLATE = '../dataset_crop/template/Donut/7907.png'    ##### H.PARAM #####

template_files = []
template_files.append(PATH_TEMPLATE)

latent_matrix_templates = get_latent_vectors(template_files)



###  Testset
PATH_TESTSET = '../dataset_crop/test/'     ##### H.PARAM #####

testset = []
testset_files = []
folders = os.listdir(PATH_TESTSET)
for fold in folders:
    path = PATH_TESTSET + fold
    files = os.listdir(path)
    tmp = []
    for f in files:
        tmp.append(path + '/' + f)
        testset_files.append(path + '/' + f)
    testset.append(tmp)



### Get latent vector
latent_matrix_testset_grouped = []
latent_matrix_testset = []
for files in testset:
    results = get_latent_vectors(files)
    latent_matrix_testset_grouped.append(results)

    for item in results:
        latent_matrix_testset.append(item)


latent_matrix_testset_grouped = np.array(latent_matrix_testset_grouped)
latent_matrix_testset = np.array(latent_matrix_testset)



### T-SNE
print('Start T-SNE')
model = TSNE(learning_rate=100)
for group in latent_matrix_testset_grouped:
    transformed = model.fit_transform(group)
    xs = transformed[:,0]
    ys = transformed[:,1]
    plt.scatter(xs,ys)
plt.show()



### Calculate cos similarity
cos_similarity = np.dot(latent_matrix_templates, latent_matrix_testset.T) / (LA.norm(latent_matrix_templates)*LA.norm(latent_matrix_testset))
sorted_index = np.argsort(cos_similarity)[0][::-1]  # sort the scores
cos_similarity = cos_similarity[0, sorted_index]
### create sorted arr of cos_similarity
sorted_cos_similarity = np.zeros(len(cos_similarity))
for idx in sorted_index:
    sorted_cos_similarity[idx] = cos_similarity[idx]



y_test = []
y_score = []

K = 51   ##### H.PARAM #####
PATH_RESULT = './result/'
for idx, value in enumerate(sorted_cos_similarity):
    if idx >= K :
        break;

    if not (os.path.isdir(PATH_RESULT)):
        os.makedirs(os.path.join(PATH_RESULT))

    order = '{0:03d}'.format(idx)

    testset_path = testset_files[sorted_index[idx]]
    testset_path_splited = testset_path.split('/')
    testset_class = testset_path_splited[len(testset_path_splited)-2]
    testset_filename = testset_path_splited[len(testset_path_splited)-1]

    template_path = PATH_TEMPLATE.split('/')
    template_class = template_path[len(template_path)-2]

    dst = PATH_RESULT + order + '_' + testset_class + '_' + testset_filename

    shutil.copyfile(testset_path, dst)

    y_score.append(value)
    if (testset_class == template_class):
        y_test.append([1])
    else:
        y_test.append([0])



### precision recall curv
precision, recall, _ = precision_recall_curve(y_test, y_score)
average_precision = average_precision_score(y_test, y_score)

plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
#plt.legend(loc="lower left")
plt.show()



print('Finished !')