import cv2
import os
import numpy as np
import numpy.linalg as LA
# import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import shutil
import time
import copy
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE

import warnings

N_DIMS = 512

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CLASSES = 7
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
model_load = True
tune_conv_layer = True
#dataset_dir = 'augment'


# Reference code from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)  # vgg 19 model is imported
        self.model.fc = nn.Linear(N_DIMS, NUM_CLASSES)
        if not tune_conv_layer:
            for name, p in self.model.named_parameters():  # freeze conv layer param
                if 'fc' not in name:
                    p.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return out


resnet = ResNet18().cuda()
# resnet.eval()

model = resnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE)),
        transforms.ToTensor(),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((IMG_SIZE)),
        transforms.ToTensor(),
    ]),
}

# Create training and validation datasets
# data_dir = '../dataset_crop/'
# # if dataset_dir == 'augment':
# #     data_dir = '../dataset_v2_augment/'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# # Create training and validation dataloaders
# dataloaders_dict = {
#     x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in
#     ['train', 'valid']}

writer = SummaryWriter(log_dir='runs_resnet')


if tune_conv_layer:
    MODEL_PATH = './model_resnet_ft_aug.pth'
else:
    MODEL_PATH = './model_resnet_ft1.pth'
if not model_load:
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=210)
    torch.save(model_ft, MODEL_PATH)
else:
    model_ft = torch.load(MODEL_PATH)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model_ft.model.fc = Identity()
model_ft.eval()



loop = 0
def get_latent_vectors(path_img_files, model):
    global  loop

    n = len(path_img_files)
    latent_matrix = np.zeros((n, N_DIMS))

    # transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   ## ???

    for index, img_path in enumerate(path_img_files):
        image_np = Image.open(img_path)
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 224), mode='constant')
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        # image_np = transform(image_np)
        image_np = Variable(image_np.unsqueeze(0))  # batch size, channel, height, width
        image_np = image_np.cuda()

        feature = model(image_np)
        feature = feature.squeeze().cpu().data.numpy()
        feature = feature.reshape((1, N_DIMS))  # Feature Flatten
        feature = feature / LA.norm(feature)  # Feature Normalization
        latent_matrix[index] = feature

        print(str(loop))
        loop += 1

        # if (index >= 1000) :
        #    break;

    return latent_matrix




### Template
# PATH_TEMPLATE = '../dataset_crop/template/Center/'  ##### H.PARAM #####
# K = 425

# PATH_TEMPLATE = '../dataset_crop/template/Edge-Loc/'  ##### H.PARAM #####
# K = 515

# PATH_TEMPLATE = '../dataset_crop/template/Loc/'  ##### H.PARAM #####
# K = 355

# PATH_TEMPLATE = '../dataset_crop/template/Donut/'  ##### H.PARAM #####
# K = 51

# PATH_TEMPLATE = '../dataset_crop/template/Edge-Ring/'  ##### H.PARAM #####
# K = 963

PATH_TEMPLATE = '../dataset_crop/template/Scratch/'    ##### H.PARAM #####
K = 114


query_latents = []

files = os.listdir(PATH_TEMPLATE)

for item in files:
    #template_files.append(PATH_TEMPLATE + item)
    tmp = get_latent_vectors([PATH_TEMPLATE+item], model_ft)
    tmp = tmp.flatten()
    query_latents.append(tmp)

query_latents = np.array(query_latents)

#latent_matrix_templates = (query_latents[0] + query_latents[1] + query_latents[2] + query_latents[3] + query_latents[4]) / 5
#latent_matrix_templates = latent_matrix_templates[np.newaxis, :]

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
    results = get_latent_vectors(files, model_ft)
    latent_matrix_testset_grouped.append(results)

    for item in results:
        latent_matrix_testset.append(item)


latent_matrix_testset_grouped = np.array(latent_matrix_testset_grouped)
latent_matrix_testset = np.array(latent_matrix_testset)



### T-SNE
print('Start T-SNE')
# model = TSNE(learning_rate=100)
# scatters = []
# legends = []
# for idx, group in enumerate(latent_matrix_testset_grouped):
#     transformed = model.fit_transform(group)
#     xs = transformed[:,0]
#     ys = transformed[:,1]
#     tmp = plt.scatter(xs, ys, s=10)
#     scatters.append(tmp)
#     legends.append(folders[idx])
# plt.legend(loc="lower left")
# plt.ylim([-65.0, 65.0])
# plt.xlim([-65.0, 65.0])
# plt.legend(scatters, legends)
# plt.show()





### Calculate cos similarity
cos_similarity1 = np.dot(query_latents[0], latent_matrix_testset.T) / (LA.norm(query_latents[0])*LA.norm(latent_matrix_testset))
cos_similarity2 = np.dot(query_latents[1], latent_matrix_testset.T) / (LA.norm(query_latents[1])*LA.norm(latent_matrix_testset))
cos_similarity3 = np.dot(query_latents[2], latent_matrix_testset.T) / (LA.norm(query_latents[2])*LA.norm(latent_matrix_testset))
cos_similarity4 = np.dot(query_latents[3], latent_matrix_testset.T) / (LA.norm(query_latents[3])*LA.norm(latent_matrix_testset))
cos_similarity5 = np.dot(query_latents[4], latent_matrix_testset.T) / (LA.norm(query_latents[4])*LA.norm(latent_matrix_testset))


cos_similarity = []
for idx in range(len(cos_similarity1)):
    tmp = [cos_similarity1[idx], cos_similarity2[idx], cos_similarity3[idx], cos_similarity4[idx], cos_similarity5[idx]]
    tmp = max(tmp)
    cos_similarity.append(tmp)

cos_similarity = np.array(cos_similarity)

# cos_similarity = (cos_similarity1 + cos_similarity2 + cos_similarity3 + cos_similarity4 + cos_similarity5) / 5
cos_similarity = cos_similarity[np.newaxis, :]

sorted_index = np.argsort(cos_similarity)[0][::-1]  # sort the scores
cos_similarity = cos_similarity[0, sorted_index]
### create sorted arr of cos_similarity
sorted_cos_similarity = np.zeros(len(cos_similarity))
for idx in sorted_index:
    sorted_cos_similarity[idx] = cos_similarity[idx]



y_test = []
y_score = []


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

    y_score.append([value])
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











