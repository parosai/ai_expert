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

import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
NUM_CLASSES = 7
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
model_load = True

# Reference code from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)  # vgg 19 model is imported
        self.vgg19.classifier = self.vgg19.classifier[0:6]
        self.vgg19.classifier.add_module('6', nn.Linear(4096, NUM_CLASSES))  # modify output class num
        for p in self.vgg19.features.parameters():  # freeze conv layer param
            p.requires_grad = False

    def forward(self, x):
        out = self.vgg19(x)
        return out


vgg19 = VGG19().cuda()
vgg19.eval()

model = vgg19
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

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
data_dir = '../dataset_crop/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in
    ['train', 'valid']}


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


MODEL_PATH = './model_ft.pth'
if not model_load:
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=500)
    torch.save(model_ft, MODEL_PATH)
else:
    model_ft = torch.load(MODEL_PATH)
model_ft.vgg19.classifier = model_ft.vgg19.classifier[:4]
model_ft.eval()


def get_latent_vectors(path_img_files, model):
    n = len(path_img_files)
    latent_matrix = np.zeros((n, 4096))

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ## ???

    for index, img_path in enumerate(path_img_files):
        image_np = Image.open(img_path)
        image_np = np.array(image_np)
        image_np = resize(image_np, (224, 224), mode='constant')
        image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
        image_np = transform(image_np)
        image_np = Variable(image_np.unsqueeze(0))  # batch size, channel, height, width
        image_np = image_np.cuda()

        feature = model(image_np)
        feature = feature.squeeze().cpu().data.numpy()
        feature = feature.reshape((1, 4096))  # Feature Flatten
        feature = feature / LA.norm(feature)  # Feature Normalization
        latent_matrix[index] = feature

        print(index, '/', n)
        # if (index >= 1000) :
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

latent_matrix_testset = get_latent_vectors(testset_files, model_ft)

# ### Template
PATH_TEMPLATE = '../dataset_crop/template/Edge-Loc/397.png'

template_files = []
template_files.append(PATH_TEMPLATE)

latent_matrix_templates = get_latent_vectors(template_files, model_ft)

### Calculate cos similarity
cos_similarity = np.dot(latent_matrix_templates, latent_matrix_testset.T) / (
        LA.norm(latent_matrix_templates) * LA.norm(latent_matrix_testset))
sorted_index = np.argsort(cos_similarity)[0][::-1]  # sort the scores
cos_similarity = cos_similarity[0, sorted_index]

### save result
THRESHOLD = 0.01935
PATH_RESULT = './result/'
for idx, value in enumerate(cos_similarity):
    if (value < THRESHOLD):
        break

    if not (os.path.isdir(PATH_RESULT)):
        os.makedirs(os.path.join(PATH_RESULT))

    src = testset_files[sorted_index[idx]]
    tmp = src.split('/')
    dst = PATH_RESULT + tmp[len(tmp) - 1]
    shutil.copyfile(src, dst)

print('Finished !')
