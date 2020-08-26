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
tune_fc_layer = True
dataset_dir = 'augment'


def weights_init_uniform_rule(m): # weight init
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

# Reference code from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)  # vgg 19 model is imported
        self.model.fc = nn.Linear(N_DIMS, NUM_CLASSES)
        if tune_fc_layer:
            self.model.fc = nn.Sequential(
                nn.Linear(N_DIMS, N_DIMS),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(N_DIMS, N_DIMS),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(N_DIMS, NUM_CLASSES)
            )
        if not tune_conv_layer:
            for name, p in self.model.named_parameters():  # freeze conv layer param
                if 'fc' not in name:
                    p.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return out


resnet = ResNet18().cuda()
# resnet.eval()
if tune_fc_layer:
    resnet.apply(weights_init_uniform_rule)

model = resnet
criterion = nn.CrossEntropyLoss().cuda()
if tune_conv_layer:
    params = list(map(lambda x: x[1], list(filter(lambda kv: 'fc' not in kv[0], model.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: 'fc' in kv[0], model.named_parameters()))))
    # optimizer = optim.Adam([{'params': params, 'lr': 1e-7}], lr=0.00001)
    optimizer = optim.Adam([{'params': base_params}, {'params': params, 'lr': 1e-7}], lr=0.00001)
else:
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
if dataset_dir == 'augment':
    data_dir = '../dataset_v2_augment/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in
    ['train', 'valid']}

writer = SummaryWriter(log_dir='runs_resnet')


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
            batch_cnt = 0
            for inputs, labels in dataloaders[phase]:
                print(str(batch_cnt), end='\r')
                batch_cnt += 1

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
            writer.add_scalar("Loss/{}".format(phase), epoch_loss, epoch)
            writer.add_scalar("Acc/{}".format(phase), epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
        if epoch % 30 == 0:
            if tune_conv_layer:
                TMP_MODEL_PATH = './model_resnet_conv_ft_tmp.pth'
                if tune_fc_layer:
                    TMP_MODEL_PATH = './model_resnet_fc_ft_tmp.pth'
            else:
                TMP_MODEL_PATH = './model_resnet_ft_tmp.pth'
            tmp_best_model = copy.deepcopy(model)
            tmp_best_model.load_state_dict(best_model_wts)
            torch.save(tmp_best_model, TMP_MODEL_PATH)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    writer.flush()
    writer.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if tune_conv_layer:
    MODEL_PATH = './model_resnet_conv_ft.pth'
    if tune_fc_layer:
        MODEL_PATH = './model_resnet_fc_ft.pth'
else:
    MODEL_PATH = './model_resnet_ft.pth'
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

if tune_fc_layer:
    model_ft.model.fc = model_ft.model.fc[:4]
else:
    model_ft.model.fc = Identity()

model_ft.eval()

loop = 0


def get_latent_vectors(path_img_files, model):
    global loop

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
# PATH_TEMPLATE = '../dataset_crop/template/Donut/7907.png'  ##### H.PARAM #####
# PATH_TEMPLATE = '../dataset_crop/template/Donut/639390.png'    ##### H.PARAM #####
# PATH_TEMPLATE = '../dataset_crop/template/Edge-Loc/682398.png'    ##### H.PARAM #####
PATH_TEMPLATE = '../dataset_crop/template/Edge-Loc/397.png'    ##### H.PARAM #####
# PATH_TEMPLATE = '../dataset_crop/template/Loc/7610.png'    ##### H.PARAM #####
# PATH_TEMPLATE = '../dataset_crop/template/Loc/7553.png'    ##### H.PARAM #####
# PATH_TEMPLATE = '../dataset_crop/template/Scratch/135.png'  ##### H.PARAM #####
# PATH_TEMPLATE = '../dataset_crop/template/Scratch/15909.png'  ##### H.PARAM #####
# PATH_TEMPLATE = '../dataset_crop/template/Edge-Ring/12634.png'

template_files = []
template_files.append(PATH_TEMPLATE)

latent_matrix_templates = get_latent_vectors(template_files, model_ft)

###  Testset
PATH_TESTSET = '../dataset_crop/test/'  ##### H.PARAM #####

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
model = TSNE(learning_rate=100)
for group in latent_matrix_testset_grouped:
    transformed = model.fit_transform(group)
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    plt.scatter(xs, ys)
plt.show()

### Calculate cos similarity
cos_similarity = np.dot(latent_matrix_templates, latent_matrix_testset.T) / (
        LA.norm(latent_matrix_templates) * LA.norm(latent_matrix_testset))
sorted_index = np.argsort(cos_similarity)[0][::-1]  # sort the scores
cos_similarity = cos_similarity[0, sorted_index]
### create sorted arr of cos_similarity
sorted_cos_similarity = np.zeros(len(cos_similarity))
for idx in sorted_index:
    sorted_cos_similarity[idx] = cos_similarity[idx]

y_test = []
y_score = []

K = 51  ##### H.PARAM #####
PATH_RESULT = './result/'
for idx, value in enumerate(sorted_cos_similarity):
    if idx >= K:
        break;

    if not (os.path.isdir(PATH_RESULT)):
        os.makedirs(os.path.join(PATH_RESULT))

    order = '{0:03d}'.format(idx)

    testset_path = testset_files[sorted_index[idx]]
    testset_path_splited = testset_path.split('/')
    testset_class = testset_path_splited[len(testset_path_splited) - 2]
    testset_filename = testset_path_splited[len(testset_path_splited) - 1]

    template_path = PATH_TEMPLATE.split('/')
    template_class = template_path[len(template_path) - 2]

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
# plt.legend(loc="lower left")
plt.show()

print('Finished !')
