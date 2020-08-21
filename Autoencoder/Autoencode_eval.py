####  https://github.com/Puayny/Autoencoder-image-similarity/

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import cv2
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import shutil
import numpy.linalg as LA
#import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE




# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# # Create a directory if not exists
# sample_dir = 'samples'
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)
# model_path_dir = 'saved_model'
# # Hyper-parameters
# image_size = 224*224
# h_dim = 400
# z_dim = 20
# num_epochs = 200
# # batch_size = 128
# batch_size = 64
# learning_rate = 1e-3



#train_folder = '/home/com15/ai_expert/dataset_crop/Train_Together/'

# training##MNIST### dataset
# def load_images(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = mpimg.imread(os.path.join(folder, filename))
#
#         # img resize
#         img = resize(img, (224, 224), mode='constant')
#
#         if img is not None:
#             images.append(img)
#     return images
#
# # print(load_images(train_folder))
#
#
# #
# # dataset = torchvision.datasets(root='../dataset_crop/train/All_Together',
# #                                      train=True,
# #                                      transform=transforms.ToTensor(),
# #                                      download=True)
# # Data loader
# data_loader = DataLoader(dataset=load_images(train_folder),
#                                           batch_size=batch_size,
#                                           shuffle=True)

# print(data_loader)
# assert False
# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=224*224, h_dim=100, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


#model = VAE().to(device)




MODEL_PATH = './autoencoder_epoch119.pth'

model_ft = torch.load(MODEL_PATH)
model_ft.eval()

model_ft.VAE.classifier = model_ft.VAE.classifier[:2]








loop = 0
def get_latent_vectors(path_img_files, model):
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

        feature = model(image_np)
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
PATH_TEMPLATE = '../dataset_crop/template/Scratch/135.png'     ##### H.PARAM #####

template_files = []
template_files.append(PATH_TEMPLATE)

latent_matrix_templates = get_latent_vectors(template_files, model_ft)



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
model = TSNE(learning_rate=100)
scatters = []
legends = []
for idx, group in enumerate(latent_matrix_testset_grouped):
    transformed = model.fit_transform(group)
    xs = transformed[:,0]
    ys = transformed[:,1]
    tmp = plt.scatter(xs, ys, s=8)
    scatters.append(tmp)
    legends.append(folders[idx])
plt.legend(scatters, legends)
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

K = 114                      ##### H.PARAM #####
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




