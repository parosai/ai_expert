

import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import shutil
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



IMG_WIDTH = 224
IMG_HEIGHT = 224


class Encoder(torch.nn.Module):
    def __init__(self, image_size=IMG_WIDTH*IMG_HEIGHT):
        super(Encoder, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

    def GetLatentLayer1(self, x):
        out = self.conv1(x)
        return out

    def GetLatentLayer2(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

    def GetLatentLayer3(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out



MODEL_PATH = './autoencoder_multilayer_epoch160.param'

encoder = Encoder()
encoder.load_state_dict(torch.load(MODEL_PATH))
encoder = encoder.to(device)
encoder.eval()



### template
# PATH_QUERY = '../dataset_crop/template/Center/'  ##### H.PARAM #####
# PATH_RESULT = './result_Center_weight_0.3_1_3/'

# PATH_QUERY = '../dataset_crop/template/Edge-Loc/'  ##### H.PARAM #####
# PATH_RESULT = './result_EdgeLoc_weight_0.3_1_3/'

# PATH_QUERY = '../dataset_crop/template/Loc/'  ##### H.PARAM #####
# PATH_RESULT = './result_Loc_weight_0.3_1_3/'

# PATH_QUERY = '../dataset_crop/template/Donut/'  ##### H.PARAM #####
# PATH_RESULT = './result_Donut_weight_0.3_1_3/'

# PATH_QUERY = '../dataset_crop/template/Edge-Ring/'  ##### H.PARAM #####
# PATH_RESULT = './result_EdgeRing_weight_0.3_1_3/'

PATH_QUERY = '../dataset_crop/template/Scratch/'    ##### H.PARAM #####
PATH_RESULT = './result_Scratch_weight_0.3_1_3/'





####(0,0,1), (1,1,1), (0.3,1,3)
alpha1=1
alpha2=1
alpha3=1


def GetQueryLatent(file):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_NEAREST)
    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    dataset_test = DataLoader(dataset=[img], batch_size=1, shuffle=False)
    latent_vector1 = 0
    latent_vector2 = 0
    latent_vector3 = 0
    ###

    for _, x in enumerate(dataset_test):
        x = np.array(x)
        x = torch.from_numpy(x).float()
        x = x.to(device)
        x = x.permute(0, 3, 1, 2)
        z1 = encoder.GetLatentLayer1(x)
        z1 = z1[0]
        z1 = z1.to('cpu')
        z1 = torch.autograd.Variable(z1, requires_grad=True)
        z1 = z1.detach().numpy()
        z1 = z1.flatten()
        z1 = z1 / LA.norm(z1)
        latent_vector1 = z1

        z2 = encoder.GetLatentLayer2(x)
        z2 = z2[0]
        z2 = z2.to('cpu')
        z2 = torch.autograd.Variable(z2, requires_grad=True)
        z2 = z2.detach().numpy()
        z2 = z2.flatten()
        z2 = z2 / LA.norm(z2)
        latent_vector2 = z2

        z3 = encoder.GetLatentLayer3(x)
        z3 = z3[0]
        z3 = z3.to('cpu')
        z3 = torch.autograd.Variable(z3, requires_grad=True)
        z3 = z3.detach().numpy()
        z3 = z3.flatten()
        z3 = z3 / LA.norm(z3)
        latent_vector3 = z3

    return latent_vector1, latent_vector2, latent_vector3



latent_vector = []

files = os.listdir(PATH_QUERY)
for f in files:
    latent_vector1, latent_vector2, latent_vector3 = GetQueryLatent(PATH_QUERY + f)
    ### concate
    tmp = np.concatenate([alpha1 * latent_vector1, alpha2 * latent_vector2, alpha3 * latent_vector3])
    latent_vector.append(tmp)


latent_vector = np.array(latent_vector)







PATH_TESTSET = '../dataset_crop/test/'     ##### H.PARAM #####

print('img load ing...')
testset = []  # grouped
testset_files = []
testset_files_img = []
folders = os.listdir(PATH_TESTSET)
for fold in folders:
    path = PATH_TESTSET + fold
    files = os.listdir(path)
    tmp = []
    for f in files:
        path_file = path + '/' + f
        tmp.append(path_file)
        testset_files.append(path_file)
        img = cv2.imread(path_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_NEAREST)
        img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        testset_files_img.append([img, fold])
    testset.append(tmp)

dataset_test = DataLoader(dataset=testset_files_img, batch_size=1, shuffle=False)



### Get latent vector
print('Get latent vector ing...')
latent_matrix_testset_grouped = {'Center':[], 'Donut':[], 'Edge-Loc':[], 'Edge-Ring':[], 'Loc':[], 'Random':[], 'Scratch':[]}
latent_matrix_testset1 = []
latent_matrix_testset2 = []
latent_matrix_testset3 = []
for _, [x, label] in enumerate(dataset_test):
    x = np.array(x)
    x = torch.from_numpy(x).float()
    x = x.to(device)
    x = x.permute(0, 3, 1, 2)

    z1 = encoder.GetLatentLayer1(x)
    z1 = z1[0]
    z1 = torch.flatten(z1)
    z1 = z1.to('cpu')
    z1 = torch.autograd.Variable(z1, requires_grad=True)
    z1 = z1.detach().numpy()
    z1 = z1 / LA.norm(z1)
    latent_matrix_testset1.append(z1)

    z2 = encoder.GetLatentLayer2(x)
    z2 = z2[0]
    z2 = torch.flatten(z2)
    z2 = z2.to('cpu')
    z2 = torch.autograd.Variable(z2, requires_grad=True)
    z2 = z2.detach().numpy()
    z2 = z2 / LA.norm(z2)
    latent_matrix_testset2.append(z2)

    z3 = encoder.GetLatentLayer3(x)
    z3 = z3[0]
    z3 = torch.flatten(z3)
    z3 = z3.to('cpu')
    z3 = torch.autograd.Variable(z3, requires_grad=True)
    z3 = z3.detach().numpy()
    z3 = z3 / LA.norm(z3)
    latent_matrix_testset3.append(z3)

    latent_matrix_testset_grouped[label[0]].append(z3)



### T-SNE
# print('T-SNE ing...')
# model = TSNE(learning_rate=100)
# scatters = []
# legends = []
# for label, tensor in latent_matrix_testset_grouped.items():
#     tensor = np.array(tensor)
#     transformed = model.fit_transform(tensor)
#     xs = transformed[:,0]
#     ys = transformed[:,1]
#     tmp = plt.scatter(xs, ys, s=8)
#     scatters.append(tmp)
#     legends.append(label)
# # plt.ylim([-65.0, 65.0])
# # plt.xlim([-65.0, 65.0])
# plt.legend(scatters, legends, loc="lower left")
# plt.show()





### concate
latent_matrix_testset_concated = []
for i in range(len(latent_matrix_testset1)):
    latent_matrix_testset_concated.append(np.concatenate([alpha1*latent_matrix_testset1[i], alpha2*latent_matrix_testset2[i], alpha3*latent_matrix_testset3[i]]))
latent_matrix_testset = np.array(latent_matrix_testset_concated)


### Calculate cos similarity
cos_similarity1 = np.dot(latent_vector[0], latent_matrix_testset.T) / (LA.norm(latent_vector[0])*LA.norm(latent_matrix_testset))
cos_similarity2 = np.dot(latent_vector[1], latent_matrix_testset.T) / (LA.norm(latent_vector[1])*LA.norm(latent_matrix_testset))
cos_similarity3 = np.dot(latent_vector[2], latent_matrix_testset.T) / (LA.norm(latent_vector[2])*LA.norm(latent_matrix_testset))
cos_similarity4 = np.dot(latent_vector[3], latent_matrix_testset.T) / (LA.norm(latent_vector[3])*LA.norm(latent_matrix_testset))
cos_similarity5 = np.dot(latent_vector[4], latent_matrix_testset.T) / (LA.norm(latent_vector[4])*LA.norm(latent_matrix_testset))


cos_similarity = []
for idx in range(len(cos_similarity1)):
    tmp = [cos_similarity1[idx], cos_similarity2[idx], cos_similarity3[idx], cos_similarity4[idx], cos_similarity5[idx]]
    tmp = max(tmp)
    cos_similarity.append(tmp)

similarity = np.array(cos_similarity)



#np.squeeze(similarity, axis=-1)
# sorted_index = np.argsort(similarity)[0][::-1]  # sort the scores
sorted_index = np.argsort(similarity)[::-1]
similarity = similarity[sorted_index]
### create sorted arr of similarity
sorted_similarity = np.zeros(len(similarity))
for idx in sorted_index:
    sorted_similarity[idx] = similarity[idx]



y_test = []
y_score = []


for idx, value in enumerate(sorted_similarity):
    # if idx >= K :
    #     break;

    if not (os.path.isdir(PATH_RESULT)):
        os.makedirs(os.path.join(PATH_RESULT))

    order = '{0:03d}'.format(idx)

    testset_path = testset_files[sorted_index[idx]]
    testset_path_splited = testset_path.split('/')
    testset_class = testset_path_splited[len(testset_path_splited)-2]
    testset_filename = testset_path_splited[len(testset_path_splited)-1]

    template_path = PATH_QUERY.split('/')
    template_class = template_path[len(template_path)-2]

    dst = PATH_RESULT + order + '_' + testset_class + '_' + testset_filename

    #shutil.copyfile(testset_path, dst)

    y_score.append([value*100])
    if (testset_class == template_class):
        y_test.append([1])
    else:
        y_test.append([0])



y_score = minmax_scale(y_score, feature_range=(0,1), axis=0)


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