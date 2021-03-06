
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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



BATCH_SIZE = 64
IMG_WIDTH = 224
IMG_HEIGHT = 224

class Encoder(torch.nn.Module):
    def __init__(self, image_size=IMG_WIDTH*IMG_HEIGHT):
        super(Encoder, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(8, 16, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(16, 16, 3, padding=1, bias=False), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Decoder(torch.nn.Module):
    def __init__(self, image_size=IMG_WIDTH*IMG_HEIGHT):
        super(Decoder, self).__init__()

        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 16, 3, 2, 1, 1, bias=False), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, 2, 1, 1, bias=False), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 3, 3, 2, 1, 1, bias=False), torch.nn.ReLU(),
        )


    def forward(self, z):
        out = self.deconv(z)
        return out


encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.train()
decoder.train()



parameters = list(encoder.parameters()) + list(decoder.parameters()) # 인코더 디코더의 파라미터를 동시에 학습시키기 위해 이를 묶음

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=0.0001)



### Load train img
PATH_TEMPLATE = '../dataset_crop/train/'     ##### H.PARAM #####


def load_images(path):
    images = []
    folders = os.listdir(path)
    for fold in folders:
        files = os.listdir(path + fold)
        for filename in files:
            template_img = cv2.imread(path+fold+'/'+filename, cv2.IMREAD_COLOR)
            template_img = cv2.resize(template_img, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_NEAREST)
            cv2.normalize(template_img, template_img, 0, 1, cv2.NORM_MINMAX)

            if template_img is not None:
                images.append(template_img)
    return images

dataset_template = DataLoader(dataset=load_images(PATH_TEMPLATE), batch_size=BATCH_SIZE, shuffle=True)




x_img = 0
y_img = 0
#latent_vector = 0

### Training
EPOCH = 100          ###  H.PARAM   ###
for epoch in range(EPOCH):
    loss = 0.
    for i, x in enumerate(dataset_template):
        x = np.array(x)
        #x = x[np.newaxis, :]
        x = torch.from_numpy(x).float()
        x = x.to(device)

        optimizer.zero_grad()
        x = x.permute(0, 3, 1, 2)
        z = encoder(x)
        #latent_vector = z
        output = decoder(z)

        x_img = x
        y_img = output

        loss = loss_func(output, x)
        loss.backward()
        optimizer.step()

    print("Epoch[{}/{}], Loss: {:.10f}".format(epoch + 1, EPOCH, loss))

x_img = x_img.permute(0, 2, 3, 1)
x_img = x_img.to('cpu')
y_img = y_img.permute(0, 2, 3, 1)
y_img = y_img.to('cpu')
y_img = torch.autograd.Variable(y_img, requires_grad=True)
y_img = y_img.detach().numpy()

fig = plt.figure()
fig_img = fig.add_subplot(1, 2, 1)

fig_img.imshow(x_img[0])
fig_img = fig.add_subplot(1, 2, 2)
fig_img.imshow(y_img[0])
plt.show()


#### save weight



##### Evaluation Testset !
encoder.eval()
decoder.eval()


### template
PATH_QUERY = '../dataset_crop/template/Loc/7610.png'     ##### H.PARAM #####
img = cv2.imread(PATH_QUERY, cv2.IMREAD_COLOR)
img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), cv2.INTER_NEAREST)
img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
dataset_test = DataLoader(dataset=[img], batch_size=1, shuffle=False)
latent_vector = 0
for _, x in enumerate(dataset_test):
    x = np.array(x)
    x = torch.from_numpy(x).float()
    x = x.to(device)
    x = x.permute(0, 3, 1, 2)
    z = encoder(x)
    z = z[0]

    z = z.to('cpu')
    z = torch.autograd.Variable(z, requires_grad=True)
    z = z.detach().numpy()

    z = z.flatten()

    z = z / LA.norm(z)

    # z = z[:]

    # z.reshape(12544, 1)

    latent_vector = z




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
latent_matrix_testset = []
for _, [x, label] in enumerate(dataset_test):
    # x_img = dataset_template.dataset[0]
    # x_img = x_img[np.newaxis, :]
    x = np.array(x)
    #x = x[np.newaxis, :]
    x = torch.from_numpy(x).float()
    x = x.to(device)
    x = x.permute(0, 3, 1, 2)
    z = encoder(x)
    z = z[0]
    #z = z.view(-1)
    z = torch.flatten(z)

    z = z.to('cpu')
    z = torch.autograd.Variable(z, requires_grad=True)
    z = z.detach().numpy()

    z = z / LA.norm(z)

    latent_matrix_testset_grouped[label[0]].append(z)
    latent_matrix_testset.append(z)



### T-SNE
print('T-SNE ing...')
model = TSNE(learning_rate=100)
scatters = []
legends = []
for label, tensor in latent_matrix_testset_grouped.items():
    tensor = np.array(tensor)
    transformed = model.fit_transform(tensor)
    xs = transformed[:,0]
    ys = transformed[:,1]
    tmp = plt.scatter(xs, ys, s=8)
    scatters.append(tmp)
    legends.append(label)
# plt.ylim([-65.0, 65.0])
# plt.xlim([-65.0, 65.0])
plt.legend(scatters, legends, loc="lower left")
plt.show()



### Calculate cos similarity
latent_matrix_testset = np.array(latent_matrix_testset)

#cos_similarity = np.dot(latent_vector, latent_matrix_testset.T) / (LA.norm(latent_vector)*LA.norm(latent_matrix_testset))
similarity = np.dot(latent_vector, latent_matrix_testset.T)
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

K = 51                      ##### H.PARAM #####
PATH_RESULT = './result/'
for idx, value in enumerate(sorted_similarity):
    if idx >= K :
        break;

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