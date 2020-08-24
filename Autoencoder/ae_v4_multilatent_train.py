
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
PATH_TEMPLATE = '../dataset_crop/valid/'     ##### H.PARAM #####


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
EPOCH = 21          ###  H.PARAM   ###
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

    if (epoch + 1) % 20 == 0:
        #### save weight
        model = encoder.cpu()
        torch.save(model.state_dict(), './autoencoder_multilayer_epoch' + str(epoch+1) + '.param')
        model = model.to(device)








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






print('Finished !')