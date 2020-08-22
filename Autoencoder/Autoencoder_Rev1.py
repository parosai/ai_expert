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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
model_path_dir = 'saved_model'
# Hyper-parameters
image_size = 224*224
h_dim = 400
z_dim = 20
num_epochs = 200
# batch_size = 128
batch_size = 64
learning_rate = 1e-3

train_folder = '/home/com15/ai_expert/dataset_crop/Train_Together/'

# training##MNIST### dataset
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))

        # img resize
        img = resize(img, (224, 224), mode='constant')

        if img is not None:
            images.append(img)
    return images

# print(load_images(train_folder))


#
# dataset = torchvision.datasets(root='../dataset_crop/train/All_Together',
#                                      train=True,
#                                      transform=transforms.ToTensor(),
#                                      download=True)
# Data loader
data_loader = DataLoader(dataset=load_images(train_folder),
                                          batch_size=batch_size,
                                          shuffle=True)

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


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Python 에서 파일을 여실 때에는 with 문을 쓰는 게 좋습니다
# with 문으로 연 파일 스트림은 프로그램이 정상적으로 종료되지 않아도 안전하게 닫히거든요
# f = open('./train_log.txt', 'a')
with open('./train_log.txt', 'a') as f:
    f.write('----------------------start-------------------')

    # Start training
    for epoch in range(num_epochs):
        for i, x in enumerate(data_loader):
            # Forward pass
            image_np = np.array(x)
            # print(data_loader)
            #print("image_np", image_np, image_np.shape)

            x = x.to(device).view(-1, image_size)
            x_reconst, mu, log_var = model(x)


            # assert False
            # image_np = torch.from_numpy(image_np).permute(2, 0, 1).float()
            # image_np = transform(image_np)
            # image_np = Variable(image_np.squeeze(0))  # batch size, channel, height, width
            # print(image_np, image_np.shape)

            # image_np = image_np.cuda()
            # x = image_np.to(device).view(-1, image_size)
            # x_reconst, mu, log_var = model(x)

            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
              .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()))

        f.write("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}\r\n"
              .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()))

        if (epoch + 1) % 10 == 0:

            # 저장할 때에는 모델을 CPU 로 보내준 다음에 해주는 게 좋습니다
            # 물론 저장하고 나서는 GPU 로 다시 보내주어야 하고요
            model = model.cpu()
            torch.save(model.state_dict(), './autoencoder_epoch' + str(epoch) + '.param')
            model = model.to(device)
            
            # 아래와 같은 코드는 모델의 인스턴스 자체를 얼리게 되는데요,
            # 이러면 나중에 불편한 점이 많습니다
            # torch.save(model, './autoencoder_epoch' + str(epoch) + '.pth')

        with torch.no_grad():
            # Save the sampled images
            z = torch.randn(batch_size, z_dim).to(device)
            out = model.decode(z).view(-1, 1, 224, 224)
            save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch + 1)))

            # Save the reconstructed images
            out, _, _ = model(x)
            x_concat = torch.cat([x.view(-1, 1, 224, 224), out.view(-1, 1, 224, 224)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))

    f.write('----------------------end-------------------')