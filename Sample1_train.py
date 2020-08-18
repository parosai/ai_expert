import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from keras.layers import Input, Dense, Flatten, Conv2D, Flatten
from keras.models import Sequential, Model
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"]='0'

########################################
PATH_TEMPLATE = './dataset_crop/templates/'
########################################

#img = cv2.imread(PATH_TEMPLATE_IMG)
#img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)


## Model structure
input_img = Input(shape=(4096,))                                   # 4096 = 64 x 64
encoded = Dense(16, activation='relu')(input_img)
decoded = Dense(4096, activation='sigmoid')(encoded)

model = Model(input_img, decoded)
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy')
model.summary()


## Load data
files = os.listdir(PATH_TEMPLATE)

x_train = []
for f in files:
    path = PATH_TEMPLATE + f
    img = load_img(PATH_TEMPLATE + f, color_mode='grayscale', target_size=(64, 64), interpolation='nearest')
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.
    img_tensor_flat = np.ravel(img_tensor, order='C')
    img_tensor_flat = np.expand_dims(img_tensor_flat, axis=0)
    x_train.append(img_tensor_flat)


## Start training
model.fit(
    x_train,
    x_train,
    batch_size=1,
    shuffle=True,
    epochs=100000)





## predict
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(16,))
decoder_layer = model.layers[-1]  # 오토인코더 모델의 마지막 레이어 얻기
decoder = Model(encoded_input, decoder_layer(encoded_input)) # 디코더 모델 생성

encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)



import matplotlib.pyplot as plt
n = 1
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


print('Finished !')