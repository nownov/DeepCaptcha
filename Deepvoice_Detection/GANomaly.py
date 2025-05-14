from keras import layers
import keras
import keras.backend as K
import os
import numpy as np
from PIL import Image
from keras.datasets import mnist
import cv2


### 레이어 정의 ####
## Generators Encoder
## 주어진 데이터(이미지)를 잠재벡터화 시키는 과정이다.

# 입력 이미지의 크기와 채널 수(RGB)를 정의한다.
# 우리는 height = 64, wudth = 64, channels = 4로 정의하였다.
input_layer = layers.Input(name='input', shape=(height, width, channels))

# 컨볼루션 레이어로, 5x5의 커널을 사용하여 다운샘플링한다.
x = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='conv_1', kernel_regularizer = 'l2')(input_layer)
x = layers.LeakyReLU(name='leaky_1')(x)

# 컨볼루션 레이어로, 3x3의 커널을 사용하여 다운샘플링한다.
x = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(name='leaky_2')(x)

x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_3', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(name='leaky_3')(x)

x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_4', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(name='leaky_4')(x)

# Pooling을 통해 각 특징 값의 평균을 계산한다.
x = layers.GlobalAveragePooling2D(name='g_encoder_output')(x)

g_e = keras.models.Model(inputs=input_layer, outputs=x)



## Generator Decoder
## 앞서 Encoder에서 추출한 잠재벡터를 입력 받아 생성 이미지로 복원하는 과정이다.

input_layer = layers.Input(name='input', shape=(height, width, channels))

x = g_e(input_layer)

y = layers.Dense(width * width * 2, name='dense')(x) 
y = layers.Reshape((width//8, width//8, 128), name='de_reshape')(y)

# 3x3의 커널을 사용하는 컨볼루션 레이어에 업샘플링한다.
y = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', name='deconv_1', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_1')(y)

y = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='deconv_2', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_2')(y)

y = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='deconv_3', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_3')(y)

# 1x1의 커널을 사용하는 컨볼류션 레이어에 tanh 활성화 함수를 사용하여 생성 이미지를 만든다.
y = layers.Conv2DTranspose(channels, (1, 1), strides=(1,1), padding='same', name='decoder_deconv_output', kernel_regularizer = 'l2', activation='tanh')(y)

g = keras.models.Model(inputs=input_layer, outputs=y)


## Encoder
## 주어진 데이터(이미지)를 잠재벡터화 시키는 과정이다.

input_layer = layers.Input(name='input', shape=(height, width, channels))

z = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='encoder_conv_1', kernel_regularizer = 'l2')(input_layer)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='encoder_conv_2', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_1')(z)
z = layers.LeakyReLU()(z)


z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='encoder_conv_3', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_2')(z)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_41', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_3')(z)
z = layers.LeakyReLU()(z)

z = layers.GlobalAveragePooling2D(name='encoder_output')(z)

encoder = keras.models.Model(input_layer, z)


## Feature extractor
## 주어진 데이터(이미지)를 잠재벡터화 시켜 특징을 추출하는 과정이다.
input_layer = layers.Input(name='input', shape=(height, width, channels))

f = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='f_conv_1', kernel_regularizer = 'l2')(input_layer)
f = layers.LeakyReLU(name='f_leaky_1')(f)

f = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='f_conv_2', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_1')(f)
f = layers.LeakyReLU(name='f_leaky_2')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_3', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_2')(f)
f = layers.LeakyReLU(name='f_leaky_3')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_4', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_3')(f)
f = layers.LeakyReLU(name='feature_output')(f)

feature_extractor = keras.models.Model(input_layer, f)


### 손실함수 정의 ###
# Adversarial loss, Contextual loss, Encoder loss의 손실함수를 정의한다.

## Adversarial Loss

# 원본 이미지와 생성 이미지를 Feature extractor를 사용하여 각 이미지의 특성을 추출한다.
# 각 이미지의 특징 간의 평균 제곱 오차를 계산한다
class AdvLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    

## Content Loss
# 원본 이미지와 생성 이미지의 특징 간 평균 제곱 오차를 계산한다.

class CntLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
## Encoder Loss
# g_e를 통해 원본 이미지의 특징을 추출한다
# encoder를 통해 생성 이미지의 특징을 추출한다
# 두 잠재벡터 간의 평균 제곱 오차를 계산한다

class EncLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(g_e(ori) - encoder(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)

### 모델 정의 ###
input_layer = layers.Input(name='input', shape=(height, width, channels))

# Generator를 사용하여 생성 이미지를 만든다.
gan = g(input_layer) # g(x)

adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cnt_loss = CntLoss(name='cnt_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])

gan_trainer = keras.models.Model(input_layer, [adv_loss, cnt_loss, enc_loss])

def loss(yt, yp):
    return yp

losses = {
    'adv_loss': loss,
    'cnt_loss': loss,
    'enc_loss': loss,
}

## 손실함수의 가중치를 결정한다.
# 우리는 Encoder Loss의 값을 50으로 주었다.

lossWeights = {'cnt_loss': 1.0, 'adv_loss': 1.0, 'enc_loss': 50.0}

gan_trainer.compile(optimizer = 'adam', loss=losses, loss_weights=lossWeights)

## Discriminator
# 주어진 이미지가 진짜인지 가짜인지 판별하는 역할을 한다.
# 생성하고 판별하는 과정에서 GANomaly 모델의 성능이 향상된다.
input_layer = layers.Input(name='input', shape=(height, width, channels))

# feature extractor를 통해 이미지의 특징을 추출한다.
f = feature_extractor(input_layer)

d = layers.GlobalAveragePooling2D(name='glb_avg')(f)
d = layers.Dense(1, activation='sigmoid', name='d_out')(d)

d = keras.models.Model(input_layer, d)

d.compile(optimizer='adam', loss='binary_crossentropy')


### 훈련 준비 과정 ###
## 훈련 데이터셋과 테스트 데이터셋을 배열로 변환한다.
x_train = []
x_train = np.array(x_train)
x_test = []
x_test = np.array(x_test)

#Load data
def reshape_x(x):
    new_x = np.empty((len(x), width, height, 4), dtype=np.uint8)  # 4 채널 (RGB) 이미지로 설정
    for i, e in enumerate(x):
        new_x[i] = cv2.resize(e, (width, height))
    return new_x/255.0


### 훈련 과정 ###
niter = 1
bz = 32

def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y]

train_data_generator = get_data_generator(x_ok, bz)

for i in range(niter):
    # Discriminator를 훈련하는 부분이다.
    x, y = train_data_generator.__next__()

    d.trainable = True

    fake_x = g.predict(x)

    d_x = np.concatenate([x, fake_x], axis=0)
    d_y = np.concatenate([np.zeros(len(x)), np.ones(len(fake_x))], axis=0)

    d_loss = d.train_on_batch(d_x, d_y)
    # Generator를 훈련하는 부분이다.
    d.trainable = False
    g_loss = gan_trainer.train_on_batch(x, y)

### Evaluation ###
# 생성 이미지에 대한 Anomaly Score를 계산한다.

# g_e를 사용하여 테스트 데이터의 특징을 추출한다
encoded = g_e.predict(x_test)
# Generator를 사용하여 테스트 데이터의 생성이미지를 만든다
gan_x = g.predict(x_test)
# 생성 이미지에 대해 특징을 추출한다.
encoded_gan = g_e.predict(gan_x)
# 원본 이미지와 생성 이미지 간의 특징 차이를 계산한다.
# 이 때 최솟값과 최댓값을 뺀 결과만을 Anomaly score로 취급하며 정규화한다.
score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
score = (score - np.min(score)) / (np.max(score) - np.min(score))


### 잠재벡터와 생성 이미지 시각화 ###

from matplotlib import pyplot as plt
from pylab import rcParams

#인코더에서 압축된 특징 값을 이미지화
i = 1
 # or 1
image = np.reshape(gan_x[200], (64, 64,4))
image = image * 127 + 127
plt.imshow(image.astype(np.uint8))

#디코더에서 생성된 이미지를 시각화
image = np.reshape(x_test[3], (64, 64,4))
image = image * 127 + 127
plt.imshow(image.astype(np.uint8))