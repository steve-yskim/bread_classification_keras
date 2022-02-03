#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt #모형 학습시 accuracy와 loss를 저장하기 위한 라이브러리입니다.

"""전처리를 위한 라이브러리"""
import os
import pandas as pd
import numpy as np
import cv2
import random
"""Keras 라이브러리"""
import tensorflow.keras as keras #keras 라이브러리입니다.
from tensorflow.keras.preprocessing.image import ImageDataGenerator #이미지 데이터를 tensor로 변한하기 위해 활용되는 라이브러리입니다.
from tensorflow.keras.layers import * #학습 모형을 구축하기 위해 활용되는 라이브러리입니다.
from tensorflow.keras import Sequential #학습 모형을 구축하기 위해 활용되는 라이브러리 입니다.
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import tensorflow.keras.backend as K

# GPU 디바이스 설정
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def multiple_hue(img):
    min_delta = 0
    max_delta = 300
    temp_image = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2HSV)
    temp_image = np.array(temp_image, dtype=np.float)
    d = random.uniform(min_delta, max_delta)
    temp_image[:, :, 0] += d
    temp_image = np.clip(temp_image, 0, 360)
    temp_image = cv2.cvtColor(np.uint8(temp_image), cv2.COLOR_HSV2BGR)
    temp_image = np.array(temp_image, dtype=np.float64)
    return temp_image
    
class Import_data:
    def __init__(self, train_path, val_path, batch_size):
        self.train_path = train_path
        self.test_path = val_path
        self.batch_size =batch_size
    def train(self):
        # generator 생성
        train_datagen = ImageDataGenerator(rescale = 1/255.,
                                          rotation_range=180,
                                          width_shift_range=0.8,
                                          height_shift_range=0.8,
                                          channel_shift_range=0,
                                          zoom_range=[1,2], ## 확대 와 축소 범위 [1-0.2 ~ 1+0.2 ]
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          brightness_range=[0.5,1.5],
                                          fill_mode='reflect',
                                          cval=0, preprocessing_function= multiple_hue)
        val_datagen = ImageDataGenerator(rescale = 1/255.)
        train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=(224, 224),
            batch_size=self.batch_size,
            interpolation='bilinear',
            color_mode ='rgb',
            class_mode='binary',
            seed = 1)
        val_generator = val_datagen.flow_from_directory(
            self.test_path,  
            target_size=(224, 224),      
            batch_size=32,
            interpolation='bilinear',  
            color_mode ='rgb',
            shuffle=False,
            class_mode='binary')

        return train_generator, val_generator

class Load_model:
    def __init__(self, train_path):
        self.num_class = len(os.listdir(train_path)) # 클래스 수

    # 모델 정의
    def build_network(self):
        # Instantiates the Inception-ResNet v2 architecture
        network = tf.keras.applications.Xception(include_top=False,
                          weights='imagenet', 
                          input_shape=(224, 224, 3))
        model = Sequential()
        model.add(network)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(rate = 0.5))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate = 0.5))
        model.add(Dense(self.num_class)) # softmax는 loss function에서 진행
        model.summary()

        return model

class Fine_tunning:
    def __init__(self, train_path, val_path, model_name, epoch, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data = Import_data(train_path, val_path, self.batch_size)
        self.train_data, self.val_data = self.data.train()
        self.load_model = Load_model(train_path)
        self.epoch = epoch
        self.model_name = model_name
        self.train_path = train_path


    def training(self):
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-3]

        # 옵티마이저 정의
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        es = tf.keras.callbacks.EarlyStopping( monitor="val_acc", mode="max", patience = 10)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=(1/np.sqrt(10)), patience=5, min_lr = 1e-6 )

        # 모델 생성
        model = self.load_model.build_network()

        # 학습모델 저장할 경로 생성
        save_folder = 'model_saved/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 훈련 중 주기적으로 모델 저장
        check_point = ModelCheckpoint(save_folder + 'model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                                      monitor='val_acc', save_best_only=True, mode='auto')                            
        # 모델 컴파일
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=optimizer,
                      metrics=['acc'])

        # 모델 학습
        history = model.fit_generator(
            self.train_data,
            steps_per_epoch=self.train_data.samples / self.train_data.batch_size,
            epochs=self.epoch,
            validation_data=self.val_data,
            validation_steps=self.val_data.samples / self.val_data.batch_size,
            callbacks=[check_point, reduce_lr, es],
            verbose=1)
        return history

    def save_accuracy(self, history):
        # 학습모델 저장 경로
        data_name = self.train_path.split('/')
        data_name = data_name[len(data_name)-3]
        save_folder = './model_saved/' + data_name + '/' + self.model_name + '_' + str(self.epoch) + '/'
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        epoch_list = list(epochs)

        # csv 저장
        df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': acc, 'validation_accuracy': val_acc},
                          columns=['epoch', 'train_accuracy', 'validation_accuracy'])
        df_save_path = save_folder + 'accuracy.csv'
        df.to_csv(df_save_path, index=False, encoding='euc-kr')

        # Accuracy 그래프 이미지 저장
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        save_path = save_folder + 'accuracy.png'
        plt.savefig(save_path)
        plt.cla()

        # Loss 그래프 이미지 저장
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        save_path = save_folder + 'loss.png'
        plt.savefig(save_path)
        plt.cla()

        # 마지막 모델을 제외하고 삭제
        name_list = os.listdir(save_folder)
        h5_list = []
        for name in name_list:
            if '.h5' in name:
                h5_list.append(name)
        h5_list.sort()
        h5_list = [save_folder + name for name in h5_list]
        for path in h5_list[:len(h5_list) - 1]:
            os.remove(path)
        K.clear_session()

if __name__ == '__main__':
    # from ops import *
    train_path = 'Bread-DATASET/train/' # 경로 마지막에 반드시 '/'를 기입해야하며합니다.
    #dataset 이후 Skin 혹은 Eye로 데이터셋 변경이 가능하며, 그 이후 디렉토리 구조는 동일합니다.
    val_path = 'Bread-DATASET/test/'
    model_name = 'Xception' # https://keras.io/api/applications/ 에 있는 모델이름
    epoch = 1
    batch_size = 32
    learning_rate = 0.0001
    with tf.device('/cpu:0'):
        fine_tunning = Fine_tunning(train_path = train_path, val_path = val_path,
                                    model_name = model_name,
                                    epoch = epoch, batch_size = batch_size, learning_rate = learning_rate)
        history = fine_tunning.training()
        fine_tunning.save_accuracy(history)
