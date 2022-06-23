import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import   Dense,LayerNormalization
import numpy as np
import os

class BaseLineModel(Model): 
    # 评估网络,输出动作
    def __init__(self):
        super().__init__()
        # resnet
        self.f1 = Dense(256, activation='elu',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42),
                           bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=42))
        self.l_1 = LayerNormalization(-1)
        self.f1_2 = Dense(32, activation='elu',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42),
                           bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=42))
        self.f1_3 = Dense(64, activation='elu',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42),
                           bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=42))
        self.f3 = Dense(2, activation=None,
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42),
                           bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=42))
        # 加载网络
        self.checkpoint_save_path = "./model_Base1/base"
        if os.path.exists(self.checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            self.load_weights(self.checkpoint_save_path)
        else:
            print('-------------train new model-----------------')
            
    @tf.function
    def call(self,x):
        x = self.f1(x)
        x = self.l_1(x)
        x1 = self.f1_2(x)
        x = self.f1_3(x1)
        y = self.f3(x) # 要考虑输入输出的维度

        return y      
    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)
