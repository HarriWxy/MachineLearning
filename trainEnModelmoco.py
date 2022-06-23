import numpy as np
import tensorflow as tf
import random
from collections import deque
import os 
import datetime
from AEmodel import AE_model,Class_model
import tensorflow.keras.mixed_precision  as mixed_precision
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from testAgri import testNet
os.environ['CUDA_VISIBLE_DEVICES']='0'

'''moco自监督'''
BATCH = 16 # 训练batch大小
Tau=0.998
    
def div_data():
    data = np.load('./data/data.npy',mmap_mode='r')
    labels = np.load('./data/label.npy',mmap_mode='r')
    data_div = deque()
    leng = data.shape[0]
    for i in random.sample(range(leng),leng//4):
        data_div.append((np.array(data[i],dtype=np.float16),int(labels[i])))
    del data,labels
    return data_div


@tf.function
def train_encoder(g_s,key_b):
    g_s_v = g_s + tf.random.normal([BATCH,405,500,3],0,0.1,tf.float16)
    tru_s = encoder_tar(g_s)
    key_b = encoder_tar(key_b)
    with tf.GradientTape() as tape: #在这个空间里面计算梯度
        h_temp = encoder_val(g_s_v)
        ac_loss = losses.MAE(tru_s,h_temp)
        ac_loss1 = tf.reduce_mean(ac_loss)
        ac_loss = optimizer_ac.get_scaled_loss(ac_loss1)
    gradients = tape.gradient(ac_loss, encoder_val.trainable_variables)
    gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply_gradients(zip(gradients, encoder_val.trainable_variables))
    with tf.GradientTape() as tape: #在这个空间里面计算梯度
        h_temp = encoder_val(g_s_v)
        ac_loss = -losses.MAE(key_b,h_temp)
        ac_loss2 = tf.reduce_mean(ac_loss)
        ac_loss = optimizer_ac.get_scaled_loss(ac_loss2)
    gradients = tape.gradient(ac_loss, encoder_val.trainable_variables)
    gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply_gradients(zip(gradients, encoder_val.trainable_variables))
    return ac_loss1 + ac_loss2

@tf.function
def train_class(g_s,tru_s):
    with tf.GradientTape() as tape: #在这个空间里面计算梯度
        h_temp = classifier(encoder_val(g_s))
        # print(h_temp)
        ac_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tru_s,h_temp)
        ac_loss1 = tf.reduce_mean(ac_loss)
        ac_loss = optimizer_ac.get_scaled_loss(ac_loss1)

    gradients = tape.gradient(ac_loss, encoder_val.trainable_variables+classifier.trainable_variables)
    gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply_gradients(zip(gradients, encoder_val.trainable_variables+classifier.trainable_variables))
    return ac_loss1

def trainNet():
    eps=0
    # tensorboard
    # train_log_dir='logs/'+datetime.datetime.now().strftime("%m%d-%H-%M")
    # train_sum_writer = tf.summary.create_file_writer(train_log_dir)
    t= 100
    while eps < 40:
        data = div_data()
        # testAcc
        minibatch = random.sample(data, BATCH)
        g_s = tf.convert_to_tensor([d[0] for d in minibatch])
        g_s = tf.reshape(g_s,[BATCH,405,500,3])
        tru_s = tf.convert_to_tensor([d[1] for d in minibatch])
        h_temp = classifier(encoder_val(g_s))
        acc = tru_s - tf.cast(tf.argmax(h_temp,-1),dtype=tf.int32)
        acc = 1 - np.count_nonzero(acc.numpy())/BATCH
        print(acc)
        # with train_sum_writer.as_default():
        #     tf.summary.scalar('acc',acc,step=eps)

        # train
        for i in range(t):
            key_b = g_s
            minibatch = random.sample(data, BATCH)
            g_s = tf.convert_to_tensor([d[0] for d in minibatch])
            tru_s = tf.convert_to_tensor([d[1] for d in minibatch])
            g_s = tf.reshape(g_s,[BATCH,405,500,3])
            simi_loss = train_encoder(g_s,key_b)
            ac_loss = train_class(g_s,tru_s)
            print("ac-loss = %f" % ac_loss,"simi-loss = %f" % simi_loss,"t=",eps*t+i)
            act_tar_var=[i*Tau+j*(1-Tau) for i, j in zip(encoder_tar.get_weights(),encoder_val.get_weights())]
            encoder_tar.set_weights(act_tar_var)
            # if i % 5 == 4:
            #     with train_sum_writer.as_default():
            #         tf.summary.scalar('CrossEn-loss',ac_loss,step=eps*t+i)
            #         tf.summary.scalar('Encoder-loss',simi_loss,step=eps*t+i)
        encoder_val.save_wei()
        classifier.save_wei()
        eps+=1
        testNet()

if __name__ == "__main__":
    # load_data()
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    encoder_tar = AE_model()
    encoder_val = AE_model()
    classifier = Class_model()
    optimizer_ac = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    trainNet()
    