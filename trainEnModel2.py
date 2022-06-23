import numpy as np
import tensorflow as tf
import random
from collections import deque
import os 
import datetime
from AEmodel2 import AE_model,Class_model
import tensorflow.keras.mixed_precision  as mixed_precision
from tensorflow.keras import losses
from tensorflow.keras import optimizers
os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT']= '2'
'''第二版模型,动量更新'''
BATCH = 16 # 训练batch大小
Tau=0.995
    
def div_data():
    data = np.load('./data/data.npy',mmap_mode='r')
    labels = np.load('./data/label.npy',mmap_mode='r')
    data_div = deque()
    leng = data.shape[0]
    for i in random.sample(range(leng),leng//4):
        re_data = np.reshape(np.array(data[i],dtype=np.float16),(405,500,3))
        data_div.append((re_data,int(labels[i])))
    del data,labels
    return data_div


@tf.function
def train_encoder(g_s):
    g_s_v = g_s + tf.random.normal([BATCH,405,500,3],0,1,tf.float16)
    tru_s = encoder_tar(g_s)
    with tf.GradientTape() as tape: #在这个空间里面计算梯度
        h_temp = encoder_val(g_s_v)
        # print(h_temp)
        ac_loss = losses.MAE(tru_s,h_temp)
        ac_loss1 = tf.reduce_mean(ac_loss)
        ac_loss = optimizer_ac.get_scaled_loss(ac_loss1)

    gradients = tape.gradient(ac_loss, encoder_val.trainable_variables)
    gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply_gradients(zip(gradients, encoder_val.trainable_variables))

    return ac_loss1

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
    data = div_data()

    # tensorboard
    train_log_dir='logs/'+datetime.datetime.now().strftime("%m%d-%H-%M")
    train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    t= 100
    while eps < 20:
        data = div_data()

        # train
        for i in range(t):
            minibatch = random.sample(data, BATCH)
            g_s = tf.convert_to_tensor([d[0] for d in minibatch],dtype=tf.float16)
            tru_s = tf.convert_to_tensor([d[1] for d in minibatch])
            simi_loss = train_encoder(g_s)
            ac_loss =train_class(g_s,tru_s)
            print("ac-loss = %f" % ac_loss,"simi-loss = %f" % simi_loss,"t=",eps*t+i)
            act_tar_var=[i*Tau+j*(1-Tau) for i, j in zip(encoder_tar.get_weights(),encoder_val.get_weights())]
            encoder_tar.set_weights(act_tar_var)
            if i % 5 == 4:
                with train_sum_writer.as_default():
                    tf.summary.scalar('crossEn-loss',ac_loss,step=eps*t+i)
                    tf.summary.scalar('Encoder-loss',simi_loss,step=eps*t+i)

        encoder_val.save_wei()
        classifier.save_wei()
        eps+=1
        # testAcc
        minibatch = random.sample(data, BATCH)
        g_s = tf.convert_to_tensor([d[0] for d in minibatch])
        tru_s = tf.convert_to_tensor([d[1] for d in minibatch])
        h_temp = classifier(encoder_val(g_s))
        acc = tru_s - tf.cast(tf.argmax(h_temp,-1),dtype=tf.int32)
        acc = 1 - np.count_nonzero(acc.numpy())/BATCH
        print(acc)

if __name__ == "__main__":
    # load_data()
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    encoder_val = AE_model()
    encoder_tar = AE_model()
    classifier = Class_model()
    optimizer_ac = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    trainNet()
    # testNet()