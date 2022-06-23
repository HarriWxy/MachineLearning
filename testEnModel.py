import numpy as np
import tensorflow as tf
import random
from collections import deque
import os 
import datetime
from Convmodel import Class_model
import tensorflow.keras.mixed_precision  as mixed_precision
# from testAgri import testNet
from tensorflow.keras import optimizers
os.environ['CUDA_VISIBLE_DEVICES']='0'

'''没有encoder直接训练softmax'''
BATCH = 16 # 训练batch大小
Tau=0.995
    
def div_data():
    '''加载并随机从整个数据集中选择1/4长度加载进入内存'''
    data = np.load('./data/data.npy',mmap_mode='r')
    labels = np.load('./data/label.npy',mmap_mode='r')
    data_div = deque()
    leng = data.shape[0]
    for i in random.sample(range(leng),leng//4):
        data_div.append((np.array(data[i],dtype=np.float16),int(labels[i])))
    del data,labels
    return data_div

@tf.function
def train_class(g_s,tru_s):
    '''训练分类器'''
    g_s = g_s + tf.random.normal([BATCH,405,500,3],0,0.1,tf.float16) # 增加噪声
    with tf.GradientTape() as tape: #在这个空间里面计算梯度,自动求导
        h_temp = classifier(g_s)
        ac_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tru_s,h_temp) # 交叉熵损失
        ac_loss1 = tf.reduce_mean(ac_loss)
        ac_loss = optimizer_ac.get_scaled_loss(ac_loss1)

    gradients = tape.gradient(ac_loss, classifier.trainable_variables)
    gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply_gradients(zip(gradients, classifier.trainable_variables))
    return ac_loss1

def trainNet():
    eps=0
    # tensorboard
    train_log_dir='logs/conv/'+datetime.datetime.now().strftime("%m%d-%H-%M")
    train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    t= 100
    while eps < 40:
        data = div_data()
        # testAcc,在每次训练循环之前从内存中采样minibatch测试准确率
        minibatch = random.sample(data, BATCH)
        g_s = tf.convert_to_tensor([d[0] for d in minibatch])
        g_s = tf.reshape(g_s,[BATCH,405,500,3]) # reshape
        tru_s = tf.convert_to_tensor([d[1] for d in minibatch])
        h_temp = classifier(g_s) # 神经网络推理
        # 用标签减去神经网络推理出的结果,0表示成功预测,1表示1预测为0,-1表示0预测为1
        acc = tru_s - tf.cast(tf.argmax(h_temp,-1),dtype=tf.int32)
        acc = 1 - np.count_nonzero(acc.numpy())/BATCH # 准确率
        print(acc)
        with train_sum_writer.as_default():
            tf.summary.scalar('acc',acc,step=eps)
        # train
        for i in range(t):
            minibatch = random.sample(data, BATCH)
            g_s = tf.convert_to_tensor([d[0] for d in minibatch])
            tru_s = tf.convert_to_tensor([d[1] for d in minibatch])
            g_s = tf.reshape(g_s,[BATCH,405,500,3])
            ac_loss =train_class(g_s,tru_s)
            print("ac-loss = %f" % ac_loss,"t=",eps*t+i)
            if i % 5 == 4:
                with train_sum_writer.as_default():
                    tf.summary.scalar('CrossEn-loss',ac_loss,step=eps*t+i)
        classifier.save_wei()
        eps+=1

if __name__ == "__main__":
    # load_data()
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    classifier = Class_model()
    optimizer_ac = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    trainNet()
    # testNet()