import numpy as np
import tensorflow as tf
import random
from collections import deque
import os 
import datetime
from Convmodel import Class_model
import tensorflow.keras.mixed_precision  as mixed_precision
# from testAgri import testNet
from tensorflow.keras import losses
from multiprocessing import Process,Queue

os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT']= '2'
'''多线程'''
BATCH = 16 # 训练batch大小
Tau=0.995
    
def div_data():
    data = np.load('./data/data.npy',mmap_mode='r')
    data_div = deque()
    leng = data.shape[0]
    for i in random.sample(range(leng),leng//3):
        re_data = np.reshape(np.array(data[i],dtype=np.float16),(405,500,3))
        data_div.append(re_data)
    del data
    return data_div

def co_data(D:Queue,BATCH:int):
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
    eps=0
    t= 100
    while eps < 40:
        data = div_data()
        for i in range(t):
            minibatch = random.sample(data, BATCH)
            D.put(minibatch)
            # t-=1
        eps+=1
        
def trainNet(D:Queue):
    @tf.function
    def train_class(g_s,tru_s):
        with tf.GradientTape() as tape: #在这个空间里面计算梯度
            h_temp = classifier(g_s)
            # print(h_temp)
            ac_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tru_s,h_temp)
            ac_loss1 = tf.reduce_mean(ac_loss)
            ac_loss = optimizer_ac.get_scaled_loss(ac_loss1)

        gradients = tape.gradient(ac_loss, classifier.trainable_variables)
        gradients = optimizer_ac.get_unscaled_gradients(gradients)
        optimizer_ac.apply_gradients(zip(gradients, classifier.trainable_variables))
        return ac_loss1
    
    # train
    # tensorboard
    # train_log_dir='logs/'+datetime.datetime.now().strftime("%m%d-%H-%M")
    # train_sum_writer = tf.summary.create_file_writer(train_log_dir)
    i=0
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    classifier = Class_model()
    optimizer_ac = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate = 1e-4))
    g_s_d = deque()
    tru_s_d = deque()
    while True:
        minibatch = D.get()
        for d in minibatch:
            g_s_d.append(d[0])
            tru_s_d.append(d[1])
        g_s = tf.convert_to_tensor(g_s_d,dtype=tf.float16)
        tru_s = tf.convert_to_tensor(tru_s_d)
        g_s_d.clear()
        tru_s_d.clear()
        ac_loss = train_class(g_s,tru_s)
        print("ac-loss = %f" % ac_loss,"t=",i)
        if i % 100 == 99:
            # with train_sum_writer.as_default():
            #     tf.summary.scalar('dis-loss',ac_loss,step=i)
            classifier.save_wei()
        i+=1

        # testAcc



if __name__ == "__main__":
    # load_data()
    D = Queue(5)
    p2 = Process(target=co_data,args=([D,BATCH]))
    p1 = Process(target=trainNet,args=([D]))
    p2.start()
    p1.start()
    p2.join()
    p1.join()
    # testNet()