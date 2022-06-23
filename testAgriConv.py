import numpy as np
import tensorflow as tf
from collections import deque
import os 
from BaseLine import BaseLineModel
# from AEmodel2 import Class_model,AE_model
from Convmodel import Class_model
import tensorflow.keras.mixed_precision  as mixed_precision
os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT']= '2'


def load_data():
    data = np.load('./data/data.npy',mmap_mode='r')
    labels = np.load('./data/label.npy',mmap_mode='r')
    return data,labels
    
def testNet():
    pre_s = []  # Memory
    # tf.random.set_seed(42)

    label_b = np.load('./data/label.npy',mmap_mode='r')
    label = np.array(label_b,dtype=np.int16)
    del label_b
    # data_b,label_b = load_data()
    data_len = label.shape[0] # 数据的长度
    k_len = data_len//64
    for i in range(k_len):
        data_b = np.load('./data/data.npy',mmap_mode='r')
        data = np.array(data_b[i*64:(i+1)*64],dtype=np.float16)#.reshape((64,405,500,3))
        pre = tf.argmax(classifier(data),-1).numpy()
        pre_s.append(pre)

    data = np.array(data_b[k_len*64:],dtype=np.float16)#.reshape((label.shape[0] % 64,405,500,3))
    pre = tf.argmax(classifier(data),-1).numpy()
    pre_s = np.reshape(pre_s,-1).tolist()
    pre_s = pre_s+pre.tolist()
    del data_b
    pre_s = np.array(pre_s)
    # t = np.array(pre_s)
    # 用标签减去神经网络推理出的结果,0表示成功预测,1表示1预测为0,-1表示0预测为1
    temp = label - pre_s
    FP =  np.count_nonzero(np.where(temp<0 ,temp,0))       # 负样本预测为正
    FN =  np.count_nonzero(np.where(temp>0 ,temp,0))  # 正样本预测为负
    acc = data_len - np.count_nonzero(temp) # 预测正确的数量
    TN = data_len - np.count_nonzero(label*2 - pre_s) # 成功预测0的数量
    TP = acc - TN # 成功预测1的数量
    # posi = np.count_nonzero(label) # 正样本数量
    # nega = data_len - posi # 负样本数量
    prec = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1_sc = 2*prec*recall/(prec+recall)
    acc = acc/data_len
    print("Accuracy=",acc,"F1-score=",F1_sc)


if __name__ == "__main__":
    # load_data()
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    # encoder = AE_model()
    # classifier = Class_model()
    # testNet()
    classifier = BaseLineModel()
    testNet()