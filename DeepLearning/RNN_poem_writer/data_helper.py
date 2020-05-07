import numpy as np
import os
from sklearn.utils import shuffle


CUR_DIR = os.getcwd()
data_file = CUR_DIR + '/data/tang.npz'

def load_data():
    file_npz = np.load(data_file)
    data = file_npz['data']
    data = shuffle(data) # shuffle train data
    ix2word = file_npz['ix2word'].item() # get index-word dict 
    word2ix = file_npz['word2ix'].item() # get word-index dict
    tmp_s = np.ones(data.shape[0], dtype=np.int32) * 8292
    data = np.c_[data, tmp_s]
    return data, ix2word, word2ix

def pick_word_index(predict):
    """
    选择概率最高的前100个词，并用轮盘赌法选取最终结果
    :param predict: 概率向量
    :return: 生成的词索引
    """
    prob = sorted(predict, reverse=True)[:100]
    word_idx = np.searchsorted(np.cumsum(prob), np.random.rand(1) * np.sum(prob))
    return int(word_idx[0])

if __name__ == "__main__":
    data, ix2word, word2ix = load_data()
    print(data.shape)
    print(data[0])
    print(len(ix2word))

