# -*- coding: UTF-8 -*-
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from config import Config
conf = Config()

def cal_accuracy(predections, labels):
    return 100 * (np.sum(np.argmax(predections,1)==labels)/predections.shape[0])

def eval_in_batches(data, eval_prediction, eval_data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < conf.batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, conf.n_class), dtype=np.float32)
    for begin in range(0, size, conf.batch_size):
      end = begin + conf.batch_size
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-conf.batch_size:]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

#　构建词汇表并存储
def build_word2id(file):
    """
        :param file: word2id保存地址
        :return: None
    """
    word2id = {'_PAD_':0}
    for path in [conf.train_path, conf.val_path]:
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    # with open(file, 'w', encoding='utf-8') as f:
    #     for w in word2id:
    #         f.write(w+'\t'+str(word2id[w])+'\n')
    return word2id

# 基于预训练好的word2vec构建训练语料中所含词语的word2vec
def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    import gensim
    n_words = max(word2id.values())+1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1.,1., [n_words, model.vector_size]))
    # print(n_words, model.vector_size)
    i=0
    for word in word2id.keys():
        try:
            # i+=1
            # if i == 5:
            #     break
            # print(word)
            # print('model[word]', model[word].shape, '\t', model[word])
            # print('word2id[word]', '\t', word2id[word])
            # print('word_vecs[word2id[word]]', word_vecs[word2id[word]].shape, '\t', word_vecs[word2id[word]])
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass

    # if save_to_path:
    #     with open(save_to_path, 'w', encoding='utf-8') as f:
    #         for vec in word_vecs:
    #             vec=[str(w) for w in vec]
    #             f.write(''.join(vec)+'\n')
    return word_vecs

# 加载语料库：train/dev/test; 生成批处理id序列
def load_corpus(path, word2id, max_sen_len=50):
    """
    :param path: 样本语料库的文件
    :return: 文本内容contents，以及分类标签labels(onehot形式)
    """
    contents,labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        print("load data "+path)
        for line in f.readlines():
            if len(line)>1:
                sp = line.strip().split()
                label = sp[0]
                content = [word2id.get(w,0) for w in sp[1:]]
                content = content[:max_sen_len]
                if len(content) < max_sen_len:
                    content += [word2id['_PAD_']] * (max_sen_len - len(content))
                labels.append(label)
                contents.append(content)
    counter = Counter(labels)
    contents = np.asarray(contents)#.astype(np.int32)
    labels = np.array([conf.classes[i] for i in labels])#.astype(np.int32)
    contents, labels = shuffle(contents, labels)
    return contents, labels

if __name__ == "__main__":
    word2id = build_word2id(conf.word2id_path)
    # print("train data")
    # train_data,train_labels = load_corpus(conf.train_path, word2id, max_sen_len=conf.max_sen_len)
    print("validation data")
    val_data, val_labels = load_corpus(conf.val_path, word2id, max_sen_len=conf.max_sen_len)
    # print("test data")
    # test_data, test_labels = load_corpus(conf.test_path, word2id, max_sen_len=conf.max_sen_len)
    # 
    # word_vecs= build_word2vec(conf.pre_word2vec_path, word2id, conf.corpus_word2vec_path)
    # print(word_vecs.shape)
    # print(word_vecs[0])
    

