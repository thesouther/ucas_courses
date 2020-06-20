# -*- coding: UTF-8 -*-
class Config:
    def __init__(self):
        self.select_model = 'model2'           #选择模型，两个选择，['model', 'model2']

        if self.select_model == 'model2':    
            self.kernel_size = [2,3,4,5] # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
            self.kernel_classes = 4      # 卷积核的种类；nlp任务中通常选择2,3,4,5
        else: 
            self.kernel_size = 4         # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
        self.seed=66478
        self.update_w2v = True           # 是否在训练中更新w2v
        self.vocab_size = 58954          # 词汇量，与word2id中的词汇量一致
        self.n_class = 2                 # 分类数：分别为pos和neg
        self.max_sen_len = 75            # 句子最大长度
        self.embedding_dim = 50          # 词向量维度
        self.batch_size = 100            # 批处理尺寸
        self.n_hidden = 256              # 隐藏层节点数
        self.n_epoch = 10                # 训练迭代周期，即遍历整个训练样本的次数
        self.opt = 'adam'                # 训练优化器：adam或者adadelta
        self.learning_rate = 0.001       # 学习率；若opt=‘adadelta'，则不需要定义学习率
        self.drop_keep_prob = 0.5        # dropout层，参数keep的比例
        self.input_channel = 1
        self.num_filters = 256           # 卷积层filter的数量
        # self.kernel_size = [3,4,5]             # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
        # self.kernel_size = 4             # 卷积核的尺寸；nlp任务中通常选择2,3,4,5
        
        self.print_per_batch = 10      # 训练过程中,每100词batch迭代，打印训练信息
        self.save = True     
        self.save_dir = './model/'       # 训练模型保存的地址
        self.train_path = '../Dataset/train.txt'
        self.val_path = '../Dataset/validation.txt'
        self.test_path = '../Dataset/test.txt'
        self.word2id_path = '../Dataset/word_to_id.txt'
        self.pre_word2vec_path = '../Dataset/wiki_word2vec_50.bin'
        self.corpus_word2vec_path = '../Dataset/corpus_word2vec.txt'
        self.classes = {'0': 0, '1': 1}  # 分类标签


