# -*- coding: UTF-8 -*-
class Config:
    def __init__(self):   
        self.seed=66478
        self.device = 'cuda:2'
        self.image_shape = 32            # channels,height,width
        self.batch_size = 32             # 批处理尺寸
        self.n_epoch = 10                # 训练迭代周期，即遍历整个训练样本的次数
        self.opt = 'adam'                # 训练优化器：adam
        self.learning_rate = 0.001       # 学习率；若opt=‘adadelta'，则不需要定义学习率

        self.test_path = './data/test/'
        
        self.set_train_params(None)

    def set_train_params(self,train_type):
        # 训练模型保存的地址
        provinces_model_path = './model/provinces.ckpt'
        area_model_path = './model/area.ckpt'
        letters_model_path = './model/letters.ckpt'
        # 数据地址
        train_provinces_path = './data/train/provinces/'
        train_area_path = './data/train/area/'
        train_letters_path = './data/train/letters/'
        val_provinces_path = './data/validation/provinces/'
        val_area_path = './data/validation/area/'
        val_letters_path = './data/validation/letters/'
        # 分类标签
        provinces_classes = ['京', '闽', '粤', '苏', '沪', '浙']
        area_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "I", "O"]
        letters_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

        self.train_type = train_type
        if self.train_type == 'provinces':
            self.print_per_batch = 5       # 训练过程中,每100次batch迭代，打印训练信息
            self.train_size = 1254+32
            self.bisic_range = 0 # 用一个字段表示数据文件开始的数字。因为area的文件是从10开始的，所以加上这个字段有利于提取数据
            self.train_path = train_provinces_path
            self.validation_path = val_provinces_path
            self.classes = provinces_classes
            self.num_classes = len(provinces_classes)
            self.labels = [i for i in range(len(provinces_classes))]
            self.model_path = provinces_model_path
            # 训练结果保存地址
            self.result_fig_path = './result/provinces.png'
            self.test_fig_names = {'1.bmp':'闽', '1510076083_953_1.bmp':'苏', '1510076148_823_1.bmp':'京','1510076240_73_1.bmp':'吉'}
        elif self.train_type == 'area':
            self.print_per_batch = 20       # 训练过程中,每100次batch迭代，打印训练信息
            self.train_size = 3383+81
            self.bisic_range = 10
            self.train_path = train_area_path
            self.validation_path = val_area_path
            self.classes = area_classes
            self.num_classes = len(area_classes)
            self.labels = [i for i in range(len(area_classes))]
            self.model_path = area_model_path
            # 训练结果保存地址
            self.result_fig_path = './result/area.png'
            self.test_fig_names = {'2.bmp':'O', '1510076083_955_4.bmp':'Q', '1510076083_956_6.bmp':'K', '1510076148_824_3.bmp':'B', '1510076213_455_2.bmp':'E', '1510076148_825_5.bmp':'I'}
        else:
            self.print_per_batch = 20       # 训练过程中,每100次batch迭代，打印训练信息
            self.train_size = 4200+200
            self.bisic_range = 0
            self.train_path = train_letters_path
            self.validation_path = val_letters_path
            self.classes = letters_classes
            self.num_classes = len(letters_classes)
            self.labels = [i for i in range(len(letters_classes))]
            self.model_path = letters_model_path
            # 训练结果保存地址
            self.result_fig_path = './result/letters.png'
            self.test_fig_names = {'1510076240_75_4.bmp':'2', '1510076083_955_5.bmp':'0', '1510076183_756_7.bmp':'8', '1510076148_824_3.bmp':'B', '1510076083_954_3.bmp':'7', '1510076183_756_6.bmp':'9'}
        self.set_model_params()

    def set_model_params(self):
        self.num_convs = 3 # 3个卷积层
        
        self.conv1_input_channels = 1
        self.conv1_num_filters = 32
        self.conv1_kerne_size = 3 
        self.conv1_kerne_stride = 1 
        self.conv1_padding = 1
        self.pool1_size = 2
        self.pool1_strides = 2

        self.conv2_input_channels = 32
        self.conv2_num_filters = 64
        self.conv2_kerne_size = 3
        self.conv2_kerne_stride = 1 
        self.conv2_padding = 1
        self.pool2_size = 2
        self.pool2_strides = 2

        self.conv3_input_channels = 64
        self.conv3_num_filters = 128
        self.conv3_kerne_size = 3
        self.conv3_kerne_stride = 1 
        self.conv3_padding = 1
        self.pool3_size = 2
        self.pool3_strides = 2

        shrink_scales = 2**self.num_convs
        img_final_size = (self.image_shape/shrink_scales)
        self.fc1_input_channels = int(img_final_size * img_final_size * 128)
        self.fc1_output_channels = 512

        self.drop_keep_prob = 0.6        # dropout层，参数keep的比例

        self.fc2_input_channels = self.fc1_output_channels
        self.fc2_output_channels = self.num_classes  # 这里最后一个全连接层的结点数量，需要根据训练类型制定好之后才能确定

        