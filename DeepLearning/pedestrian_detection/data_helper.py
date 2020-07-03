#-*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pandas as pd
import imageio 

def compose_gif(conf): 
    """讲图片转为gif"""
    print("start compose gif...")
    img_paths = []
    for i in range(4500,5056):
        i_path = conf.test_result_path + str(i) +'.jpg.jpg'
        img_paths.append(i_path)
    gif_images = [] 
    for path in img_paths: 
        gif_images.append(imageio.imread(path)) 
    imageio.mimsave(conf.result_gif_path,gif_images,fps=20)
    print("compose gif finished!")

def video2im(conf):
    """
    提取视频帧作为图片保存
    """
    train_img_path = conf.train_img_path
    test_img_path = conf.test_img_path
    video_path = conf.town_centre_path
    img_shrink_factor = conf.img_shrink_factor

    frame = 0
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('total frame count: ', length)

    while True:
        check,img = cap.read()
        if check:
            if frame <4500:
                path = train_img_path
            else: 
                path = test_img_path
            img = cv2.resize(img, (1920//img_shrink_factor, 1080//img_shrink_factor))
            cv2.imwrite(os.path.join(path, str(frame)+'.jpg'), img)

            frame+=1
            print('processed: ', frame, end='\r')
        else:
            break
    cap.release()

def extract_GT(conf):
    """
    从TownCentre-groundtruth.top提取前4500帧图像的行人位置信息
    """
    GT = pd.read_csv(conf.town_centre_info_path,header=None)
    indent = lambda x,y: ''.join([' ' for _ in range(y)]) + x

    name = 'pedestrian'
    train_size = conf.train_size
    factor = conf.img_shrink_factor
    width,height = conf.img_width//factor, conf.img_height//factor
    depth = conf.img_depth
    # print(width, height)

    for frame_number in range(train_size):
        Frame = GT.loc[GT[1] == frame_number]
        x1 = list(Frame[8])
        y1 = list(Frame[11])
        x2 = list(Frame[10])
        y2 = list(Frame[9])
        points = [[(round(x1_), round(y1_)), (round(x2_), round(y2_))] for x1_,y1_,x2_,y2_ in zip(x1,y1,x2,y2)]

        with open(os.path.join(conf.trianval_xml_path, str(frame_number) + '.xml'), 'w') as file:
            file.write('<annotation>\n')
            file.write(indent('<filename>' + str(frame_number) + '.jpg' + '</filename>\n',1))
            file.write(indent('<size>\n',1))
            file.write(indent('<width>' + str(width) + '</width>\n',2))
            file.write(indent('<height>' + str(height) + '</height>\n',2))
            file.write(indent('<depth>' + str(depth) + '</depth>\n',2))
            file.write(indent('</size>\n',1))

            for point in points:
                top_left = point[0]
                bottom_right = point[1]

                if top_left[0] > bottom_right[0]:
                    xmax, xmin = min(top_left[0] // factor,width), max(bottom_right[0] // factor,0)
                else:
                    xmin, xmax = max(top_left[0] // factor,0), min(bottom_right[0] // factor,width)
                if top_left[1] > bottom_right[1]:
                    ymax, ymin = min(top_left[1] // factor,height), max(bottom_right[1] // factor,0)
                else:
                    ymin, ymax = max(top_left[1] // factor,0), min(bottom_right[1] // factor,height)

                file.write(indent('<object>\n', 1))
                file.write(indent('<name>' + name + '</name>\n', 2))
                file.write(indent('<bndbox>\n', 2))
                file.write(indent('<xmin>' + str(xmin) + '</xmin>\n', 3))
                file.write(indent('<ymin>' + str(ymin) + '</ymin>\n', 3))
                file.write(indent('<xmax>' + str(xmax) + '</xmax>\n', 3))
                file.write(indent('<ymax>' + str(ymax) + '</ymax>\n', 3))
                file.write(indent('</bndbox>\n', 2))
                file.write(indent('</object>\n', 1))

            file.write('</annotation>\n')
        print('File:', frame_number, end = '\r')  

def extract_trainval(conf):
    """
    提取用于训练的所有帧的文件名
    """
    file_list = os.listdir(conf.train_img_path)
    with open(conf.trianval_path, 'w') as file:
        for img in file_list:
            img = img.strip().split('.')[0]
            file.write(img+'\n')

def prepare_data():
    from config import Config
    conf=Config()
    print("start extracting images from video")
    video2im(conf)

    print("start extracting pedestrian info.")
    extract_GT(conf)

    print("start extracting trian image names.")
    extract_trainval(conf)

if __name__ == "__main__":
    prepare_data()
