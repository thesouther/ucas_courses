class Config:
    def __init__(self):
        self.img_width = 1920
        self.img_height = 1080
        self.img_depth = 3
        self.town_centre_path = './data/TownCentre/TownCentreXVID.avi' 
        self.town_centre_info_path = './data/TownCentre/TownCentre-groundtruth.top'
        self.train_img_path = './data/train_images/'
        self.test_img_path = './data/test_images/'
        self.img_shrink_factor = 2
        self.train_size = 4500
        self.train_data_rate = 0.95

        self.trianval_xml_path = './data/annotations/xmls/'
        self.trianval_path = './data/annotations/trainval.txt'

        self.label_map_path = './data/annotations/label_map.pbtxt'
        self.annotations_dir = './data/annotations/'

        self.train_record_path = './data/train_data.record'
        self.val_record_path = './data/val_data.record'
        self.test_result_path = './result/images/'
        self.result_gif_path = './result/images.gif'

        self.pd_path = 'pb_output/frozen_inference_graph.pb'
        