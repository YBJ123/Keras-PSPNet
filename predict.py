# import the necessary packages
import os
import random

import cv2
import keras.backend as K
import numpy as np

from config import img_rows,img_cols,num_classes,rgb_image_path,inference_output_path_base
from pspnet_model import build_pspnet
from utils import colorful,vis_segmentation
from matplotlib import pyplot as plt

if __name__ == '__main__':
    
    # 指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 加载模型
    model_weights_path = 'models/pspnet50_model_20190214.18-0.2068.hdf5'
    pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols))
    pspnet50_model.load_weights(model_weights_path)
    pspnet50_model.summary()

    # 测试图片
    # img_path_test = '/home/datalab/ex_disk1/bulang/segmentation_data_repository/data_shm_id_photo/Test_set/merge_test'
    # test_images = [f for f in os.listdir(img_path_test) if
    #                os.path.isfile(os.path.join(img_path_test, f)) and f.endswith('.png')]
    filename = 'test_names.txt'
    with open(filename, 'r') as f:
        names = f.read().splitlines()

    # 定义推理输出路径
    output_dir = "test_output_20190214"
    inference_out_path = os.path.join(inference_output_path_base,output_dir)
    if not os.path.exists(inference_out_path):
        os.makedirs(inference_out_path)

    for i in range(len(names)):
        name = names[i]

        image_path = os.path.join(rgb_image_path, name)
        image_bgr = cv2.imread(image_path,1)
        image_bgr = cv2.resize(image_bgr,(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
     
        x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
        x_test[0, :, :, 0:3] = image_bgr/255.0

        out = pspnet50_model.predict(x_test)
        out = np.reshape(out, (img_rows, img_cols, num_classes))
        out = np.argmax(out, axis=2)

        result_rgb = colorful(out)

        # 同时展示原图/预测结果/预测结果叠加原图，并保存
        vis_seg_save_path = os.path.join(inference_out_path,name.split('.')[0]+"_predict_compare.png")
        vis_segmentation(image_rgb,result_rgb,vis_seg_save_path)
        print("generating: {}".format(vis_seg_save_path))

    K.clear_session()

