import os
import random
from random import shuffle

import cv2
import numpy as np

from keras.utils import Sequence

from config import img_rows, img_cols, batch_size, num_classes,rgb_image_path, mask_img_path,unknown_code
from utils import generate_random_trimap, random_choice, safe_crop, make_trimap_for_batch_y,random_rescale_image_and_mask

class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        # filename会是"train_names.txt"或"val_names.txt"
        # "train_names.txt"、"val_names.txt"中图片名，例如"0-100.png"
        filename = '{}_names.txt'.format(usage)
        with open(filename, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        # 每一个epoch，从前往后读取self.ids，依据id，读取self.names
        # idx应为第几个batch，i为该次batch的起始点
        i = idx * batch_size
        # length为当前batch的大小
        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, 3), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, num_classes), dtype=np.uint8)

        for i_batch in range(length):
            ###normal
            img_name = self.names[i] # xx.jpg
            img_name_prefix,useless = os.path.splitext(img_name)
            mask_name = img_name_prefix+'.png'

            image_path = os.path.join(rgb_image_path, img_name)
            image = cv2.imread(image_path,1)
            mask_path = os.path.join(mask_img_path, mask_name)
            mask = cv2.imread(mask_path,0)

            ###temp
            # img_name = self.names[i] # xx.jpg
            # image_path = os.path.join(rgb_image_path, img_name)
            # image = cv2.imread(image_path,1)

            # img_name_prefix = img_name.split('split')[0][0:-1]
            # mask_name = img_name_prefix+'.png'
            # mask_path = os.path.join(mask_img_path, mask_name)
            # mask = cv2.imread(mask_path,0)
            ##mask = (mask!=0)*255

            # 随机缩放image和mask，0.5~2.0
            image,mask = random_rescale_image_and_mask(image,mask)

            # 实时处理alpha，得到trimap:128/0/255
            trimap = generate_random_trimap(mask)

            # 定义随机剪裁尺寸
            crop_size = (512,512)
            # 获得剪裁的起始点，其目的是为了保证剪裁的图像中包含未知像素
            x, y = random_choice(trimap, crop_size)      

            # 剪裁image，到指定剪裁尺寸crop_size，并缩放到(img_rows,img_cols)
            image = safe_crop(image, x, y, crop_size)
            # 剪裁trimap，到指定剪裁尺寸crop_size，并缩放到(img_rows,img_cols)
            trimap = safe_crop(trimap, x, y, crop_size)

            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                trimap = np.fliplr(trimap)

            ### save the image/trimap crop patch
            # patch_save_dir = "show_data_loader"
            # image_patch_path = "show_data_loader" + '/' + img_name_prefix + '_image_' + str(i_batch) + '.png'
            # trimap_patch_path = "show_data_loader" + '/' + img_name_prefix + '_trimap_' + str(i_batch) + '.png'
            # cv2.imwrite(image_patch_path,image)
            # cv2.imwrite(trimap_patch_path,trimap)

            batch_x[i_batch] = image/255.0
            batch_y[i_batch] = make_trimap_for_batch_y(trimap) 

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)

def train_gen():
    return DataGenSequence('train')

def valid_gen():
    return DataGenSequence('valid')

if __name__ == '__main__':
    pass
