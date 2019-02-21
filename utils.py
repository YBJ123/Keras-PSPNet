import multiprocessing
import os
import random

import cv2
import keras.backend as K
from keras.utils import to_categorical

import numpy as np
from tensorflow.python.client import device_lib

from config import num_classes, rgb_image_path, mask_img_path,img_rows, img_cols,unknown_code,min_scale,max_scale

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

def set_npy_weights(weights_path, model):
    npy_weights_path = os.path.join("pretrained_weights", "npy", weights_path + ".npy")
    print(npy_weights_path)
    json_path = os.path.join("pretrained_weights", "keras", weights_path + ".json")
    print(json_path)
    h5_path = os.path.join("pretrained_weights", "keras", weights_path + ".h5")
    print(h5_path)

    print("Importing weights from %s" % npy_weights_path)
    weights = np.load(npy_weights_path,encoding="latin1").item()

    for layer in model.layers:
        print(layer.name)
        if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
            mean = weights[layer.name]['mean'].reshape(-1)
            variance = weights[layer.name]['variance'].reshape(-1)
            scale = weights[layer.name]['scale'].reshape(-1)
            offset = weights[layer.name]['offset'].reshape(-1)
            model.get_layer(layer.name).set_weights([scale, offset, mean, variance])
        elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
            try:
                weight = weights[layer.name]['weights']
                model.get_layer(layer.name).set_weights([weight])
            except Exception as err:
                try:
                    biases = weights[layer.name]['biases']
                    model.get_layer(layer.name).set_weights([weight,
                                                             biases])
                except Exception as err2:
                    print(err2)
        if layer.name == 'activation_52':
            break

# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()

# Plot the training and validation loss + accuracy
def plot_training(history,pic_name='train_val_loss.png'):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(history.history['acc'],label="train_acc")
    plt.plot(history.history['val_acc'],label="val_acc")
    plt.title("Train/Val Loss and Train/Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Acc")
    plt.legend(loc="upper left")
    plt.savefig(pic_name)

def random_rescale_image_and_mask(image,mask,min_scale = min_scale, max_scale = max_scale):
    rows = image.shape[0]
    cols = image.shape[1]
    # print("image.shape:{}".format(image.shape))
    # print("mask.shape:{}".format(mask.shape))
    # print("rows:{},cols:{}".format(rows,cols))
    ratio = random.uniform(min_scale,max_scale)
    # print("ratio:{}".format(ratio))
    new_rows = int(ratio*rows)
    new_cols = int(ratio*cols)
    # print("new_rows:{},new_cols:{}".format(new_rows,new_cols))
    image = cv2.resize(image, dsize=(new_cols, new_rows), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, dsize=(new_cols, new_rows), interpolation=cv2.INTER_LINEAR)
    # print("image.shape:{}".format(image.shape))
    # print("mask.shape:{}".format(mask.shape))
    return image,mask

def generate_random_trimap(alpha):
    ### 非0区域置为255，然后膨胀及收缩，多出的部分为128区域
    ### 优点：如果有一小撮头发为小于255，但大于0的，那通过该方法，128区域会覆盖到该一小撮头发部分
    mask = alpha.copy()                             # 0~255
    # 非纯背景置为255
    mask = ((mask!=0)*255).astype(np.float32)       # 0.0和255.0
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 如果尺寸过小(总面积小于500*500)，则减半膨胀和腐蚀的程度
    if(alpha.shape[0]* alpha.shape[1] < 250000):
        dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(5, 7))   # 膨胀少点
        erode = cv2.erode(mask, kernel, iterations=np.random.randint(7, 10))    # 腐蚀多点
    else:
        dilate = cv2.dilate(mask, kernel, iterations=np.random.randint(10, 15)) # 膨胀少点
        erode = cv2.erode(mask, kernel, iterations=np.random.randint(15, 20))   # 腐蚀多点

    ### for循环生成trimap，特别慢
    # for row in range(mask.shape[0]):
    #     for col in range(mask.shape[1]):
    #         # 背景区域为第0类
    #         if(dilate[row,col]==255 and mask[row,col]==0):
    #             img_trimap[row,col]=128
    #         # 前景区域为第1类
    #         if(mask[row,col]==255 and erode[row,col]==0):
    #             img_trimap[row,col]=128

    ### 操作矩阵生成trimap，特别快
    # ((mask-erode)==255.0)*128  腐蚀掉的区域置为128
    # ((dilate-mask)==255.0)*128 膨胀出的区域置为128
    # + erode 整张图变为255/0/128
    img_trimap = ((mask-erode)==255.0)*128 + ((dilate-mask)==255.0)*128 + erode

    return img_trimap.astype(np.uint8)

# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(img_rows, img_cols)):
    crop_height, crop_width = crop_size
    # np.where(arry)：输出arry中‘真’值的坐标(‘真’也可以理解为非零)
    # 返回：(array([]),array([])) 第一个array([])是行坐标，第二个array([])是列坐标
    y_indices, x_indices = np.where(trimap == unknown_code)
    # 未知像素的数量
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        # 任取一个未知像素的坐标
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        # 为下面的剪裁提供起始点
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y

def safe_crop(mat, x, y, crop_size=(img_rows, img_cols)):
    # 例如：crop_height = 640，crop_width = 640
    crop_height, crop_width = crop_size
    # 对于alpha，先建立尺寸为(crop_height, crop_width)的全0数组
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    # 对于fg,bg,image，先建立尺寸为(crop_height, crop_width,3)的全0数组
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    # 注意：这里是函数名为safe_crop的原因！
    # 若(y+crop_height)超出了mat的范围，则也不会报错，直接取到mat的边界即停止
    # 因此crop的尺寸不一定是(crop_height,crop_height)，有可能小于(crop_height,crop_height)
    crop = mat[y:y+crop_height, x:x+crop_width]
    # 得到crop的尺寸
    h, w = crop.shape[:2]
    # 将crop所包含的内容，赋给ret
    # 当然，ret其余部分为0
    ret[0:h, 0:w] = crop
    # 缩放到(img_rows,img_cols)，即(320,320)
    if crop_size != (img_rows, img_cols):
        # dsize即指的是Size(width，height)
        print("crop_size != (512,512)")
        ret = cv2.resize(ret, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return ret

def make_trimap_for_batch_y(trimap):
    for row in range(trimap.shape[0]):
        for col in range(trimap.shape[1]):
            # 背景区域为第0类
            if(trimap[row,col]==0):
                trimap[row,col]=0
            # 前景区域为第1类
            if(trimap[row,col]==255):
                trimap[row,col]=1
            # 128区域为第2类
            if(trimap[row,col]==128): 
                trimap[row,col]=2
    trimap = to_categorical(trimap, 3) 
    return trimap

def colorful(out):
    result_rgb = np.empty((img_rows, img_cols, 3), dtype=np.uint8)
    for row in range(img_rows):
        for col in range(img_cols):
            # 背景区域为第0类
            if(out[row,col]==0):
                result_rgb[row,col,0]=0
                result_rgb[row,col,1]=0
                result_rgb[row,col,2]=0
            # 前景区域为第1类
            if(out[row,col]==1):
                result_rgb[row,col,0]=255
                result_rgb[row,col,1]=255
                result_rgb[row,col,2]=255
            # 128区域为第2类
            if(out[row,col]==2): 
                result_rgb[row,col,0]=128
                result_rgb[row,col,1]=128
                result_rgb[row,col,2]=128
    return result_rgb

def vis_segmentation(image, seg_map, save_path_name = "examples.png"):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = seg_map
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  # ax = plt.subplot(grid_spec[3])
  # legend_elements = [Line2D([0], [0], color='black', lw=4, label='Background'),
  #                    Line2D([0], [0], color='gray', lw=4, label='Unknow Area'),
  #                    Line2D([0], [0], color='white', lw=4, label='Foreground')]
  # ax.legend(handles=legend_elements,loc = "center")
  # plt.axis('off')
  # plt.show()

  plt.savefig(save_path_name)
  plt.close('all')
  
if __name__ == '__main__':
    ### test generator_random_trimap()
    # img_mask = cv.imread('trimap_test_out/supervisely4847.png',0) 
    # i = 0
    # for i in list(range(10)):
    #     generator_random_trimap(img_mask,i)

    ### test random_rescale_image_and_mask()
    image  = cv2.imread("./temp/image/supervisely5641.jpg",1)
    mask  = cv2.imread("./temp/mask/supervisely5641.png",0)
    image,mask = random_rescale_image_and_mask(image,mask)
    cv2.imwrite("./temp/image/image_new.png",image)
    cv2.imwrite("./temp/mask/mask_new.png",mask)




