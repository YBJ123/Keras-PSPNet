# Keras-PSPNet

## Statement

- For part of PSPNet model, referenced from [Vladkryvoruchko/PSPNet-Keras-tensorflow](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow)
- For the part of data_generator and trainning method, referenced from [foamliu/Deep-Image-Matting](https://github.com/foamliu/Deep-Image-Matting)

## Project structure

### Folders

- [``models``](): path where save the keras model
- [``pretrained_weights``](): contains pretrained_wrights
- [``logs``](): path where save the trainning logs

### Files

- [``config.py``](https://github.com/liminn/Keras-PSPNet/blob/master/config.py): configure the input/trainning/dataset information
- [``data_generator.py``](https://github.com/liminn/Keras-PSPNet/blob/master/data_generator.py): customize keras data_generator of PSPNet model
- [``pspnet_model.py``](https://github.com/liminn/Keras-PSPNet/blob/master/pspnet_model.py): defines the PSPNet model
- [``train.py``](https://github.com/liminn/Keras-PSPNet/blob/master/train.py): defines the trainning progress
- [``predict.py``](https://github.com/liminn/Keras-PSPNet/blob/master/predict.py): defines the inference/predict progress
- [``utils.py``](https://github.com/liminn/Keras-PSPNet/blob/master/utils.py): contains all the useful function for [``data_generator.py``](https://github.com/liminn/Keras-PSPNet/blob/master/data_generator.py) and [``predict.py``](https://github.com/liminn/Keras-PSPNet/blob/master/predict.py)
- [``train_names.txt``](https://github.com/liminn/Keras-PSPNet/blob/master/train_names.txt): contains all the trainning image names
- [``valid_names.txt``](https://github.com/liminn/Keras-PSPNet/blob/master/valid_names.txt): contains all the valid image names