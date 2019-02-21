import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from pspnet_model import build_pspnet
from customize import CustomizeModelCheckpoint
from config import img_rows,img_cols,num_classes, patience, batch_size, epochs, num_train_samples, num_valid_samples, checkpoint_models_path_base
from data_generator import train_gen, valid_gen
from utils import get_available_cpus, get_available_gpus, plot_training,set_npy_weights

import tensorflow as tf
import os

if __name__ == '__main__':

    # Use GPU:1
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Set model save path
    model_dir = 'pspnet50_512x512_20190220_supervisely/'
    checkpoint_models_path = checkpoint_models_path_base + '/' + model_dir
    if not os.path.exists(checkpoint_models_path):
        os.makedirs(checkpoint_models_path)
    
    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_save_path = checkpoint_models_path + 'pspnet50.{epoch:02d}-{val_loss:.4f}.hdf5'
    #model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 2), verbose=1)

    # Set pretrained model path
    pretrained_path = "pspnet50_ade20k" 
    #pretrained_path = 'models/pspnet50_512x512_20190214/pspnet50_model_20190214.22-0.2034.hdf5'

    # Load our model, added support for Multi-GPUs
    num_gpu = len(get_available_gpus())
    if num_gpu >= 2:
        with tf.device("/cpu:0"):
            pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols))
            if pretrained_path is not None:
                set_npy_weights(weights_path=pretrained_path, model = pspnet50_model)
                #pspnet50_model.load_weights(pretrained_path)
            else:
                pass
        final = multi_gpu_model(pspnet50_model, gpus=num_gpu)
        # rewrite the callback: saving through the original model and not the multi-gpu model.
        model_checkpoint = CustomizeModelCheckpoint(pspnet50_model,model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
    else:
        pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols))
        if pretrained_path is not None:
            set_npy_weights(weights_path=pretrained_path, model = pspnet50_model)
            #pspnet50_model.load_weights(pretrained_path)
        else:
            pass
        final = pspnet50_model

    # finetune the whole network together.
    for layer in final.layers:
        layer.trainable = True

    Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    final.compile(optimizer='nadam', loss = 'categorical_crossentropy', metrics=['accuracy'])  # 此处metric为None
    final.summary()

    # Summarize then go!
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 4))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    final.fit_generator(
                        generator=train_gen(),
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_gen(),
                        validation_steps=num_valid_samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=workers,
                        #initial_epoch = 56
                        )
