
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from pspnet_model import build_pspnet
from config import img_rows,img_cols,num_classes, patience, batch_size, epochs, num_train_samples, num_valid_samples, checkpoint_models_path_base
from data_generator import train_gen, valid_gen
from utils import get_available_cpus, get_available_gpus, plot_training,set_npy_weights

import tensorflow as tf
import os

if __name__ == '__main__':

    # Use GPU:1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set model save path
    model_dir = 'pspnet50_512x512_20190214'
    checkpoint_models_path = os.path.join(checkpoint_models_path_base,model_dir)
    if not os.path.exists(checkpoint_models_path):
        os.makedirs(checkpoint_models_path)

    # Set pretrained model path
    pretrained_path = "pspnet50_ade20k" 
   
    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model_names = checkpoint_models_path + 'pspnet50.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 2), verbose=1)

    # Load our model
    if pretrained_path is not None:
        pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols))
        set_npy_weights(weights_path=pretrained_path, model = pspnet50_model) 
    else:
        pspnet50_model = build_pspnet(num_classes, resnet_layers=50, input_shape=(img_rows,img_cols))


    Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    pspnet50_model.compile(optimizer='nadam', loss = 'categorical_crossentropy', metrics=['accuracy'])  # 此处metric为None
    pspnet50_model.summary()

    # Summarize then go!
    #num_gpu = get_available_gpus
    num_gpu = 1
    num_cpu = get_available_cpus()
    workers = int(round(num_cpu / 4))

    # Final callbacks
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    history = pspnet50_model.fit_generator(
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

    #plot_training(history,pic_name='pspnet50_train_val_loss_acc.png')
