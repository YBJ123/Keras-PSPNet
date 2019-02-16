img_rows = 512
img_cols = 512
unknown_code = 128

num_classes = 3

# 训练集占总数据集的80%
num_train_samples = 8835
# 验证集占总数据集的20%         
num_valid_samples = 2209          
batch_size = 8
epochs = 100
patience = 10

# Set model save path
checkpoint_models_path_base = 'models'
# Set inference output path
inference_output_path_base = "inference_output"

# train image path
#rgb_image_path = '/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/T-Net-Person/image'
rgb_image_path = "../SemanticHumanMatting/data_shm_id_photo/Training_set/composite"
# train mask path
#mask_img_path = '/home/datalab/ex_disk1/bulang/segmentation_data_repository/原始数据/T-Net-Person/mask'
mask_img_path = "../SemanticHumanMatting/data_shm_id_photo/Training_set/alpha"
