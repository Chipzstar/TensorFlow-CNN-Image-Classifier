import os
from tqdm import tqdm

# Check what folder to use for training and testing images
CHISOM_ALIGNED_TRAIN_DIR = 'C:/Users/chiso/MEGA/data/aligned_train'
YISI_ALIGNED_TRAIN_DIR = 'E:/MegaSync/data/aligned_train'


def get_directories():
    if os.path.exists(CHISOM_ALIGNED_TRAIN_DIR):
        return CHISOM_ALIGNED_TRAIN_DIR
    else:
        return YISI_ALIGNED_TRAIN_DIR


def create_labels_file(labels_dir):
    with open('labels.txt', 'w') as file:
        for folder in tqdm(os.listdir(labels_dir)):
            file.write(str(folder + "\n"))
    # After leaving the above block of code, the file is closed


ALIGNED_TRAIN_DIR = get_directories()

create_labels_file(ALIGNED_TRAIN_DIR)
