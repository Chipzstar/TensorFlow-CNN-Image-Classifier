#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[1]:


import tensorflow as tf
import datetime
import time
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
from random import shuffle, randint, seed
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.framework import graph_util
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# In[2]:


import bz2
import argparse
from tensorflow.keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

# In[3]:


print(f'OpenCV version: {cv2.__version__}')
print(f'Tensorflow version: {tf.__version__}')

# In[4]:


## Gets the repo for aligned images

# !rm -rf sample_data
# !git clone https://github.com/pbaylies/stylegan-encoder


# # PREPROCESSING

# In[5]:


# Check what folder to use for training and testing images
CHISOM_TRAIN_DIR = 'C:/Users/chiso/MEGA/data/train'
CHISOM_TEST_DIR = 'C:/Users/chiso/MEGA/data/test'
CHISOM_ALIGNED_TRAIN_DIR = 'C:/Users/chiso/MEGA/data/aligned_train'
CHISOM_ALIGNED_TEST_DIR = 'C:/Users/chiso/MEGA/data/aligned_test'

YISI_TRAIN_DIR = 'E:/MegaSync/data/train'
YISI_TEST_DIR = 'E:/MegaSync/data/test'
YISI_ALIGNED_TRAIN_DIR = 'E:/MegaSync/data/aligned_train'
YISI_ALIGNED_TEST_DIR = 'E:/MegaSync/data/aligned_test'


def get_directories():
    if os.path.exists(CHISOM_TRAIN_DIR) and os.path.exists(CHISOM_TEST_DIR) and os.path.exists(
            CHISOM_ALIGNED_TRAIN_DIR) and os.path.exists(CHISOM_ALIGNED_TEST_DIR):
        return CHISOM_TRAIN_DIR, CHISOM_TEST_DIR, CHISOM_ALIGNED_TRAIN_DIR, CHISOM_ALIGNED_TEST_DIR
    else:
        return YISI_TRAIN_DIR, YISI_TEST_DIR, YISI_ALIGNED_TRAIN_DIR, YISI_ALIGNED_TEST_DIR


# ### Useful Parameters

# In[6]:


DATE = datetime.datetime.now().strftime('%d-%b-%Y')
TRAIN_DIR, TEST_DIR, ALIGNED_TRAIN_DIR, ALIGNED_TEST_DIR = get_directories()
IMG_SIZE = 50
LR = 1e-3
MODEL_PATH = f'models/{DATE}/'
MODEL_NAME = 'ImageClassifier-keras-6-Conv-Layer-{}.model'.format(int(time.time()))
TENSORBOARD = TensorBoard(log_dir=f'logs\\{MODEL_NAME}')
NUM_CLASSES = len(next(os.walk(ALIGNED_TRAIN_DIR))[1])

# In[7]:


# ### Walkthrough of Subfolders in Train Directory:

# In[8]:


# Only the root
ROOTS = next(os.walk(ALIGNED_TRAIN_DIR))[0]
print(f"Roots = {ROOTS}")

# In[9]:


# Only the directories
DIRS = next(os.walk(ALIGNED_TRAIN_DIR))[1]
print(DIRS)

# In[10]:


# Only the files
"""for root, dirs, files in os.walk(ALIGNED_TRAIN_DIR):
    for name in files:
        print(name.split('.')[0]) # filters the file name by file extension and the copy_number
        
"""

# ### Generation of Image classes

# In[11]:


# Used for abbreviating the class names NOT USED

"""def get_class_labels():
    labels = []
    for root, dirs, files in os.walk(TRAIN_DIR):
        path = root.split(os.sep)
        for folder in dirs:
            name = folder.split()
            class_label = "".join([letter[0] for letter in name])
            labels.append(class_label)
    return labels"""

# ### One-Hot Encoding

# In[12]:


LABELS = next(os.walk(ALIGNED_TRAIN_DIR))[1]  # all the class labels (pornstar names) to be used
LABELS = np.reshape(LABELS, (-1, 1))  # reshapes array from 1D to 2D array
mlb = MultiLabelBinarizer()
encoded_labels = np.array(mlb.fit_transform(LABELS))
# dict(zip(LABELS.flatten(), encoded_labels))


# In[13]:


# img.split('.')[0].split('(')[0]  # filters the file name by file extension and the copy_number
"""
Labelled training data
"""


def create_train_data():
    training_data = []
    # iterate over each image-class (subfolder) in training directory
    for folder in tqdm(os.listdir(ALIGNED_TRAIN_DIR)):
        full_path = f'{ALIGNED_TRAIN_DIR}/{folder}'
        # iterate over each image in each subfolder
        for img in os.listdir(full_path):
            img_name = str(folder)  # the sub-folder is used as the image name for each image
            img_name = img_name.strip()  # removes any leading and trailing whitespaces from the img name
            label = mlb.transform([[img_name]])  # encodes the label of the image using MultiLabelBinarizer
            label = label.flatten()  # converts encoded label from 2D to 1D array
            # print(f'Image: {img} - Encoding:{label}')
            path = os.path.join(full_path, img)  # full path of the image
            # feature extraction
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            img = tf.cast(img, tf.float32)  # change data type of image to float32
            training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


# In[14]:


"""
Unlabelled test data
"""


def process_test_data():
    img_ids = list(range(len(os.listdir(ALIGNED_TEST_DIR))))  # generates list of ID numbers
    shuffle(img_ids)  # randomly assorted
    img_ids = iter(img_ids)
    testing_data = []
    for img in tqdm(os.listdir(ALIGNED_TEST_DIR)):
        path = os.path.join(ALIGNED_TEST_DIR, img)
        img_num = next(img_ids)
        print(f"ID: {img_num} - Image: {img}")
        # feature extraction
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32)
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data


# ### Generate Training and Testing data

# In[15]:


# train_data = create_train_data()
# test_data = process_test_data()
# if train/test data already exists
train_data = np.load('train_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)

# In[16]:

print(len(train_data))
print(len(test_data))


# # BUILDING THE MODEL

# ### Implementation of Convoluted Neural Network

# In[17]:


def create_cnn_model():
    # tf.reset_default_graph()
    model = Sequential()
    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    # INPUT LAYER
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    # HIDDEN LAYER 1
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # HIDDEN LAYER 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # HIDDEN LAYER 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully Connected
    model.add(Flatten())  # converts the 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # reduces overfitting

    # OUTPUT LAYER
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# # TRAINING THE NETWORK

# ### Cross Validation Train/Test Split

# In[18]:


train = train_data[:-(NUM_CLASSES * 10)]  # sample train data
test = train_data[-(NUM_CLASSES * 10):]

# In[19]:


train_X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # train features (images)
train_Y = np.array([i[1] for i in train])  # train labels

test_X = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # test features (images)
test_Y = np.array([i[1] for i in test])  # test labels

# ### Feature Scaling (Normalization)

# In[20]:


# Have to divide by 255 
train_X = train_X / 255.0
test_X = test_X / 255.0

# In[21]:


print(f"train data: {train_X.shape}")
print(f"train labels: {train_Y.shape}")
print(f"test data: {test_X.shape}")
print(f"test labels: {test_Y.shape}")

# ### Frequency distribution of classes being used in "test data"

# In[22]:


enc = []
for img in test:
    enc.append(img[1])

enc = np.array(enc)
test_labels = mlb.inverse_transform(enc)
c = Counter(test_labels)
print(c)

# In[23]:


"""MODEL = create_cnn_model()
MODEL.summary()
history = MODEL.fit(train_X, train_Y, batch_size=32, epochs=100, validation_data=(test_X, test_Y), verbose=2, callbacks=[TENSORBOARD])"""

# ## Saving Model

# In[24]:


# MODEL.save(f'{MODEL_PATH}')


# ## Load Model

# In[25]:


MODEL = tf.keras.models.load_model(f'{MODEL_PATH}')

# ### Convert model to TensorFlow Lite format

# In[36]:


"""converter = tf.lite.TFLiteConverter.from_keras_model(MODEL)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)"""

# # RESULTS

# In[27]:


CLASS_INDEX = dict(zip([np.argmax(x) for x in encoded_labels], LABELS.flatten()))
# CLASS_INDEX = dict(sorted(CLASS_INDEX.items()))
LABELS = LABELS.flatten()
IMAGE_IDs = []

# ### Graph Plot of Predicted Classes

# In[28]:


fig = plt.figure(figsize=(20, 10))
results = {cls: [] for cls in LABELS}

# iterate over each image in test_sample
# get the model's class prediction of the image
for num, data in enumerate(test_data):
    data[0] = data[0] / 255.0
    img_data = data[0]
    img_num = data[1]
    y = fig.add_subplot(6, 6, num + 1)
    orig = img_data
    data = img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    model_out = MODEL.predict([data]).flatten()
    index = np.argmax(model_out)
    # generate output dictionary
    results = {LABELS[i]: results.get(LABELS[i]) + [model_out[i]] for i in range(NUM_CLASSES)}
    IMAGE_IDs.append(img_num)

    # cross-reference the predicted class-index to its class-label (for each test image)
    class_label = CLASS_INDEX.get(index, 'Invalid class!')
    print(f"Image ID: {img_num}\t | Prediction: {class_label}")

    y.imshow(orig, cmap='gray')
    plt.title(f'{img_num}: {class_label}')
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
# plt.savefig('Class Results')


# In[29]:


imgs = [img.split('.')[0] for img in next(os.walk(ALIGNED_TEST_DIR))[2]]

# ### Tabulated Prediction Probabilities

# In[30]:


# Creates a HeatMap using the seaborn library
cm = sns.light_palette("blue", as_cmap=True)
df = pd.DataFrame.from_dict(results, orient='index', columns=imgs)
df.style.format("{:.2%}").set_caption('Confidence Values').background_gradient(cmap=cm)

# In[31]:


"""
Re-structures the results dictionary so that each class_label points to another dictionary {k, v}
where k = the Image_Id number and v = the confidence value
"""


def gen_results(results):
    my_dict = {}
    for cls in LABELS:
        probs = iter(results[cls])
        my_dict.update({cls: {}})
        for k in IMAGE_IDs:
            my_dict[cls][int(k)] = next(probs)

    return my_dict


# In[32]:


def get_top5(results, ID=1):
    results = gen_results(results)
    probs = np.array([(results[k][ID]) for k in results])
    # print(f'Reverse: {(-probs).argsort()} - {sorted(probs, reverse=True)}')
    indices = (-probs).argsort()[
              :5]  # sorts probabilities (largest - smallest) + returns their corresponding array indices
    top_5 = [CLASS_INDEX.get(i) for i in indices]
    return top_5


# In[33]:


Image_ID = 7
TOP_5 = get_top5(results, Image_ID)
print(TOP_5)


# # Get Overall Accuracy

# In[34]:


def get_overall_accuracy(results):
    i = 0
    num_correct = 0
    total = len(test_data)  # total number of images
    keys = results.keys()
    class_labels = []

    for ID in IMAGE_IDs:  # loop through each image ID
        predictions = []
        for key in list(keys):  # for each model in the results dictionary
            prob = results[key].get(ID)
            predictions.append(prob)
        max_index = np.argmax(predictions)  # max index
        label = CLASS_INDEX.get(max_index, 'Invalid class!')
        class_labels.append(label)

    for img in os.listdir(ALIGNED_TEST_DIR):
        img = img.split('.')[0].strip()  # gets the class name of the image file
        if img == class_labels[i]:
            num_correct += 1
            # print(f"Image name: {img} - predicted label: {class_labels[i]}")
        print(f"Image name: {img} - predicted label: {class_labels[i]}")
        i += 1

    accuracy = round((num_correct / total) * 100, 2)
    return f'{accuracy}%'


# # Overall Accuracy

# In[35]:


get_overall_accuracy(gen_results(results))

# In[35]:
