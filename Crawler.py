#!/usr/bin/env python
# coding: utf-8

import os
import re
import numpy as np
import requests as req
from bs4 import BeautifulSoup

#%%

CHISOM_FILE_PATH = os.path.normpath('C:/Users/chiso/MEGA/data/new')
YISI_FILE_PATH = os.path.normpath('E:/MegaSync/data/new')


def get_file_path(folder_name):
    if os.path.exists(CHISOM_FILE_PATH):
        file_path = CHISOM_FILE_PATH
    else:
        file_path = YISI_FILE_PATH

    full_path = os.path.join(file_path, folder_name)
    try:
        os.mkdir(full_path)
    except OSError as error:
        print(error)
    return full_path


"""Ensures correct numbering system is maintained"""


def file_numbering(path):
    number = 0
    if os.path.exists(path):
        try:
            files = os.listdir(path)  # list all files in file path
            num_files = len(files)  # number of files in folder
            regex = re.compile(r'\d+')  # regular expression to extract only the number from filename(s)
            file_numbers = np.array([regex.findall(file) for file in files]).astype(int)  # store all file numbers in an array
            number = max(file_numbers.flatten()) if file_numbers.flatten().size > 0 else 0  # grab the largest value in array
            if number != num_files:  # compare the max value in array to num_files, if they are different then some images may be missing or deleted from folder
                print("WARNING: files in this directory are numbered incorrectly!")
                print(f"Number of files in folder: {num_files}")
                print(f"File number of last file: {number}")
        except OSError as error:
            print(error)

    return number


def check_prev_model_name():
    if 'MODEL_NAME' in globals():  # Check if the variable 'MODEL NAME' exists in the program
        return True
    return False


def get_model_name():
    response = input(f"Use previous model name: {MODEL_NAME}? y/n")
    if response == 'y' or response == 'Y':
        return MODEL_NAME
    return input("Enter name of new model: ").title()


def find_list_items(soup2):
    locator = "div#wrapper div#content div#main ul#tiles li.thumbwook a" # CSS Selector

    images = soup2.select(locator) # selects all items found in the locator
    image_links = [image.attrs['href'] for image in images] # selects the 'href' attribute value
    return image_links


def download_images(file_path, file_name):
    print(file_path)
    r2 = req.get(URL)  # makes a http request to the web site url
    soup2 = BeautifulSoup(r2.text, "html.parser")  # parses the website
    image_files = find_list_items(soup2)
    init_num = file_numbering(file_path)
    for index, img_link in enumerate(image_files):
        if index == NUM_IMAGES:
            break
        img_data = req.get(img_link).content
        with open(f"{file_path}/{file_name} ({str(init_num+index+1)}).jpg", 'wb+') as f:
            f.write(img_data)
            print(f"Image '{file_name}{init_num+index+1}' was successfully written!")


is_finished = False
while not is_finished:
    URL = input("Enter the URL to download images from: ")
    MODEL_NAME = input("Enter the name of the model: ").title() if not check_prev_model_name() else get_model_name()
    NUM_IMAGES = int(input("Number of images to select: "))
    FILE_NAME = MODEL_NAME.lower().split(" ")[0]
    FILE_PATH = get_file_path(MODEL_NAME)
    download_images(FILE_PATH, FILE_NAME)
    CONTINUE = input("\nContinue? y/n")
    if not (CONTINUE == 'y' or CONTINUE == 'Y'):
        is_finished = True
