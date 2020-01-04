#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import requests as req
import os
import re
import numpy as np


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
            number = max(file_numbers.flatten()) if file_numbers.flatten().size > 0 else 0
            # number = max(file_numbers.flatten())  # grab the largest value in array
            if number != num_files:  # compare the max value in array to num_files, if they are different then some images may be missing or deleted from folder
                print("WARNING: files in this directory are numbered incorrectly!")
                print(f"Number of files in folder: {num_files}")
                print(f"File number of last file: {number}")
        except OSError as error:
            print(error)

    return number


def find_list_items(soup2):
    locator = "div#wrapper div#content div#main ul#tiles li.thumbwook a"  # CSS Selector

    # image = soup2.select_one(locator) # selects one item from the locator
    # image_link = image.attrs['src'] # selects the 'src' attribute value

    images = soup2.select(locator)  # selects all items found in the locator
    image_links = [image.attrs['href'] for image in images]  # selects the 'href' attribute value
    return image_links


def download_images(file_path, file_name):
    print(file_path)
    r2 = req.get(URL)
    soup2 = BeautifulSoup(r2.text, "html.parser")
    image_files = find_list_items(soup2)
    init_num = file_numbering(file_path)
    for index, img_link in enumerate(image_files):
        img_data = req.get(img_link).content
        with open(f"{file_path}/{file_name} ({str(init_num+index+1)}).jpg", 'wb+') as f:
            f.write(img_data)
            print(f"Image '{file_name}{init_num+index+1}' was successfully written!")


URL = input("Enter the URL to download images from: ")
MODEL_NAME = input("Enter the name of the model: ").title()
FILE_NAME = MODEL_NAME.lower().split(" ")[0]
FILE_PATH = get_file_path(MODEL_NAME)
download_images(FILE_PATH, FILE_NAME)
