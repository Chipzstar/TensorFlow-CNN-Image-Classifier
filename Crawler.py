#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import requests as req
import os

CHISOM_FILE_PATH = os.path.normpath('C:/Users/chiso/MEGA/data/new')
YISI_FILE_PATH = os.path.normpath('E:/MegaSync/data/new')


def get_file_path(folder_name):
    if os.path.exists(CHISOM_FILE_PATH):
        file_path = CHISOM_FILE_PATH
    else:
        file_path = YISI_FILE_PATH

    full_path = os.path.join(file_path, folder_name)
    os.mkdir(full_path)
    return full_path


def find_list_items(soup2):
    locator = "div#wrapper div#content div#main ul#tiles li.thumbwook img"  # CSS Selector

    # image = soup2.select_one(locator) # selects one item from the locator
    # image_link = image.attrs['src'] # selects the 'src' attribute value

    images = soup2.select(locator)  # selects all items found in the locator
    image_links = [image.attrs['src'] for image in images]  # selects the 'src' attribute value

    return image_links


def download_images(file_path, file_name):
    r2 = req.get(URL)
    soup2 = BeautifulSoup(r2.text, "html.parser")
    image_files = find_list_items(soup2)
    for index, img_link in enumerate(image_files):
        img_data = req.get(img_link).content
        with open(f"{file_path}/{file_name}_{str(index+1)}.jpg", 'wb+') as f:
            f.write(img_data)
            print(f"File {index+1} was successfully written!")


# example url: https://www.pornpics.com/galleries/curvy-lady-frees-her-big-boobs-and-butt-from-bra-and-panties-in-her-office/"
URL = input("Enter the URL to download images from: ")
MODEL_NAME = input("Enter the name of the model: ")
FILE_NAME = MODEL_NAME.lower().split(" ")[0]
print(FILE_NAME)
download_images(get_file_path(MODEL_NAME), FILE_NAME)
