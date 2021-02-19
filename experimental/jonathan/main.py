#!/usr/bin/env python3
import cv2
import os
from time import perf_counter

cwd = os.getcwd()
base_dir = (
    "/home/jonathan/Documents/uav/hawk-eye/data_generation/assets/competition-2019/"
)
image_dir = cwd + "/sliced_images/"


def preprocessing():
    if os.isdir(image_dir):
        for f in os.listdir(image_dir):
            os.remove(os.path.join(image_dir, f))
    else:
        os.mkdir(image_dir)


def process(image, num):
    img = cv2.imread(image)
    images = []
    h, w, c = img.shape
    for y in range(0, h - 512, 512):
        for x in range(0, w - 512, 512):
            images.append(img[y : y + 512, x : x + 512])
    for i in range(0, len(images)):
        name_of_img = image_dir + str(num) + str(i) + ".png"
        cv2.imwrite(name_of_img, images[i])


def main():
    t1 = perf_counter()
    # doing a conuter since my computer sucks and slows down really bad when trying to do the whole folder
    processed = 0
    for f in os.listdir(base_dir):
        # slice the image
        if processed > 1:
            break
        else:
            image_file_name = os.path.join(base_dir, f)
            process(image_file_name, processed)
        processed += 1
    t2 = perf_counter()
    time = t2 - t1
    print(f"total time elasped for splitting 2 images is {time}")


if __name__ == "__main__":
    main()
