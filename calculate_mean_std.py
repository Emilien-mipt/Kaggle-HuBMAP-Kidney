"""
Calculation Of standard deviation and mean (per channel) over all images of the image dataset
"""
import os

import cv2

PATH = "./data/hubmap-1024x1024/train"
SIZE = 1024


def get_mean_std_opencv(path, size):
    sum_mean = 0.0
    sum_std = 0.0
    file_name_list = os.listdir(path)
    for file_name in file_name_list:
        print("Processing file ", file_name)
        file_path = f"{path}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (size, size))
        mean, std = cv2.meanStdDev(resized_image)
        sum_mean += mean
        sum_std += std
    avg_mean = sum_mean / len(file_name_list)
    avg_std = sum_std / len(file_name_list)
    return avg_mean / 255.0, avg_std / 255.0


def main():
    MEAN, STD = get_mean_std_opencv(PATH, SIZE)

    print("MEAN: ", MEAN, "STD: ", STD)


if __name__ == "__main__":
    main()
