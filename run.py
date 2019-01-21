import math

import cv2

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 1/21/2019
"""


def main():
    file_name = "test.jpg"

    read_and_show_image(file_name)

    split_channel_blue(file_name)

    show_gray_image(file_name)

    rotate_image(file_name, 90)

    half_width_image(file_name)

    img = cv2.imread(file_name, 1)
    height, width, channels = img.shape
    resize_image(file_name, math.floor(width / 2), height)

    show_image_edges(file_name)

    segment_image(file_name)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_and_show_image(file_name):
    color = cv2.imread(file_name, 1)
    cv2.imshow("image", color)


def split_channel_blue(file_name):
    color = cv2.imread(file_name, 1)
    b, g, r = cv2.split(color)
    b_split = cv2.merge([b, b, b])
    cv2.imshow("blue", b_split)


def show_gray_image(file_name):
    color = cv2.imread(file_name, 1)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("gray.jpg", gray)
    cv2.imshow("gray scale", gray)


def rotate_image(file_name, angel):
    img = cv2.imread(file_name, 1)
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angel, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imshow("rotated", rotated)


def half_width_image(file_name):
    img = cv2.imread(file_name, 1)
    resize_image = cv2.resize(img, (0, 0), fx=0.5, fy=1)
    cv2.imshow("resize image", resize_image)


def resize_image(file_name, width, height):
    img = cv2.imread(file_name, 1)
    resize_image = cv2.resize(img, (width, height))
    cv2.imshow("resize image 2", resize_image)


def show_image_edges(file_name):
    img = cv2.imread(file_name, 1)
    edges = cv2.Canny(img, 100, 70)
    cv2.imshow("Canny edges", edges)


def segment_image(file_name):
    img = cv2.imread(file_name, 0)
    thresh_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow("segment by adaptive threshold", thresh_adapt)


if __name__ == '__main__':
    main()
