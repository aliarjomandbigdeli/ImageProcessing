import math
import cv2
import numpy as np

"""
.. sectionauthor:: Ali ArjomandBigdeli  <https://github.com/aliarjomandbigdeli>
.. since:: 1/21/2019
"""


def main():
    file_name = "test.jpg"
    video_file_name = "test.avi"

    read_and_show_image(file_name)

    split_channel_blue(file_name)

    show_gray_image(file_name)

    gaussian_blur_gray_scale_image(file_name)

    rotate_image(file_name, 90)

    half_width_image(file_name)

    img = cv2.imread(file_name, 1)
    height, width, channels = img.shape
    resize_image(file_name, math.floor(width / 2), height)

    show_image_edges(file_name)

    segment_image(file_name)

    face_detection(file_name)

    show_frame_of_a_video(video_file_name, 5)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_and_show_image(file_name):
    color = cv2.imread(file_name, 1)
    cv2.imshow("image", color)


def split_channel_blue(file_name):
    color = cv2.imread(file_name, 1)
    blue_img = color.copy()
    blue_img[:, :, 1] = 0
    blue_img[:, :, 2] = 0
    cv2.imshow('Blue channel', blue_img)

    # just show the value of channel blue
    # red = color[:, :, 2]
    # green = color[:, :, 1]
    # blue = color[:, :, 0]
    # cv2.imshow("blue", blue)

    # b, g, r = cv2.split(color)
    # cv2.imshow("blue", b)


def show_gray_image(file_name):
    color = cv2.imread(file_name, 1)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("gray.jpg", gray)
    cv2.imshow("gray scale", gray)


def gaussian_blur_gray_scale_image(file_name):
    color = cv2.imread(file_name, 1)
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("gaussian blur", blur)


def rotate_image(file_name, angel):
    # img = cv2.imread(file_name, 1)
    # rotated = cv2.rotate(img, angel)
    # cv2.imshow('rotated image', rotated)

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
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # removing noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # finding sure foreground area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    cv2.imshow('segmentation', img)


def face_detection(file_name):
    img = cv2.imread(file_name, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    path = "data/haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40, 40))
    print("number of faces that are detected: {}".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("face detection", img)


def show_frame_of_a_video(file_name, number_of_frames):
    cap = cv2.VideoCapture(file_name)
    frames = []
    for i in range(number_of_frames):
        ret, frame = cap.read()
        cv2.imshow("0.5 sec delay in video, frame{}".format(i + 1), frame)
        cv2.waitKey(500)

    for i in range(number_of_frames):
        ret, frame = cap.read()
        frames.append(frame)

    for i in range(len(frames)):
        cv2.imshow("5 first frame with 0.5 sec display delay{}".format(i + 1), frames[i])
        cv2.waitKey(500)

    cap.release()


if __name__ == '__main__':
    main()
