import cv2


def read_and_show_image(file_name):
    color = cv2.imread(file_name, 1)
    cv2.imshow(file_name, color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    read_and_show_image("test.jpg")


if __name__ == '__main__': main()
