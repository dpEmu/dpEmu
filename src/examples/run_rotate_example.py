import imutils
import cv2


def rotate(image, angle):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imshow("Rotated (Correct)", rotated)
    cv2.waitKey(0)


def main():
    # load the image from disk
    image = cv2.imread("demo/landscape.png")
    rotate(image, 123)

    # loop over the rotation angles
    # for angle in np.arange(0, 360, 15):
    #     rotated = imutils.rotate(image, angle)
    #     cv2.imshow("Rotated (Problematic)", rotated)
    #     cv2.waitKey(0)

    # loop over the rotation angles again, this time ensuring
    # no part of the image is cut off
    # for angle in np.arange(0, 360, 15):
    #     rotated = imutils.rotate_bound(image, angle)
    #     cv2.imshow("Rotated (Correct)", rotated)
    #     cv2.waitKey(0)


if __name__ == "__main__":
    main()
