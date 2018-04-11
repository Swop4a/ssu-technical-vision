import cv2
import matplotlib.pyplot as plt
import sys

print(cv2.__version__)
print(sys.version)

image1 = cv2.imread('input1.jpg')
image2 = cv2.imread('input2.jpg')

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

image3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:18], None, flags=2)

plt.gcf().set_size_inches((plt.gcf().get_size_inches()[0] * 2, plt.gcf().get_size_inches()[1] * 2))
plt.imshow(image3)
plt.show()
