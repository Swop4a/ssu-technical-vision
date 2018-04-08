import cv2
import numpy as np
import sys

print (cv2.__version__)
print (sys.version)

image = cv2.imread(sys.path[0] + "/input.jpg")
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

height, width, colors = np.shape(lab_image)
for py in range(0, height):
    print("Step #", py)
    for px in range(0, width):
        lab_image[py][px][1] = (lab_image[py][px][1] - 128) * 1.5 + 128
        lab_image[py][px][2] = (lab_image[py][px][2] - 128) * 1.5 + 128

result = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
cv2.imwrite("output.jpg", result)
