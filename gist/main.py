import cv2
import sys
import numpy
from matplotlib import pyplot as plt

print (cv2.__version__)
print (sys.version)

image = cv2.imread(sys.path[0] + "/input.jpg") * 1.0
# retval, outputImg = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

new_image = image
alpha, betta = 1.1, 1.0
new_image[:, :, :] *= alpha
new_image[:, :, :] += betta

new_image = numpy.array(new_image, dtype='float64').astype('float32')
cv2.imwrite("output.jpg", new_image)

color = ('b', 'r', 'g')
for i, col in enumerate(color):
    hist = cv2.calcHist([new_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()
