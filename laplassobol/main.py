import cv2
import sys

print (cv2.__version__)
print (sys.version)

image = cv2.imread(sys.path[0] + "/input.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (3, 3), 0)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobelxy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)


cv2.imwrite("laplacian.jpg", laplacian)
cv2.imwrite("sobelx.jpg", sobelx)
cv2.imwrite("sobely.jpg", sobely)
cv2.imwrite("sobelxy.jpg", sobelxy)
