import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('car.png', 0)

img_float32 = np.float32(img)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.subplot(222), plt.imshow(magnitude_spectrum.copy(), cmap='gray')

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
SQUARE = 20

for i in range(rows):
    for j in range(cols):
        if magnitude_spectrum[i, j] >= 255.0 and not (abs(crow - i) < SQUARE or abs(ccol - j) < SQUARE):
            magnitude_spectrum[i - SQUARE: i + SQUARE, j - SQUARE: j + SQUARE] = 0
            fshift[i - SQUARE: i + SQUARE, j - SQUARE: j + SQUARE] = 0

plt.subplot(223), plt.imshow(magnitude_spectrum, cmap='gray')
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(224), plt.imshow(img_back, cmap='gray')

plt.show()
cv2.imwrite('out.png', img_back)
