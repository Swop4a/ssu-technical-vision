import numpy as np
import cv2
from matplotlib import pyplot as plt


def plot(input_image, magnitude_spectrum):
    plt.subplot(121), plt.imshow(input_image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


image = cv2.imread('car.png', 0)

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plot(image, magnitude_spectrum)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
dx, dy = 30, 30
mask[crow - dx:crow + dx, ccol - dy:ccol + dy] = 1

# apply mask and inverse DFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
image_back = cv2.idft(f_ishift)
image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

plot(image, image_back)
