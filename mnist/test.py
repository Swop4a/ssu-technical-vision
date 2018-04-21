from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

image = Image.open('img/digit_7_(0).png')
image_data = list(image.getdata())
image_data = np.array(image_data)
image_data.shape = (image.height, image.width)
image = image_data.tolist()

X_test = np.array([image])
X_test = X_test.reshape(1, 28 * 28)
X_test = X_test.astype('float32')
X_test /= 255

print("[INFO] loading network...")
model = load_model('model.h5')

prediction = model.predict(X_test, batch_size=128, verbose=1)

prediction = prediction.round()[0].tolist()
result = 0
for digit in range(10):
    result += digit * prediction[digit]
print('Data class is: ', int(result))
