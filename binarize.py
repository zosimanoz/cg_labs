"""

Binary Images

"""

# using numpy
import cv2

# using matplotlib
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
# apply thresholding to convert grayscale to binary image
ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

w, h = thresh.shape[:2]
max_pixel_value = thresh.max()
print(f" max pix value : {max_pixel_value}")

newImage = np.zeros([w, h])

for i in range(w):
    for j in range(h):
        newImage[i, j] = max_pixel_value - thresh[i, j]


f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(thresh, cmap=plt.get_cmap("binary"))
f.add_subplot(1, 2, 2)
plt.imshow(newImage, cmap=plt.get_cmap("binary"))
plt.show()
