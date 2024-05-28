"""

Image thresholding

"""

# using numpy
import cv2

# using matplotlib
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("lena_gray.bmp", cv2.IMREAD_GRAYSCALE)
w, h = img.shape[:2]

newImage = np.zeros([w, h])

# plt.imshow(newImage, cmap=plt.get_cmap("gray"))
# plt.show()

for i in range(w):
    for j in range(h):
        if img[i, j] > 127:
            newImage[i, j] = 127
        else:
            newImage[i, j] = 90

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(img, cmap=plt.get_cmap("gray"))
f.add_subplot(1, 2, 2)
plt.imshow(newImage, cmap=plt.get_cmap("gray"))
plt.show()
