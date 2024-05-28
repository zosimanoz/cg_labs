"""

Histogram Equalizer in Python

"""

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("beans.png", cv2.IMREAD_GRAYSCALE)
w, h = img.shape[:2]
newImage = np.zeros([w, h])


def freq(lst):
    d = {}
    for i in lst:
        for j in i:
            if d.get(j):
                d[j] += 1
            else:
                d[j] = 1
    return d


def probability(d, total_pixels):
    l = {}
    for i in range(256):
        value = d.get(i)
        if value != None:
            l[i] = value / total_pixels
    return l


def equalizer(d, l):
    f_dic = {}
    last_sum = 0
    for i in range(l):
        if d.get(i):
            prob = d.get(i)
            last_sum = last_sum + ((l - 1) * prob)
            f_dic[i] = math.floor(last_sum)
    return f_dic


def replace_values(f_dic, img):
    print(f_dic)
    for i in range(len(img)):
        for j in range(len(img)):
            vv = f_dic.get(img[i][j])
            if vv != None:
                newImage[i][j] = vv
    return newImage


def histogramEqualization(img):
    freqq = freq(img)
    area = img.shape
    total_pixels = area[0] * area[1]
    prob = probability(freqq, total_pixels)
    f_dic = equalizer(prob, 256)
    new_img = replace_values(f_dic, img)
    return new_img


new_img = histogramEqualization(img)

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(img, cmap=plt.get_cmap("gray"))
f.add_subplot(1, 2, 2)
plt.imshow(new_img, cmap=plt.get_cmap("gray"))
plt.show()
