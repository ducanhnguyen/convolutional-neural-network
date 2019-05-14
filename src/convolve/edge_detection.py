import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import color

import src.convolve.convolve as c

"""
Implementation of edge detection

Link: https://en.wikipedia.org/wiki/Sobel_operator
"""

img = imageio.imread("../../img/house.jpg")

img = color.rgb2gray(img)# convert image to black-white image
print(img.shape)

# Sobel operator
Lx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
Gx = c.convolve2d(img, Lx)

Ly = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
Gy = c.convolve2d(img, Ly)

delta_L = np.sqrt(Gx ** 2, Gy ** 2)

plt.imshow(delta_L, cmap='gray')
plt.show()
