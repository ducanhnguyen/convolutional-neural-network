import imageio
import matplotlib.pyplot as plt
import numpy as np

import src.convolve.convolve as c

"""
Implementation of Gaussian blur.

Link: https://en.wikipedia.org/wiki/Gaussian_blur
"""

img3d = img = imageio.imread("../../img/lena.png")

img3d = img3d[1:30, :, :]  # resize the original image due to the poor performance

filter = c.generate_gaussian_matrix_2d_filter()

y = np.zeros(shape=img3d.shape)

y[:, :, 0] = c.convolve2d(img3d[:, :, 0], filter)  # red
y[:, :, 1] = c.convolve2d(img3d[:, :, 1], filter)  # green
y[:, :, 2] = c.convolve2d(img3d[:, :, 2], filter)  # blue

y = y.astype('int')

plt.imshow(y)
plt.show()
