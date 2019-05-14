import imageio
from skimage import color

import src.convolve.convolve as c
import src.utils

"""
Implementation of Gaussian blur.

Link: https://en.wikipedia.org/wiki/Gaussian_blur
"""

img = imageio.imread('../../img/lena.png')
img2d = color.rgb2gray(img)

img2d = img2d[1:100, :]  # reduce the size of the original image due to the poor performance

filter = c.generate_gaussian_matrix_2d_filter(std=0.1)
img2 = c.convolve2d(img2d, filter)

filter = c.generate_gaussian_matrix_2d_filter(std=1)
img3 = c.convolve2d(img2d, filter)

src.utils.show_images(images=[img2, img3], cols=2, titles=['std = 0.1', 'std = 1'])
