from __future__ import division
import numpy as np
from PIL import Image


def get_scaled_noisy_image(image):

    return (image - np.min(image)) * 255.0 / (np.max(image) - np.min(image))

    # image_grid = np.zeros(shape=image.flatten().shape)
    #
    # for vertex_index, vertex in enumerate(image.flatten()):
    #     if vertex > 0:
    #         image_grid[vertex_index] = 255
    # return np.reshape(image_grid, image.shape)


def get_image_from_sample(sampled_grid):
    image_grid = np.zeros(shape=sampled_grid.flatten().shape)

    for vertex_index, vertex in enumerate(sampled_grid.flatten()):
        if vertex == 1:
            image_grid[vertex_index] = 1

    return np.reshape(image_grid, sampled_grid.shape)


def save_grid_image_to_location(image_grid, location, image_mode="1"):
    img = Image.new(image_mode, image_grid.T.shape)
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = int(image_grid.T[i][j])

    # img.show()
    img.save(location)


def get_scaled_image(image):

    scaled_image = np.zeros(shape=image.shape)
    for row_pixel_index in xrange(len(image)):
        for column_pixel_index, pixel in enumerate(image[row_pixel_index]):
            if pixel == 255:
                scaled_image[row_pixel_index][column_pixel_index] = 1
            elif pixel == 0:
                scaled_image[row_pixel_index][column_pixel_index] = -1

    return scaled_image


def get_image_from_location(location):
    im = np.asarray(Image.open(location).convert("L"))

    img = Image.new('L', im.T.shape)
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = im.T[i][j]

    return np.asarray(img)


def get_bias_params_from_image(image, bias_term):

    bias_param = np.zeros_like(image)
    for pixel_index, pixel in enumerate(image):
        bias_param[pixel_index] = bias_term * pixel

    return bias_param


def main():
    print get_image_from_location("../data/denoising_images/im_noisy.png").shape


if __name__ == '__main__':
    main()
