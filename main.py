from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
from skimage.transform import warp

def generate_synthetic_data():
    image = shepp_logan_phantom()
    image = np.stack([image for _ in range(2)], axis=0)
    image[1] = warp(image[1], lambda coords: (coords[0] + 5*np.sin(coords[1]/10), coords[1]))
    image[1] = random_noise(image[1])
    return image

def visualize(image):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image[0], cmap='gray')
    axs[1].imshow(image[1], cmap='gray')
    plt.show()

if __name__ == '__main__':
    images = generate_synthetic_data()
    visualize(images)
