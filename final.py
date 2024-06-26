import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.PGM'):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(np.array(img))
    return np.array(images)


def segment_image(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    blurred_image_uint8 = cv2.convertScaleAbs(blurred_image)

    _, segmented_image = cv2.threshold(blurred_image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return segmented_image


def calculate_phase_amplitude(images):
    num_images = len(images)
    height, width = images[0].shape
    phase_images = np.zeros((height, width), dtype=np.float32)
    amplitude_images = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            c_accumulator = 0
            s_accumulator = 0

            for index, img in enumerate(images, start=1):
                nu = (2 * index - 1) * np.pi / num_images
                intensity = img[y, x]
                c_accumulator += intensity * np.cos(nu)
                s_accumulator += intensity * np.sin(nu)

            phase = (np.pi / 2) + np.arctan2(s_accumulator, c_accumulator)
            amplitude = (2 / num_images) * np.sqrt(c_accumulator ** 2 + s_accumulator ** 2)

            phase_images[y, x] = phase
            amplitude_images[y, x] = amplitude

    segmented_amplitude = segment_image(amplitude_images)
    _, mask = cv2.threshold(segmented_amplitude, 1, 255, cv2.THRESH_BINARY)

    phase_copy = np.copy(phase_images)
    phase_copy[mask == 0] = 0

    phase_copy = phase_copy.astype(np.float32)
    segmented_amplitude = segmented_amplitude.astype(np.float32)

    phase_overlay = cv2.addWeighted(phase_copy, 1.0, segmented_amplitude, 0.0, 0)

    return phase_overlay, amplitude_images


def visualize_amplitude_and_phase(phase_image, amplitude_image):
    # Visualize Amplitude Image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(amplitude_image, cmap='jet')
    plt.title('Amplitude Image')
    plt.colorbar()

    # Visualize Phase Image
    plt.subplot(1, 2, 2)
    plt.imshow(phase_image, cmap='jet', vmin=0, vmax=2 * np.pi)
    plt.title('Phase Image')
    plt.colorbar(label='Phase (radians)')

    plt.show()


def main():
    folder = "gate_sziv"
    gated_images = load_images_from_folder(folder)
    phase_images, amplitude_images = calculate_phase_amplitude(gated_images)
    visualize_amplitude_and_phase(phase_images, amplitude_images)


if __name__ == "__main__":
    main()
