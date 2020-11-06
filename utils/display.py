import numpy as np
import cv2


def saliency_vis(img: np.array, grad: np.array) -> np.array:
    """
    :param img: input image, values in range [0, 1]
    :param grad: gradient map, values in range [0, 1]
    :return: img * grad, values in range [0, 255], uint8
    """
    img = (255 * img).copy()
    output = (img * grad).astype(np.uint8)
    return output


def cam_vis(img: np.array, amap: np.array, weight: float = 0.3) -> np.array:
    """
    :param img: input image, values in range [0, 1]
    :param amap: activation map, values in range [0, 1]
    :param weight: weighting factor between the image an activation map
    :return: overlapping between input and activation, values in range [0, 255], uint8
    """
    # get colormap
    amap = (amap * 255).astype(np.uint8)
    amap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
    amap = amap[...,::-1]

    # combine colormap and image
    img  = (255 * img).copy()
    combined = weight * img + (1 - weight) * amap
    return combined.astype(np.uint8)
