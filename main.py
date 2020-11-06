import torch
import torchvision

import numpy as np
import cv2
import argparse
import ast
import algos
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchvision import transforms
from PIL import Image


# define parser
parser = argparse.ArgumentParser()
parser.add_argument("--classes", type=str, default="data/classes.txt")
parser.add_argument("--img", type=str, default="data/hen.jpg")
parser.add_argument("--algo", type=str, default="saliency")
parser.add_argument("--model", type=str, default="ResNet18")
args = parser.parse_args()

# define input image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# define unnormalize transformation
unnormalize = transforms.Compose([
    transforms.Normalize(
        mean=[0., 0., 0.],
        std=[1./0.229, 1./0.224, 1./0.225]
    ),
    transforms.Normalize(
        mean=[-0.485, -0.456, -0.406],
        std=[1., 1., 1.]
    )
])

if __name__ == "__main__":
    # read imagenet classes file
    with open(args.classes, "rt") as fin:
        classes = ast.literal_eval(fin.read())

    # read image
    img = Image.open(args.img)
    tensor_img = transform(img)

    # define model
    model = None
    if args.model == "ResNet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()

    # define visualization algorithms
    saliency = algos.Saliency(model=model, conv1=model.conv1, fc=model.fc)
    cam = algos.CAM(model=model, avgpool=model.avgpool, fc=model.fc)

    # run algorithms
    idx = saliency(tensor_img)
    idx = cam(tensor_img)
    print(f"Predicted class is {classes[idx]}")

    # get outputs of the algorithms
    grad = saliency.get_hard_grad()
    amap = cam.get_activation_map()

    # get visualization
    tensor_img = unnormalize(tensor_img)
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)

    vis_saliency = utils.saliency_vis(img, grad)
    vis_cam = utils.cam_vis(img, amap)
    vis_img = (255 * img).astype(np.uint8)

    # display results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(vis_img)
    ax[0].set_title("Original image")

    ax[1].imshow(vis_saliency)
    ax[1].set_title("Saliency map (hard)")
    ax[1].set_xticks([])

    ax[2].imshow(vis_cam)
    ax[2].set_title("CAM")
    ax[2].set_xticks([])

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f"Predicted class: {classes[idx]}")
    plt.show()
