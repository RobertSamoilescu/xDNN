import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CAM(object):
    def __init__(self, model: nn.Module, avgpool: nn.Module, fc: nn.Module):
        self.model = model
        self.avgpool = avgpool
        self.fc = fc

        # set hook functions
        self.input_avgpool = None
        self.avgpool.register_forward_hook(self._hook_forward)

        # variable to save the input image
        self.image = None
        self.idx = None

    def _hook_forward(self, module, input, output):
        self.input_avgpool = input[0].clone()

    def __call__(self, img: torch.tensor) -> int:
        # set model to evaluation
        self.model.zero_grad()
        self.model.eval()

        # make a copy of the input image
        self.image = img.clone()
        img = self.image.unsqueeze(0)

        # pass input through the network
        output = self.model(img)

        # get the most probable class
        self.idx = torch.argmax(output).item()
        return self.idx

    def get_activation_map(self):
        # get the weights of the fc layer
        # corresponding to the most probable class
        weights = self.fc.weight.data[self.idx]

        # reshape the weights & the activation map
        weights = weights.reshape(1, -1, 1, 1)
        self.input_avgpool = self.input_avgpool

        # multiply the activation map with the weights
        activation_map = self.input_avgpool * weights
        activation_map = activation_map.sum(dim=[1], keepdim=True)

        # reshape the activation map to the input shape
        c, h, w = self.image.shape
        activation_map = F.interpolate(activation_map, mode='bilinear', size=(h, w), align_corners=True)
        activation_map = activation_map.squeeze(0)

        # normalize
        min_val, max_val = activation_map.min(), activation_map.max()
        activation_map = (activation_map - min_val)/(max_val - min_val)

        # transform to numpy
        activation_map = activation_map.detach().cpu().numpy()
        activation_map = activation_map.transpose((1, 2, 0))
        return activation_map

