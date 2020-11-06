import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, backward


class Saliency(object):
    def __init__(self, model: nn.Module, conv1: nn.Module, fc: nn.Module):
        self.model = model
        self.conv1 = conv1
        self.fc = fc

        # hook backward function to the first layer
        self.grad_first_layer = None
        self.conv1.register_backward_hook(self._hook_backward)

        # hook forward function to the last layer
        self.output_last_layer = None
        self.fc.register_forward_hook(self._hook_forward)

        # store image
        self.image = None

    def _hook_backward(self, module, grad_in, grad_out):
        self.grad_first_layer = grad_in[0].clone()

    def _hook_forward(self, module, input, output):
        self.output_last_layer = output[0].clone()

    def __call__(self, img: torch.tensor) -> int:
        # set model to evaluation
        self.model.zero_grad()
        self.model.eval()

        # save input image
        self.image = img

        # transform input image to variable that requires
        # gradient computation
        img = Variable(img.unsqueeze(0), requires_grad=True)

        # pass image through the model
        output = self.model(img)

        # select the highest activation
        idx = output.argmax().item()

        # define mask
        mask = torch.zeros_like(output)
        mask[0, idx] = 1

        # compute gradients
        output.backward(gradient=mask)

        # return the index of the most probable class
        return idx

    def get_soft_grad(self) -> np.array:
        grad = self.grad_first_layer.clone()
        grad = grad.squeeze(0)

        grad = torch.abs(grad)
        grad = torch.max(grad, dim=0).values
        grad = grad.cpu().numpy()
        quantile = np.quantile(grad, 0.90)

        grad *= (grad >= quantile).astype(np.float)
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        grad = np.expand_dims(grad, axis=2)
        return grad

    def get_hard_grad(self):
        grad = self.get_soft_grad()
        grad[grad > 0] = 1
        return grad