import torch

# a tutorial dedicated to custom layers:
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None