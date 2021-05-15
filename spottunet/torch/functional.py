import torch
import torch.nn.functional as F
from torch.autograd import Variable

from dpipe.torch import get_device


def sample_gumbel(shape, eps=1e-20, device='cuda'):
    if device == torch.device('cpu'):
        u = torch.rand(shape, requires_grad=False, device=device)
    else:
        u = torch.cuda.FloatTensor(shape).uniform_()
    return -Variable(torch.log(-torch.log(u + eps) + eps))


def gumbel_softmax_sample(logits, temperature, use_gumbel=True):
    y = logits + sample_gumbel(logits.size(), device=get_device(logits)) if use_gumbel else logits
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, use_gumbel=True, temperature=5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, use_gumbel=use_gumbel)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
