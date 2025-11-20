import torch.nn as nn

###############################################################################
# Activation Functions
###############################################################################


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()


class Glu(nn.Module):
    def __init__(self, dim):
        super(Glu, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()
