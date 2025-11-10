import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

x = torch.randint(high=2, size=(16, 16, 16)).unsqueeze(0).unsqueeze(0).float().cuda()
mask = F.interpolate(x, scale_factor=4, mode='nearest')
