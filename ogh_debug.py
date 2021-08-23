# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from utils import training, data
from utils.one_hot_graph import OneHotConv, OneHotGraph
import torch


# %%
input_channels = 5
x_size = 2


# %%
ohg_conv = OneHotConv(input_channels, 5, 1)


# %%
x = torch.randn(x_size, input_channels)
onehots = [torch.eye(x_size)]
index = torch.randint(0,2, (x_size, x_size))
index = index + index.T
index = index.to_sparse()._indices()
adjs = [index]

# %%
print(f"OneHots:\n{onehots}")
print(f"x:\n{x}")

# %%
x, onehots = ohg_conv.forward(x, onehots, index, torch.ones(len(x)), torch.tensor([x_size]), adjs)
print(f"OneHots:\n{onehots}")
print(f"x:\n{x}")

# %%




# %%
