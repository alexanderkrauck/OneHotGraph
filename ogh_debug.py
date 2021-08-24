# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from utils import training, data
from utils.one_hot_graph import AttentionOneHotConv, OneHotGraph
import torch
from utils.baselines import IsomorphismOneHotGraph_Baseline
import utils.training


# %%
input_channels = 5
x_size = 2


# %%
data_module = data.DataModule("tox21_original", split_mode = "predefined", workers = 2)
one_hot_channels=8
ohg_conv = IsomorphismOneHotGraph_Baseline(data_module, 128, 2, 0.2, 2, 0.2, one_hot_channels=one_hot_channels).to("cuda")


# %%
a = []
for epoch in range(20):
    for n, p in ohg_conv.named_parameters():
        if n == "ohg.convs.0.mlp.0.weight":
            a.append(p)
    opt = torch.optim.Adam(ohg_conv.parameters(), lr=1e-3, weight_decay=1e-8)
    utils.training.train(ohg_conv, opt, data_module.make_train_loader(), data_module.num_classes, epoch, device="cuda", use_tqdm=True)

# %%
print(a[0][:, : -one_hot_channels].detach().abs().mean())
print(a[0][:, -one_hot_channels:].detach().abs().mean())

# %%
x, onehots = ohg_conv.forward(x, onehots, index, torch.ones(len(x)), torch.tensor([x_size]), adjs)
print(f"OneHots:\n{onehots}")
print(f"x:\n{x}")

# %%




# %%
