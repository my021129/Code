import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from model import Siren, Homography  # 你自定义的模型模块
from util import get_mgrid, apply_homography, VideoFitting  # 自定义工具函数


# ========================
# 🧠 单应性 + SIREN 网络训练函数
# ========================
def train_homography(path, total_steps=3000, verbose=True, steps_til_summary=100):
    transform = Compose([
        Resize(512),
        ToTensor(),
        Normalize(torch.tensor([0.5] * 3), torch.tensor([0.5] * 3))
    ])

    dataset = VideoFitting(path, transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)

    g = Homography(hidden_features=256, hidden_layers=2).cuda()
    f = Siren(in_features=2, out_features=3, hidden_features=256,
              hidden_layers=4, outermost_linear=True).cuda()

    optim = torch.optim.Adam(chain(g.parameters(), f.parameters()), lr=1e-4)

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (dataset.H * dataset.W) // 4
    loss_history = []  # 新增：记录每步损失

    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))

        xy = model_input[start:end, :-1]
        t = model_input[start:end, -1:]

        h = g(t)
        xy_warped = apply_homography(xy, h)
        pred = f(xy_warped)

        loss = ((pred - ground_truth[start:end]) ** 2).mean()
        loss_history.append(loss.item())  # 保存损失

        if verbose and step % steps_til_summary == 0:
            print(f"Step [{step:04d}/{total_steps}]: loss={loss.item():.4f}")

        optim.zero_grad()
        loss.backward()
        optim.step()

    return f, loss_history  # 返回损失记录



# 🚀 训练模型
f, loss_history = train_homography('./data/vis/homography', total_steps=3000)

# 📉 绘制训练损失曲线
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label='Training Loss', color='blue')
plt.xlabel('Training Step')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 🎨 可视化重建图像
with torch.no_grad():
    xy = get_mgrid([512, 1024], [-1.5, -2.0], [1.5, 2.0]).cuda()
    output = f(xy).view(512, 1024, 3).cpu().numpy()
    output = np.clip(output, -1, 1) * 0.5 + 0.5

    plt.figure(figsize=(18, 18))
    plt.imshow(output)
    plt.axis('off')
    plt.show()

