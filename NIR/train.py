import os
import numpy as np
from itertools import chain
import torch
from torch.utils.data import DataLoader
import imageio

from model import Siren
from util import get_mgrid, jacobian, VideoFitting


def train(path, total_steps, lambda_interf=0.01, lambda_flow=0.02, verbose=True, steps_til_summary=100):
    g = Siren(in_features=3, out_features=3, hidden_features=256,
              hidden_layers=5, outermost_linear=True).cuda()
    f1 = Siren(in_features=3, out_features=3, hidden_features=256,
               hidden_layers=5, outermost_linear=True).cuda()
    f2 = Siren(in_features=3, out_features=1, hidden_features=256,
               hidden_layers=5, outermost_linear=True).cuda()

    optim = torch.optim.Adam(lr=1e-4, params=chain(g.parameters(), f1.parameters(), f2.parameters()))

    v = VideoFitting(path)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) // 4
    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))

        xyt = model_input[start:end].requires_grad_()
        h = g(xyt)
        xy_, w = xyt[:, :-1] + h[:, :-1], h[:, [-1]]
        o_scene = torch.sigmoid(f1(torch.cat((xy_, w), -1)))     #代表场景
        o_rain = torch.sigmoid(f2(xyt))                                 #代表雨
        o = (1 - o_rain) * o_scene + o_rain
        loss_recon = (o - ground_truth[start:end]).abs().mean()
        loss_interf = o_rain.abs().mean()
        loss_flow = jacobian(h, xyt).abs().mean()
        loss = loss_recon + lambda_interf * loss_interf + lambda_flow * loss_flow

        if not step % steps_til_summary and verbose:
            print(f"Step [{step:04d}/{total_steps}]: recon={loss_recon:.8f}, interf={loss_interf:.4f}, flow={loss_flow:.4f}")

        optim.zero_grad()
        loss.backward()
        optim.step()

    return g, f1, f2, v.video


def save_outputs(g, f1, f2, orig, out_dir='./data'):
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        N, _, H, W = orig.size()
        xyt = get_mgrid([H, W, N]).cuda()

        h = g(xyt)
        xy_, w = xyt[:, :-1] + h[:, :-1], h[:, [-1]]
        o_scene = torch.sigmoid(f1(torch.cat((xy_, w), -1)))
        o_rain = torch.sigmoid(f2(xyt))

        o_scene = o_scene.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().numpy()
        o_rain = o_rain.view(H, W, N).permute(2, 0, 1).cpu().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        o_rain = (o_rain * 255).astype(np.uint8)
        orig = orig.permute(0, 2, 3, 1).cpu().numpy()
        orig = (orig * 255).astype(np.uint8)

        orig = [frame for frame in orig]
        o_scene = [frame for frame in o_scene]
        o_rain = [frame for frame in o_rain]

        fn_orig = os.path.join(out_dir, 'rain_orig.mp4')
        fn_scene = os.path.join(out_dir, 'rain_scene.mp4')
        fn_rain = os.path.join(out_dir, 'rain_interf.mp4')

        imageio.mimwrite(fn_orig, orig, fps=1, format='FFMPEG')
        imageio.mimwrite(fn_scene, o_scene, fps=1, format='FFMPEG')
        imageio.mimwrite(fn_rain, o_rain, fps=1, format='FFMPEG')

        print(f"Saved videos to: \n- {fn_orig}\n- {fn_scene}\n- {fn_rain}")


if __name__ == '__main__':
    g, f1, f2, orig = train('./data/rain', total_steps=5000)
    save_outputs(g, f1, f2, orig)
