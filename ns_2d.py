import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import torch
from tqdm import tqdm
import cv2
from torch.nn.functional import pad

K_dx = torch.Tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
]) / 8
K_dy = torch.Tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]) / 8
K_smooth = torch.Tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16
K_dxy = torch.Tensor([
    [1, 0, -1],
    [0, 0, 0],
    [-1, 0, 1]
]) / 4
K_ddx = torch.Tensor([
    [1, 2, 1],
    [-2, -4, -2],
    [1, 2, 1]
]) / 4
K_ddy = torch.Tensor([
    [1, -2, 1],
    [2, -4, 2],
    [1, -2, 1]
]) / 4


def flow_vis(U):
    hsv = np.zeros((*U.shape[2:4], 3))
    hsv[..., 0] = np.arctan2(U[0, 0], -U[0, 1]) / (2 * np.pi) + 0.5
    vel = (U ** 2).sum(axis=[0, 1]) ** 0.5
    hsv[..., 1] = vel / vel.max()
    hsv[..., 2] = 1.0
    rgb = hsv_to_rgb(hsv)
    return rgb


def conv(X, K):
    return torch.conv2d(
        pad(X, (1, 1, 1, 1), mode='replicate'),
        torch.broadcast_to(K, (1, 1, 3, 3)),
        padding='valid'
    )


def solve_continuity(U, threshold=0.0001, max_n=100):
    err = np.nan
    for i in range(max_n):
        du_ddx = conv(U[:, [0]], K_ddx)
        dv_dxy = conv(U[:, [1]], K_dxy)
        du_dxy = conv(U[:, [0]], K_dxy)
        dv_ddy = conv(U[:, [1]], K_ddy)
        du_dc = conv(du_ddx + dv_dxy, K_smooth)
        dv_dc = conv(dv_ddy + du_dxy, K_smooth)
        U[:, [0]] += du_dc
        U[:, [1]] += dv_dc
        err = (du_dc ** 2 + dv_dc ** 2).sum()
        if err < threshold:
            break
    return i, err


n_height, n_width = 100, 100
U = torch.zeros((1, 2, n_height, n_width))
U[:, 1, 25:75, 25:75] = 0.2
n_pts = 30
pts = torch.rand((2, n_pts))
pts[0] *= n_height - 1
pts[1] *= n_width - 1
ij = torch.zeros(pts.shape, dtype=int)

out = cv2.VideoWriter('ns_2d.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (U.shape[3], U.shape[2]))

for i in tqdm(range(1000)):

    # boundary conditions
    U[:, :, [0, -1]] = 0
    U[:, :, :, [0, -1]] = 0
    U[:, :, 45:55, 45:55] = 0
    U[:, 0, 25, 40:55] = 0.2
    U[:, 0, 75, 45:60] = -0.2

    # write to video
    ij[:] = torch.round(pts).to(int)
    if i % 5 == 0:
        img = (flow_vis(U) * 255).astype(np.uint8)
        img[ij[0], ij[1]] = [0, 0, 0]
        out.write(img[..., [2, 1, 0]])

    # smooth
    U[:, [0]] = conv(U[:, [0]], K_smooth)
    U[:, [1]] = conv(U[:, [1]], K_smooth)

    # clip velocity
    vel = (U ** 2).sum(axis=[0, 1]) ** 0.5
    U[:, :, vel > 1] /= vel[vel > 1]

    # calc NS fields
    du_dx = conv(U[:, [0]], K_dx)
    du_dy = conv(U[:, [0]], K_dy)
    dv_dx = conv(U[:, [1]], K_dx)
    dv_dy = conv(U[:, [1]], K_dy)
    du_dt = -U[:, [0]] * du_dx - U[:, [1]] * du_dy
    dv_dt = -U[:, [0]] * dv_dx - U[:, [1]] * dv_dy

    # update velocity
    U[:, [0]] += du_dt
    U[:, [1]] += dv_dt

    # solve continuity
    solve_continuity(U, 0.001, 100)

    # flow particles
    pts += U[0, :, ij[0], ij[1]]
    pts[pts < 0] = 0
    pts[0, pts[0] > n_height - 1] = n_height - 1
    pts[1, pts[1] > n_width - 1] = n_width - 1

out.release()
