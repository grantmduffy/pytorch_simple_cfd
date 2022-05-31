import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import torch
from tqdm import tqdm
import cv2


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


def solve_continuity(U, threshold=0.0001, max_n=100):
    err = None
    for i in range(max_n):
        du_ddx = torch.conv2d(U[:, [0]], torch.broadcast_to(K_ddx, (1, 1, 3, 3)), padding='same')
        dv_dxy = torch.conv2d(U[:, [1]], torch.broadcast_to(K_dxy, (1, 1, 3, 3)), padding='same')
        du_dxy = torch.conv2d(U[:, [0]], torch.broadcast_to(K_dxy, (1, 1, 3, 3)), padding='same')
        dv_ddy = torch.conv2d(U[:, [1]], torch.broadcast_to(K_ddy, (1, 1, 3, 3)), padding='same')
        du_dc = torch.conv2d(du_ddx + dv_dxy, torch.broadcast_to(K_smooth, (1, 1, 3, 3)), padding='same')
        dv_dc = torch.conv2d(dv_ddy + du_dxy, torch.broadcast_to(K_smooth, (1, 1, 3, 3)), padding='same')
        # du_dc = du_ddx + dv_dxy
        # dv_dc = dv_ddy + du_dxy
        U[:, [0]] += du_dc
        U[:, [1]] += dv_dc
        err = (du_dc ** 2 + dv_dc ** 2).sum()

        if err < threshold:
            break
    return i + 1, err


def flow_vis(U):
    hsv = np.zeros((*U.shape[2:4], 3))
    hsv[..., 0] = np.arctan2(U[0, 0], -U[0, 1]) / (2 * np.pi) + 0.5
    vel = (U ** 2).sum(axis=[0, 1]) ** 0.5
    hsv[..., 1] = vel / vel.max()
    hsv[..., 2] = 1.0
    rgb = hsv_to_rgb(hsv)
    return rgb


def temp_vis(T):
    cmap = plt.get_cmap('inferno')
    val = T[0, 0] / T.max()
    img = (cmap(val) * 255).astype(np.uint8)
    return img[..., ::-1]


def humid_vis(W, precip):
    cmap = plt.get_cmap('Blues')
    cmap_precip = plt.get_cmap('gnuplot2')
    val = W[0, 0] / W.max()
    img = (cmap(val ** 0.2) * 255).astype(np.uint8)
    val = precip[0, 0] / precip.max()
    m_precip = val > 0.05
    img_precip = (cmap_precip(val ** 0.5) * 255).astype(np.uint8)
    img[m_precip] = img_precip[m_precip]
    return img[..., ::-1]


# Setup arrays and outputs
n_width, n_height = 500, 100
K_evap = 0.01
H_vap = 0.1
roof = torch.linspace(0, 1, n_height) ** 0.02
U = torch.zeros((1, 2, n_height, n_width))
T = torch.zeros((1, 1, n_height, n_width))
T[0, 0] = torch.from_numpy(np.random.normal(0, 0.03, (n_height, n_width)) ** 2)
W = torch.zeros((1, 1, n_height, n_width))

out = cv2.VideoWriter('humidity.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (U.shape[3], U.shape[2]))
cont_hist = []

for i in tqdm(range(3000)):

    # Set boundary conditions
    U[:, :, [0, -1]] = 0
    U[:, :, 0, [0, -1]] = 0
    U[:, :, 1, [0, -1]] = 0
    T[0, 0, :len(roof)] *= roof[:, None]
    T[0, 0, -1] += 0.005

    # Surface Evaporation
    W[0, 0, -1] += K_evap * (T[0, 0, -1] - W[0, 0, -1])
    T[0, 0, -1] -= H_vap * K_evap * (T[0, 0, -1] - W[0, 0, -1])

    # Precipitation
    precip = np.maximum(W - T, 0)
    W -= precip

    # Navier Stokes for incompressible inviscid flow
    du_dx = torch.conv2d(U[:, [0]], torch.broadcast_to(K_dx, (1, 1, 3, 3)), padding='same')
    du_dy = torch.conv2d(U[:, [0]], torch.broadcast_to(K_dy, (1, 1, 3, 3)), padding='same')
    dv_dx = torch.conv2d(U[:, [1]], torch.broadcast_to(K_dx, (1, 1, 3, 3)), padding='same')
    dv_dy = torch.conv2d(U[:, [1]], torch.broadcast_to(K_dy, (1, 1, 3, 3)), padding='same')
    du_dt = -U[:, [0]] * du_dx - U[:, [1]] * du_dy
    dv_dt = -U[:, [0]] * dv_dx - U[:, [1]] * dv_dy
    du_dt -= T - T.mean()  # Temp dependent body force

    # Convection of heat
    dT_dx = torch.conv2d(T, torch.broadcast_to(K_dx, (1, 1, 3, 3)), padding='same')
    dT_dy = torch.conv2d(T, torch.broadcast_to(K_dy, (1, 1, 3, 3)), padding='same')
    dT_dt = -U[:, [0]] * dT_dx - U[:, [1]] * dT_dy

    # Convection of moisture
    dW_dx = torch.conv2d(W, torch.broadcast_to(K_dx, (1, 1, 3, 3)), padding='same')
    dW_dy = torch.conv2d(W, torch.broadcast_to(K_dy, (1, 1, 3, 3)), padding='same')
    dW_dt = -U[:, [0]] * dW_dx - U[:, [1]] * dW_dy

    # Update fields
    U[:, [0]] += du_dt
    U[:, [1]] += dv_dt
    T += dT_dt
    W += dW_dt

    # Smooth
    U[:, [0]] = torch.conv2d(U[:, [0]], torch.broadcast_to(K_smooth, (1, 1, 3, 3)), padding='same')
    U[:, [1]] = torch.conv2d(U[:, [1]], torch.broadcast_to(K_smooth, (1, 1, 3, 3)), padding='same')
    T = torch.conv2d(T, torch.broadcast_to(K_smooth, (1, 1, 3, 3)), padding='same')
    W = torch.conv2d(W, torch.broadcast_to(K_smooth, (1, 1, 3, 3)), padding='same')

    # Solve continuity
    cont_hist.append(solve_continuity(U, 0.01, 100))

    # Write to video
    if i % 5 == 0:
        # img = (flow_vis(U) * 255).astype(np.uint8)
        # img = temp_vis(T)
        img = humid_vis(W, precip)
        out.write(img)

        # plt.figure()
        # plt.title('Water')
        # plt.imshow(W[0, 0])
        # plt.figure()
        # plt.title('Temp')
        # plt.imshow(T[0, 0])
        # plt.show()

out.release()

# Plot continuity convergence
cont_hist = np.array(cont_hist)
plt.figure()
plt.subplot(211)
plt.plot(cont_hist[:, 0])
plt.subplot(212)
plt.plot(cont_hist[:, 1])
plt.show()


