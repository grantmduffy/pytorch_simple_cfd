import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import torch
from tqdm import tqdm
import cv2
from matplotlib.colors import ListedColormap


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
    rgb[land] *= 0.5
    return rgb


def temp_vis(T):
    cmap = plt.get_cmap('inferno')
    val = T[0, 0] / T.max()
    img = cmap(val)
    img[land] *= 0.5
    img = (img * 255).astype(np.uint8)
    return img[..., ::-1]


c = np.array([163, 183, 209, 256])[None, :] * np.ones((256, 1)) / 256
c[:, 3] = np.linspace(0, 1, 256)
rain_cmap = ListedColormap(c)


def humid_vis(W, precip, T):
    cmap = plt.get_cmap('Blues')
    cmap_precip = plt.get_cmap('gnuplot2')
    humidity = W[0, 0] / (T[0, 0] + 1e-6)
    img = cmap(1 - humidity)
    rain = np.cumsum(precip[0, 0], axis=0)
    # print(float(precip.max()), float(rain.max()), float(W.max()), float(T.max()))
    rain /= 1e-5
    val = precip[0, 0] / precip.max()
    m_precip = val > 0.05
    img_precip = cmap_precip(val ** 0.5)
    # img[m_precip] = img_precip[m_precip]
    img_rain = rain_cmap(rain)
    img[..., :3] = img[..., :3] * (1 - img_rain[..., [-1]]) + img_rain[..., :3] * img_rain[..., [-1]]
    img[land, :] = 0.0
    return (img * 255).astype(np.uint8)[..., ::-1]


def random_land(n_width, n_height):
    h = np.cumsum(np.random.randint(-2, 3, 500))[None, :]
    y = np.arange(100)[::-1, None]
    M = y <= h
    return M


# Setup arrays and outputs
n_width, n_height = 500, 100
K_evap = 0.001
H_vap = 0.10
roof = torch.linspace(0, 1, 70) ** 0.05
U = torch.zeros((1, 2, n_height, n_width))
T = torch.zeros((1, 1, n_height, n_width))
T[0, 0] = torch.from_numpy(np.random.normal(0, 0.01, (n_height, n_width)) ** 2)
W = torch.zeros((1, 1, n_height, n_width))
land = random_land(n_width, n_height)

plt.figure()
plt.imshow(land)
plt.show()

out_temp = cv2.VideoWriter('flow_temp.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (U.shape[3], U.shape[2]))
out_vel = cv2.VideoWriter('flow_vel.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (U.shape[3], U.shape[2]))
out_vap = cv2.VideoWriter('flow_vap.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (U.shape[3], U.shape[2]))

cont_hist = []

for i in tqdm(range(3000)):

    # Set boundary conditions
    U[:, :, [0, -1]] = 0
    U[:, :, 0, [0, -1]] = 0
    U[:, :, 1, [0, -1]] = 1.0
    U[:, :, land] = 0
    T[0, 0, :len(roof)] *= roof[:, None]

    # Surface heating
    i_surface = np.argmax(land, axis=0)
    i_surface[i_surface == 0] = n_height - 1
    T[0, 0, i_surface, range(n_width)] += 0.0005

    # Surface Evaporation
    evap = K_evap * (T[0, 0, -1] - W[0, 0, -1])
    W[0, 0, -1] += evap
    T[0, 0, -1] -= H_vap * evap

    # Precipitation
    precip = np.maximum(W - T, 0)
    W -= precip
    T += H_vap * precip

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
        out_vel.write((flow_vis(U) * 255).astype(np.uint8))
        out_temp.write(temp_vis(T))
        out_vap.write(humid_vis(W, precip, T))

        # plt.figure()
        # plt.title('Water')
        # plt.imshow(W[0, 0])
        # plt.figure()
        # plt.title('Temp')
        # plt.imshow(T[0, 0])
        # plt.show()

out_temp.release()
out_vel.release()
out_vap.release()

# Plot continuity convergence
cont_hist = np.array(cont_hist)
plt.figure()
plt.subplot(211)
plt.plot(cont_hist[:, 0])
plt.subplot(212)
plt.plot(cont_hist[:, 1])
plt.show()


