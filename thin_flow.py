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
    hsv[..., 0] = np.arctan2(U[0, 0], -U[1, 0]) / (2 * np.pi) + 0.5
    vel = (U[:2, :] ** 2).sum(axis=[0, 1]) ** 0.5
    hsv[..., 1] = vel / vel.max()
    hsv[..., 2] = 1.0
    rgb = hsv_to_rgb(hsv)
    return rgb


def conv(X, *K):
    return torch.conv2d(
        pad(X, (1, 1, 1, 1), mode='replicate'),
        torch.stack(K)[:, None],
        padding='valid'
    )


def solve_continuity(U, bc, threshold=0.0001, max_n=100, step_size=0.1):
    err = np.nan
    for i in range(max_n):
        # boundary conditions
        bc(U)

        # derivatives
        dU_dxy = conv(U, K_dx, K_dy)
        # [du0_dxx, du0_dxy, du1_dxx, du1_dxy]
        du_dxxy = conv(dU_dxy[[0, 2]].reshape((4, 1, n_height, n_width)), K_dx)
        # [dv0_dxy, dv0_dyy, dv1_dxy, dv1_dyy]
        dv_dxyy = conv(dU_dxy[[1, 3]].reshape((4, 1, n_height, n_width)), K_dy)

        # div0 = dU_dxy[[0, 1], [0, 1]].sum(dim=0)
        # div1 = dU_dxy[[2, 3], [0, 1]].sum(dim=0)
        #
        # delta_U = du_dxxy + dv_dxyy + torch.cat([dU_dxy[4, :, None], -dU_dxy[4, :, None]])
        # delta_w = -((dU_dxy[[0, 1], [0, 1]] - dU_dxy[[2, 3], [0, 1]]).sum(dim=0) + U[4, 0]) / 2

        # plt.figure()
        # plt.subplot(221)
        # plt.title('div0')
        # plt.imshow(div0)
        # plt.subplot(222)
        # plt.title('div1')
        # plt.imshow(div1)
        # plt.subplot(223)
        # plt.title('w')
        # plt.imshow(U[4, 0])
        # plt.subplot(224)
        # plt.title('delta w')
        # plt.imshow(delta_w)
        # plt.tight_layout()
        #
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.title('u0')
        # plt.imshow(U[0, 0])
        # plt.subplot(2, 2, 2)
        # plt.title('v0')
        # plt.imshow(U[1, 0])
        # plt.subplot(2, 2, 3)
        # plt.title('delta u0')
        # plt.imshow(delta_U[0, 0])
        # plt.subplot(2, 2, 4)
        # plt.title('delta v0')
        # plt.imshow(delta_U[1, 0])
        # plt.tight_layout()
        #
        # plt.figure()
        # plt.subplot(2, 2, 1)
        # plt.title('u1')
        # plt.imshow(U[2, 0])
        # plt.subplot(2, 2, 2)
        # plt.title('v1')
        # plt.imshow(U[3, 0])
        # plt.subplot(2, 2, 3)
        # plt.title('delta u1')
        # plt.imshow(delta_U[2, 0])
        # plt.subplot(2, 2, 4)
        # plt.title('delta v1')
        # plt.imshow(delta_U[3, 0])
        # plt.tight_layout()
        #
        # plt.show()

        # U[:4] += delta_U * 0.1
        # U[4] += delta_w * 0.1

        U[:4] += step_size * (du_dxxy + dv_dxyy + torch.cat([dU_dxy[4, :, None], -dU_dxy[4, :, None]]))
        U[4] += -step_size * ((dU_dxy[[0, 1], [0, 1]] - dU_dxy[[2, 3], [0, 1]]).sum(dim=0) + U[4, 0] / 2)

        # (du0_dx + dv0_dy + w)^2 + (du1_dx + dv1_dy - w)^2
        err = ((dU_dxy[[0, 2], 0] + dU_dxy[[1, 3], 1] + torch.cat([U[4], -U[4]])) ** 2).max()
        # print(err)

        # U[:] = conv(U, K_smooth)

        if err < threshold:
            break
    # print(err.item(), i)
    return i, err


def bc(U):
    U[:, :, [0, -1]] = 0
    U[:, :, :, [0, -1]] = 0
    U[:, :, 45:55, 45:55] = 0


n_height, n_width = 100, 100
U = torch.zeros((5, 1, n_height, n_width))
# U[4, :, 55:60, 55:60] = 0.5
U[[1, 3], :, 40:60, :20] = 0.5
# U[4, :, 10:20, 10:20] = 0.1
# U[0, :, 30:40, 10:20] = 0.1
# U[0, :, 40:50, 10:20] = -0.1
# U[1, :, 60:70, 10:20] = 0.1
# U[1, :, 60:70, 20:30] = -0.1
# U[2, :, 10:20, 50:60] = 0.1
# U[2, :, 20:30, 50:60] = -0.1
# U[3, :, 40:50, 50:60] = 0.1
# U[3, :, 40:50, 60:70] = -0.1
# U[4, :, 60:70, 60:70] = 0.1
# U[4, :, 80:90, 80:90] = -0.1
n_pts = 100
pts = torch.rand((3, n_pts))
pts[0] *= n_height - 1
pts[1] *= n_width - 1
ij = torch.zeros(pts.shape, dtype=int)
img = np.zeros((n_height, n_width, 3), np.uint8)

out = cv2.VideoWriter('thin_flow.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (U.shape[3], U.shape[2]))

for i in tqdm(range(1000)):

    # flow particles
    pt_vels = U[:2, 0, ij[0], ij[1]]
    pts[:2] += pt_vels
    pts[pts < 0] = 0
    pts[0, pts[0] > n_height - 1] = n_height - 1
    pts[1, pts[1] > n_width - 1] = n_width - 1

    # write to video
    ij[:] = torch.round(pts).to(int)
    if i % 10 == 0:
        img[:] = (flow_vis(U) * 255).astype(np.uint8)
        img[ij[0], ij[1]] = [0, 0, 0]
        out.write(img[..., [2, 1, 0]])

        # plt.figure()
        # plt.imshow(img)
        # plt.streamplot(np.arange(n_width), np.arange(n_height), U[1, 0], U[0, 0])
        # plt.streamplot(np.arange(n_width), np.arange(n_height), U[3, 0], U[2, 0])
        # plt.quiver(pts[1], pts[0], pt_vels[1], -pt_vels[0])
        # plt.show()

    # smooth
    # U[:] = conv(U, K_smooth)

    # clip velocity
    vel0 = (U[[0, 1], :] ** 2).sum(axis=[0, 1]) ** 0.5
    vel1 = (U[[2, 3], :] ** 2).sum(axis=[0, 1]) ** 0.5
    U[:2, :, vel0 > 1] /= vel0[vel0 > 1]
    U[2:4, :, vel1 > 1] /= vel1[vel1 > 1]

    # calc NS fields
    dU_dxy = conv(U, K_dx, K_dy)
    # delta_u = U[[0, 2]] * dU_dxy[[0, 2], 0][:, None] + U[[1, 3]] * dU_dxy[[0, 2], 1][:, None] + U[[4]] * (U[[2]] - U[[0]])
    # delta_v = U[[0, 2]] * dU_dxy[[1, 3], 0][:, None] + U[[1, 3]] * dU_dxy[[1, 3], 1][:, None] + U[[4]] * (U[[3]] - U[[1]])
    # plt.figure()
    # plt.subplot(221)
    # plt.title('u0')
    # plt.imshow(U[0, 0])
    # plt.subplot(222)
    # plt.title('v0')
    # plt.imshow(U[1, 0])
    # plt.subplot(223)
    # plt.title('delta u0')
    # plt.imshow(delta_u[0, 0])
    # plt.subplot(224)
    # plt.title('delta v0')
    # plt.imshow(delta_v[0, 0])`
    # plt.show()

    # du_dt = -u * du_dx - v * du_dy - w * (u1 - u0)
    U[[0, 2]] -= U[[0, 2]] * dU_dxy[[0, 2], 0][:, None] + U[[1, 3]] * dU_dxy[[0, 2], 1][:, None] + U[[4]] * (U[[2]] - U[[0]])
    # dv_dt = -u * dv_dx - v * dv_dy - w * (v1 - v0)
    U[[1, 3]] -= U[[0, 2]] * dU_dxy[[1, 3], 0][:, None] + U[[1, 3]] * dU_dxy[[1, 3], 1][:, None] + U[[4]] * (U[[3]] - U[[1]])
    # dw_dt = -u * dw_dx - v * dw_dy
    U[4, 0] -= 0.5 * U[[0, 2], 0].sum(dim=0) * dU_dxy[4, 0] + 0.5 * U[[1, 3], 0].sum(axis=0) * dU_dxy[4, 1]

    # solve continuity
    solve_continuity(U, bc, 1e-6, 100)

    # plt.figure()
    # # plt.imshow(U[1, 0])
    # plt.imshow(img)
    # plt.streamplot(np.arange(n_width), np.arange(n_height), U[1, 0], U[0, 0])
    # plt.streamplot(np.arange(n_width), np.arange(n_height), U[3, 0], U[2, 0])
    # plt.show()

out.release()
