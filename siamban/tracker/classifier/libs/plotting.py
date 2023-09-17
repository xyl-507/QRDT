import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_tensor(a: torch.Tensor, fig_num=None, title=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    plt.cla()
    plt.imshow(a_np)
    plt.axis('off')
    plt.axis('equal')
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.pause(0.001)


def plot_graph(a: torch.Tensor, fig_num=None, title=None):
    """Plot graph. Data is a 1D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim > 1:
        raise ValueError
    plt.figure(fig_num)
    # plt.tight_layout()
    plt.cla()
    plt.plot(a_np)
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.pause(0.001)


def show_tensor_np(a: torch.Tensor, fig_num=None, title=None, i=None):  # xyl 20220927
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    # a_np = a.squeeze().cpu().clone().detach().numpy()
    num = a.size
    w_h= int(np.sqrt(num))
    a_np = a.reshape(w_h, w_h)
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    plt.cla()
    plt.imshow(a_np)
    plt.axis('off')  # 关闭坐标轴
    plt.axis('equal')  # x，y刻度一样
    if title is not None:
        plt.title(title)
    plt.savefig('/home/xyl/xyl-code/siamban-DROL/demo/response/pscore_st/classification_{:06d}.png'.format(i), pad_inches=0, dpi=300)
    plt.draw()
    plt.pause(0.001)
