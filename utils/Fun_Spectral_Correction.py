import numpy as np
import torch


def Fun_Spectral_Correction(T, p, q, h, index, sort_order, m, n, device):
    ga = []
    t = 1
    while t <= T:
        if t <= p * T:
            a = 0
            ga.append(a)
        elif p * T < t <= T:
            t_q = (T - p * T) * q + p * T
            a = (-h / (p * T - t_q) ** 2) * (t - t_q) ** 2 + h
            if a < 0:
                a = 0
            ga.append(a)
        t += 1

    fa = 1 + torch.tensor(ga, device=device)
    Fa = fa[sort_order]  # 排序
    Fa = Fa[index]  # 索引
    Fa = Fa.reshape(m, n)  # 重塑形状

    return Fa