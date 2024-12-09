import tqdm
from skimage import exposure
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import torch

# 定义安全的除法操作，任何异常值都赋值为nan
def safe_divide(numerator, denominator):
    result = torch.div(numerator, denominator)
    result[denominator == 0] = torch.nan  # 将除以零的结果设置为nan
    return result

def Fun_All_Factors2(A, ck, batch_size, device):
    # 提取各个波段
    Ar = A[:, :, 2, :]  # 第三波段 (红光)
    Ag = A[:, :, 1, :]  # 第二波段 (绿光)
    Ab = A[:, :, 0, :]  # 第一波段 (蓝光)
    Are = A[:, :, 3, :]  # 第四波段 (红边)
    Anir = A[:, :, 4, :]  # 第五波段 (近红外)

    # 计算颜色矩阵
    R = torch.mean(Ar, dim=(0, 1))
    G = torch.mean(Ag, dim=(0, 1))
    Rstd = torch.std(Ar, dim=(0, 1), unbiased=False)
    Gstd = torch.std(Ag, dim=(0, 1), unbiased=False)
    Bstd = torch.std(Ab, dim=(0, 1), unbiased=False)
    Rmean = R
    Gmean = G
    Bmean = torch.mean(Ab, dim=(0, 1))  # 直接计算

    # 植被指数计算
    NIR = torch.mean(Anir, dim=(0, 1))
    RE = torch.mean(Are, dim=(0, 1))
    NDVI = safe_divide(NIR - R, NIR + R)
    EVI2 = 2.5 * safe_divide(NIR - R, 1 + NIR + 2.4 * R)
    RVI = safe_divide(NIR, R)
    DVI = NIR - R
    RDVI = safe_divide(NIR - R, torch.sqrt(NIR + R))
    MSR = safe_divide(NIR / R - 1, torch.sqrt(NIR / R + 1))
    MCARI = safe_divide(NIR - RE - 0.2 * (NIR - G), NIR / RE)
    OSAVI = safe_divide(1.16 * (NIR - R), NIR + R + 0.16)
    WDRVI = safe_divide(0.1 * NIR - R, 0.1 * NIR + R)

    S1, S2, S3, S4, S5, S6 = [], [], [], [], [], []

    for i in range(batch_size):
        gray_Anir = exposure.rescale_intensity(Anir[:, :, i].cpu().numpy(), out_range=(0, 1))
        image_uint8 = (gray_Anir * 63).astype(np.uint8)
        # 纹理指数计算
        glcms1 = graycomatrix(image_uint8, levels=64, distances=[1],
                              angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        # np.transpose(glcms1, (0, 1, 3, 2))
        # stats = graycoprops(glcms1, prop=['contrast', 'correlation', 'energy', 'homogeneity'])
        ga1 = glcms1[:, :, 0, 0]
        ga2 = glcms1[:, :, 0, 1]
        ga3 = glcms1[:, :, 0, 2]
        ga4 = glcms1[:, :, 0, 3]
        energya1 = 0
        energya2 = 0
        energya3 = 0
        energya4 = 0
        homogeneity1 = 0
        homogeneity2 = 0
        homogeneity3 = 0
        homogeneity4 = 0

        for e in range(64):
            for f in range(64):
                energya1 = energya1 + ga1[e, f] ** 2
                homogeneity1 = homogeneity1 + (1 / (1 + (e - f) ** 2)) * ga1[e, f]
                energya2 = energya2 + ga2[e, f] ** 2
                homogeneity2 = homogeneity2 + (1 / (1 + (e - f) ** 2)) * ga2[e, f]
                energya3 = energya3 + ga3[e, f] ** 2
                homogeneity3 = homogeneity3 + (1 / (1 + (e - f) ** 2)) * ga3[e, f]
                energya4 = energya4 + ga4[e, f] ** 2
                homogeneity4 = homogeneity4 + (1 / (1 + (e - f) ** 2)) * ga4[e, f]
        s1 = np.sum(graycoprops(glcms1, 'contrast'))
        s2 = np.sum(graycoprops(glcms1, 'correlation'))
        s3 = np.sum(graycoprops(glcms1, 'energy'))
        s4 = np.sum(graycoprops(glcms1, 'homogeneity'))
        s5 = 0.00001 * (energya1 + energya2 + energya3 + energya4)
        s6 = 0.0001 * (homogeneity1 + homogeneity2 + homogeneity3 + homogeneity4)
        S1.append(s1)
        S2.append(s2)
        S3.append(s3)
        S4.append(s4)
        S5.append(s5)
        S6.append(s6)
    S1 = torch.tensor(S1, device=device)
    S2 = torch.tensor(S2, device=device)
    S3 = torch.tensor(S3, device=device)
    S4 = torch.tensor(S4, device=device)
    S5 = torch.tensor(S5, device=device)
    S6 = torch.tensor(S6, device=device)

    # 合并这些张量，结果形状将是 22 x 64
    outmat = torch.stack([NDVI, EVI2, RVI, DVI, RDVI, MSR,
                              MCARI, OSAVI, WDRVI, NIR, S1, S2, S3, S4,
                              S5, S6, Rmean, Gmean, Bmean, Rstd, Gstd, Bstd], dim=0)

    # 转置得到 64 x 22
    outmat = outmat.t()

    return outmat