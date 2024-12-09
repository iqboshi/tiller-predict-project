import tqdm
from skimage import exposure
import numpy as np
from skimage.feature import graycomatrix, graycoprops
def Fun_All_Factors2(A, m0, n0, ck):
    # 提取各个波段
    Ar = A[:, :, 2]  # 第三波段 (红光)
    Ag = A[:, :, 1]  # 第二波段 (绿光)
    Ab = A[:, :, 0]  # 第一波段 (蓝光)
    Are = A[:, :, 3]  # 第四波段 (红边)
    Anir = A[:, :, 4]  # 第五波段 (近红外)
    gray_Anir = exposure.rescale_intensity(Anir, out_range=(0, 1))  # 归一化

    # 初始化各个记录矩阵
    Rstd = np.zeros((int(m0 / ck), int(n0 / ck)))
    Gstd = np.zeros((int(m0 / ck), int(n0 / ck)))
    Bstd = np.zeros((int(m0 / ck), int(n0 / ck)))
    Rmean = np.zeros((int(m0 / ck), int(n0 / ck)))
    Gmean = np.zeros((int(m0 / ck), int(n0 / ck)))
    Bmean = np.zeros((int(m0 / ck), int(n0 / ck)))
    NDVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    EVI2_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    RVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    DVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    RDVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    MSR_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    MCARI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    OSAVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    WDRVI_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    NIR_record = np.zeros((int(m0 / ck), int(n0 / ck)))
    S1 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S2 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S3 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S4 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S5 = np.zeros((int(m0 / ck), int(n0 / ck)))
    S6 = np.zeros((int(m0 / ck), int(n0 / ck)))

    # 滑动窗口
    with tqdm.tqdm(total=int(m0 / ck), desc="数据保存") as pbar:
        for i in range(int(m0 / ck)):
            for j in range(int(n0 / ck)):
                # 计算窗口位置
                h = int(ck - (ck - 1) / 2 + i * ck)
                l = int(ck - (ck - 1) / 2 + j * ck)
                w = int(h - (ck - 1) / 2) - 1
                e = int(l - (ck - 1) / 2) - 1

                # 提取窗口
                window_Ar = Ar[w:w + ck, e:e + ck]
                window_Ag = Ag[w:w + ck, e:e + ck]
                window_Ab = Ab[w:w + ck, e:e + ck]
                window_Are = Are[w:w + ck, e:e + ck]
                window_Anir = Anir[w:w + ck, e:e + ck]
                window_gray_Anir = gray_Anir[w:w + ck, e:e + ck]

                # 计算颜色矩阵
                R = np.mean(window_Ar)
                G = np.mean(window_Ag)
                Rstd[i, j] = np.std(window_Ar)
                Gstd[i, j] = np.std(window_Ag)
                Bstd[i, j] = np.std(window_Ab)
                Rmean[i, j] = R
                Gmean[i, j] = G
                Bmean[i, j] = np.mean(window_Ab)

                # 植被指数计算
                NIR = np.mean(window_Anir)
                RE = np.mean(window_Are)
                NDVI = (NIR - R) / (NIR + R)
                EVI2 = 2.5 * (NIR - R) / (1 + NIR + 2.4 * R)
                RVI = NIR / R
                DVI = NIR - R
                RDVI = (NIR - R) / np.sqrt(NIR + R)
                MSR = (NIR / R - 1) / np.sqrt(NIR / R + 1)
                MCARI = (NIR - RE - 0.2 * (NIR - G)) / (NIR / RE)
                OSAVI = (1.16 * (NIR - R)) / (NIR + R + 0.16)
                WDRVI = (0.1 * NIR - R) / (0.1 * NIR + R)

                NDVI_record[i, j] = NDVI
                EVI2_record[i, j] = EVI2
                RVI_record[i, j] = RVI
                DVI_record[i, j] = DVI
                RDVI_record[i, j] = RDVI
                MSR_record[i, j] = MSR
                MCARI_record[i, j] = MCARI
                OSAVI_record[i, j] = OSAVI
                WDRVI_record[i, j] = WDRVI
                NIR_record[i, j] = NIR

                image_uint8 = (window_gray_Anir * 63).astype(np.uint8)
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
                S1[i, j] = s1
                S2[i, j] = s2
                S3[i, j] = s3
                S4[i, j] = s4
                S5[i, j] = s5
                S6[i, j] = s6
            pbar.update(1)
    # 将记录矩阵展平并拼接
    NDVI_record = NDVI_record.flatten()
    EVI2_record = EVI2_record.flatten()
    RVI_record = RVI_record.flatten()
    DVI_record = DVI_record.flatten()
    RDVI_record = RDVI_record.flatten()
    MSR_record = MSR_record.flatten()
    MCARI_record = MCARI_record.flatten()
    OSAVI_record = OSAVI_record.flatten()
    WDRVI_record = WDRVI_record.flatten()
    NIR_record = NIR_record.flatten()
    S1 = S1.flatten()
    S2 = S2.flatten()
    S3 = S3.flatten()
    S4 = S4.flatten()
    S5 = S5.flatten()
    S6 = S6.flatten()
    Rmean = Rmean.flatten()
    Gmean = Gmean.flatten()
    Bmean = Bmean.flatten()
    Rstd = Rstd.flatten()
    Gstd = Gstd.flatten()
    Bstd = Bstd.flatten()

    # 拼接所有因子
    outmat = np.column_stack((NDVI_record, EVI2_record, RVI_record, DVI_record, RDVI_record, MSR_record,
                              MCARI_record, OSAVI_record, WDRVI_record, NIR_record, S1, S2, S3, S4,
                              S5, S6, Rmean, Gmean, Bmean, Rstd, Gstd, Bstd))

    return outmat