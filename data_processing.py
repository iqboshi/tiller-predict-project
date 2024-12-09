import os.path
import numpy as np
import torch
import tifffile as tiff
import tqdm
from scipy.io import savemat
from utils.Fun_Spectral_Correction import Fun_Spectral_Correction
from utils.Fun_All_Factors2 import Fun_All_Factors2
from sklearn.preprocessing import MinMaxScaler
from skimage import exposure


def custom_scale_numpy(array, min_val=0, max_val=1):
    # 计算每个通道的最小值和最大值
    min_array = np.min(array, axis=(0, 1), keepdims=True)
    max_array = np.max(array, axis=(0, 1), keepdims=True)

    # 处理最大值和最小值相等的情况，防止除以零
    scale = max_array - min_array
    scale[scale == 0] = 1  # 避免除以零，对于常数数组，此步骤将 scale 设置为 1

    # 执行缩放
    scaled_array = (array - min_array) / scale
    scaled_array = scaled_array * (max_val - min_val) + min_val

    # 如果原始最大值和最小值相同，将数组设置为 min_val
    scaled_array[max_array == min_array] = min_val

    return scaled_array
def process_fn(batchs, device):
    batchs = batchs.permute(2, 3, 1, 0)
    patch_size, _, C, batch_size = batchs.shape
    Ar = batchs[:, :, 2, :]
    Ag = batchs[:, :, 1, :]
    Ab = batchs[:, :, 0, :]
    Are = batchs[:, :, 3, :]
    Anir = batchs[:, :, 4, :]

    # div: 25, 25, 64
    div = Anir - Ar

    div_record = torch.mean(div, dim=(0, 1), keepdim=True)

    _, binEdges = np.histogram(div_record.flatten().cpu(), bins=5)

    p0 = torch.ones_like(div_record)
    p0 = torch.where((binEdges[0] < div_record) & (div_record <= binEdges[1]), 0.4, p0)
    p0 = torch.where((binEdges[1] < div_record) & (div_record <= binEdges[2]), 0.35, p0)
    p0 = torch.where((binEdges[2] < div_record) & (div_record <= binEdges[3]), 0.3, p0)
    p0 = torch.where((binEdges[3] < div_record) & (div_record <= binEdges[4]), 0.25, p0)
    p0 = torch.where((binEdges[4] < div_record) & (div_record <= binEdges[5]), 0.2, p0)

    # 自适应光谱增强
    # 使用 torch.unique 获取唯一值和逆索引
    Anir_unique, l_idx = torch.unique(Anir, return_inverse=True)
    # 对唯一值进行排序，获取索引（虽然 unique 已经排序了，这一步确保理解 argsort 的使用）
    Anir_sort = torch.argsort(Anir_unique)
    T = len(Anir_sort)

    p, q, h = p0, 0.6, 0.15

    Fa = [Fun_Spectral_Correction(T, p[:, :, i].item(), q, h, l_idx[:, :, i], Anir_sort, patch_size, patch_size, device) for i in range(batch_size)]

    Fa = torch.stack(Fa, dim=2)

    Anir_adj = Anir * Fa

    # 图像阈值分割
    R = Anir_adj - Ar
    R = [np.clip(MinMaxScaler((0, 1)).fit_transform(R[:, :, i].cpu().numpy()), 0, 1) for i in range(batch_size)]
    R = [exposure.equalize_adapthist(r) for r in R]
    R = torch.tensor(np.stack(R, axis=2)).to(device)

    rm_ad = torch.mean(R, dim=(0, 1), keepdim=True)
    NDVI_adj = Anir_adj - Ar / Anir_adj + Ar

    maxndvi = torch.amax(NDVI_adj, dim=(0, 1), keepdim=True)
    minndvi = torch.amin(NDVI_adj, dim=(0, 1), keepdim=True)
    meanndvi = torch.mean(NDVI_adj, dim=(0, 1), keepdim=True)
    cv = ((meanndvi - minndvi) / (maxndvi - meanndvi)) ** 2

    # 计算 c 和 c_a 的张量
    c = torch.where(cv > 1, cv, -cv)  # 如果 cv > 1，c = cv 否则 c = -cv
    c_a = torch.where(cv > 1, 2 * c * (10 ** -2), 2 * (1 / c) * (10 ** -2))

    # 更新 rm_ad
    l = rm_ad + c_a

    W = torch.zeros_like(R)
    # 向量化条件赋值
    W = torch.where(R > l, Anir_adj + 0.1, Anir_adj - 0.1)
    W = [torch.tensor(MinMaxScaler((0, 1)).fit_transform(W[:, :, i].cpu()), device=device) for i in range(batch_size)]
    W = torch.stack(W, dim=2)
    Ar_w = torch.where(W < 0.5, torch.zeros_like(W), torch.ones_like(W))

    Mask = (Ar_w == 0)
    Mask = Mask.float()
    shape_record = [(1 - torch.sum(Mask[:, :, i]) / (patch_size * patch_size)) for i in range(batch_size)]
    shape_record = torch.stack(shape_record, dim=0)

    masked_image = batchs * Ar_w.unsqueeze(2)

    # 提取指数
    outmat = Fun_All_Factors2(masked_image, patch_size, batch_size, device)

    # 将 outmat 和 shape_record1 进行水平拼接
    all_factors = torch.cat((outmat, shape_record.unsqueeze(1)), dim=1)

    return all_factors

def process_patches_in_batches(patches, process_fn, batch_size):
    """
    分批处理 patch，防止一次处理太多导致显存溢出
    :param patches: (N, C, patch_size, patch_size) 的张量
    :param process_fn: 用于处理每批 patch 的函数，输入和输出为 (N, C, patch_size, patch_size)
    :param batch_size: 每次处理的 patch 数量
    :return: 处理后的 patches，形状同输入
    """
    device = patches.device
    N = patches.shape[0]
    processed_list = []
    with tqdm.tqdm(total=N // batch_size + 1, desc="处理中（batch/batchs）") as pbar:
        for start_idx in range(51*batch_size, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch = patches[start_idx:end_idx]  # (batch_size, C, patch_size, patch_size)
            processed_batch = process_fn(batch, device) # 用户自定义处理函数，比如卷积、特征提取等
            processed_list.append(processed_batch.cpu())

            del batch, processed_batch
            torch.cuda.empty_cache()
            pbar.update(1)
    return torch.cat(processed_list, dim=0).to(device)

def process_large_image(image_np, patch_size, process_fn, batch_size, device='cuda'):
    """
    将大图 (H, W, C) 切分为 patch，并使用 GPU 上的分批处理完成处理。
    :param image_np: 输入图像，NumPy 数组，形状为 (H, W, C)，C为通道数
    :param patch_size: 每个 patch 的大小 (patch_size x patch_size)
    :param process_fn: 用户自定义的处理函数，输入 (N, C, patch_size, patch_size)，返回同形状结果
    :param batch_size: 每批处理的patch数量
    :param device: 运行设备，'cuda'或'cpu'
    :return: 处理后的图像，NumPy数组，(H, W, C)
    """
    # 转为张量，并放到指定设备
    # image_np 形状 (H, W, C)
    image = torch.from_numpy(image_np).float().to(device)

    H, W, C = image.shape
    # 确保H和W可以整除patch_size，如不行需先对图像进行padding
    assert H % patch_size == 0 and W % patch_size == 0, "H和W需要是patch_size的整数倍"

    image = image.permute(2, 0, 1)
    # 使用 unfold 将图像分解为 patch
    # unfold(维度, kernel_size, step)
    # 沿H方向分patch
    patches = image.unfold(1, patch_size, patch_size)
    # 沿W方向再分patch
    patches = patches.unfold(2, patch_size, patch_size)
    # 此时patches形状为 (C, num_patches_h, num_patches_w, patch_size, patch_size)

    # 调整维度顺序为 (num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4)

    # 合并num_patches_h和num_patches_w为N
    num_patches_h = patches.size(0)
    num_patches_w = patches.size(1)
    patches = patches.reshape(num_patches_h * num_patches_w, C, patch_size, patch_size)  # (N, C, patch_size, patch_size)

    # 分批处理patch
    processed_patches = process_patches_in_batches(patches, process_fn, batch_size)  # (N, C, patch_size, patch_size)

    return processed_patches

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    dir_name = 'D:\py_project\data\\11.06'
    raw_pic_name = 'composite_image_vZeaz9yO.tif'
    input_path = os.path.join(dir_name, raw_pic_name)
    patch_size = 25
    batch_size = 64

    T1 = tiff.imread(input_path).astype('float16')
    H, W, C = T1.shape
    T1 = T1[:H - H % patch_size, : W - W % patch_size]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processed_image = process_large_image(T1, patch_size, process_fn, batch_size, device)
    savemat('yu3' + '_all_factors.mat', {'all_factors': processed_image.cpu().numpy()})
    pass