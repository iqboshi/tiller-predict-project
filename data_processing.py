import os.path
import numpy as np
import torch
import tifffile as tiff

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

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch = patches[start_idx:end_idx]  # (batch_size, C, patch_size, patch_size)
        processed_batch = process_fn(batch) # 用户自定义处理函数，比如卷积、特征提取等
        processed_list.append(processed_batch)

    return torch.cat(processed_list, dim=0)

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

    # 重组回原图
    processed_patches = processed_patches.view(num_patches_h, num_patches_w, C, patch_size, patch_size)
    # 调整维度为 (C, num_patches_h * patch_size, num_patches_w * patch_size)
    processed_patches = processed_patches.permute(2, 0, 3, 1, 4).contiguous()
    processed_image = processed_patches.view(C, H, W)

    # 转回 (H, W, C)
    processed_image_np = processed_image.cpu().numpy().transpose(1, 2, 0)
    return processed_image_np

if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    dir_name = 'D:\python_project\data'
    raw_pic_name = 'composite_image_vZeaz9yO.tif'
    input_path = os.path.join(dir_name, raw_pic_name)
    patch_size = 25
    batch_size = 64

    T1 = tiff.imread(input_path).astype('float32')
    H, W, C = T1.shape
    T1 = T1[:H - H % patch_size, : W - W % patch_size]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def example_process_fn(patches):
        # patches: (N, C, patch_size, patch_size)
        return patches * 2

    processed_image = process_large_image(T1, patch_size, example_process_fn, batch_size, device)