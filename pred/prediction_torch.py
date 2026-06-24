#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
from time import time
from scipy.ndimage import label, generate_binary_structure

from Unet3_torch import UNet3D   # ← 使用你最终的优化模型
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ============================================================
# I/O 工具
# ============================================================
def read_dat(path):
    return np.fromfile(path, dtype=np.float32)


def write_dat(path, arr):
    arr.astype(np.float32).tofile(path)


# ============================================================
# 边缘平滑 mask（滑窗融合用）
# ============================================================
def get_mask(overlap, n1, n2, n3):
    sc = np.ones((n1, n2, n3), dtype=np.float32)
    sp = np.zeros((overlap,), dtype=np.float32)

    sig = 0.5 / ((overlap / 4) ** 2 + 1e-12)
    for k in range(overlap):
        ds = k - overlap + 1
        sp[k] = np.exp(-ds * ds * sig)

    # three dims
    for k in range(overlap):
        sc[k, :, :] *= sp[k]
        sc[-k - 1, :, :] *= sp[k]
        sc[:, k, :] *= sp[k]
        sc[:, -k - 1, :] *= sp[k]
        sc[:, :, k] *= sp[k]
        sc[:, :, -k - 1] *= sp[k]

    return sc


# ============================================================
# 后处理： 3D 连通域最小体素过滤
# ============================================================
def connected_component_filter(pred, min_size=50):
    """pred: (D,H,W) 概率图"""
    binary = (pred > 0.5).astype(np.uint8)
    structure = generate_binary_structure(3, 2)
    labeled, num = label(binary, structure)

    out = np.zeros_like(binary)
    for i in range(1, num + 1):
        region = (labeled == i)
        if region.sum() >= min_size:
            out[region] = 1
    return out


# ============================================================
# 归一化函数：与 DataGenerator 完全一致
# ============================================================
def normalize(vol):
    mean, std = vol.mean(), vol.std()
    return (vol - mean) / (std + 1e-6)


# ============================================================
# 修复 deep-supervision 输出格式
# ============================================================
def forward_main_output(model, x):
    """统一深监督结构输出：返回 main 输出"""
    out = model(x)
    # 你的模型：若启用 deep supervision，则返回 (o1,o2,o3)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# ============================================================
# 单体积滑窗预测
# ============================================================
@torch.no_grad()
def predict_single_volume(model, input_path, output_path,
                          input_shape=(128, 128, 128),
                          block_size=(128, 128, 128),
                          overlap=16,
                          min_cc_size=50,
                          device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Predict] Device: {device}")

    vol = read_dat(input_path)
    D, H, W = input_shape

    if vol.size != D*H*W:
        print(f"[Predict] Skip {input_path}: Shape mismatch")
        return

    vol = vol.reshape(input_shape).astype(np.float32)
    vol = normalize(vol)  # 与训练保持一致

    n1, n2, n3 = block_size
    osz = overlap
    stride = (n1 - osz, n2 - osz, n3 - osz)

    c1 = int(np.ceil((D - osz) / stride[0]))
    c2 = int(np.ceil((H - osz) / stride[1]))
    c3 = int(np.ceil((W - osz) / stride[2]))

    # pad
    p1 = stride[0]*c1 + osz
    p2 = stride[1]*c2 + osz
    p3 = stride[2]*c3 + osz

    pad_vol = np.zeros((p1, p2, p3), dtype=np.float32)
    pad_vol[:D, :H, :W] = vol

    pred_map   = np.zeros_like(pad_vol)
    weight_map = np.zeros_like(pad_vol)

    mask = get_mask(osz, n1, n2, n3)

    model.eval()
    model.to(device)

    print(f"[Predict] Sliding window…  ({c1} x {c2} x {c3}) blocks)")
    t0 = time()

    for i1 in range(c1):
        for i2 in range(c2):
            for i3 in range(c3):

                b1, e1 = i1*stride[0], i1*stride[0] + n1
                b2, e2 = i2*stride[1], i2*stride[1] + n2
                b3, e3 = i3*stride[2], i3*stride[2] + n3

                block = pad_vol[b1:e1, b2:e2, b3:e3]

                inp = torch.from_numpy(block[None,None,...]).float().to(device)

                out = forward_main_output(model, inp)
                out = torch.sigmoid(out).cpu().numpy()[0, 0]

                pred_map[b1:e1,b2:e2,b3:e3] += out * mask
                weight_map[b1:e1,b2:e2,b3:e3] += mask

    fused = pred_map / (weight_map + 1e-6)
    fused = fused[:D, :H, :W]

    filtered = connected_component_filter(fused, min_size=min_cc_size)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_dat(output_path, filtered.astype(np.float32))

    print(f"[Predict] Saved => {output_path}, time {time()-t0:.1f}s")


# ============================================================
# 批量预测
# ============================================================
def batch_predict(model, input_dir, output_dir,
                  input_shape=(128, 128, 128),
                  block_size=(128,128,128),
                  overlap=16,
                  min_cc_size=50,
                  device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".dat")]
    print(f"[BatchPredict] {len(files)} volumes found")

    for f in files:
        ip = os.path.join(input_dir, f)
        op = os.path.join(output_dir, f.replace(".dat", "_pred3.dat"))
        predict_single_volume(
            model, ip, op,
            input_shape=input_shape,
            block_size=block_size,
            overlap=overlap,
            min_cc_size=min_cc_size,
            device=device
        )


# ============================================================
# 主程序（加载权重 + 批量预测）
# ============================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "/root/autodl-tmp/base/check_zcq17/model_best.pth"
    input_dir = "/autodl-fs/data/data/test/"
    output_dir = "/root/autodl-tmp/base/data/output_pred/"

    print(f"[Main] Device: {device}")
    print(f"[Main] Loading model from: {ckpt_path}")

    # 与训练脚本统一
    model = UNet3D(in_ch=1, base=16, deep_super=True)

    # --------- 安全加载模型（兼容 strict=False + weights_only）---------
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    filtered = {k: v for k, v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(filtered, strict=False)

    print(f"[Main] Loaded {len(filtered)} parameters.")

    batch_predict(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        input_shape=(128, 128, 128),
        block_size=(128,128,128),
        overlap=16,
        min_cc_size=50,
        device=device,
    )
