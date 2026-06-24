import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy.ndimage as ndi

def normalize(vol):
    vol = vol.astype(np.float32)
    mean, std = vol.mean(), vol.std()
    return (vol - mean) / (std + 1e-6)

def random_flip_rot(img, lbl):
    # img/lbl shape: (C=1, D, H, W)
    if random.random() < 0.5:
        img = img[:, :, :, ::-1].copy()
        lbl = lbl[:, :, :, ::-1].copy()
    if random.random() < 0.5:
        img = img[:, :, ::-1, :].copy()
        lbl = lbl[:, :, ::-1, :].copy()
    k = random.randint(0, 3)
    img = np.rot90(img, k, axes=(2, 3)).copy()
    lbl = np.rot90(lbl, k, axes=(2, 3)).copy()
    return img, lbl

def intensity_aug(vol):
    if random.random() < 0.7:
        mul = 0.9 + 0.2 * random.random()
        vol = vol * mul
    if random.random() < 0.3:
        bias = 0.1 * (random.random() - 0.5)
        vol = vol + bias
    if random.random() < 0.2:
        sigma = random.uniform(0.5, 1.0)
        vol = ndi.gaussian_filter(vol, sigma)
    return vol

class DataGenerator(Dataset):
    """
    带断层优先采样的数据集：
    - 可配置 patch_size（cube）
    - fault_sampling_ratio: 0~1，采样断层优先样本比例
    - vol_shape: 原始体积装载时 reshape 的目标形状 (Z,Y,X)
    """

    def __init__(self, dpath, lpath, ids, vol_shape=(128,128,128), patch_size=64, transform=True, fault_sampling_ratio=0.5):
        self.dpath = dpath
        self.lpath = lpath
        self.vol_shape = tuple(vol_shape)
        self.patch_size = int(patch_size)
        self.transform = transform
        self.ids = ids
        self.fault_sampling_ratio = float(fault_sampling_ratio)

    def __len__(self):
        return len(self.ids)

    def _load_volume(self, sid):
        d = os.path.join(self.dpath, f"{sid}.dat")
        l = os.path.join(self.lpath, f"{sid}.dat")
        gx = np.fromfile(d, dtype=np.float32)
        fx = np.fromfile(l, dtype=np.float32)
        try:
            gx = gx.reshape(self.vol_shape)
            fx = fx.reshape(self.vol_shape)
        except Exception as e:
            raise RuntimeError(f"Failed to reshape volume {sid} to {self.vol_shape}: {e}")
        return gx, fx

    def _sample_patch(self, gx, fx):
        D, H, W = fx.shape
        ps = self.patch_size

        # choose center
        if random.random() < self.fault_sampling_ratio:
            pos = np.argwhere(fx > 0.5)
            if len(pos) > 0:
                z, y, x = pos[random.randint(0, len(pos) - 1)]
            else:
                z, y, x = D//2, H//2, W//2
        else:
            z = random.randint(0, D-1)
            y = random.randint(0, H-1)
            x = random.randint(0, W-1)

        # compute patch corner so that center is inside patch and patch inside volume
        z0 = max(0, min(D - ps, z - ps // 2))
        y0 = max(0, min(H - ps, y - ps // 2))
        x0 = max(0, min(W - ps, x - ps // 2))
        z1, y1, x1 = z0 + ps, y0 + ps, x0 + ps

        pimg = gx[z0:z1, y0:y1, x0:x1]
        plbl = fx[z0:z1, y0:y1, x0:x1]

        # Safety: if volume smaller than ps, pad
        if pimg.shape != (ps, ps, ps):
            pimg_p = np.zeros((ps, ps, ps), dtype=pimg.dtype)
            plbl_p = np.zeros((ps, ps, ps), dtype=plbl.dtype)
            pimg_p[:pimg.shape[0], :pimg.shape[1], :pimg.shape[2]] = pimg
            plbl_p[:plbl.shape[0], :plbl.shape[1], :plbl.shape[2]] = plbl
            pimg, plbl = pimg_p, plbl_p

        return pimg, plbl

    def __getitem__(self, idx):
        sid = self.ids[idx]

        gx, fx = self._load_volume(sid)
        gx = normalize(gx)

        gx, fx = self._sample_patch(gx, fx)
        gx = gx[None]  # channel dim
        fx = fx[None]

        if self.transform:
            gx[0] = intensity_aug(gx[0])
            gx, fx = random_flip_rot(gx, fx)

        return torch.from_numpy(gx).float(), torch.from_numpy(fx).float()


if __name__ == "__main__":
    # ---------------------------------------------------------
    #  快速跑通测试：验证 DataGenerator 数据读取与切割策略
    # ---------------------------------------------------------
    import tempfile
    import shutil

    print("🚀 开始测试 DataGenerator...")

    # 1. 使用 tempfile 创建一个阅后即焚的临时文件夹，模拟你的 autodl 数据集环境
    temp_dir = tempfile.mkdtemp()
    try:
        dpath = os.path.join(temp_dir, "seis")
        lpath = os.path.join(temp_dir, "fault")
        os.makedirs(dpath, exist_ok=True)
        os.makedirs(lpath, exist_ok=True)

        # 2. 伪造一个 128x128x128 的 3D 体积数据，并保存为 1.dat (与你定义的读取逻辑一致)
        vol_shape = (128, 128, 128)
        dummy_seis = np.random.randn(*vol_shape).astype(np.float32)
        dummy_fault = np.random.randint(0, 2, vol_shape).astype(np.float32)

        dummy_seis.tofile(os.path.join(dpath, "1.dat"))
        dummy_fault.tofile(os.path.join(lpath, "1.dat"))

        # 3. 实例化 DataGenerator
        # 设置 patch_size=64, 意味着我们会从 128 的大图中随机裁剪出 64 的立方体 (Cube) 参与训练
        dataset = DataGenerator(dpath, lpath, ids=[1], vol_shape=vol_shape, patch_size=64, transform=True,
                                fault_sampling_ratio=0.8)

        # 4. 抽取第一个样本，触发 _sample_patch (基于断层目标的优先裁剪) 和数据增强
        img, lbl = dataset[0]

        print(f"➡️ 裁剪后的输入数据 (Image) 形状: {img.shape}")
        print(f"➡️ 裁剪后的断层标签 (Label) 形状: {lbl.shape}")

        # 验证通道扩展 (C=1) 和裁剪尺寸 (64x64x64) 是否正确
        assert img.shape == (1, 64, 64, 64), "❌ Image 形状不正确！"
        assert lbl.shape == (1, 64, 64, 64), "❌ Label 形状不正确！"
        print("✅ DataGenerator 与数据增强测试通过！")

    finally:
        # 清理临时文件夹，保持环境整洁
        shutil.rmtree(temp_dir)