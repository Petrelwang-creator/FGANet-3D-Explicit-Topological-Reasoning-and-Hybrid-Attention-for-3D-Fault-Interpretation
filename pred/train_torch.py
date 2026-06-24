#!/usr/bin/env python
# coding: utf-8

import os
import time
import csv
from pathlib import Path
import random
import numpy as np

import torch
from torch import autocast, cuda
from torch.utils.data import DataLoader

from faultseg_torch import DataGenerator
from Unet3_torch import UNet3D
from loss_utils import CombinedLoss, dice_from_logits

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_load_checkpoint(model, optimizer, scaler, ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0, 0.0
    print("Resuming from:", ckpt_path)
    data = torch.load(ckpt_path, map_location="cpu")
    epoch = data.get("epoch", 0)
    best_val = data.get("best_val", 0.0)
    state = data.get("state_dict", data)
    model_state = {k:v for k,v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(model_state, strict=False)
    if optimizer is not None and "optimizer" in data:
        try:
            optimizer.load_state_dict(data["optimizer"])
        except Exception:
            print("Warning: failed to load optimizer state (skipping).")
    if scaler is not None and "scaler" in data:
        try:
            scaler.load_state_dict(data["scaler"])
        except Exception:
            print("Warning: failed to load scaler state (skipping).")
    print(f"Loaded parameters: {len(model_state)}/{len(model.state_dict())}")
    return epoch, best_val


def train(
    train_data_dir,
    train_label_dir,
    val_data_dir,
    val_label_dir,
    out_dir,
    vol_dim=(128,128,128),
    patch_size=64,
    batch=1,
    epochs=200,
    lr=1e-3,
    accum_steps=1,
    num_workers=4,
    fault_sampling_ratio=0.5,
    seed=42,
    early_stop_patience=30
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # collect ids robustly (assume .dat names are integers)
    def ids_from_dir(d):
        items = [f for f in os.listdir(d) if str(f).endswith('.dat')]
        ids = []
        for f in items:
            s = Path(f).stem
            try:
                ids.append(int(s))
            except:
                # fallback: keep filename
                ids.append(s)
        return sorted(ids)

    train_ids = ids_from_dir(train_data_dir)
    val_ids   = ids_from_dir(val_data_dir)

    print(f"Train samples: {len(train_ids)} Val samples: {len(val_ids)}")

    train_set = DataGenerator(train_data_dir, train_label_dir, train_ids,
                              vol_shape=vol_dim, patch_size=patch_size,
                              transform=True, fault_sampling_ratio=fault_sampling_ratio)
    val_set   = DataGenerator(val_data_dir, val_label_dir, val_ids,
                              vol_shape=vol_dim, patch_size=patch_size,
                              transform=False, fault_sampling_ratio=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    loader_kwargs = {"batch_size": batch, "shuffle": True, "num_workers": num_workers if use_cuda else 0, "pin_memory": True if use_cuda else False}
    val_loader_kwargs = {"batch_size": 1, "shuffle": False, "num_workers": 0, "pin_memory": False}

    train_loader = DataLoader(train_set, **loader_kwargs)
    val_loader   = DataLoader(val_set, **val_loader_kwargs)

    model = UNet3D(in_channels=1,out_channels=1, base=8,use_cbam=True, use_nonlocal=True).to(device)

    # loss & optimizer
    loss_fn = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # Resume if exists
    last_epoch, best_val = safe_load_checkpoint(model, optimizer, scaler, os.path.join(out_dir, "model_best.pth"))

    start_epoch = last_epoch + 1 if last_epoch else 1
    best_val_dice = best_val or 0.0
    no_improve = 0

    log = os.path.join(out_dir, "train_log.csv")
    with open(log, "w") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "lr"])

    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        TL, TD = 0.0, 0.0
        optimizer.zero_grad()
        it = 0

        for gx, fx in train_loader:
            gx, fx = gx.to(device), fx.to(device)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                out = model(gx)
                if isinstance(out, tuple):
                    o1, o2, o3 = out
                    loss = loss_fn(o1, fx) + 0.7*loss_fn(o2, fx) + 0.5*loss_fn(o3, fx)
                    main = o1
                else:
                    loss = loss_fn(out, fx)
                    main = out

                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (it + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            TL += loss.item() * accum_steps
            TD += dice_from_logits(main.detach(), fx).item()
            it += 1

        TL /= max(1, len(train_loader))
        TD /= max(1, len(train_loader))

        # validation
        model.eval()
        VL, VD = 0.0, 0.0
        with torch.no_grad():
            for gx, fx in val_loader:
                gx, fx = gx.to(device), fx.to(device)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    out = model(gx)
                    if isinstance(out, tuple):
                        o1, o2, o3 = out
                        loss = (loss_fn(o1, fx) + loss_fn(o2, fx) + loss_fn(o3, fx)) / 3.0
                        main = o1
                    else:
                        loss = loss_fn(out, fx)
                        main = out
                VL += loss.item()
                VD += dice_from_logits(main, fx).item()

        VL /= max(1, len(val_loader))
        VD /= max(1, len(val_loader))

        scheduler.step(VL)

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"[Epoch {ep}/{epochs}] train_loss={TL:.4f} val_loss={VL:.4f} train_dice={TD:.4f} val_dice={VD:.4f} LR={lr_now:.3e} time={elapsed:.1f}s")

        with open(log, "a") as f:
            csv.writer(f).writerow([ep, TL, VL, TD, VD, lr_now])

        # save checkpoint (last)
        ck = {
            "epoch": ep,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "val_loss": VL,
            "val_dice": VD
        }
        torch.save(ck, os.path.join(out_dir, "model_last.pth"))

        # save best
        if VD > best_val_dice:
            best_val_dice = VD
            torch.save(ck, os.path.join(out_dir, "model_best.pth"))
            print(f"New best model saved (val_dice={VD:.4f})")
            no_improve = 0
        else:
            no_improve += 1

        # periodic snapshot
        if ep % 5 == 0:
            torch.save({"state_dict": model.state_dict(), "epoch": ep}, os.path.join(out_dir, f"model_ep{ep}.pth"))

        # early stopping
        if early_stop_patience and no_improve >= early_stop_patience:
            print(f"Early stopping: no improvement for {no_improve} epochs.")
            break

    print("Training finished. Best val dice:", best_val_dice)


# if __name__ == "__main__":
#     train(
#         train_data_dir="/root/autodl-fs/data/train/seis/",
#         train_label_dir="/root/autodl-fs/data/train/fault/",
#         val_data_dir="/root/autodl-fs/data/validation/seis/",
#         val_label_dir="/root/autodl-fs/data/validation/fault/",
#         out_dir="/root/autodl-tmp/base/check_zcq",
#         vol_dim=(128,128,128),
#         patch_size=64,
#         epochs=200,
#         batch=1,
#         num_workers=4,
#         accum_steps=1,
#         fault_sampling_ratio=0.5,
#         seed=42,
#         early_stop_patience=40
#     )


if __name__ == "__main__":
    # ---------------------------------------------------------
    #  快速跑通测试：一键验证整个训练闭环
    # ---------------------------------------------------------
    import tempfile
    import shutil

    print(" 开始快速验证主训练流程 (自动生成 Mock 数据)...")

    temp_dir = tempfile.mkdtemp()
    try:
        # 1. 在临时目录下构建标准的训练/验证集目录结构
        train_d = os.path.join(temp_dir, "train", "seis")
        train_l = os.path.join(temp_dir, "train", "fault")
        val_d = os.path.join(temp_dir, "val", "seis")
        val_l = os.path.join(temp_dir, "val", "fault")
        out_dir = os.path.join(temp_dir, "check_zcq18_test")

        for d in [train_d, train_l, val_d, val_l, out_dir]:
            os.makedirs(d, exist_ok=True)

        # 2. 伪造极其微小的数据体积 (32x32x32) 确保 CPU/笔记本 也能在几秒内跑完测试
        mock_dim = (32, 32, 32)

        # 写入 2 个假训练样本
        for i in range(2):
            np.random.randn(*mock_dim).astype(np.float32).tofile(os.path.join(train_d, f"{i}.dat"))
            np.random.randint(0, 2, mock_dim).astype(np.float32).tofile(os.path.join(train_l, f"{i}.dat"))

        # 写入 1 个假验证样本
        for i in range(1):
            np.random.randn(*mock_dim).astype(np.float32).tofile(os.path.join(val_d, f"{i}.dat"))
            np.random.randint(0, 2, mock_dim).astype(np.float32).tofile(os.path.join(val_l, f"{i}.dat"))

        # 3. 触发主训练函数 (特意调小了 epochs 和 patch_size)
        print("🏃‍♂️ 数据就绪，启动极速训练循环...")
        train(
            train_data_dir=train_d,
            train_label_dir=train_l,
            val_data_dir=val_d,
            val_label_dir=val_l,
            out_dir=out_dir,
            vol_dim=mock_dim,
            patch_size=16,  #  切片缩小至 16，避免 OOM 并加快验证
            epochs=1,  #  仅测 1 个 epoch，确保反向传播没抛异常
            batch=1,
            num_workers=0,  #
            accum_steps=1,
            fault_sampling_ratio=0.5,
            seed=42,
            early_stop_patience=2
        )
        print("✅ 训练主流程 快速跑通！全链路无报错！")

    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)