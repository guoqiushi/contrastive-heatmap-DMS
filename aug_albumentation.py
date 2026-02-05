import cv2
import albumentations as A

# aug = A.Compose(
#     [
#         A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.10, rotate_limit=6,
#                            border_mode=cv2.BORDER_REFLECT_101, p=0.35),
#
#         A.LongestMaxSize(max_size=800, p=1.0),
#         A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
#
#         A.OneOf([
#             A.RandomGamma(gamma_limit=(70, 140), p=1.0),
#             A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
#             A.CLAHE(clip_limit=(1, 3), tile_grid_size=(8, 8), p=1.0),
#         ], p=0.7),
#
#         A.OneOf([
#             A.MotionBlur(blur_limit=(3, 9), p=1.0),
#             A.GaussianBlur(blur_limit=(3, 7), p=1.0),
#             A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.3), p=1.0),
#         ], p=0.3),
#
#         A.OneOf([
#             A.ISONoise(color_shift=(0.0, 0.01), intensity=(0.05, 0.25), p=1.0),
#             # A.GaussNoise(...)  # 若你用 A2.x，注意参数签名（你前面已经遇到过）
#         ], p=0.25),
#
#         A.ImageCompression(quality_range=(35, 90), p=0.2),
#
#         # A.OneOf([
#         #     A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.03, 0.12),
#         #                     hole_width_range=(0.03, 0.12), fill=0, p=1.0),
#         #     A.GridDropout(ratio=0.3, unit_size_min=24, unit_size_max=72, fill=0, p=1.0),
#         # ], p=0.15),
#     ],
#     bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
# )

import random
import math
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


def _clip_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


# ----------------------------
# 1) IR_AGC_AE: gain + bias + gamma
# ----------------------------
class IR_AGC_AE(ImageOnlyTransform):
    """
    Global AGC/AE style: gain + bias + gamma (non-linear).
    Works for gray (H,W) or gray3/rgb (H,W,3).
    """
    def __init__(
        self,
        gain_range=(0.75, 1.40),
        bias_range=(-20, 20),
        gamma_range=(0.75, 1.40),
        p=0.7
    ):
        super().__init__(p=p)
        self.gain_range = gain_range
        self.bias_range = bias_range
        self.gamma_range = gamma_range

    def apply(self, img, **params):
        x = img.astype(np.float32)

        gain = random.uniform(*self.gain_range)
        bias = random.uniform(*self.bias_range)
        gamma = random.uniform(*self.gamma_range)

        # gain + bias
        x = x * gain + bias
        x = np.clip(x, 0, 255)

        # gamma in [0,1] domain
        x01 = x / 255.0
        x01 = np.power(x01, gamma)
        x = x01 * 255.0

        return _clip_u8(x)


# ----------------------------
# 2) Vignetting: dark corners / uneven illumination
# ----------------------------
class IR_Vignetting(ImageOnlyTransform):
    """
    Radial vignetting: center bright, corners darker.
    strength_range: larger -> stronger darkening.
    """
    def __init__(self, strength_range=(0.15, 0.55), p=0.25):
        super().__init__(p=p)
        self.strength_range = strength_range

    def apply(self, img, **params):
        h, w = img.shape[:2]
        strength = random.uniform(*self.strength_range)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r = r / (r.max() + 1e-6)              # [0,1]
        mask = 1.0 - strength * (r ** 2)      # center ~1, corners smaller

        x = img.astype(np.float32)
        if x.ndim == 2:
            x *= mask
        else:
            x *= mask[..., None]

        return _clip_u8(x)


# ----------------------------
# 3) Hotspot + bloom: local saturated reflection + halo
# ----------------------------
class IR_HotSpotBloom(ImageOnlyTransform):
    """
    Add 0~N bright spots with gaussian halo (bloom).
    Useful for IR retro-reflection / glasses / skin oil / screen reflection.
    """
    def __init__(
        self,
        num_spots=(0, 2),
        radius_range=(12, 80),
        intensity_range=(25, 120),
        blur_ksize=(21, 61),   # bloom blur kernel size range
        p=0.35
    ):
        super().__init__(p=p)
        self.num_spots = num_spots
        self.radius_range = radius_range
        self.intensity_range = intensity_range
        self.blur_ksize = blur_ksize

    def apply(self, img, **params):
        h, w = img.shape[:2]
        x = img.astype(np.float32)

        n = random.randint(self.num_spots[0], self.num_spots[1])
        if n <= 0:
            return img

        mask = np.zeros((h, w), np.float32)
        for _ in range(n):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            r = random.randint(self.radius_range[0], self.radius_range[1])
            inten = random.uniform(*self.intensity_range)
            cv2.circle(mask, (cx, cy), r, float(inten), -1)

        # bloom halo
        kmin, kmax = self.blur_ksize
        k = random.randint(kmin // 2, kmax // 2) * 2 + 1  # odd
        halo = cv2.GaussianBlur(mask, (k, k), 0)

        if x.ndim == 2:
            x = x + halo
        else:
            x = x + halo[..., None]

        return _clip_u8(x)


# ----------------------------
# Compose builder
# ----------------------------
def build_ir_light_augmenter(p_all=1.0) -> A.Compose:
    """
    IR light pipeline:
      - (optional) CLAHE (local)
      - AGC/AE (global non-linear)
      - vignetting (uneven illum)
      - hotspot/bloom (local saturation/reflection)
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=800, p=1.0),
            A.PadIfNeeded(min_height=800, min_width=800, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            # 先做 AGC/AE（全局），再 CLAHE（局部）通常更像“先曝光再局部拉对比”
            IR_AGC_AE(gain_range=(0.75, 1.40), bias_range=(-20, 20), gamma_range=(0.75, 1.40), p=0.75),

            # CLAHE 是 albumentations 内置
            A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.35),

            IR_Vignetting(strength_range=(0.15, 0.55), p=0.25),

            IR_HotSpotBloom(num_spots=(0, 2), radius_range=(12, 80), intensity_range=(25, 120), p=0.35),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),

    )








