
import common
import losses
import augment
import numpy as np
import cv2
import math
import logger
import eval_tool
from aug_albumentation import build_ir_light_augmenter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torchvision.ops import roi_align
import random
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from dbface_ghost_cl import DBFace



def _gray_and_grad_2ch(img: np.ndarray) -> np.ndarray:
    """Build 2-channel input: [I, |∇I|].
    - I: grayscale intensity in [0,1]
    - |∇I|: Sobel gradient magnitude, per-image normalized to [0,1]
    Returns: float32 array of shape [H, W, 2].
    """
    if img is None:
        raise ValueError("img is None")

    # Accept HxW (gray) or HxWxC (color)
    if img.ndim == 2:
        I = img.astype(np.float32)
    else:
        # Albumentations returns RGB by default; common.imread may return BGR.
        # cvtColor handles either as long as channels=3, but to be safe we assume BGR/RGB both fine for gray conversion.
        I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    I = I / 255.0  # [0,1]

    # Optional: suppress noise before gradients (helps night IR). Keep very light.
    # I_blur = cv2.GaussianBlur(I, (3, 3), 0)
    I_blur = I

    gx = cv2.Sobel(I_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(I_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy + 1e-12)

    # Per-image normalization to stabilize the scale across AGC/AE and scenes
    mag = mag / (mag.max() + 1e-6)

    x = np.stack([I, mag], axis=-1).astype(np.float32)  # [H,W,2]
    return x


def _convert_first_conv_inplace(model: nn.Module, in_ch: int = 2) -> None:
    """Convert the first Conv2d layer in a model to accept `in_ch` channels.
    Copies weights by:
      - if old_in==1: duplicate -> [w, w]
      - if old_in==3: RGB->gray then duplicate -> [wgray, wgray]
      - else: keep first `in_ch` channels if possible, or repeat as needed
    """
    first_name, first_conv = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            first_name, first_conv = name, m
            break
    if first_conv is None:
        raise RuntimeError("No Conv2d found in model; cannot convert input channels.")

    if first_conv.in_channels == in_ch:
        return

    old = first_conv
    new = nn.Conv2d(
        in_channels=in_ch,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups if old.groups == 1 else old.groups,  # keep groups (expect 1)
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )

    with torch.no_grad():
        w = old.weight.data  # [out, in, k, k]
        out_c, old_in, k1, k2 = w.shape

        if old_in == 1:
            if in_ch == 2:
                w2 = torch.cat([w, w], dim=1)
            else:
                w2 = w.repeat(1, in_ch, 1, 1) / float(in_ch)
        elif old_in == 3:
            wr, wg, wb = w[:, 0:1], w[:, 1:2], w[:, 2:3]
            wgray = 0.2989 * wr + 0.5870 * wg + 0.1140 * wb  # [out,1,k,k]
            if in_ch == 2:
                w2 = torch.cat([wgray, wgray], dim=1)
            else:
                w2 = wgray.repeat(1, in_ch, 1, 1) / float(in_ch)
        else:
            # generic fallback
            if old_in >= in_ch:
                w2 = w[:, :in_ch].contiguous()
            else:
                rep = (in_ch + old_in - 1) // old_in
                w2 = w.repeat(1, rep, 1, 1)[:, :in_ch].contiguous()

        new.weight.data.copy_(w2)
        if old.bias is not None:
            new.bias.data.copy_(old.bias.data)

    # assign back to parent
    parent = model
    parts = first_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


class LDataset(Dataset):
    def __init__(self, labelfile, imagesdir, mean, std, width=800, height=800):
        
        self.width = width
        self.height = height
        # self.items = common.load_webface(labelfile, imagesdir)
        self.items = common.load_webface_folder(labelfile,imagesdir)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgfile, objs = self.items[index]

        image = common.imread(imgfile)

        # 1) Geometry-only augmentation (crop/resize/flip), keep objs consistent for CL two-views
        # image, objs = augment.webface_no_colorjittering(image, objs, self.width, self.height, keepsize=0)


        # 2) Photometric/IR-style augmentation: generate two views with shared geometry
        aug = build_ir_light_augmenter()

        # Albumentations expects bboxes as a list. Each bbox is (x_min, y_min, x_max, y_max) in pixels
        # when bbox_params.format == "pascal_voc".
        bboxes_in = [o.box for o in objs]
        labels_in = list(range(len(objs)))  # use obj index as id so we can map back after any filtering

        out1 = aug(image=image, bboxes=bboxes_in, labels=labels_in)
        image_v1 = out1["image"]
        bboxes_aug = out1["bboxes"]
        labels_aug = out1["labels"]

        out2 = aug(image=image, bboxes=bboxes_in, labels=labels_in)
        image_v2 = out2["image"]

        # Write back augmented boxes to objs (and drop boxes that were filtered by Albumentations)
        if len(bboxes_aug) == 0:
            log.info("{} has no bboxes after augmentation, index={}".format(imgfile, index))
            return self[random.randint(0, len(self.items)-1)]

        new_objs = []
        for bb, oid in zip(bboxes_aug, labels_aug):
            o = objs[int(oid)]
            o.x, o.y, o.r, o.b = bb[0], bb[1], bb[2], bb[3]
            new_objs.append(o)
        objs = new_objs

        if image is None or image_v1 is None or image_v2 is None:
            log.info("{} is empty, index={}".format(imgfile, index))
            return self[random.randint(0, len(self.items)-1)]

        keepsize            = 12
        # image, objs = augment.webface(image, objs, self.width, self.height, keepsize=0)

        # norm (two views) + build 2-channel structural input: I + |∇I|
        # 1) build [H,W,2] in [0,1]
        image_v1 = _gray_and_grad_2ch(image_v1)
        image_v2 = _gray_and_grad_2ch(image_v2)

        # 2) per-channel normalize (mean/std should be length-2 lists)
        image_v1 = ((image_v1 - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)).astype(np.float32)
        image_v2 = ((image_v2 - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)).astype(np.float32)

        posweight_radius    = 2
        stride              = 4
        fm_width            = self.width // stride
        fm_height           = self.height // stride

        heatmap_gt          = np.zeros((1,     fm_height, fm_width), np.float32)
        heatmap_posweight   = np.zeros((1,     fm_height, fm_width), np.float32)
        keep_mask           = np.ones((1,     fm_height, fm_width), np.float32)
        reg_tlrb            = np.zeros((1 * 4, fm_height, fm_width), np.float32)
        reg_mask            = np.zeros((1,     fm_height, fm_width), np.float32)
        distance_map        = np.zeros((1,     fm_height, fm_width), np.float32) + 1000
        landmark_gt         = np.zeros((1 * 10,fm_height, fm_width), np.float32)
        landmark_mask       = np.zeros((1,     fm_height, fm_width), np.float32)

        hassmall = False
        for obj in objs:
            isSmallObj = obj.area < keepsize * keepsize

            if isSmallObj:
                cx, cy = obj.safe_scale_center(1 / stride, fm_width, fm_height)
                keep_mask[0, cy, cx] = 0
                w, h = obj.width / stride, obj.height / stride

                x0 = int(common.clip_value(cx - w // 2, fm_width-1))
                y0 = int(common.clip_value(cy - h // 2, fm_height-1))
                x1 = int(common.clip_value(cx + w // 2, fm_width-1) + 1)
                y1 = int(common.clip_value(cy + h // 2, fm_height-1) + 1)
                if x1 - x0 > 0 and y1 - y0 > 0:
                    keep_mask[0, y0:y1, x0:x1] = 0
                hassmall = True

        for obj in objs:

            classes = 0
            cx, cy = obj.safe_scale_center(1 / stride, fm_width, fm_height)
            reg_box = np.array(obj.box) / stride
            isSmallObj = obj.area < keepsize * keepsize

            if isSmallObj:
                if obj.area >= 5 * 5:
                    distance_map[classes, cy, cx] = 0
                    reg_tlrb[classes*4:(classes+1)*4, cy, cx] = reg_box
                    reg_mask[classes, cy, cx] = 1
                continue

            w, h = obj.width / stride, obj.height / stride
            x0 = int(common.clip_value(cx - w // 2, fm_width-1))
            y0 = int(common.clip_value(cy - h // 2, fm_height-1))
            x1 = int(common.clip_value(cx + w // 2, fm_width-1) + 1)
            y1 = int(common.clip_value(cy + h // 2, fm_height-1) + 1)
            if x1 - x0 > 0 and y1 - y0 > 0:
                keep_mask[0, y0:y1, x0:x1] = 1

            w_radius, h_radius = common.truncate_radius((obj.width, obj.height))
            gaussian_map = common.draw_truncate_gaussian(heatmap_gt[classes, :, :], (cx, cy), h_radius, w_radius)

            mxface = 300
            miface = 25
            mxline = max(obj.width, obj.height)
            gamma = (mxline - miface) / (mxface - miface) * 10
            gamma = min(max(0, gamma), 10) + 1
            common.draw_gaussian(heatmap_posweight[classes, :, :], (cx, cy), posweight_radius, k=gamma)

            range_expand_x = math.ceil(w_radius)
            range_expand_y = math.ceil(h_radius)

            min_expand_size = 3
            range_expand_x = max(min_expand_size, range_expand_x)
            range_expand_y = max(min_expand_size, range_expand_y)

            icx, icy = cx, cy
            reg_landmark = None
            fill_threshold = 0.3
			
            if obj.haslandmark:
                reg_landmark = np.array(obj.x5y5_cat_landmark) / stride
                x5y5 = [cx]*5 + [cy]*5
                rvalue = (reg_landmark - x5y5)
                landmark_gt[0:10, cy, cx] = np.array(common.log(rvalue)) / 4
                landmark_mask[0, cy, cx] = 1

            if not obj.rotate:
                for cx in range(icx - range_expand_x, icx + range_expand_x + 1):
                    for cy in range(icy - range_expand_y, icy + range_expand_y + 1):
                        if cx < fm_width and cy < fm_height and cx >= 0 and cy >= 0:
                            
                            my_gaussian_value = 0.9
                            gy, gx = cy - icy + range_expand_y, cx - icx + range_expand_x
                            if gy >= 0 and gy < gaussian_map.shape[0] and gx >= 0 and gx < gaussian_map.shape[1]:
                                my_gaussian_value = gaussian_map[gy, gx]
                                
                            distance = math.sqrt((cx - icx)**2 + (cy - icy)**2)
                            if my_gaussian_value > fill_threshold or distance <= min_expand_size:
                                already_distance = distance_map[classes, cy, cx]
                                my_mix_distance = (1 - my_gaussian_value) * distance

                                if my_mix_distance > already_distance:
                                    continue

                                distance_map[classes, cy, cx] = my_mix_distance
                                reg_tlrb[classes*4:(classes+1)*4, cy, cx] = reg_box
                                reg_mask[classes, cy, cx] = 1

        # if hassmall:
        #     common.imwrite("test_result/keep_mask.jpg", keep_mask[0]*255)
        #     common.imwrite("test_result/heatmap_gt.jpg", heatmap_gt[0]*255)
        #     common.imwrite("test_result/keep_ori.jpg", (image*self.std+self.mean)*255)
        # pad gt boxes for ROIAlign-based contrastive learning
        max_faces = 32
        gt_boxes = np.full((max_faces, 4), -1, dtype=np.float32)  # xyxy in resized image (800x800)
        nfaces = min(len(objs), max_faces)
        for i in range(nfaces):
            x1, y1, x2, y2 = objs[i].box
            gt_boxes[i] = np.array([x1, y1, x2, y2], dtype=np.float32)

        return (
            T.to_tensor(image_v1),
            T.to_tensor(image_v2),
            heatmap_gt,
            heatmap_posweight,
            reg_tlrb,
            reg_mask,
            landmark_gt,
            landmark_mask,
            nfaces,
            gt_boxes,
            keep_mask
        )



class App(object):
    def __init__(self, labelfile, imagesdir):

        self.width, self.height = 800, 800
        self.mean = [0.44, 0.0]
        self.std = [0.28, 1.0]  # intensity normalized like before; grad is already ~[0,1]
        self.batch_size = 4
        self.lr = 1e-4
        self.gpus = [0] #[0, 1, 2, 3]
        self.gpu_master = self.gpus[0]
        self.model = DBFace(has_landmark=True, wide=64, has_ext=True, upmode="UCBA")
        self.model.init_weights()
        # Enable 2-channel input (I + |∇I|)
        _convert_first_conv_inplace(self.model, in_ch=2)
        self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.cuda(device=self.gpu_master)
        self.model.train()

        self.focal_loss = losses.FocalLoss()
        self.giou_loss = losses.GIoULoss()
        self.landmark_loss = losses.WingLoss(w=2)
        self.train_dataset = LDataset(labelfile, imagesdir, mean=self.mean, std=self.std, width=self.width, height=self.height)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=24)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.per_epoch_batchs = len(self.train_loader)
        self.iter = 0
        self.epochs = 150

        # weights for new objectives
        self.lambda_ghost = 1.0
        self.lambda_cl = 0.05
        self.cl_tau = 0.2
        self.ghost_topk = 20
        self.neg_topk = 32


    def set_lr(self, lr):

        self.lr = lr
        log.info(f"setting learning rate to: {lr}")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr



    @staticmethod
    def _dilate_mask(mask: torch.Tensor, k: int = 7) -> torch.Tensor:
        # mask: [B,1,H,W] in {0,1}
        pad = k // 2
        return (F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad) > 0).to(mask.dtype)

    @staticmethod
    def _draw_gaussian_torch(dst: torch.Tensor, cx: int, cy: int, sigma: float = 2.0, k: float = 1.0):
        # dst: [H,W]
        H, W = dst.shape
        radius = max(1, int(3 * sigma))
        x0, x1 = max(0, cx - radius), min(W - 1, cx + radius)
        y0, y1 = max(0, cy - radius), min(H - 1, cy + radius)

        xs = torch.arange(x0, x1 + 1, device=dst.device, dtype=torch.float32)
        ys = torch.arange(y0, y1 + 1, device=dst.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)) * float(k)

        patch = dst[y0:y1 + 1, x0:x1 + 1]
        dst[y0:y1 + 1, x0:x1 + 1] = torch.maximum(patch, g)

    def _build_ghost_targets(self, hm_det: torch.Tensor, face_mask: torch.Tensor,
                             topk: int = 20, thr: float = 0.30, sigma: float = 2.0) -> torch.Tensor:
        """
        Online pseudo-labeling for ghost head:
        - take high center scores outside GT face regions as ghost positives
        hm_det: [B,1,H,W] sigmoid center heatmap (detector confidence)
        face_mask: [B,1,H,W] {0,1} GT face region mask (should be dilated)
        returns ghost_target: [B,1,H,W] in [0,1]
        """
        B, _, H, W = hm_det.shape
        ghost_t = torch.zeros((B, 1, H, W), device=hm_det.device, dtype=hm_det.dtype)

        for b in range(B):
            score = hm_det[b, 0].clone()
            score[face_mask[b, 0] > 0] = 0
            score[score < thr] = 0

            nz = (score > 0).view(-1)
            if nz.sum().item() == 0:
                continue

            ksel = min(topk, int(nz.sum().item()))
            vals, idx = torch.topk(score.view(-1), k=ksel)
            for v, idv in zip(vals, idx):
                cy = int(idv.item() // W)
                cx = int(idv.item() % W)
                self._draw_gaussian_torch(ghost_t[b, 0], cx, cy, sigma=sigma, k=float(v.item()))
        return ghost_t

    def _sample_ghost_negatives(self, feat: torch.Tensor, ghostness: torch.Tensor, face_mask: torch.Tensor,
                                topk: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample ghost-guided hard negatives from feature map.
        feat: [B,C,H,W]
        ghostness: [B,1,H,W] sigmoid ghost map
        face_mask: [B,1,H,W] {0,1} mask to exclude face regions (dilated)
        returns:
            neg_z: [M,C] pooled negatives
            neg_w: [M] ghostness weights in [0,1]
        """
        B, C, H, W = feat.shape
        neg_feats = []
        neg_wts = []

        for b in range(B):
            score = ghostness[b, 0].clone()
            score[face_mask[b, 0] > 0] = -1  # exclude
            flat = score.view(-1)
            # keep only positive candidates
            valid = flat > 0
            if valid.sum().item() == 0:
                continue
            ksel = min(topk, int(valid.sum().item()))
            vals, idx = torch.topk(flat, k=ksel)
            ys = (idx // W).long()
            xs = (idx % W).long()
            # gather feature vectors at selected points
            f = feat[b, :, ys, xs].transpose(0, 1)  # [K,C]
            neg_feats.append(f)
            neg_wts.append(vals.clamp(min=0).to(f.dtype))

        if len(neg_feats) == 0:
            return torch.empty((0, C), device=feat.device, dtype=feat.dtype), torch.empty((0,), device=feat.device, dtype=feat.dtype)

        return torch.cat(neg_feats, dim=0), torch.cat(neg_wts, dim=0)

    def _gghn_infonce(self, z1: torch.Tensor, z2: torch.Tensor,
                      neg_z: torch.Tensor, neg_w: torch.Tensor,
                      tau: float = 0.2, alpha: float = 5.0) -> torch.Tensor:
        """
        Ghost-Guided Hard-Negative InfoNCE.
        z1, z2: [N,D] (paired positives, same order)
        neg_z: [M,D] ghost negatives
        neg_w: [M] ghostness in [0,1]
        """
        if z1.numel() == 0:
            return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

        # normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # in-batch logits
        logits_in = (z1 @ z2.t()) / tau  # [N,N]
        # positive is diagonal
        pos = logits_in.diag()

        # ghost logits (weighted)
        if neg_z.numel() > 0:
            neg_z = F.normalize(neg_z, dim=1)
            logits_g = (z1 @ neg_z.t()) / tau  # [N,M]
            w = (1.0 + alpha * neg_w.clamp(min=0, max=1)).clamp(min=1e-6)
            logits_g = logits_g + torch.log(w).unsqueeze(0)  # weight in denom
            denom = torch.logsumexp(torch.cat([logits_in, logits_g], dim=1), dim=1)
        else:
            denom = torch.logsumexp(logits_in, dim=1)

        loss = (-pos + denom).mean()
        return loss

    def train_epoch(self, epoch):
        
        for indbatch, (images_v1, images_v2, heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, num_faces, gt_boxes, keep_mask) in enumerate(self.train_loader):

            self.iter += 1

            batch_objs = int(num_faces.sum().item()) if torch.is_tensor(num_faces) else sum(num_faces)
            batch_size = self.batch_size

            if batch_objs == 0:
                batch_objs = 1

            heatmap_gt          = heatmap_gt.to(self.gpu_master)
            heatmap_posweight   = heatmap_posweight.to(self.gpu_master)
            keep_mask           = keep_mask.to(self.gpu_master)
            reg_tlrb            = reg_tlrb.to(self.gpu_master)
            reg_mask            = reg_mask.to(self.gpu_master)
            landmark_gt         = landmark_gt.to(self.gpu_master)
            landmark_mask       = landmark_mask.to(self.gpu_master)
            images_v1           = images_v1.to(self.gpu_master)
            images_v2           = images_v2.to(self.gpu_master)
            gt_boxes            = gt_boxes.to(self.gpu_master)

            # Forward two views in one pass for efficiency
            images_cat = torch.cat([images_v1, images_v2], dim=0)
            outs = self.model(images_cat)

            # unpack outputs (has_landmark=True):
            # center, box, landmark, ghost, feat
            hm_cat, tlrb_cat, landmark_cat, ghost_cat, feat_cat = outs

            bsz = images_v1.shape[0]
            hm, hm2 = hm_cat[:bsz], hm_cat[bsz:]
            tlrb, _ = tlrb_cat[:bsz], tlrb_cat[bsz:]
            landmark, _ = landmark_cat[:bsz], landmark_cat[bsz:]
            ghost, ghost2 = ghost_cat[:bsz], ghost_cat[bsz:]
            feat, feat2 = feat_cat[:bsz], feat_cat[bsz:]

            hm = hm.sigmoid()
            ghost_sig = ghost.sigmoid()
            hm = torch.clamp(hm, min=1e-4, max=1-1e-4)
            tlrb = torch.exp(tlrb)

            hm_loss = self.focal_loss(hm, heatmap_gt, heatmap_posweight, keep_mask=keep_mask) / batch_objs
            reg_loss = self.giou_loss(tlrb, reg_tlrb, reg_mask) * 5
            landmark_loss = self.landmark_loss(landmark, landmark_gt, landmark_mask) * 0.1

            # ---- Ghost-Heatmap supervision (online pseudo-label) ----
            # face region mask (dilated) to avoid mining around real faces
            face_mask = (heatmap_gt > 0).to(hm.dtype)
            face_mask = self._dilate_mask(face_mask, k=9)

            ghost_target = self._build_ghost_targets(hm.detach(), face_mask, topk=self.ghost_topk, thr=0.30, sigma=2.0)
            ghost_sig = torch.clamp(ghost_sig, min=1e-4, max=1-1e-4)
            ghost_loss = F.binary_cross_entropy(ghost_sig, ghost_target)

            # ---- Ghost-Guided Hard-Negative Contrastive Learning ----
            # Build ROIs from padded gt_boxes (same geometry for both views)
            stride = 4.0
            rois = []
            for bi in range(images_v1.shape[0]):
                nf = int(num_faces[bi].item()) if torch.is_tensor(num_faces) else int(num_faces[bi])
                if nf <= 0:
                    continue
                b = gt_boxes[bi, :nf]  # [nf,4] xyxy in input space
                # filter invalid
                b = b[b[:, 0] >= 0]
                if b.numel() == 0:
                    continue
                bi_col = torch.full((b.shape[0], 1), float(bi), device=b.device, dtype=b.dtype)
                rois.append(torch.cat([bi_col, b], dim=1))
            if len(rois) > 0:
                rois = torch.cat(rois, dim=0)  # [N,5] with batch idx
                # ROIAlign expects coords in input scale; we set spatial_scale=1/stride for feat map
                roi_f1 = roi_align(feat, rois, output_size=(1, 1), spatial_scale=1.0/stride, aligned=True)  # [N,C,1,1]
                roi_f2 = roi_align(feat2, rois, output_size=(1, 1), spatial_scale=1.0/stride, aligned=True)

                z1 = roi_f1.flatten(1)
                z2 = roi_f2.flatten(1)

                # sample ghost negatives from view1 feature + ghost map
                neg_f, neg_w = self._sample_ghost_negatives(feat, ghost_sig.detach(), face_mask, topk=self.neg_topk)

                # lightweight projection head (on-the-fly)
                # NOTE: if you want to save/restore it, move it into DBFace and add to optimizer
                proj_dim = 128
                if not hasattr(self, "_proj"):
                    self._proj = nn.Sequential(
                        nn.Linear(z1.shape[1], z1.shape[1], bias=False),
                        # nn.BatchNorm1d(z1.shape[1]),
                        nn.LayerNorm(z1.shape[1]),
                        nn.ReLU(inplace=True),
                        nn.Linear(z1.shape[1], proj_dim, bias=True),
                    ).to(z1.device)
                    self.optimizer.add_param_group({"params": self._proj.parameters(), "lr": self.lr})

                z1p = self._proj(z1)
                z2p = self._proj(z2)
                neg_z = self._proj(neg_f) if neg_f.numel() > 0 else neg_f

                cl_loss = self._gghn_infonce(z1p, z2p, neg_z, neg_w, tau=self.cl_tau, alpha=5.0)
            else:
                cl_loss = torch.tensor(0.0, device=hm.device, dtype=hm.dtype)

            loss = hm_loss + reg_loss + landmark_loss + self.lambda_ghost * ghost_loss + self.lambda_cl * cl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_flt = epoch + indbatch / self.per_epoch_batchs

            if indbatch % 10 == 0:
                log.info(
                    f"iter: {self.iter}, lr: {self.lr:g}, epoch: {epoch_flt:.2f}, loss: {loss.item():.2f}, hm_loss: {hm_loss.item():.2f}, "
                    f"box_loss: {reg_loss.item():.2f}, lmdk_loss: {landmark_loss.item():.5f}, ghost_loss: {ghost_loss.item():.4f}, cl_loss: {cl_loss.item():.4f}"
                )

            if indbatch % 1000 == 0:
                log.info("save hm")
                hm_image = hm[0, 0].cpu().data.numpy()
                common.imwrite(f"{jobdir}/imgs/hm_image.jpg", hm_image * 255)
                common.imwrite(f"{jobdir}/imgs/hm_image_gt.jpg", heatmap_gt[0, 0].cpu().data.numpy() * 255)

                # Reconstruct a uint8 visualization image from normalized tensors.
                # NOTE: current input is 2-channel (I, |∇I|). OpenCV imencode only supports 1/3/4 channels,
                # so we visualize the intensity channel as a 3-channel image for bbox drawing/saving.
                vis = (images_v1[0].permute(1, 2, 0).cpu().data.numpy() * np.array(self.std, dtype=np.float32) + np.array(self.mean, dtype=np.float32))
                if vis.ndim == 3 and vis.shape[2] == 2:
                    I = np.clip(vis[:, :, 0] * 255.0, 0, 255).astype(np.uint8)  # [H,W]
                    image = np.stack([I, I, I], axis=-1)  # [H,W,3]
                else:
                    image = np.clip(vis * 255.0, 0, 255).astype(np.uint8)
                outobjs = eval_tool.detect_images_giou_with_netout(hm, tlrb, landmark, threshold=0.1, ibatch=0)

                im1 = image.copy()
                for obj in outobjs:
                    common.drawbbox(im1, obj)
                common.imwrite(f"{jobdir}/imgs/train_result.jpg", im1)



    def train(self):

        lr_scheduer = {
            1: 1e-3,
            2: 2e-3,
            3: 1e-3,
            60: 1e-4,
            120: 1e-5
        }

        # webface
        self.model.train()
        for epoch in range(self.epochs):

            if epoch in lr_scheduer:
                self.set_lr(lr_scheduer[epoch])

            self.train_epoch(epoch)
            file = f"{jobdir}/models/{epoch + 1}.pth"
            common.mkdirs_from_file_path(file)
            torch.save(self.model.module.state_dict(), file)


trial_name = "small-H-dense-wide64-UCBA-keep12-ignoresmall"
jobdir = f"jobs/{trial_name}"

log = logger.create(trial_name, f"{jobdir}/logs/{trial_name}.log")
# app = App("webface/label_replaced.txt", "")
app = App('label_list','')
app.train()