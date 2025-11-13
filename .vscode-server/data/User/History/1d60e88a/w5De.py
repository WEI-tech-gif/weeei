import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors, dist2bbox
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================== 配置 ==================
WEIGHTS = "runs_agropest12/baseline_yolo11n2/weights/best.pt"
IMAGE_PATH = "valid/images/ants-332-_jpg.rf.91ec27962da84acbdb6f13920275d913.jpg"
IMG_SIZE = 640
TARGET_CLASS_NAME = "Ants"
IOU_THRESHOLD = 0.3      # 选 best_k 时的 IoU 下限
SHRINK = 0.5             # ROI 收缩比例: 越小越贴身体(建议 0.3~0.5)
LAMBDA_LOC = 0.5         # 分类 vs 定位 IoU 权重


# ================== 工具函数 ==================
def load_image(path, size):
    img_bgr = cv2.imread(path)
    assert img_bgr is not None, f"Cannot read image: {path}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size))
    img_float = img_rgb.astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(img_float)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    return img_rgb, img_float, tensor


def get_best_ants_box_640(yolo_model, img_path, img_size, target_class_name):
    """用官方 YOLO 推理一次，选出 Ants 类置信度最高的框，并映射到 640x640。"""
    with torch.no_grad():
        res = yolo_model(img_path, imgsz=img_size, verbose=False)[0]

    names = res.names if hasattr(res, "names") else yolo_model.names
    if isinstance(names, dict):
        id2name = {int(k): v for k, v in names.items()}
    else:
        id2name = {i: n for i, n in enumerate(names)}

    ants_cls = None
    for i, n in id2name.items():
        if str(n).lower() == target_class_name.lower():
            ants_cls = i
            break
    assert ants_cls is not None, f"Class '{target_class_name}' not found."

    boxes = res.boxes
    assert boxes is not None and boxes.xyxy.numel() > 0, "No detections."

    xyxy = boxes.xyxy.cpu()
    cls = boxes.cls.cpu().long()
    conf = boxes.conf.cpu()

    mask = cls == ants_cls
    assert mask.any(), f"No '{target_class_name}' detections in this image."

    xyxy_ants = xyxy[mask]
    conf_ants = conf[mask]

    best_idx = int(conf_ants.argmax())
    best_box = xyxy_ants[best_idx].tolist()
    best_conf = float(conf_ants[best_idx])

    print(f"Use class_index={ants_cls} ({id2name.get(ants_cls)})")
    print(f"Best {target_class_name} box conf={best_conf:.4f}, xyxy(orig)={best_box}")

    # 映射到 640x640
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    sx = img_size / w0
    sy = img_size / h0
    x1, y1, x2, y2 = best_box
    best_box_640 = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    print(f"Best {target_class_name} box xyxy(640x640)={best_box_640}")

    return best_box_640, ants_cls


def box_iou_xyxy(boxes, box):
    """boxes: [K,4], box: [4]"""
    x1 = torch.maximum(boxes[:, 0], box[0])
    y1 = torch.maximum(boxes[:, 1], box[1])
    x2 = torch.minimum(boxes[:, 2], box[2])
    y2 = torch.minimum(boxes[:, 3], box[3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    area2 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    return inter / (area1 + area2 - inter + 1e-6)


# ================== 手写 Raw Pred 模型 ==================
class YoloRawPredModel(nn.Module):
    """
    手动执行 DetectionModel 的骨干和 neck，遇到 Detect 时：
      - 使用 cv2/cv3 + DFL + make_anchors + dist2bbox 解码
      - 返回 [B, K, 4+nc]，前4为 xyxy（输入尺度），后面为各类 logits（未 sigmoid）
    """

    def __init__(self, det_model: nn.Module):
        super().__init__()
        self.model = det_model.model  # nn.ModuleList

        detect_idx = None
        detect_module = None
        for i, m in enumerate(self.model):
            if isinstance(m, Detect):
                detect_idx = i
                detect_module = m
                break
        assert detect_idx is not None, "Detect layer not found."

        self.detect_idx = detect_idx
        self.detect = detect_module

    def forward(self, x):
        x = x.requires_grad_(True)
        outputs = []

        with torch.enable_grad():
            for i, m in enumerate(self.model):
                if i == self.detect_idx:
                    det = self.detect
                    f = det.f
                    if not isinstance(f, (list, tuple)):
                        f = [f]

                    feats = [outputs[j] for j in f]
                    nl = det.nl
                    no = det.no
                    reg_max = det.reg_max
                    nc = det.nc
                    assert len(feats) == nl

                    # 1) head：cv2/cv3
                    p = []
                    for j in range(nl):
                        feat = feats[j]
                        pj = torch.cat((det.cv2[j](feat), det.cv3[j](feat)), dim=1)
                        p.append(pj)  # [B,no,Hj,Wj]

                    bs = p[0].shape[0]

                    # 2) flatten concat -> [B,no,K]
                    x_cat = torch.cat(
                        [pi.view(bs, no, -1) for pi in p],
                        dim=2
                    )  # [B,no,K]

                    # 3) box logits / cls logits
                    box_logits = x_cat[:, : reg_max * 4, :]   # [B,4*reg_max,K]
                    cls_logits = x_cat[:, reg_max * 4 :, :]   # [B,nc,K]

                    # 4) anchors / strides
                    anchors, strides = make_anchors(p, det.stride, 0.5)
                    anchors = anchors.transpose(0, 1).contiguous()  # [2,K]
                    strides = strides.transpose(0, 1).contiguous()  # [1,K]

                    anchors = anchors.to(x_cat.device)
                    strides = strides.to(x_cat.device)

                    # 5) DFL + dist2bbox -> xywh
                    box_dist = det.dfl(box_logits)           # [B,4,K]
                    box_xywh = dist2bbox(
                        box_dist, anchors.unsqueeze(0), xywh=True, dim=1
                    )                                       # [B,4,K]
                    box_xywh = box_xywh * strides           # [B,4,K]

                    # 6) xywh -> xyxy
                    x_c, y_c, w, h = box_xywh[:, 0], box_xywh[:, 1], box_xywh[:, 2], box_xywh[:, 3]
                    x1 = x_c - w / 2
                    y1 = y_c - h / 2
                    x2 = x_c + w / 2
                    y2 = y_c + h / 2
                    box_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [B,4,K]

                    # 7) [B,K,4+nc]
                    pred = torch.cat((box_xyxy, cls_logits), dim=1)  # [B,4+nc,K]
                    pred = pred.permute(0, 2, 1).contiguous()        # [B,K,4+nc]
                    return pred

                # 非 Detect 层
                f = getattr(m, "f", -1)
                if isinstance(f, int):
                    x_in = x if f == -1 else outputs[f]
                else:
                    x_in = [x if j == -1 else outputs[j] for j in f]

                x = m(x_in)
                outputs.append(x)

        raise RuntimeError("Reached end of model without hitting Detect.")


# ================== 主流程 ==================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) NMS 得到 best_box
    yolo = YOLO(WEIGHTS)
    best_box_640, ants_cls = get_best_ants_box_640(
        yolo_model=yolo,
        img_path=IMAGE_PATH,
        img_size=IMG_SIZE,
        target_class_name=TARGET_CLASS_NAME,
    )
    best_box = torch.tensor(best_box_640, dtype=torch.float32, device=device)

    # 2) RawPred 模型
    det_model = yolo.model.to(device).eval()
    cam_model = YoloRawPredModel(det_model).to(device).eval()

    # 3) target_layer：Detect 使用的最小 stride 分支
    detect_layer = cam_model.detect
    f = detect_layer.f
    if not isinstance(f, (list, tuple)):
        f = [f]
    stride_tensor = detect_layer.stride
    hi_scale_idx = int(torch.argmin(stride_tensor).item())
    hi_stride = float(stride_tensor[hi_scale_idx].item())
    hi_layer_idx = int(f[hi_scale_idx])
    target_layer = cam_model.model[hi_layer_idx]

    print(
        f"Target layer index={hi_layer_idx}, "
        f"stride={int(hi_stride)}, "
        f"name={target_layer.__class__.__name__}"
    )

    # 4) 输入
    img_rgb, img_float, inp = load_image(IMAGE_PATH, IMG_SIZE)
    inp = inp.to(device)
    inp.requires_grad_(True)

    # 5) hooks
    activations = {}
    gradients = {}

    def fwd_hook(module, input, output):
        activations["value"] = output

    def bwd_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    try:
        h2 = target_layer.register_full_backward_hook(bwd_hook)
    except AttributeError:
        h2 = target_layer.register_backward_hook(bwd_hook)

    # 6) 前向，得到 raw preds
    cam_model.zero_grad()
    if inp.grad is not None:
        inp.grad.zero_()
    preds0 = cam_model(inp)[0]              # [K,4+nc]

    boxes = preds0[:, :4]
    cls_logits = preds0[:, 4:]
    cls_for_ants = cls_logits[:, ants_cls]

    # 7) 用 IoU * logit 选 best_k
    ious = box_iou_xyxy(boxes, best_box)
    base_scores = torch.relu(cls_for_ants)
    cand_scores = ious * base_scores
    if IOU_THRESHOLD > 0:
        cand_scores = torch.where(
            ious >= IOU_THRESHOLD,
            cand_scores,
            torch.full_like(cand_scores, -1e9),
        )
    best_k = int(cand_scores.argmax().item())
    if cand_scores.max() < -1e8:
        best_k = int(base_scores.argmax().item())

    print(f"Best match anchor idx={best_k}, "
          f"logit={float(cls_for_ants[best_k]):.4f}, "
          f"IoU={float(ious[best_k]):.4f}")

    # 8) 标量目标：该预测的类别 logit + 定位 IoU
    def iou_single(b, g):
        x1 = torch.maximum(b[0], g[0]); y1 = torch.maximum(b[1], g[1])
        x2 = torch.minimum(b[2], g[2]); y2 = torch.minimum(b[3], g[3])
        inter = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
        area_b = (b[2]-b[0]).clamp(min=0)*(b[3]-b[1]).clamp(min=0)
        area_g = (g[2]-g[0]).clamp(min=0)*(g[3]-g[1]).clamp(min=0)
        return inter / (area_b + area_g - inter + 1e-6)

    loc_iou = iou_single(boxes[best_k], best_box)
    target = cls_for_ants[best_k] + LAMBDA_LOC * loc_iou

    cam_model.zero_grad()
    if inp.grad is not None:
        inp.grad.zero_()
    target.backward(retain_graph=False)

    # 9) 取激活和梯度
    h1.remove()
    h2.remove()
    assert "value" in activations and "value" in gradients, "Hooks failed."

    A = activations["value"][0]    # [C,H,W]
    dA = gradients["value"][0]     # [C,H,W]
    C, H, W = A.shape

    # 10) 构造“框内椭圆高斯”软 ROI（关键改动）
    bx1, by1, bx2, by2 = best_box_640

    # 转到特征图坐标
    bx1_f, bx2_f = bx1 / hi_stride, bx2 / hi_stride
    by1_f, by2_f = by1 / hi_stride, by2 / hi_stride
    cx_f = 0.5 * (bx1_f + bx2_f)
    cy_f = 0.5 * (by1_f + by2_f)

    # 椭圆半轴 = 框宽/高 * SHRINK / 2
    ax = max((bx2_f - bx1_f) * SHRINK / 2.0, 1.0)
    ay = max((by2_f - by1_f) * SHRINK / 2.0, 1.0)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=A.device),
        torch.arange(W, device=A.device),
        indexing="ij",
    )

    # 椭圆距离
    ell = ((xs - cx_f) / ax) ** 2 + ((ys - cy_f) / ay) ** 2
    soft_mask = torch.exp(-0.5 * ell)  # 椭圆中心 1，边缘衰减

    # 限制在原始检测框内（避免越界到奇怪位置）
    box_mask = (
        (xs >= bx1_f) & (xs <= bx2_f) &
        (ys >= by1_f) & (ys <= by2_f)
    ).float()

    roi_mask_2d = soft_mask * box_mask
    roi_mask_2d = roi_mask_2d / (roi_mask_2d.max() + 1e-6)  # 归一化到 [0,1]
    roi_mask_3d = roi_mask_2d.unsqueeze(0)                  # [1,H,W]

    # 11) ROI 内的正梯度 -> 通道权重
    grad_pos = torch.relu(dA) * roi_mask_3d
    weights = grad_pos.mean(dim=(1, 2))                     # [C]

    # 标准 Grad-CAM，再乘软 mask 收紧到蚂蚁身体
    cam = (weights.view(-1, 1, 1) * A).sum(dim=0)           # [H,W]
    cam = torch.relu(cam)
    cam = cam * roi_mask_2d                                 # 框外/边缘抑制

    # 归一化 + resize
    cam_min, cam_max = cam.min(), cam.max()
    if (cam_max - cam_min) > 1e-6:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam_np = cam.detach().cpu().numpy()
    cam_np = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE))

    # 12) 可视化
    cam_image = show_cam_on_image(img_float, cam_np, use_rgb=True)
    x1_i, y1_i, x2_i, y2_i = map(int, best_box_640)
    cv2.rectangle(cam_image, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)

    out_path = "yolo11_gradcam_ants_box_soft_roi.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"Saved local Grad-CAM to {out_path}")


if __name__ == "__main__":
    main()
