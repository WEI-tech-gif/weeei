

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors, dist2bbox
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

# ================== 配置 ==================
WEIGHTS = "runs_agropest12/baseline_yolo11n2/weights/best.pt"
IMAGE_PATH = "valid/images/ants-19-_jpg.rf.f2bc23a96f199526bd21d3614f6a0431.jpg"
IMG_SIZE = 640
TARGET_CLASS_NAME = "Ants"
IOU_THRESHOLD = 0.3  # IoU>阈值的预测参与加权


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

    # 映射到 640x640（通用写法，哪怕现在就是 640 图）
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    sx = img_size / w0
    sy = img_size / h0
    x1, y1, x2, y2 = best_box
    best_box_640 = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    print(f"Best {target_class_name} box xyxy(640x640)={best_box_640}")

    return best_box_640, ants_cls

class YoloRawPredModel(nn.Module):
    """
    使用 DetectionModel 的骨干 + Detect 内部子模块（cv2/cv3/dfl）手写 decode：
      - 不调用 Detect.forward/_inference（避免 smart_inference_mode/no_grad）
      - 输出 [B, K, 4+nc]，前4维为 xyxy，其后是每类概率（sigmoid 后）
    """

    def __init__(self, det_model: nn.Module):
        super().__init__()
        # 直接复用内部 ModuleList
        self.model = det_model.model

        detect_idx = None
        detect_module = None
        for i, m in enumerate(self.model):
            if isinstance(m, Detect):
                detect_idx = i
                detect_module = m
                break
        assert detect_idx is not None, "Detect layer not found in model."

        self.detect_idx = detect_idx
        self.detect = detect_module

    def forward(self, x):
        # ★ 确保输入在计算图里
        x = x.requires_grad_(True)

        outputs = []

        # ★ 强制开启梯度，防止外层环境(no_grad / inference_mode)污染
        with torch.enable_grad():
            for i, m in enumerate(self.model):
                if i == self.detect_idx:
                    det = self.detect
                    f = det.f
                    if not isinstance(f, (list, tuple)):
                        f = [f]

                    # Detect 的多尺度输入特征
                    feats = [outputs[j] for j in f]
                    nl = det.nl
                    no = det.no
                    reg_max = det.reg_max
                    nc = det.nc

                    assert len(feats) == nl, "Number of input features != nl"

                    # 1) 头部：cv2/cv3
                    p = []
                    for j in range(nl):
                        feat = feats[j]
                        pj = torch.cat((det.cv2[j](feat), det.cv3[j](feat)), dim=1)
                        p.append(pj)  # [B,no,Hj,Wj]

                    bs = p[0].shape[0]

                    # 2) flatten + concat -> [B,no,K]
                    x_cat = torch.cat(
                        [pi.view(bs, no, -1) for pi in p],
                        dim=2
                    )  # [B,no,K]

                    # 3) 拆 box / cls
                    box_logits = x_cat[:, : reg_max * 4, :]   # [B,4*reg_max,K]
                    cls_logits = x_cat[:, reg_max * 4 :, :]   # [B,nc,K]

                    # 4) anchors / strides（按你上一版已验证可用的写法）
                    anchors, strides = make_anchors(p, det.stride, 0.5)
                    # anchors: [nl,2,HW] -> [2,K]
                    anchors = anchors.transpose(0, 1).contiguous().view(2, -1)
                    # strides: [nl,1,HW] -> [1,K]
                    strides = strides.transpose(0, 1).contiguous().view(1, -1)

                    anchors = anchors.to(x_cat.device)       # [2,K]
                    strides = strides.to(x_cat.device)       # [1,K]

                    # 5) DFL + dist2bbox -> xywh
                    box_dist = det.dfl(box_logits)           # [B,4,K]
                    box_xywh = dist2bbox(
                        box_dist,
                        anchors.unsqueeze(0),                # [1,2,K]
                        xywh=True,
                        dim=1,
                    )                                       # [B,4,K]
                    box_xywh = box_xywh * strides           # [B,4,K]

                    # 6) xywh -> xyxy
                    x_c = box_xywh[:, 0]
                    y_c = box_xywh[:, 1]
                    w = box_xywh[:, 2]
                    h = box_xywh[:, 3]
                    x1 = x_c - w / 2
                    y1 = y_c - h / 2
                    x2 = x_c + w / 2
                    y2 = y_c + h / 2
                    box_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [B,4,K]

                    # 7) cls 概率
                    cls_scores = cls_logits.sigmoid()        # [B,nc,K]

                    # 8) 拼 [B,K,4+nc]
                    pred = torch.cat((box_xyxy, cls_scores), dim=1)  # [B,4+nc,K]
                    pred = pred.permute(0, 2, 1).contiguous()        # [B,K,4+nc]

                    # ★ 非常关键：这里的 pred 现在有 grad_fn
                    return pred

                # 非 Detect 层：按 m.f 取输入
                f = getattr(m, "f", -1)
                if isinstance(f, int):
                    if f == -1:
                        x_in = x
                    else:
                        x_in = outputs[f]
                else:
                    x_in = [x if j == -1 else outputs[j] for j in f]

                x = m(x_in)
                outputs.append(x)

        raise RuntimeError("Reached end of model without hitting Detect.")
class YoloBestMatchTarget:
    """
    针对 YoloRawPredModel 输出的 [1,K,4+nc]:
      - 计算每个预测框与 best_box 的 IoU
      - 用 IoU * cls_score 选出“最匹配”的那个预测
      - 只对这个位置的该类分数做 Grad-CAM
    这样热力图会紧贴那个检测框，而不是整片区域。
    """

    def __init__(self, best_box_xyxy, class_index: int, iou_threshold: float = 0.3):
        self.best_box = torch.tensor(best_box_xyxy, dtype=torch.float32)
        self.class_index = int(class_index)
        self.iou_thr = float(iou_threshold)

    def __call__(self, model_output):
        # model_output: [1,K,4+nc] 或 [K,4+nc]
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]

        if model_output.ndim == 3:
            pred = model_output[0]  # [K,4+nc]
        else:
            pred = model_output     # [K,4+nc]

        box = self.best_box.to(pred.device)

        boxes = pred[:, :4]                            # [K,4]
        cls_scores = pred[:, 4 + self.class_index]     # [K]

        # IoU(best_box, boxes[i])
        x1 = torch.maximum(boxes[:, 0], box[0])
        y1 = torch.maximum(boxes[:, 1], box[1])
        x2 = torch.minimum(boxes[:, 2], box[2])
        y2 = torch.minimum(boxes[:, 3], box[3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        area2 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        iou = inter / (area1 + area2 - inter + 1e-6)

        # 综合考虑 IoU 和类别分数，选最“像” best_box 的预测点
        score = cls_scores * torch.clamp(iou, min=0.0)

        # 如果我们希望略微过滤掉胡乱点，可以加一个 IoU 阈值：
        if self.iou_thr > 0:
            score = torch.where(iou >= self.iou_thr, score, torch.zeros_like(score))

        # 找 score 最大的那个 index
        best_idx = int(score.argmax().item())

        # 防止全为 0 的极端情况：如果 score 全 0，就退化为纯 cls_scores 最大
        if score.max() <= 0:
            best_idx = int(cls_scores.argmax().item())

        # ★ Grad-CAM 的 loss：只对这个预测点的该类分数求梯度
        loss = cls_scores[best_idx]
        return loss



# ================== 主流程 ==================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) 正常 YOLO 推理，拿 best Ants 框（640x640）
    yolo = YOLO(WEIGHTS)
    best_box_640, ants_cls = get_best_ants_box_640(
        yolo_model=yolo,
        img_path=IMAGE_PATH,
        img_size=IMG_SIZE,
        target_class_name=TARGET_CLASS_NAME,
    )

    # 2) DetectionModel
    det_model = yolo.model.to(device).eval()

    # 3) Raw Pred 模型（完全可微）
    cam_model = YoloRawPredModel(det_model).to(device).eval()

    # 4) 选 Detect 使用的最高分辨率特征层作为 Grad-CAM target_layer
    detect_layer = cam_model.detect
    f = detect_layer.f
    if not isinstance(f, (list, tuple)):
        f = [f]

    stride = detect_layer.stride  # [nl]
    hi_scale_idx = int(torch.argmin(stride).item())  # 最小 stride -> 最高分辨率
    hi_layer_idx = int(f[hi_scale_idx])
    target_layer = cam_model.model[hi_layer_idx]

    print(
        f"Target layer index={hi_layer_idx}, "
        f"stride={int(stride[hi_scale_idx])}, "
        f"name={target_layer.__class__.__name__}"
    )

    # 5) 准备输入
    img_rgb, img_float, inp = load_image(IMAGE_PATH, IMG_SIZE)
    inp = inp.to(device)

    # 6) Grad-CAM
    cam = GradCAM(
        model=cam_model,
        target_layers=[target_layer],
    )


    targets = [YoloBestMatchTarget(best_box_640, ants_cls, IOU_THRESHOLD)]


    grayscale_cam = cam(
        input_tensor=inp,
        targets=targets
    )[0]  # [H,W]

    grayscale_cam = scale_cam_image(grayscale_cam)

    # 7) 可视化 + 画出 best_box
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    x1, y1, x2, y2 = map(int, best_box_640)
    cv2.rectangle(cam_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_path = "yolo11_gradcam_ants_iou_highres_fixed.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM to {out_path}")


if __name__ == "__main__":
    main()
