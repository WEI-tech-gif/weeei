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
IMG_SIZE = 640
IOU_THRESHOLD = 0.3  # 选 raw pred 中 best match anchor 的 IoU 阈值

CLASS2IMG = {
    "Ants":          "valid/images/ants-332-_jpg.rf.91ec27962da84acbdb6f13920275d913.jpg",
    "Bees":          "valid/images/bees-36-_jpg.rf.9cbaf3ef3f5c06bf4dee596c17be194a.jpg",
    "Beetles":       "valid/images/beetle-53-_jpg.rf.d3bc80a8ef757c517cf647d5d9fccb16.jpg",
    "Caterpillars":  "valid/images/catterpillar-111-_jpg.rf.63af6b68e9d97d3d014f3677a9cbd4eb.jpg",
    "Earthworms":    "valid/images/earthworms-132-_jpg.rf.edcb9ed1f6204188616440b567cf9409.jpg",
    "Earwigs":       "valid/images/earwig-146-_jpg.rf.4abdf59f83b367d578302a024759e6cf.jpg",
    "Grasshoppers":  "valid/images/grasshopper-118-_jpg.rf.ff4538cd01423433f8df21d5101d8ca2.jpg",
    "Moths":         "valid/images/moth-58-_jpg.rf.0560db8fa8c10372ba164f011eef33ef.jpg",
    "Slugs":         "valid/images/slug-256-_jpg.rf.dfb459d23051f491d7d444af6417a986.jpg",
    "Snails":        "valid/images/snail-47-_jpg.rf.2d501ea127e9dca115fe402e9b2ccfc0.jpg",
    "Wasps":         "valid/images/wasps-example.jpg",
    "Weevils":       "valid/images/weevils-example.jpg",
}

CLASS_NAMES = list(CLASS2IMG.keys())

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


def map_names(names_obj):
    if isinstance(names_obj, dict):
        return {int(k): v for k, v in names_obj.items()}
    else:
        return {i: n for i, n in enumerate(names_obj)}


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
    执行 backbone+neck，到 Detect 时手动解码：
      - 使用 cv2/cv3 + DFL + make_anchors + dist2bbox
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

                    assert len(feats) == nl, "Number of input features != nl"

                    # head：cv2/cv3
                    p = []
                    for j in range(nl):
                        feat = feats[j]
                        pj = torch.cat((det.cv2[j](feat), det.cv3[j](feat)), dim=1)
                        p.append(pj)  # [B,no,Hj,Wj]

                    bs = p[0].shape[0]

                    # flatten concat -> [B,no,K]
                    x_cat = torch.cat(
                        [pi.view(bs, no, -1) for pi in p],
                        dim=2
                    )  # [B,no,K]

                    # 拆 box logits / cls logits
                    box_logits = x_cat[:, : reg_max * 4, :]   # [B,4*reg_max,K]
                    cls_logits = x_cat[:, reg_max * 4 :, :]   # [B,nc,K]

                    # anchors / strides
                    anchors, strides = make_anchors(p, det.stride, 0.5)
                    anchors = anchors.transpose(0, 1).contiguous().view(2, -1).to(x_cat.device)
                    strides = strides.transpose(0, 1).contiguous().view(1, -1).to(x_cat.device)

                    # DFL + dist2bbox -> xywh
                    box_dist = det.dfl(box_logits)  # [B,4,K]
                    box_xywh = dist2bbox(
                        box_dist,
                        anchors.unsqueeze(0),
                        xywh=True,
                        dim=1,
                    )  # [B,4,K]
                    box_xywh = box_xywh * strides   # [B,4,K]

                    # xywh -> xyxy
                    x_c = box_xywh[:, 0]
                    y_c = box_xywh[:, 1]
                    w = box_xywh[:, 2]
                    h = box_xywh[:, 3]
                    x1 = x_c - w / 2
                    y1 = y_c - h / 2
                    x2 = x_c + w / 2
                    y2 = y_c + h / 2
                    box_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [B,4,K]

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


def get_target_layer_and_stride(cam_model: YoloRawPredModel):
    det = cam_model.detect
    f = det.f
    if not isinstance(f, (list, tuple)):
        f = [f]
    stride_tensor = det.stride  # [nl]
    hi_scale_idx = int(torch.argmin(stride_tensor).item())  # 最小 stride
    hi_stride = float(stride_tensor[hi_scale_idx].item())
    hi_layer_idx = int(f[hi_scale_idx])
    target_layer = cam_model.model[hi_layer_idx]
    return target_layer, hi_stride


# ================== 单类别 CAM 生成（返回该类图像） ==================
def run_cam_for_class(
    class_name,
    class_idx,
    img_path,
    yolo_res,
    cam_model,
    target_layer,
    hi_stride,
    img_float,
    base_inp,
    img_size,
    iou_threshold,
    device,
):
    boxes = yolo_res.boxes
    if boxes is None or boxes.xyxy.numel() == 0:
        print(f"[Skip] {class_name}: no detections at all in this image.")
        return None

    xyxy = boxes.xyxy.cpu()
    cls = boxes.cls.cpu().long()
    conf = boxes.conf.cpu()

    mask = cls == class_idx
    if not mask.any():
        print(f"[Skip] No '{class_name}' detections in {img_path}.")
        return None

    xyxy_cls = xyxy[mask]
    conf_cls = conf[mask]

    best_idx = int(conf_cls.argmax())
    best_box = xyxy_cls[best_idx].tolist()
    best_conf = float(conf_cls[best_idx])
    print(f"[{class_name}] {img_path} best det conf={best_conf:.4f}, xyxy(orig)={best_box}")

    # 映射到 640x640
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    sx = img_size / w0
    sy = img_size / h0
    x1, y1, x2, y2 = best_box
    best_box_640 = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

    # hooks
    activations, gradients = {}, {}

    def fwd_hook(module, input, output):
        activations["value"] = output

    def bwd_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    try:
        h2 = target_layer.register_full_backward_hook(bwd_hook)
    except AttributeError:
        h2 = target_layer.register_backward_hook(bwd_hook)

    # 为该类别构造输入
    inp = base_inp.clone().detach().to(device).requires_grad_(True)

    cam_model.zero_grad()
    preds = cam_model(inp)            # [1,K,4+nc]
    preds0 = preds[0]                 # [K,4+nc]
    pred_boxes = preds0[:, :4]        # [K,4]
    cls_logits = preds0[:, 4:]        # [K,nc]

    best_box_t = torch.tensor(best_box_640, dtype=torch.float32, device=device)

    # 选 best anchor（IoU * logit）
    ious = box_iou_xyxy(pred_boxes, best_box_t)
    cls_for_c = cls_logits[:, class_idx]
    scores = ious * torch.relu(cls_for_c)
    if iou_threshold > 0:
        scores = torch.where(
            ious >= iou_threshold,
            scores,
            torch.full_like(scores, -1e9),
        )
    best_k = int(scores.argmax().item())
    if scores.max() < -1e8:
        best_k = int(cls_for_c.argmax().item())

    # IoU 作为定位项
    def iou_single(b, g):
        x1 = torch.maximum(b[0], g[0]); y1 = torch.maximum(b[1], g[1])
        x2 = torch.minimum(b[2], g[2]); y2 = torch.minimum(b[3], g[3])
        inter = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
        area_b = (b[2]-b[0]).clamp(min=0)*(b[3]-b[1]).clamp(min=0)
        area_g = (g[2]-g[0]).clamp(min=0)*(g[3]-g[1]).clamp(min=0)
        return inter / (area_b + area_g - inter + 1e-6)

    loc_iou = iou_single(pred_boxes[best_k], best_box_t)
    lambda_loc = 0.6
    target = cls_for_c[best_k] + lambda_loc * loc_iou

    cam_model.zero_grad()
    if inp.grad is not None:
        inp.grad.zero_()
    target.backward(retain_graph=False)

    # 取消 hooks
    h1.remove()
    h2.remove()
    if "value" not in activations or "value" not in gradients:
        print(f"[Warn] Hooks failed for class {class_name}")
        return None

    A = activations["value"][0]   # [C,H,W]
    dA = gradients["value"][0]    # [C,H,W]
    C, H, W = A.shape

    # bbox -> 特征图坐标
    bx1, by1, bx2, by2 = best_box_640
    sx1, sx2 = bx1 / hi_stride, bx2 / hi_stride
    sy1, sy2 = by1 / hi_stride, by2 / hi_stride

    # 正梯度平均 + 中心格点强化
    shrink = 0.5
    cx = 0.5 * (bx1 + bx2)
    cy = 0.5 * (by1 + by2)
    bw = (bx2 - bx1) * shrink
    bh = (by2 - by1) * shrink

    rsx1 = int(max(0, min(W - 1, (cx - bw / 2) / hi_stride)))
    rsx2 = int(max(0, min(W - 1, (cx + bw / 2) / hi_stride)))
    rsy1 = int(max(0, min(H - 1, (cy - bh / 2) / hi_stride)))
    rsy2 = int(max(0, min(H - 1, (cy + bh / 2) / hi_stride)))
    if rsx2 <= rsx1:
        rsx2 = min(W - 1, rsx1 + 1)
    if rsy2 <= rsy1:
        rsy2 = min(H - 1, rsy1 + 1)

    roi_grad = dA[:, rsy1:rsy2 + 1, rsx1:rsx2 + 1]
    if roi_grad.numel() == 0:
        roi_grad = dA

    roi_grad_pos = torch.relu(roi_grad)
    w_roi = roi_grad_pos.mean(dim=(1, 2))        # [C]

    cx_f = int(((bx1 + bx2) / 2) / hi_stride)
    cy_f = int(((by1 + by2) / 2) / hi_stride)
    cx_f = max(0, min(W - 1, cx_f))
    cy_f = max(0, min(H - 1, cy_f))
    w_center = torch.relu(dA[:, cy_f, cx_f])     # [C]

    alpha_center = 0.3
    weights = (1 - alpha_center) * w_roi + alpha_center * w_center  # [C]

    cam = (weights.view(-1, 1, 1) * A).sum(dim=0)  # [H,W]
    cam = torch.relu(cam)

    # 椭圆高斯软 mask
    cx_box = 0.5 * (sx1 + sx2)
    cy_box = 0.5 * (sy1 + sy2)
    ax = max((sx2 - sx1) * shrink / 2.0, 1.0)
    ay = max((sy2 - sy1) * shrink / 2.0, 1.0)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=cam.device),
        torch.arange(W, device=cam.device),
        indexing="ij",
    )
    ell = ((xs - cx_box) / ax) ** 2 + ((ys - cy_box) / ay) ** 2
    soft_mask = torch.exp(-0.5 * ell)

    box_mask = ((xs >= sx1) & (xs <= sx2) &
                (ys >= sy1) & (ys <= sy2)).float()
    soft_mask = soft_mask * box_mask
    soft_mask = soft_mask / (soft_mask.max() + 1e-6)

    cam = cam * soft_mask

    # 归一化 + resize
    cam_min, cam_max = cam.min(), cam.max()
    if (cam_max - cam_min) > 1e-6:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam_np = cam.detach().cpu().numpy()
    cam_np = cv2.resize(cam_np, (img_size, img_size))

    # 叠加可视化（RGB）
    cam_image = show_cam_on_image(img_float, cam_np, use_rgb=True)

    # 画框 + 类别标题
    x1_i, y1_i, x2_i, y2_i = map(int, best_box_640)
    cv2.rectangle(cam_image, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
    cv2.putText(
        cam_image,
        class_name,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return cam_image.astype(np.uint8)


# ================== 主流程：每类一张图 -> 12 宫格 ==================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) 初始化 YOLO（共享权重）
    yolo = YOLO(WEIGHTS)
    det_model = yolo.model.to(device).eval()

    # 2) RawPred 模型 & target layer（共享一次）
    cam_model = YoloRawPredModel(det_model).to(device).eval()
    target_layer, hi_stride = get_target_layer_and_stride(cam_model)
    print(f"Target layer: {target_layer.__class__.__name__}, stride={hi_stride}")

    # 3) 类别映射
    id2name = map_names(getattr(det_model, "names", getattr(yolo, "names", {})))
    name2id = {v.lower(): k for k, v in id2name.items()}
    print("Model classes:", id2name)

    grid_imgs = []

    # 4) 对每个类别用自己的样例图跑 CAM
    for cname in CLASS_NAMES:
        img_path = CLASS2IMG.get(cname)
        if not img_path:
            print(f"[Skip] No image path configured for {cname}")
            sub = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 40
            cv2.putText(sub, f"{cname}: NoPath", (20, IMG_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            grid_imgs.append(sub)
            continue

        cid = name2id.get(cname.lower())
        if cid is None:
            print(f"[Skip] Class '{cname}' not in model label set.")
            sub = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 40
            cv2.putText(sub, f"{cname}: N/A", (20, IMG_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            grid_imgs.append(sub)
            continue

        # YOLO 检测（该类样例图）
        with torch.no_grad():
            yolo_res = yolo(img_path, imgsz=IMG_SIZE, verbose=False)[0]

        # 预处理图像
        try:
            _, img_float, base_inp = load_image(img_path, IMG_SIZE)
        except AssertionError as e:
            print(f"[Skip] {cname}: {e}")
            sub = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 40
            cv2.putText(sub, f"{cname}: ReadFail", (20, IMG_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            grid_imgs.append(sub)
            continue

        cam_img = run_cam_for_class(
            class_name=cname,
            class_idx=cid,
            img_path=img_path,
            yolo_res=yolo_res,
            cam_model=cam_model,
            target_layer=target_layer,
            hi_stride=hi_stride,
            img_float=img_float,
            base_inp=base_inp,
            img_size=IMG_SIZE,
            iou_threshold=IOU_THRESHOLD,
            device=device,
        )

        if cam_img is None:
            sub = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 40
            cv2.putText(sub, f"No {cname}", (20, IMG_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            grid_imgs.append(sub)
        else:
            grid_imgs.append(cam_img)

    # 5) 拼 3x4 宫格
    assert len(grid_imgs) == len(CLASS_NAMES)
    rows, cols = 3, 4
    H, W = IMG_SIZE, IMG_SIZE
    grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)

    for idx, sub in enumerate(grid_imgs):
        r = idx // cols
        c = idx % cols
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = sub

    out_path = "yolo11_gradcam_12classes_grid_perclass.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[Saved] 12-class Grad-CAM grid -> {out_path}")


if __name__ == "__main__":
    main()
