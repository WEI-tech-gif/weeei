import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors, dist2bbox

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

# ================== 基本配置 ==================
WEIGHTS = "runs_agropest12/baseline_yolo11n2/weights/best.pt"
IMAGE_PATH = "valid/images/ants-8-_jpg.rf.d164ba1daadd8148414cdc98438576a1.jpg"
IMG_SIZE = 640
TARGET_CLASS_NAME = "Ants"
IOU_MATCH_THRESHOLD = 0.5  # raw预测与best box的IoU下限


# ================== 工具函数 ==================
def load_image(path, size):
    img_bgr = cv2.imread(path)
    assert img_bgr is not None, f"Cannot read image: {path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size))
    img_float = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).float()
    return img_rgb, img_float, tensor


def box_iou_xyxy(boxes1, box2):
    """
    boxes1: [N,4], box2: [4]  -> IoU: [N]
    """
    x1 = torch.maximum(boxes1[:, 0], box2[0])
    y1 = torch.maximum(boxes1[:, 1], box2[1])
    x2 = torch.minimum(boxes1[:, 2], box2[2])
    y2 = torch.minimum(boxes1[:, 3], box2[3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (box2[2] - box2[0]).clamp(min=0) * (box2[3] - box2[1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-6
    return inter / union


# =============== 从 Detect 训练输出手动 decode（可导版本） ===============
def decode_from_train_outputs(det: Detect, feats_list):
    """
    det: ultralytics.nn.modules.head.Detect 实例
    feats_list: Detect 训练分支输出的 list[Tensor], 每个 [B, no, H, W]
    返回: preds_decoded: [B, N, 4 + nc]，最后维度 [x,y,w,h, cls_probs...]
    （注意：这里没有 obj，按 YOLOv10/11 head 结构来）
    """
    assert isinstance(feats_list, (list, tuple)), f"Expected list, got {type(feats_list)}"
    bs = feats_list[0].shape[0]
    no = det.no  # reg_max*4 + nc

    # 展平各层: [B, no, H, W] -> [B, no, H*W] 然后在锚点维拼起来
    x_cat = torch.cat([f.view(bs, no, -1) for f in feats_list], dim=2)  # [B, no, N]

    # anchors & strides（如果还没建，就用 make_anchors 建一下）
    if det.anchors is None or det.anchors.numel() == 0 or det.strides is None or det.strides.numel() == 0:
        # det.stride: [nl]，标准 YOLO11n 是 [8,16,32]，训练时已写进模型，一般这里是有值的
        anchors, strides = make_anchors(feats_list, det.stride, 0.5)
        # make_anchors 返回 [2,N] 之类的格式，这里和官方 Detect._inference 对齐
        det.anchors, det.strides = anchors.transpose(0, 1), strides.transpose(0, 1)

    box, cls = x_cat.split((det.reg_max * 4, det.nc), dim=1)  # box:[B,4*reg_max,N], cls:[B,nc,N]

    # DFL -> 距离分布转实际距离，得到 [B,4,N]
    box = det.dfl(box)

    # 距离 -> bbox (xywh 或 xyxy)，返回 [B,4,N]
    dbox = det.decode_bboxes(box, det.anchors.unsqueeze(0)) * det.strides  # 与官方一致

    # 拼回 [B,4+nc,N]，然后转成 [B,N,4+nc]
    preds = torch.cat((dbox, cls.sigmoid()), dim=1)  # [B,4+nc,N]
    preds = preds.permute(0, 2, 1).contiguous()      # [B,N,4+nc]

    return preds


# ================== 手写可导 YOLO 前向图 ==================
class YoloCAMModel(nn.Module):
    """
    完整复制 ultralytics BaseModel._forward_once 的图结构，
    但：
    - 不调用 _predict_once / _inference / smart_inference_mode
    - Detect 直接走“训练分支”：cv2+cv3 拼接，不做 NMS，保持可导

    forward 返回 Detect 训练分支输出 list[Tensor]，供 decode_from_train_outputs 使用。
    """
    def __init__(self, det_model):
        super().__init__()
        self.model = det_model.model  # nn.ModuleList
        self.detect = None
        for m in self.model:
            if isinstance(m, Detect):
                self.detect = m
        assert self.detect is not None, "Detect head not found in model."

    def forward(self, x):
        outputs = []
        for m in self.model:
            # 处理 from 属性（拓扑连接）
            f = getattr(m, "f", -1)

            if isinstance(f, int):
                if f == -1:
                    x_in = x
                else:
                    x_in = outputs[f]
            else:
                # 多输入，例如 concat，或 Detect 接收多个层
                x_in = [x if j == -1 else outputs[j] for j in f]

            if isinstance(m, Detect):
                # **强制使用训练分支逻辑**，不看 m.training，不走 _inference
                assert isinstance(x_in, (list, tuple)), "Detect expects list of feature maps"
                z = []
                for i in range(m.nl):
                    z.append(torch.cat((m.cv2[i](x_in[i]), m.cv3[i](x_in[i])), 1))  # [B,no,H,W]
                x = z  # list[Tensor]
            else:
                x = m(x_in)

            outputs.append(x)

        # 最终 x 应该是 Detect 的训练输出 list[Tensor]
        if not isinstance(x, (list, tuple)):
            raise RuntimeError(f"YoloCAMModel final output must be list, got {type(x)}")
        return x  # list[Tensor] for Detect levels


# ================== Grad-CAM Target：只针对 1 个框 ==================
class SingleBoxTarget:
    """
    只针对 decode 后第 k 个预测框、指定类别 c 的得分做目标：
    loss = cls_prob[k, c]
    """
    def __init__(self, det_head: Detect, box_index: int, class_index: int):
        self.det = det_head
        self.k = int(box_index)
        self.c = int(class_index)

    def __call__(self, model_output):
        # model_output 是 YoloCAMModel 的输出: list[Tensor]
        if not isinstance(model_output, (list, tuple)):
            raise RuntimeError(f"Unexpected model_output type: {type(model_output)}")

        preds = decode_from_train_outputs(self.det, model_output)[0]  # [N,4+nc]
        boxes = preds[:, :4]
        cls = preds[:, 4:]

        K, dim = preds.shape
        nc = cls.shape[1]

        if self.c < 0 or self.c >= nc:
            # 类别索引异常，退化全局最大
            return cls.max()

        if self.k < 0 or self.k >= K:
            # index 异常，退化成该类最大
            return cls[:, self.c].max()

        # 只对这个预测框、这个类别的分数做目标
        score = cls[self.k, self.c]
        return score


# ================== 主逻辑 ==================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 加载 YOLO 模型（用于推理 & 提取底层 det_model）
    yolo = YOLO(WEIGHTS)
    det_model = yolo.model.to(device)
    det_model.eval()  # 推理用 eval，CAM 我们自己手写 forward，不依赖 training flag

    # 2. 获取类别索引
    names = getattr(det_model, "names", getattr(yolo, "names", None))
    if isinstance(names, dict):
        id_to_name = {int(k): v for k, v in names.items()}
    else:
        id_to_name = {i: n for i, n in enumerate(names)}

    class_index = None
    for i, n in id_to_name.items():
        if str(n).lower() == TARGET_CLASS_NAME.lower():
            class_index = i
            break
    assert class_index is not None, f"Class {TARGET_CLASS_NAME} not found. Got: {id_to_name}"
    print(f"Use class_index={class_index} ({id_to_name[class_index]})")

    # 3. 读图
    img_rgb, img_float, inp = load_image(IMAGE_PATH, IMG_SIZE)
    inp = inp.to(device)

    # 4. 正常 YOLO 推理一次（带 NMS），选出 Ants 置信度最高的框
    with torch.no_grad():
        result = yolo(IMAGE_PATH, imgsz=IMG_SIZE, conf=0.001, verbose=False)[0]

    best_conf = -1.0
    best_box_xyxy = None

    if result.boxes is not None and len(result.boxes) > 0:
        b_xyxy = result.boxes.xyxy.cpu().numpy()
        b_conf = result.boxes.conf.cpu().numpy()
        b_cls = result.boxes.cls.cpu().numpy().astype(int)

        for xyxy, conf, cid in zip(b_xyxy, b_conf, b_cls):
            if cid == class_index and conf > best_conf:
                best_conf = float(conf)
                best_box_xyxy = xyxy.tolist()

    assert best_box_xyxy is not None, "No Ants detection found in this image."
    print(f"Best Ants box conf={best_conf:.4f}, xyxy={best_box_xyxy}")

    best_box_t = torch.tensor(best_box_xyxy, dtype=torch.float32, device=device)

    # 5. 构建可导 YOLO 计算图（YoloCAMModel）
    cam_model = YoloCAMModel(det_model).to(device)

    # 6. 从 cam_model 里选 Grad-CAM 的 target layer（Detect 前最后一个卷积块）
    modules = list(det_model.model)
    detect_idx = None
    for i, m in enumerate(modules):
        if isinstance(m, Detect):
            detect_idx = i
            break
    assert detect_idx is not None, "Detect layer not found."

    target_layer = None
    for m in reversed(modules[:detect_idx]):
        name = m.__class__.__name__.lower()
        if any(k in name for k in ["conv", "c2f", "c3", "bottleneck", "spp", "sppf", "cbam"]):
            target_layer = m
            break
    if target_layer is None:
        target_layer = modules[detect_idx - 1]
    print("Target layer for CAM:", target_layer.__class__.__name__)

    # 7. 用 YoloCAMModel（无梯度要求）解码一遍 raw 预测，找到与 best_box 对应的那个 index
    with torch.no_grad():
        train_feats = cam_model(inp)                          # list[Tensor], 训练分支输出
        preds_decoded = decode_from_train_outputs(cam_model.detect, train_feats)[0]  # [N,4+nc]

    boxes_all = preds_decoded[:, :4]
    cls_all = preds_decoded[:, 4:]
    assert class_index < cls_all.shape[1], "class_index out of range in decoded preds"

    ant_scores = cls_all[:, class_index]
    ious = box_iou_xyxy(boxes_all, best_box_t)

    mask = (ious >= IOU_MATCH_THRESHOLD)
    if mask.any():
        scores_masked = ant_scores.clone()
        scores_masked[~mask] = -1e9
        best_k = int(scores_masked.argmax().item())
    else:
        best_k = int(ant_scores.argmax().item())

    print(f"Matched raw pred index={best_k}, score={float(ant_scores[best_k]):.4f}, "
          f"box={boxes_all[best_k].tolist()}")

    # 8. 创建 Grad-CAM（使用我们手写的 cam_model）
    cam = GradCAM(
        model=cam_model,
        target_layers=[target_layer],
    )

    # 只针对这个预测框 + 类别的分数做目标
    targets = [SingleBoxTarget(cam_model.detect, best_k, class_index)]

    # 9. 计算 CAM
    grayscale_cam = cam(input_tensor=inp, targets=targets)[0]  # [H,W]
    grayscale_cam = scale_cam_image(grayscale_cam)

    # 10. 叠加到原图，并画出 best_box，检查是否只高亮蚂蚁
    cam_image = show_cam_on_image(
        img_float,      # 0~1
        grayscale_cam,  # 0~1
        use_rgb=True
    )

    x1, y1, x2, y2 = map(int, best_box_xyxy)
    cv2.rectangle(cam_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_path = "yolo11_gradcam_ants_single_box_strict.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM image to {out_path}")


if __name__ == "__main__":
    main()
