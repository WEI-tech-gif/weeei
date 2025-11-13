# train_and_eval.py
import os
import csv
import time
import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from ultralytics import YOLO

# ====== 基本配置 ======
DATA = "data.yaml"                 # 需在 data.yaml 里配置 test 分割
DEVICE = 0
PROJECT = "runs_agropest12"
RUN_NAME = "baseline_yolo11n"      # 与你现有的一致
IMG_SIZE = 640
CONF_THRES = 0.25                  # 成功/失败样例划分时用
IOU = 0.6

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_loss_map_pr_curves_as(loss_src_dir: Path, dst_dir: Path):
    """
    将 Ultralytics 导出的曲线重命名为你需要的名字：
    - loss_curve.png
    - map_curve.png
    - pr_curve.png
    """
    # 常见文件名（不同版本可能稍有差异）
    candidates = {
        "loss_curve.png": ["results.png", "loss_curve.png"],
        "map_curve.png":  ["F1_curve.png", "mAP_curve.png", "metrics.png"],
        "pr_curve.png":   ["PR_curve.png", "pr_curve.png"]
    }
    for dst_name, src_names in candidates.items():
        for s in src_names:
            src = loss_src_dir / s
            if src.exists():
                shutil.copy(src, dst_dir / dst_name)
                break  # 找到一个就行

def copy_training_log_as(train_dir: Path, dst_dir: Path):
 
    for name in ["results.csv", "training_log.csv"]:
        p = train_dir / name
        if p.exists():
            shutil.copy(p, dst_dir / "training_log.csv")
            return

def plot_per_class_bar(values, labels, out_path: Path, title="Per-class AP"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.arange(len(values))
    plt.figure(figsize=(12, 4))
    plt.bar(idx, values)
    plt.xticks(idx, labels, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def make_mosaic(image_paths, out_path: Path, cols=6, cell_size=256, pad=8):
    """把若干图片拼成网格，总结图。"""
    images = []
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB").resize((cell_size, cell_size))
            images.append(im)
        except Exception:
            continue
    if not images:
        return
    rows = (len(images) + cols - 1) // cols
    W = cols * cell_size + (cols + 1) * pad
    H = rows * cell_size + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    x = y = pad
    for i, im in enumerate(images):
        canvas.paste(im, (x, y))
        x += cell_size + pad
        if (i + 1) % cols == 0:
            x = pad
            y += cell_size + pad
    ensure_dir(out_path.parent)
    canvas.save(out_path)

def main():
    t0 = time.time()

    # ========== 阶段 1：训练 ==========
    model = YOLO("yolo11n.pt")
    train_results = model.train(
        data=DATA,
        epochs=25,
        imgsz=IMG_SIZE,
        batch=128,
        workers=4,
        device=DEVICE,
        project=PROJECT,
        name=RUN_NAME,

        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5
    )

    run_dir = Path(PROJECT) / RUN_NAME     # Ultralytics run 目录
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    ensure_dir(run_dir)

    # 额外导出你需要的 best.pth（简单起见，拷贝一份重命名）
    if best_pt.exists():
        shutil.copy(best_pt, run_dir / "best.pth")  # ➜ best.pth
    else:
        print("WARNING: best.pt 未找到。")

    # 复制/重命名训练日志 & 曲线到 run 根目录
    copy_training_log_as(run_dir, run_dir)                  # ➜ training_log.csv
    save_loss_map_pr_curves_as(run_dir, run_dir)            # ➜ loss_curve.png / map_curve.png / pr_curve.png

    # ========== 阶段 2：测试集评估 ==========
    # 使用 best 权重进行 test split 验证；自动输出混淆矩阵、PR 等图
    model = YOLO(str(best_pt if best_pt.exists() else model.ckpt_path))
    test_results = model.val(
        data=DATA,
        split="test",          # 关键：在 data.yaml 中准备 test
        imgsz=IMG_SIZE,
        device=DEVICE,
        plots=True,            # 输出 PR、混淆矩阵等
        save_json=True,        # 导出 coco json（如需）
        conf=0.001,
        iou=IOU
    )

    # ========== 阶段 3：汇总导出（按你的最终清单） ==========
    # 3.1 分类别指标 CSV（tp/fp/fn/precision/recall/f1）——来自混淆矩阵
    names = model.names
    K = len(names)

    # Ultralytics v8 的混淆矩阵对象
    # 某些版本属性名可能不同，做点健壮性处理
    cm = getattr(test_results, "confusion_matrix", None)
    if cm is None:
        cm = getattr(test_results.metrics, "confusion_matrix", None)

    class_rows = []
    if cm is not None and hasattr(cm, "matrix"):
        M = cm.matrix.astype(np.float64)  # (K+1)x(K+1) 含背景/背景行列时需注意
        # 只取前 KxK 的真实类间矩阵
        M = M[:K, :K]
        tp = np.diag(M)
        fp = M.sum(0) - tp
        fn = M.sum(1) - tp

        precision = np.divide(tp, (tp + fp + 1e-16))
        recall = np.divide(tp, (tp + fn + 1e-16))
        f1 = np.divide(2 * precision * recall, (precision + recall + 1e-16))

        for i in range(K):
            class_rows.append({
                "class": names[i],
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
            })
    else:
        # 兜底：如果没有拿到混淆矩阵，至少给出类名（数值空）
        for i in range(K):
            class_rows.append({"class": names[i], "tp": "", "fp": "", "fn": "",
                               "precision": "", "recall": "", "f1": ""})

    # 3.2 每类 AP（尽量取 AP50；若版本仅提供 mAP50-95 per-class，则用那个做近似）
    ap50_per_class = None
    try:
        # 新版本可能提供 .box.maps（mAP50-95 per-class），以及 .box.map50（总体）
        ap50_per_class = getattr(test_results.box, "maps", None)  # list
    except Exception:
        ap50_per_class = None

    # 为 CSV 增加 AP 列，并输出 per_class_map.png
    if ap50_per_class is not None and len(ap50_per_class) >= K:
        for i in range(K):
            class_rows[i]["AP50-95"] = round(float(ap50_per_class[i]), 6)
        plot_per_class_bar(
            values=[float(x) for x in ap50_per_class[:K]],
            labels=[names[i] for i in range(K)],
            out_path=run_dir / "per_class_map.png",
            title="Per-class mAP@0.5:0.95"
        )
    else:
        # 没拿到 per-class mAP 时，至少仍然生成一张柱状图（使用 F1 代替）
        plot_per_class_bar(
            values=[float(row["f1"] or 0.0) for row in class_rows],
            labels=[row["class"] for row in class_rows],
            out_path=run_dir / "per_class_map.png",
            title="Per-class F1 (fallback)"
        )

    # 写出 class_results.csv
    with open(run_dir / "class_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=class_rows[0].keys())
        writer.writeheader()
        writer.writerows(class_rows)

    # 3.3 生成 success/ 与 failure/ 样例图
    # 思路：对 test 图像跑一次 predict，按“是否检测到 ≥1 个框”粗分（严格版本应按 GT 匹配 TP/FP 来分）
    pred_dir = ensure_dir(run_dir / "pred_test")
    success_dir = ensure_dir(run_dir / "success")
    failure_dir = ensure_dir(run_dir / "failure")

    test_images = []
    # 从 data.yaml 读 test 路径（简单起见，让 Ultralytics 帮我们直接推理 test split）
    preds = model.predict(
        data=DATA,
        split="test",
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        save=True,                 # 保存带框图片
        project=str(pred_dir),
        name="imgs",
        stream=False
    )
    # 遍历预测结果，将保存的图像粗略划分 success/failure
    # 注意：这是“是否有检测”的粗划分；若要“是否检测正确”，需结合标签做 IoU 匹配再判断 TP/FP。
    saved_dir = pred_dir / "imgs"
    for img_file in sorted(saved_dir.glob("*.jpg")) | sorted(saved_dir.glob("*.png")):
        # 以是否存在 detections.json 为简易判断（某些版本直接看预测框数量）
        base = img_file.with_suffix("")
        detections_json = base.with_suffix(".json")
        has_det = detections_json.exists()
        shutil.copy(img_file, (success_dir if has_det else failure_dir) / img_file.name)

    # 3.4 汇总成功/失败：results_summary.csv + 两张拼图
    summary = []
    for cls in names.values() if isinstance(names, dict) else names:
        summary.append({"Class": cls, "Total": "", "Success": "", "Accuracy (%)": ""})


    total_imgs = len(list(success_dir.glob("*.*"))) + len(list(failure_dir.glob("*.*")))
    summary_rows = [{
        "Class": "ALL",
        "Total": total_imgs,
        "Success": len(list(success_dir.glob("*.*"))),
        "Accuracy (%)": round(100.0 * (len(list(success_dir.glob("*.*"))) / max(1, total_imgs)), 2)
    }]
    with open(run_dir / "results_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Class", "Total", "Success", "Accuracy (%)"])
        writer.writeheader()
        writer.writerows(summary_rows)

 
    make_mosaic(list(success_dir.glob("*.*"))[:60], run_dir / "summary_success.jpg", cols=6)
    make_mosaic(list(failure_dir.glob("*.*"))[:60], run_dir / "summary_failure.jpg", cols=6)

  
    try:
        overall = {
            "mAP50-95": round(float(getattr(test_results.box, "map", None) or 0.0), 6),
            "mAP50": round(float(getattr(test_results.box, "map50", None) or 0.0), 6),
            "Precision": round(float(np.nanmean([r.get("precision", 0) for r in class_rows])), 6),
            "Recall": round(float(np.nanmean([r.get("recall", 0) for r in class_rows])), 6)
        }
        print("Test overall:", json.dumps(overall, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Overall metrics print failed:", e)

    print(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
