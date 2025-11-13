# train_and_eval.py
import os
import csv
import json
import time
import shutil
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


DATA = "data.yaml"        
DEVICE = 0
PROJECT = "runs_agropest12"
RUN_NAME = "baseline_yolo11n"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU = 0.6


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_split_dir_from_data_yaml(data_yaml_path: str, split_key: str) -> Path:
    p = Path(data_yaml_path).resolve()
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    split_path = cfg.get(split_key)
    assert split_path, f"{split_key} 未在 {data_yaml_path} 中定义"
    split_dir = (p.parent / split_path).resolve()  # 相对 data.yaml 的路径
    assert split_dir.exists(), f"{split_key} 路径不存在: {split_dir}"
    return split_dir

def copy_training_log_as(train_dir: Path, dst_dir: Path):
    for name in ["results.csv", "training_log.csv"]:
        p = train_dir / name
        if p.exists():
            shutil.copy(p, dst_dir / "training_log.csv")
            return

def save_loss_map_pr_curves_as(src_dir: Path, dst_dir: Path):
    mapping = {
        "loss_curve.png": ["results.png", "loss_curve.png"],
        "map_curve.png":  ["mAP_curve.png", "metrics.png", "F1_curve.png"],
        "pr_curve.png":   ["PR_curve.png", "pr_curve.png"]
    }
    for dst_name, candidates in mapping.items():
        for s in candidates:
            sp = src_dir / s
            if sp.exists():
                shutil.copy(sp, dst_dir / dst_name)
                break

def plot_per_class_bar(values, labels, out_path: Path, title="Per-class score"):
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
    images = []
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB").resize((cell_size, cell_size))
            images.append(im)
        except Exception:
            pass
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

    model = YOLO("yolo11n.pt")
    _ = model.train(
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
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5, shear=2.0,
        flipud=0.0, fliplr=0.5
    )

    run_dir = Path(PROJECT) / RUN_NAME
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    ensure_dir(run_dir)

    if best_pt.exists():
        shutil.copy(best_pt, run_dir / "best.pth")

 
    copy_training_log_as(run_dir, run_dir)            
    save_loss_map_pr_curves_as(run_dir, run_dir)       # loss_curve.png / map_curve.png / pr_curve.png

    # -------- 使用 best 权重做测试集评估 --------
    model = YOLO(str(best_pt if best_pt.exists() else "yolo11n.pt"))
    test_results = model.val(
        data=DATA,
        split="test",       # 关键：data.yaml 里要有 test:
        imgsz=IMG_SIZE,
        device=DEVICE,
        plots=True,         # 会生成 PR/混淆矩阵等
        save_json=True,
        conf=0.001,
        iou=IOU
    )

   
    names = model.names
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    K = len(names)

    # 兼容不同版本的混淆矩阵取法
    cm = getattr(test_results, "confusion_matrix", None)
    if cm is None:
        cm = getattr(test_results.metrics, "confusion_matrix", None)

    class_rows = []
    if cm is not None and hasattr(cm, "matrix"):
        M = cm.matrix.astype(np.float64)
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
        for i in range(K):
            class_rows.append({"class": names[i], "tp": "", "fp": "", "fn": "",
                               "precision": "", "recall": "", "f1": ""})

    # 每类 mAP（尽量取 maps；拿不到则用 F1 代替画图）
    ap_per_class = None
    try:
        ap_per_class = getattr(test_results.box, "maps", None)  # mAP50-95 per-class
    except Exception:
        ap_per_class = None

    if ap_per_class is not None and len(ap_per_class) >= K:
        for i in range(K):
            class_rows[i]["AP50-95"] = round(float(ap_per_class[i]), 6)
        plot_per_class_bar(
            values=[float(x) for x in ap_per_class[:K]],
            labels=names,
            out_path=run_dir / "per_class_map.png",
            title="Per-class mAP@0.5:0.95"
        )
    else:
        plot_per_class_bar(
            values=[float(r["f1"] or 0.0) for r in class_rows],
            labels=names,
            out_path=run_dir / "per_class_map.png",
            title="Per-class F1 (fallback)"
        )

    with open(run_dir / "class_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=class_rows[0].keys())
        writer.writeheader()
        writer.writerows(class_rows)

    # -------- 生成 success/ 与 failure/ 示例（已修正 predict 用法） --------
    pred_dir = ensure_dir(run_dir / "pred_test")
    success_dir = ensure_dir(run_dir / "success")
    failure_dir = ensure_dir(run_dir / "failure")

    test_images_dir = get_split_dir_from_data_yaml(DATA, "test")
    pred_results = model.predict(
        source=str(test_images_dir),     # ✅ 正确指定测试集目录
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        save=True,                       # 保存带框图片
        project=str(pred_dir),
        name="imgs",
        stream=False
    )

    for r in pred_results:
        save_img_path = Path(r.save_dir) / Path(r.path).name
        has_det = (getattr(r, "boxes", None) is not None) and (r.boxes is not None) and (r.boxes.shape[0] > 0)
        dst_dir = success_dir if has_det else failure_dir
        if save_img_path.exists():
            shutil.copy(save_img_path, dst_dir / save_img_path.name)

    success_imgs = list(success_dir.glob("*.jpg")) + list(success_dir.glob("*.png"))
    failure_imgs = list(failure_dir.glob("*.jpg")) + list(failure_dir.glob("*.png"))

    make_mosaic(success_imgs[:60], run_dir / "summary_success.jpg", cols=6)
    make_mosaic(failure_imgs[:60], run_dir / "summary_failure.jpg", cols=6)

    # 汇总表（整体）
    total_imgs = len(success_imgs) + len(failure_imgs)
    with open(run_dir / "results_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Class", "Total", "Success", "Accuracy (%)"])
        writer.writeheader()
        writer.writerow({
            "Class": "ALL",
            "Total": total_imgs,
            "Success": len(success_imgs),
            "Accuracy (%)": round(100.0 * (len(success_imgs) / max(1, total_imgs)), 2)
        })

    # 打印总体指标
    try:
        overall = {
            "mAP50-95": round(float(getattr(test_results.box, "map", 0.0) or 0.0), 6),
            "mAP50": round(float(getattr(test_results.box, "map50", 0.0) or 0.0), 6),
            "Precision(mean)": round(float(np.nanmean([r.get("precision", 0) for r in class_rows])), 6),
            "Recall(mean)": round(float(np.nanmean([r.get("recall", 0) for r in class_rows])), 6)
        }
        print("Test overall:", json.dumps(overall, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Overall metrics print failed:", e)

    print(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
