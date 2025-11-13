# train_and_eval.py
import csv
import json
import time
import shutil
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import os
# ================= 基本配置 =================
DATA = "data.yaml"          # 需在 data.yaml 中配置 train/val/test
DEVICE = 0
PROJECT = "runs_agropest12"
RUN_NAME = "baseline_yolo11n"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU = 0.6
def get_abs_split_dir(data_yaml_path: str, split_key: str) -> Path:
    """
    把 data.yaml 里的 train/val/test 路径解析成【绝对路径】返回。
    解析规则：相对于 data.yaml 所在目录进行拼接，再 .resolve() 规范化。
    """
    data_yaml = Path(data_yaml_path).resolve()
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw = cfg.get(split_key)
    assert raw, f"{split_key} 未在 {data_yaml_path} 中定义"

    # 若 raw 是绝对路径，直接用；否则相对 data.yaml 的父目录
    raw_path = Path(raw)
    abs_dir = raw_path if raw_path.is_absolute() else (data_yaml.parent / raw_path)
    abs_dir = abs_dir.resolve()
    assert abs_dir.exists() and abs_dir.is_dir(), f"{split_key} 路径不存在: {abs_dir}"
    print(f"[INFO] {split_key} 绝对路径: {abs_dir}")
    return abs_dir
# ================= 小工具 =================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_training_log_as(run_dir: Path):
    # Ultralytics 通常在 run 根目录下写 results.csv
    src = run_dir / "results.csv"
    if src.exists():
        shutil.copy(src, run_dir / "training_log.csv")

def save_loss_map_pr_curves_as(src_dir: Path):
    # 不同版本文件名可能不同，这里做个容错映射
    mapping = {
        "loss_curve.png": ["results.png", "loss_curve.png"],
        "map_curve.png":  ["mAP_curve.png", "metrics.png", "F1_curve.png"],
        "pr_curve.png":   ["PR_curve.png", "pr_curve.png"],
    }
    for dst_name, candidates in mapping.items():
        for c in candidates:
            p = src_dir / c
            if p.exists():
                shutil.copy(p, src_dir / dst_name)
                break

def plot_per_class_bar(values, labels, out_path: Path, title="Per-class score"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(values))
    plt.figure(figsize=(12, 4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def resolve_split_dir(data_yaml_path: str, split_key: str, run_dir: Path | None = None) -> Path | None:

    
    data_yaml = Path(data_yaml_path).resolve()
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw = cfg.get(split_key)
    if not raw:
        print(f"[WARN] {split_key} 未在 {data_yaml_path} 中定义")
        return None

    candidates: list[Path] = []
    raw_path = Path(raw)

   
    if raw_path.is_absolute():
        candidates.append(raw_path)


    candidates.append((data_yaml.parent / raw_path).resolve())

 
    candidates.append((Path.cwd() / raw_path).resolve())


    try:
        script_dir = Path(__file__).parent.resolve()
        candidates.append((script_dir / raw_path).resolve())
    except NameError:
        pass  

 
    if run_dir is not None:
        candidates.append((run_dir / raw_path).resolve())
        candidates.append((run_dir.parent / raw_path).resolve())

 
    search_roots = [data_yaml.parent, data_yaml.parent.parent]
    for root in search_roots:
        if root.exists():
            for p in root.rglob(f"{split_key}/images"):
                if p.is_dir():
                    candidates.append(p.resolve())


    seen = set()
    for c in candidates:
        if str(c) in seen:
            continue
        seen.add(str(c))
        if c.exists() and c.is_dir():
            print(f"[INFO]Parse to the {split_key} directory: {c}")
            return c

    print(f"[WARN] The available {split_key} directory was not found. Tried：")
    for c in candidates:
        print("  -", c)
    return None

def resolve_split_dir(data_yaml_path: str, split_key: str) -> Path | None:

    data_yaml = Path(data_yaml_path).resolve()
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get(split_key)
    if not raw:
        return None

    raw_path = Path(raw)

    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

  
    cand2 = (data_yaml.parent / raw_path).resolve()
    if cand2.exists():
        return cand2


    cand3 = (Path.cwd() / raw_path).resolve()
    if cand3.exists():
        return cand3

    return None


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
    run_dir.mkdir(parents=True, exist_ok=True)

    # 产出：best.pth（重命名一份）
    if best_pt.exists():
        shutil.copy(best_pt, run_dir / "best.pth")

    # 产出：训练日志和曲线
    copy_training_log_as(run_dir)          # training_log.csv
    save_loss_map_pr_curves_as(run_dir)    # loss_curve.png / map_curve.png / pr_curve.png

    # -------- 测试集评估（严格用 Ultralytics 的 val）--------
    model = YOLO(str(best_pt if best_pt.exists() else "yolo11n.pt"))
    val = model.val(
        data=DATA,
        split="test",                  # 关键：直接让 Ultralytics 读取 test
        imgsz=IMG_SIZE,
        device=DEVICE,
        plots=True,                    # 会生成 PR、混淆矩阵等
        save_json=True,
        conf=0.001,
        iou=IOU
    )

    # 抽取总览指标（与你的示例一致）
    m = val.results_dict
    P = float(m.get("metrics/precision(B)", 0.0))
    R = float(m.get("metrics/recall(B)", 0.0))
    AUC_05 = float(m.get("metrics/mAP50(B)", 0.0))
    AUC_5095 = float(m.get("metrics/mAP50-95(B)", 0.0))
    F1 = 2 * P * R / (P + R + 1e-12)

    # 混淆矩阵 & class-wise
    names = model.names
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    K = len(names)

    cm = getattr(val, "confusion_matrix", None)
    class_rows = []
    acc = float("nan")
    if cm is not None and hasattr(cm, "matrix"):
        M = cm.matrix.astype(float)
        # 有的版本最后一行/列可能是“背景”，这里只取 KxK
        M = M[:K, :K]
        tp = np.diag(M)
        fp = M.sum(0) - tp
        fn = M.sum(1) - tp
        precision = tp / (tp + fp + 1e-16)
        recall = tp / (tp + fn + 1e-16)
        f1 = 2 * precision * recall / (precision + recall + 1e-16)
        acc = float(np.trace(M) / (M.sum() + 1e-12))
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
            class_rows.append({
                "class": names[i], "tp": "", "fp": "", "fn": "",
                "precision": "", "recall": "", "f1": ""
            })

    # 每类 mAP（若有）
    ap_per_class = None
    try:
        ap_per_class = getattr(val.box, "maps", None)  # mAP50-95 per-class
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
        # 退化使用 F1 画柱状图，至少有一张图
        plot_per_class_bar(
            values=[float(r["f1"] or 0.0) for r in class_rows],
            labels=names,
            out_path=run_dir / "per_class_map.png",
            title="Per-class F1 (fallback)"
        )

    # 写出 class_results.csv
    with open(run_dir / "class_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=class_rows[0].keys())
        writer.writeheader()
        writer.writerows(class_rows)
    # -------- success / failure 示例图（使用绝对路径）--------
    pred_root = ensure_dir(run_dir / "pred_test")
    success_dir = ensure_dir(run_dir / "success")
    failure_dir = ensure_dir(run_dir / "failure")

    # 关键：把 test 解析为【绝对路径】并传给 predict(source=...)
    test_images_abs = get_abs_split_dir(DATA, "test")  # e.g. /root/your_project/test/images

    preds = model.predict(
        source=str(test_images_abs),   # ✅ 绝对路径，避免 CWD/.. 带来的歧义
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        save=True,
        project=str(pred_root),
        name="imgs",
        stream=False
    )

    for r in preds:
        saved = Path(r.save_dir) / Path(r.path).name  # save=True 输出的带框图片
        has_det = (getattr(r, "boxes", None) is not None) and (r.boxes is not None) and (r.boxes.shape[0] > 0)
        if saved.exists():
            (success_dir if has_det else failure_dir).mkdir(parents=True, exist_ok=True)
            saved_dst = (success_dir if has_det else failure_dir) / saved.name
            # 同名覆盖也没关系，如需去重可先判断是否存在
            shutil.copy(saved, saved_dst)

    # 拼接展示图
    success_imgs = list(success_dir.glob("*.jpg")) + list(success_dir.glob("*.png"))
    failure_imgs = list(failure_dir.glob("*.jpg")) + list(failure_dir.glob("*.png"))
    make_mosaic(success_imgs[:60], run_dir / "summary_success.jpg", cols=6)
    make_mosaic(failure_imgs[:60], run_dir / "summary_failure.jpg", cols=6)

    # 汇总 CSV
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


    # 打印总体测试指标（与你给的参考输出一致）
    overall = {
        "Precision": round(P, 6),
        "Recall": round(R, 6),
        "F1": round(F1, 6),
        "Accuracy": round(acc, 6),
        "AUC@0.5": round(AUC_05, 6),
        "AUC@0.5:0.95": round(AUC_5095, 6),
    }
    print("Test overall:", json.dumps(overall, ensure_ascii=False, indent=2))

    print(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
