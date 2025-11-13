from ultralytics import YOLO

def main():
    # 1) 用自定义结构初始化
    model = YOLO("yolo11n_cbam.yaml")

    # 2) 可选：从官方 yolo11n.pt 加载 backbone 权重（注意通道要对上）
    try:
        model.load("yolo11n.pt")
    except Exception:
        print("Warning: could not load yolo11n.pt weights; training from scratch.")

    # 3) 用原始平衡数据集训练，和 baseline 做对比
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project="runs_agropest12",
        name="baseline_yolo11n_cbam",
    )

    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()
