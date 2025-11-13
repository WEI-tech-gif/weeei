from ultralytics import YOLO

def main():

    model = YOLO("yolo11n_cbam.yaml")

 
    try:
        model.load("yolo11n.pt")
    except Exception:
        print("Warning: could not load yolo11n.pt weights; training from scratch.")

 
    model.train(
        data="data.yaml",
        epochs=25,
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        project="runs_agropest12",
        name="baseline_yolo11n_cbam",
    )

    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()
