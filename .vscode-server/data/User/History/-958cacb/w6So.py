import time
from ultralytics import YOLO

def main():
    t0 = time.time()

    model = YOLO("yolo11n.pt")

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,               
        project="runs_agropest12",
        name="baseline_yolo11n",
     
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

 
    metrics = model.val()
    print(metrics)

    print(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
