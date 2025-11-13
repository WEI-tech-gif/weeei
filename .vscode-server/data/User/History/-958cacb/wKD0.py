import time
from pathlib import Path
from ultralytics import YOLO

DATA = "data.yaml"
DEVICE = 0
PROJECT = "runs_agropest12"

def stage1():
    model = YOLO("yolo11n.pt")
    name = "y11n_stage1"
    model.train(
        data=DATA,
        epochs=40,
        imgsz=640,
        batch=128,              
        workers=4,
        device=DEVICE,
        project=PROJECT,
        name=name,

  
        mosaic=0.5,
        mixup=0.05,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.20, shear=1.0,
        flipud=0.0, fliplr=0.5,
    )
    best = Path(PROJECT) / name / "weights" / "best.pt"
    return best

def stage2(best_weight_path: Path):
    assert best_weight_path.exists(), f"The best weight of stage 1 cannot be found：{best_weight_path}"
    model = YOLO(str(best_weight_path))
    name = "y11n_stage2_finetune"
    model.train(
        data=DATA,
        epochs=20,             
        imgsz=640,
        batch=64,
        workers=4,
        device=DEVICE,
        project=PROJECT,
        name=name,


        mosaic=0.0,
        mixup=0.0,
        degrees=0.0, translate=0.0, scale=0.10, shear=0.0,
        flipud=0.0, fliplr=0.5,

    
        lr0=0.001,
        lrf=0.01,
    )
    return Path(PROJECT) / name / "weights" / "best.pt"

def main():
    t0 = time.time()
    best1 = stage1()
    print(f"[Stage-1 complete] best: {best1}")

    best2 = stage2(best1)
    print(f"[Stage-2 complete] best: {best2}")


    final_model = YOLO(str(best2))
    metrics = final_model.val(data=DATA, imgsz=640, device=DEVICE)
    print(metrics)
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
