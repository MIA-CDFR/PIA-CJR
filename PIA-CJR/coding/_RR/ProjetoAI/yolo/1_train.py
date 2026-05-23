from ultralytics import YOLO

def main():
    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=1024,
        batch=4,
        workers=0
    )

if __name__ == "__main__":
    main()