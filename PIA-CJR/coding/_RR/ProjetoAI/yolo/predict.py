from ultralytics import YOLO

def main():
    model = YOLO("runs/segment/train-50/weights/best.pt")

    results = model(
        source="../input_images",
        save=True,
        conf=0.5
    )

    print("Inferência concluída!")

if __name__ == "__main__":
    main()