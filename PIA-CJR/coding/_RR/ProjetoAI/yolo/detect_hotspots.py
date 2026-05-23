import cv2
import numpy as np

def main():
    # carregar imagem
    image_path = "rectified_panels/panel_0.png"

    img = cv2.imread(image_path)

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur para reduzir ruído
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # threshold adaptativo
    mean_temp = np.mean(blur)

    #threshold_value = int(mean_temp * 1.25)
    mean = np.mean(blur)
    std = np.std(blur)

    threshold_value = int(mean + 2.5 * std)    

    _, thresh = cv2.threshold(
        blur,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # remover pequenos ruídos
    kernel = np.ones((5,5), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # encontrar hotspots
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # desenhar deteções
    output = img.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        # ignorar ruído pequeno
        # if area < 20:
        #     continue
        if area < 20 or area > 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            output,
            (x, y),
            (x+w, y+h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            output,
            "HOTSPOT",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            2
        )

    # guardar resultado
    cv2.imwrite("hotspot_panel/hotspot_result.png", output)

    print("Hotspots detetados!")

if __name__ == "__main__":
    main()    