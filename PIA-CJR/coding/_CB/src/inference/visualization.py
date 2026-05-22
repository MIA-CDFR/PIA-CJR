from PIL import ImageDraw


def save_annotated_image(
    image,
    label,
    confidence,
    output_path
):

    draw = ImageDraw.Draw(image)

    text = f"{label} ({confidence:.2f}%)"

    draw.text(
        (10, 10),
        text,
        fill=(255, 0, 0)
    )

    image.save(output_path)