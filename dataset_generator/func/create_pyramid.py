from PIL import Image


def create_pyramid(
    image_path: str, labels: list[list[int]], window_size: int, resize_factor: float
) -> tuple[list[Image.Image], list[list[int]]]:
    def round_to_even(number: float) -> int:
        return 2 * round(number / 2)

    original_image = Image.open(image_path)
    w, h = original_image.size
    scaled_labels = []
    for label in labels:
        scaled_labels.append(
            [n / w if i % 2 == 0 else n / h for i, n in enumerate(label)]
        )

    pyramid_images = []
    pyramid_labels = []

    while w >= window_size and h >= window_size:
        pyramid_image = original_image.resize((w, h), Image.LANCZOS)
        pyramid_label = []
        for label in scaled_labels:
            pyramid_label.append(
                [n * w if i % 2 == 0 else n * h for i, n in enumerate(label)]
            )

        pyramid_images.append(pyramid_image)
        pyramid_labels.append(pyramid_label)

        w = round_to_even(w * resize_factor)
        h = round_to_even(h * resize_factor)

    return pyramid_images, pyramid_labels
