import matplotlib.pyplot as plt
from config.config import TARGET_W, TARGET_H
from src.utils import polygon_from_anchors


def visualize_sample(image, true_mask, pred_mask, anchors):
    """
    Visualize image, true mask, predicted mask and anchor polygon
    """
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(true_mask.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask + Anchors")
    plt.imshow(pred_mask.squeeze(), cmap="gray")

    polygon = polygon_from_anchors(
        anchors,
        TARGET_W,
        TARGET_H
    )
    xs, ys = zip(*polygon)
    plt.plot(xs + (xs[0],), ys + (ys[0],), color="red", linewidth=2)

    plt.axis("off")
    plt.show()