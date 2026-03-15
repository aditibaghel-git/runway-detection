import numpy as np
import matplotlib.path as mplPath

def polygon_from_anchors(anchors, image_w, image_h):
    """
    Convert normalized anchor vector to polygon points
    """
    return [
        (anchors[0] * image_w, anchors[1] * image_h),
        (anchors[2] * image_w, anchors[3] * image_h),
        (anchors[4] * image_w, anchors[5] * image_h),
        (anchors[6] * image_w, anchors[7] * image_h),
        (anchors[8] * image_w, anchors[9] * image_h),
        (anchors[10] * image_w, anchors[11] * image_h),
    ]


def calculate_iou(y_true, y_pred):
    """
    Compute Intersection over Union for binary masks
    """
    y_true = (y_true > 0.5)
    y_pred = (y_pred > 0.5)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    if union == 0:
        return 0.0

    return intersection / union


def mask_points_in_polygon(segmentation_mask, polygon):
    """
    Ratio of predicted mask pixels lying inside predicted polygon
    """
    points = np.argwhere(segmentation_mask > 0)

    if points.size == 0:
        return 0.0

    path = mplPath.Path(polygon)
    inside = path.contains_points(points[:, ::-1])

    return np.sum(inside) / len(points)


def boolean_score(true_anchors, polygon, image_w, image_h):
    """
    Check if control points lie inside predicted polygon
    """
    path = mplPath.Path(polygon)

    ctl_points = [
        (true_anchors[8] * image_w, true_anchors[9] * image_h),
        (true_anchors[10] * image_w, true_anchors[11] * image_h),
    ]

    return np.all(path.contains_points(ctl_points))