import pandas as pd
import numpy as np
from config.config import *
from src.utils import (
    polygon_from_anchors,
    calculate_iou,
    mask_points_in_polygon,
    boolean_score
)


def evaluate(model, validation_generator):
    preds = model.predict(validation_generator, steps=len(validation_generator))
    pred_masks, pred_anchors = preds

    true_masks = []
    true_anchors = []
    filenames = []

    for i in range(len(validation_generator)):
        X, outputs = validation_generator[i]

        true_masks.append(outputs["mask_output"])
        true_anchors.append(outputs["anchor_output"])

        idxs = validation_generator.indices[
            i * BATCH_SIZE:(i + 1) * BATCH_SIZE
        ]
        filenames.extend([
            validation_generator.image_files[idx] for idx in idxs
        ])

    true_masks = np.concatenate(true_masks)
    true_anchors = np.concatenate(true_anchors)

    results = []

    for i in range(len(pred_masks)):
        polygon = polygon_from_anchors(
            pred_anchors[i], TARGET_W, TARGET_H
        )

        iou = calculate_iou(
            true_masks[i, :, :, 0],
            pred_masks[i, :, :, 0]
        )

        anchor_cov = mask_points_in_polygon(
            pred_masks[i, :, :, 0],
            polygon
        )

        bool_sc = boolean_score(
            true_anchors[i],
            polygon,
            TARGET_W,
            TARGET_H
        )

        results.append({
            "Image Name": filenames[i],
            "IOU score": iou,
            "Anchor Score": anchor_cov,
            "Boolean score": int(bool_sc)
        })

    df = pd.DataFrame(results)

    df.loc[len(df)] = [
        "Mean Score",
        df["IOU score"].mean(),
        df["Anchor Score"].mean(),
        df["Boolean score"].mean()
    ]

    df.to_csv("outputs/runway_detection_evaluation.csv", index=False)
    print("Evaluation saved to outputs/runway_detection_evaluation.csv")
    print(df)

    return df