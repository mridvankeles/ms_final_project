import json
from tqdm import tqdm
import numpy as np
import pandas as pd



def calculate_polygon_area(polygon):
    """
    Calculate the area of a polygon using the Shoelace formula.
    """
    x = polygon[0::2]  # x-coordinates
    y = polygon[1::2]  # y-coordinates
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))



def extract_annotation_features(metric_path,annFile,resFile):
    print("--- annotation info features are extracting.. ---")
    metric_df = pd.read_csv(metric_path)
    metric_df["image_id"] = [str(img_id).zfill(12) for img_id in metric_df["image_id"]]
    metric_df["image_class"] = [str(img_id)[:2] for img_id in metric_df["image_id"]]

    with open(annFile, "r") as f:
        test_json = json.load(f)

    with open(resFile, "r") as f:
        pred_json = json.load(f)
        
    annotatations = test_json["annotations"]
    preds = pred_json

    annot_df = pd.DataFrame(annotatations)
    pred_df = pd.DataFrame(preds)

    annot_df["image_id"] = [str(img).zfill(12) for img in annot_df["image_id"]]
    

    labels_stats = (annot_df.groupby("image_id").agg(label_ins_number = ("bbox","count"),label_ins_area=("area","sum"),label_ins_area_mean=("area","mean")).reset_index())
        
    # Compute area for each prediction segmentation mask
    areas = []
    for annotation in preds:
        segmentations = annotation['segmentation']  # List of segmentation polygons
        total_area = 0
        for polygon in segmentations:
            total_area += calculate_polygon_area(polygon)
        areas.append({'image_id': annotation['image_id'], 'area': total_area})

    # Create a DataFrame for segmentation areas
    preds_area_df = pd.DataFrame(areas)

    # Calculate total area per image
    preds_stats = (
        preds_area_df.groupby('image_id')
        .agg(pred_ins_area=('area', 'sum'))
        .reset_index()
    )

    pred_df["area"] = preds_area_df["area"]

    # Calculate total area per image
    preds_stats = (
        pred_df.groupby("image_id").agg(pred_ins_number = ("bbox","count"),pred_ins_area=("area","sum"),pred_ins_area_mean=("area","mean")).reset_index()
    )

    preds_stats["image_id"] = [str(img).zfill(12) for img in preds_stats["image_id"]] 

    # Merge with the main DataFrame
    metric_df["pred_ins_number"] =    preds_stats["pred_ins_number"]
    metric_df["pred_ins_area"] =      preds_stats["pred_ins_area"]
    metric_df["pred_ins_area_mean"] = preds_stats["pred_ins_area_mean"]

    metric_df["label_ins_number"] = labels_stats["label_ins_number"]
    metric_df["label_ins_area"] = labels_stats["label_ins_area"]
    metric_df["label_ins_area_mean"] = labels_stats["label_ins_area_mean"]

    metric_df["instance_number_diff"] = labels_stats["label_ins_number"]  - preds_stats["pred_ins_number"]
    metric_df["instance_area_diff"] = labels_stats["label_ins_area"] - preds_stats["pred_ins_area"]
    metric_df["instance_area_mean_diff"] = labels_stats["label_ins_area_mean"] - preds_stats["pred_ins_area_mean"]
    metric_df.fillna(0.,inplace= True)
    metric_df.to_csv(metric_path,index=False) 