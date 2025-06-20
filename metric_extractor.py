import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.evaluation_labelcheck import evaluate_images
import numpy as np
import os
from tqdm import tqdm
from pycocotools.coco import COCO


def extract_metric_features(metric_path,gt_test_annot,pred_file,model_name):
    
    if not os.path.exists(metric_path):
        print("------metrics are extracting------")
        metric_df = evaluate_images(gt_test_annot,pred_file,model_name,metric_path)
        metric_df["image_id"] = [str(img_id).zfill(12) for img_id in metric_df["image_id"]]
        gt_coco = COCO(gt_test_annot)
        dt_coco = gt_coco.loadRes(pred_file)

        image_ids = dt_coco.getImgIds(catIds=dt_coco.getCatIds())
        score_dict = {"image_id":[],"score":[]}
        for image_id in tqdm(image_ids):
            img = dt_coco.loadImgs(image_id)[0]
            annotation_ids = dt_coco.getAnnIds(imgIds=img['id'])
            annotations = dt_coco.loadAnns(annotation_ids)
            score_dict["image_id"].append(str(image_id).zfill(12))
            score_dict["score"].append(annotations[0]["score"])
        
        score_df = pd.DataFrame(score_dict)
        combined_df = pd.merge(metric_df, score_df, on='image_id',how="left")
        combined_df.fillna(0.,inplace=True)
        combined_df.to_csv(metric_path,index=False)
    print("metrics already ready")


    