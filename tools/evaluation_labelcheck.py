import sys

import os
import shutil
sys.path.append(os.getcwd())

from pycocotools.coco import COCO
from pycocotools import mask as cocomask

import multiprocessing as mp
original_pool = mp.Pool
def single_process_pool(*args,**kwargs):
    kwargs["processes"] = 1
    return original_pool(*args,**kwargs)

mp.Pool = single_process_pool

from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from hisup.utils.metrics.polis import PolisEval
from hisup.utils.metrics.cIoU import calc_IoU
from hisup.config import cfg
from hisup.config.paths_catalog import DatasetCatalog
import mlflow
import pandas as pd

import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np
import os
import shutil
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
from tqdm import tqdm
import torch
os.environ["OMP_NUM_THREADS"]="1"


def browse_folder(entry):
    folder_path = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folder_path)

def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)


def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    _,polis_dict = polisEval.evaluate()
    return polis_dict

def boundary_eval(annFile, resFile):
    #mp.set_start_method("spawn",force=True)
    dilation_ratio = 0.02 # default settings 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = cocoGt.getImgIds()
    catIds = cocoGt.getCatIds()
    imgIds = imgIds[:]
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [100]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ious_dict = {}
    for imgId in imgIds:
        for catId in catIds:
            ious = cocoEval.computeBoundaryIoU(imgId,catId)
            ious_dict[str(imgId)]=np.mean(ious)

    ious = {"image_id": list(ious_dict.keys()),"boundry_iou":ious_dict.values()}
    
    return ious

def compute_IoU_cIoU(input_json, gti_annotations):
    # Ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Predictions annotations
    submission_file = json.loads(open(input_json).read())
    coco = COCO(gti_annotations)
    coco = coco.loadRes(submission_file)


    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    bar = tqdm(image_ids)

    list_iou = []
    list_ciou = []
    pss = []
    iou_dict = {
        "image_id": [],
        "iou":[]
    }
    ciou_dict = {
        "image_id": [],
        "ciou":[]
    }
    for image_id in bar:

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        N = 0
        for _idx, annotation in enumerate(annotations):
            try:
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            except Exception:
                import ipdb; ipdb.set_trace()
            m = cocomask.decode(rle)
            if _idx == 0:
                mask = m.reshape((img['height'], img['width']))
                N = len(annotation['segmentation'][0]) // 2
            else:
                mask = mask + m.reshape((img['height'], img['width']))
                N = N + len(annotation['segmentation'][0]) // 2

        mask = mask != 0


        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        N_GT = 0
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            if _idx == 0:
                mask_gti = m.reshape((img['height'], img['width']))
                N_GT = len(annotation['segmentation'][0]) // 2
            else:
                mask_gti = mask_gti + m.reshape((img['height'], img['width']))
                N_GT = N_GT + len(annotation['segmentation'][0]) // 2

        mask_gti = mask_gti != 0

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        pss.append(ps)
        iou_dict["image_id"].append(str(image_id))
        ciou_dict["image_id"].append(str(image_id))

        iou_dict["iou"].append(iou)
        ciou_dict["ciou"].append(iou * ps)

        bar.set_description("iou: %2.4f, c-iou: %2.4f, ps:%2.4f" % (np.mean(list_iou), np.mean(list_ciou), np.mean(pss)))
        bar.refresh()

    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))


    return iou_dict,ciou_dict


def eval_model(model_name,gt_annot,dt_annot):
    global metric_df

    print(f"------ evaluating model : {model_name} -------")
    cfg.merge_from_file(f"outputs/{model_name}/config.yml")
    cfg.freeze()

    polis_dict = polis_eval(gt_annot, dt_annot)
    iou_dict,ciou_dict = compute_IoU_cIoU(dt_annot, gt_annot)
    biou_dict = boundary_eval(gt_annot, dt_annot)

    polis_df =  pd.DataFrame(polis_dict)
    iou_df = pd.DataFrame(iou_dict)
    ciou_df = pd.DataFrame(ciou_dict)
    biou_df = pd.DataFrame(biou_dict)

    polis_df['image_id'] = polis_df['image_id'].astype(str)
    biou_df['image_id'] = biou_df['image_id'].astype(str)
    metric_df =pd.merge(polis_df,biou_df,on="image_id",how="inner")
    metric_df =pd.merge(metric_df,iou_df,on="image_id",how="left")
    metric_df =pd.merge(metric_df,ciou_df,on="image_id",how="left")
    metric_df.fillna(0.,inplace=True)
    return metric_df



def visualize_segmentation(image_id, source_folder, label_json, prediction_json, dest_folder):
    image_path = os.path.join(source_folder, f"{str(image_id)}.tif")
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        return

    with open(label_json) as f:
        labels = json.load(f)

    with open(prediction_json) as f:
        predictions = json.load(f)

    image = Image.open(image_path)
    
    # Etiket ve tahmin poligonları
    label_annotations = [ann for ann in  labels['annotations'] if int(ann['image_id']) == int(image_id)]
    prediction_annotations = [ann for ann in predictions if int(ann['image_id']) == int(image_id)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Sol: Etiket Poligonları
    axes[0].imshow(image)
    for ann in label_annotations:
        segmentation = ann['segmentation']
        for poly in segmentation:
            polygon = Polygon(np.array(poly).reshape(-1, 2), fill=True,linewidth=1,color="green", alpha=0.7)
            axes[0].add_patch(polygon)
    axes[0].set_title("Label")

    # Sağ: Tahmin Poligonları
    axes[1].imshow(image)
    for ann in prediction_annotations:
        segmentation = ann['segmentation']
        for poly in segmentation:
            polygon = Polygon(np.array(poly).reshape(-1, 2), fill=True,linewidth=1,color="red", alpha=0.7)
            axes[1].add_patch(polygon)
    axes[1].set_title("Prediction")

    fig_path = os.path.join(dest_folder, f"{image_id}_comparison.jpg")
    plt.savefig(fig_path)
    plt.close()
    
    print(f"Saved comparison for {image_id} to {fig_path}")



def evaluate_images(gt_test_annot,dt_annot,model_name,metric_save_path):

    metric_df = eval_model(model_name,gt_test_annot,dt_annot)
    metric_df.to_csv(metric_save_path,index=False)
    return metric_df
    
