import numpy as np
import pandas as pd
import logging

from hisup.config import cfg
from hisup.detector import BuildingDetector
from hisup.dataset.build import build_test_dataset,build_train_dataset
from hisup.utils.checkpoint import DetectronCheckpointer
from hisup.utils.comm import to_single_device
from tqdm import tqdm
import torch
import cv2
import json


def extract_hisup_losses(HISUP_CONFIG_PATH,LATENT_FEATURES_PATH,gt_file):
    hisup_losses = {"loss_jloc":[],"loss_joff":[],"loss_mask":[],"loss_afm":[],"loss_remask":[],"total_loss":[]}

    weights=[8.0,0.25,1.0,0.1,1.0]

    print("--------hisup losses are extracting--------")
    cfg.merge_from_file(HISUP_CONFIG_PATH)
    cfg.freeze()
    logger = logging.getLogger("encoding")

    train_dataset = build_train_dataset(cfg)
    model_hisup_train = BuildingDetector(cfg, test=False)
    model_hisup_train = model_hisup_train.to("cuda")

    with open(gt_file, "r") as f:
        test_json = json.load(f)
    annotatations = test_json["annotations"]
    annot_df = pd.DataFrame(annotatations)

    checkpointer = DetectronCheckpointer(cfg,
                                    model_hisup_train,
                                    save_dir=cfg.OUTPUT_DIR,
                                    save_to_disk=True,
                                    logger=logger)
    _ = checkpointer.load()


    model_hisup_train = model_hisup_train.train()
    for i, (images,annotations) in enumerate(tqdm(train_dataset)):
        loss_dict, feature_maps = model_hisup_train(images.to("cuda"),to_single_device(annotations, "cuda")) 
        loss_list = {key: loss.detach().cpu().item() for key,loss in loss_dict.items()}
        hisup_losses["loss_jloc"].append(loss_list["loss_jloc"])
        hisup_losses["loss_joff"].append(loss_list["loss_joff"])
        hisup_losses["loss_mask"].append(loss_list["loss_mask"])
        hisup_losses["loss_afm"].append(loss_list["loss_afm"])
        hisup_losses["loss_remask"].append(loss_list["loss_remask"])

    hisup_losses_df = pd.DataFrame(hisup_losses,columns=["loss_jloc","loss_joff","loss_mask","loss_afm","loss_remask"])
    hisup_losses_df["total_loss"] = np.average(hisup_losses_df[["loss_jloc","loss_joff","loss_mask","loss_afm","loss_remask"]],axis=1,weights=weights)
    hisup_losses_df.to_csv(LATENT_FEATURES_PATH,index=False)

def extract_hisup_features(HISUP_CONFIG_PATH,LATENT_FEATURES_PATH,loss_path):

    hisup_features = {"image_id":[],"mask_pred":[],"mask_feature":[],"ground_truth_mask":[]}

    print("--------hisup features-pd_mask-gt_mask are extracting--------")
    cfg.merge_from_file(HISUP_CONFIG_PATH)
    cfg.freeze()
    logger = logging.getLogger("encoding")


    test_dataset, gt_file = build_test_dataset(cfg)
    model_hisup_test = BuildingDetector(cfg, test=True)
    model_hisup_test = model_hisup_test.to("cuda")


    checkpointer = DetectronCheckpointer(cfg,
                                    model_hisup_test,
                                    save_dir=cfg.OUTPUT_DIR,
                                    save_to_disk=True,
                                    logger=logger)
    _ = checkpointer.load()
    
    
    model_hisup_test = model_hisup_test.eval()
    for i, (images, annotations) in enumerate(tqdm(test_dataset)): 
        with torch.no_grad():
            output, features = model_hisup_test(images.to("cuda"), to_single_device(annotations, "cuda"))

            image_id = annotations[0]["filename"].split(".")[0]
            hisup_features["image_id"].append(image_id)
                
            np_gt_mask=cv2.resize(annotations[0]["mask"].cpu().numpy().round(), (128,128)) 
            
            tensor_gt_mask=torch.tensor(np_gt_mask,dtype=np.long).to("cuda")
            tensor_mask_pred=torch.tensor(features["np_remask_pred"]).to("cuda")

            min_val = features["np_remask_feature"].min()
            max_val = features["np_remask_feature"].max()
            norm_fea = (features["np_remask_feature"]-min_val)/(max_val-min_val)
            tensor_mask_feature = torch.tensor(norm_fea).to("cuda")
            
            # append the mask to hisup_features
            hisup_features["ground_truth_mask"].append(tensor_gt_mask)
            hisup_features["mask_pred"].append(tensor_mask_pred)
            hisup_features["mask_feature"].append(tensor_mask_feature)

    
    hisup_maps = [torch.stack([m1.squeeze(),m2.squeeze(),m3.squeeze()],dim=0) for m1,m2,m3 in zip(
                                                            hisup_features["ground_truth_mask"],
                                                            hisup_features["mask_pred"],
                                                            hisup_features["mask_feature"],
                                                          )]
    
    
    
    loss_df = pd.read_csv(loss_path)
    feature_df = pd.DataFrame(hisup_features["image_id"],columns=["image_id"])
    feature_df = pd.concat([feature_df,loss_df],axis=1)
    feature_df.to_csv(loss_path,index=False)
    
    maps_tensor=torch.stack(hisup_maps,dim=0)
    print(maps_tensor.shape)
    torch.save(maps_tensor,LATENT_FEATURES_PATH)#remask
