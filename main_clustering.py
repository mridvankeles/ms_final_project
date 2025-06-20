import torch.optim as optim
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from training_autoencoders import combine_embeddings
from latent_extractor import extract_hisup_features,extract_hisup_losses
from metric_extractor import extract_metric_features
from info_feature_extractor import extract_annotation_features
from analyze_clusters import analyze_cluster_distribution
from tools.visualize_clusters import visualize_df_clusters,visualize_clusters_pca


# --- Configuration ---

# TABULAR METRIC AND INFO PATHS
TABULAR_FEATURES_PATH = 'outputs/crowdai_hrnet48_whumix/cluster_outputs/tabular_features.csv'
TABULAR_COMPRESSED_PATH = "outputs/crowdai_hrnet48_whumix/cluster_outputs/tabular_latent.pt"

# HISUP LATENT PATHS
HISUP_CONFIG_PATH = "outputs/crowdai_hrnet48_whumix/config.yml" #hisup output config
HISUP_LOSS_PATH = "outputs/crowdai_hrnet48_whumix/cluster_outputs/hisup_loss.csv" 
LATENT_FEATURES_PATH = "outputs/crowdai_hrnet48_whumix/cluster_outputs/pred_gt_mask.pt"
LATENT_COMPRESSED_PATH = 'outputs/crowdai_hrnet48_whumix/cluster_outputs/conv_latent.pt'

RIGHT_LABELED_IMG_FOLDER = 'data//right_labeled/'
WRONG_LABELED_IMG_FOLDER = 'data//wrong_labeled/'

mode = "val"
model_name = "crowdai_hrnet48_whumix" 
image_folder = r"data/whumix_val/images" #input image path
ann_file = r"data/whumix_val/val.json" #input image json
res_file = r"outputs/crowdai_hrnet48_whumix/crowdai_whumix_val_10ep.json" # prediction json of hisup model

# shows combined features importances by using random forest model.
compress_tabular=True
compress_mask = True # add image features with compressed format

supervised_info=True
visualize_clusters = False # save clusters to folder

# Clustering parameters
CLUSTER_MODEL = "kmeans"  # "kmeans","dbscan","agg_cluster" 
NUM_CLUSTERS = [3,4,6,7,8,9,10] # how many cluster want to model's cluster
VIZ_NUM_CLUSTERS = ["dbc"] # which cluster you want to visualize to folder from clustered data (if using dbscan : "dbc")
INSIDE_VIZ_NUM_CLUSTERS = [-1,1] # which cluster instances do you like to see inside the chosen clusters (if using dbscan (-1,1))

TABULAR_FEATURES_DROP = [] # tabular feature select.

cluster_folder = rf"outputs/crowdai_hrnet48_whumix/clusters_viz/clusters_tabular_features"
cluster_plot_path = rf"outputs/crowdai_hrnet48_whumix/clusters_viz/pca_plots_combined"


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not os.path.exists(TABULAR_FEATURES_PATH):
        extract_metric_features(TABULAR_FEATURES_PATH,ann_file,res_file,model_name)
        extract_annotation_features(TABULAR_FEATURES_PATH,annFile=ann_file,resFile=res_file)
        
        
        
    if not os.path.exists(HISUP_LOSS_PATH):
        extract_hisup_losses(HISUP_CONFIG_PATH,HISUP_LOSS_PATH,gt_file=ann_file)
    if not os.path.exists(LATENT_FEATURES_PATH):
        extract_hisup_features(HISUP_CONFIG_PATH,LATENT_FEATURES_PATH,HISUP_LOSS_PATH)

    
    tabular_embeddings,conv_embeddings = combine_embeddings(
        TABULAR_FEATURES_PATH,
        HISUP_LOSS_PATH,
        TABULAR_COMPRESSED_PATH,
        LATENT_FEATURES_PATH,
        LATENT_COMPRESSED_PATH,
        compress_tabular=compress_tabular,
        compress_mask=compress_mask

    )
    combined_embeddings = None


    if tabular_embeddings is not None and conv_embeddings is not None:
        print("preparing combine embeddings for clustering")
        np.random.seed(42) 
        scaler = StandardScaler()
        tabular_df = pd.DataFrame(tabular_embeddings)
        if len(TABULAR_FEATURES_DROP)!=0:
            tabular_df =tabular_df.filter(items = TABULAR_FEATURES_DROP)
        
        if conv_embeddings=="empty":
            combined_embeddings = tabular_df
        else:
            latent_df = pd.DataFrame(conv_embeddings)
            latent_df.columns = ["latent_"+str(i) for i in range(0,len(latent_df.columns))]
            combined_embeddings = pd.concat([tabular_df, latent_df], axis=1)

        combined_embeddings_scaled = scaler.fit_transform(combined_embeddings.drop("image_id",axis=1))
        combined_embeddings_df = pd.DataFrame(combined_embeddings_scaled,columns=combined_embeddings.columns.to_list().remove("image_id"))
        combined_embeddings_df.to_csv(LATENT_COMPRESSED_PATH.split(".")[0]+"scaled_conv_tabular_latent.csv",index=False)
    else:
        print("No embeddings available for clustering.")

    
    if combined_embeddings is not None:
        
        # Add label error column
        false_list = os.listdir(WRONG_LABELED_IMG_FOLDER)
        false_names = [i.split(".")[0] for i in false_list]
        combined_embeddings["label_error"] = 0
        combined_embeddings.loc[combined_embeddings["image_id"].isin(false_names),"label_error"]=1

        #feature importance için
        if supervised_info:
            from sklearn.ensemble import RandomForestClassifier
            X = combined_embeddings.drop(columns=["image_id",'label_error'],axis=1)
            y = combined_embeddings['label_error']
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
            rf.fit(X, y)
            importancess = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("-- features importances --")
            print(importancess)

        #--- CLUSTERING ---
        if CLUSTER_MODEL == "dbscan":
            cluster_model = DBSCAN(eps=0.5, min_samples=5)
            cluster_model.fit(combined_embeddings_scaled)
            combined_embeddings[f"cluster_dbc"] = cluster_model.labels_
            cluster_labels = cluster_model.labels_
        else:
            for cluster in NUM_CLUSTERS:
                if CLUSTER_MODEL == "kmeans":
                    cluster_model = KMeans(n_clusters=cluster, random_state=42,init="random")
                    cluster_model.fit(combined_embeddings_scaled)
                    cluster_labels = cluster_model.fit_predict(combined_embeddings_scaled)
                    
                
                elif CLUSTER_MODEL == "agg_cluster":
                    cluster_model = AgglomerativeClustering(n_clusters=cluster, linkage='average')
                    cluster_model.fit(combined_embeddings_scaled)
                    cluster_labels = cluster_model.fit_predict(combined_embeddings_scaled)

                combined_embeddings[f"cluster_{cluster}"] = cluster_labels
        
        if supervised_info:
            tsne =TSNE(n_components=2,random_state=42,init="random")
            X_tsne = tsne.fit_transform(X)
            plt.scatter(X_tsne[:,0],X_tsne[:,1],c=cluster_labels,cmap="viridis")
            plt.title(f"tsne: visualziation of clusters {CLUSTER_MODEL}")
            plt.savefig("cluster_viz.jpg")


        visualize_clusters_pca(combined_embeddings,cluster_model.labels_,save_path=cluster_plot_path)
        combined_embeddings.to_csv(LATENT_COMPRESSED_PATH.split(".")[0]+"conv_tabular_latent_clustered.csv",index=False)
        wrong_result_df=analyze_cluster_distribution(combined_embeddings,WRONG_LABELED_IMG_FOLDER) # pca viz ve siluate score
        print(wrong_result_df)
        
        # -- CLUSTER VISUALIZATION --
        # Create base output directory if it doesn't exist 
        os.makedirs(cluster_folder, exist_ok=True)
        if visualize_clusters:
            visualize_df_clusters(combined_embeddings,cluster_folder,
                                num_clusters=VIZ_NUM_CLUSTERS,
                                image_folder=image_folder,
                                label_json=ann_file,
                                pred_json=res_file,
                                cluster_select_list=INSIDE_VIZ_NUM_CLUSTERS)