import os
from PIL import Image, ImageDraw
from tools.evaluation_labelcheck import visualize_segmentation
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def visualize_df_clusters(df,cluster_folder,num_clusters,image_folder,label_json,pred_json,cluster_select_list = None): 
    # Iterate over each cluster and save images
    for i in num_clusters:
        cluster_folder = rf"{cluster_folder}/clusters_{i}"
        # Create base output directory if it doesn't exist
        os.makedirs(cluster_folder, exist_ok=True)
        if cluster_select_list is None:
            cluster_select_list = df[f'cluster_{i}'].unique()
        
        for cluster in cluster_select_list:
            cluster_dir = os.path.join(cluster_folder, f'cluster_{cluster}')
            os.makedirs(cluster_dir, exist_ok=True)

            cluster_images = df[df[f'cluster_{i}'] == cluster]['image_id']

            for image_id in tqdm(cluster_images):
                print("imageid:",image_id)
                visualize_segmentation(image_id, image_folder, label_json, pred_json, cluster_dir)

        print("Images saved successfully to their respective cluster folders.")


def visualize_clusters_pca(df_scaled,cluster_labels,save_path):
    # Calculate the silhouette score
    os.makedirs(save_path, exist_ok=True)
    sil_score = silhouette_score(df_scaled, cluster_labels)
    print(f'Silhouette Score: {sil_score:.2f}')

    # Optional: Visualize the clusters using PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)

    plt.figure(figsize=(10,6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title(f'Silhouette Score: {sil_score:.2f}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f"{save_path}/{sil_score}.jpg")