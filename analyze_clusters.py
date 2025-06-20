import os
import pandas as pd

def analyze_cluster_distribution(df, folder_path):
    """
    Analyzes the cluster distribution of images from a specified folder within a DataFrame
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'image_id' and cluster columns (e.g., 'cluster_2', 'cluster_3')
    folder_path (str): Path to the folder containing images to analyze
    
    Returns:
    dict: Cluster distribution information for each cluster column, containing:
        - total_images: Total matched images
        - cluster_distribution: Dictionary of {cluster_number: count}
        - percentages: Dictionary of {cluster_number: percentage}
    """
    
    # Get image IDs from folder (strip extensions)
    folder_images = [f.split(".")[0]  for f in os.listdir(folder_path)]
    print(len(folder_images))
    
    # Find matching rows in DataFrame
    matched_df = df[df['image_id'].isin(folder_images)]
    
    # Check for unmatched images
    unmatched_images = set(folder_images) - set(matched_df['image_id'])
    if unmatched_images:
        print(f"Warning: {len(unmatched_images)} images not found in DataFrame")
    
    # Identify cluster columns
    cluster_columns = [col for col in df.columns if col.startswith('cluster_')]
    
    # Calculate distributions
    results = {}
    for col in cluster_columns:
        cluster_counts = matched_df[col].value_counts().to_dict()
        
        results[col] = {
            'total_images': len(matched_df),
            'cluster_distribution': cluster_counts,
            'percentages': {k: (v / len(matched_df)) * 100 for k, v in cluster_counts.items()}
            }
    
    # Print results
    for col, data in results.items():
        print(f"\n {col} distribution:")
        print(f"Total matched images: {data['total_images']}")
        print(df[f"{col}"].value_counts())
        for cluster, count in data['cluster_distribution'].items():
            print(f"Cluster {cluster}: {count} images ({data['percentages'][cluster]:.1f}%)")
    
    return results