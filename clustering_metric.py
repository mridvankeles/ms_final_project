import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
        
def analyze_clusters(self,cluster_base_path,mislabeled_folder, true_labeled_folder):
    cluster_stats = []
    
    # Ensure paths exist
    if not os.path.exists(cluster_base_path) or not os.path.exists(mislabeled_folder) or not os.path.exists(true_labeled_folder):
        print("Error: One or more paths do not exist.")
        return None
    
    cluster_folders = [f for f in os.listdir(cluster_base_path) if os.path.isdir(os.path.join(cluster_base_path, f))]
    
    mislabeled_images_set = set(os.path.splitext(i)[0] for i in os.listdir(mislabeled_folder))
    true_labeled_images_set = set(os.path.splitext(i)[0] for i in os.listdir(true_labeled_folder))
    
    total_images = 0
    for cluster in cluster_folders:
        cluster_path = os.path.join(cluster_base_path, cluster)
        images=set(os.listdir(cluster_path))
        cluster_base_names = set(i.split("_")[0] for i in images)
        total_images += len(images)
        
        mislabeled_images = cluster_base_names.intersection(mislabeled_images_set)
        true_labeled_images = cluster_base_names.intersection(true_labeled_images_set)
        
        total = len(images)
        mislabeled_count = len(mislabeled_images)
        true_labeled_count = len(true_labeled_images)
        
        cluster_stats.append({
            'Cluster': cluster,
            'Total Images': total,
            'Mislabeled': mislabeled_count,
            'True Labeled': true_labeled_count,
            'Mislabeled Ratio': mislabeled_count / total if total > 0 else 0,
            'True Labeled Ratio': true_labeled_count / total if total > 0 else 0,
            'Cluster Size Ratio': total / total_images if total_images > 0 else 0
        })
    
    df = pd.DataFrame(cluster_stats)
    
    if df.empty:
        print("No data to display. Check folder paths and contents.")
        return None

    # Sort by cluster number
    df['Cluster_Num'] = df['Cluster'].str.extract('(\d+)').astype(int)
    df = df.sort_values('Cluster_Num')
    df = df.drop('Cluster_Num', axis=1)

    #Create visualizations
    #create_cluster_analysis_plots(df)
    
    return df

def create_cluster_analysis_plots(df):
    # Create output directory for plots
    plot_path = 'outputs/label_error_analysis_cluster2/plots'
    os.makedirs(plot_path, exist_ok=True)
    
    # 1. Cluster Size Distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Cluster', y='Cluster Size Ratio')
    plt.title('Cluster Size Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Proportion of Total Images')
    plt.tight_layout()
    plt.savefig(f'{plot_path}/cluster_size_distribution.png')
    plt.close()

    # 2. Label Distribution within Clusters
    plt.figure(figsize=(12, 6))
    df_melted = pd.melt(df, 
                        id_vars=['Cluster'],
                        value_vars=['Mislabeled Ratio', 'True Labeled Ratio'],
                        var_name='Label Type',
                        value_name='Ratio')
    sns.barplot(data=df_melted, x='Cluster', y='Ratio', hue='Label Type')
    plt.title('Label Distribution within Clusters')
    plt.xticks(rotation=45)
    plt.ylabel('Ratio')
    plt.tight_layout()
    plt.savefig(f'{plot_path}/label_distribution.png')
    plt.close()

    # 3. Scatter plot of cluster size vs mislabeled ratio
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Cluster Size Ratio'], df['Mislabeled Ratio'])
    plt.xlabel('Cluster Size Ratio')
    plt.ylabel('Mislabeled Ratio')
    plt.title('Cluster Size vs Mislabeled Ratio')
    for i, row in df.iterrows():
        plt.annotate(f"Cluster {row['Cluster']}", 
                    (row['Cluster Size Ratio'], row['Mislabeled Ratio']))
    plt.tight_layout()
    plt.savefig(f'{plot_path}/size_vs_mislabeled.png')
    plt.close()

    # Calculate and print summary statistics
    print("\nCluster Analysis Summary:")
    print("-" * 50)
    print(f"Total number of clusters: {len(df)}")
    print(f"Average cluster size: {df['Total Images'].mean():.2f} images")
    print(f"Average mislabeled ratio: {df['Mislabeled Ratio'].mean():.2%}")
    print("\nTop 3 clusters with highest mislabeled ratio:")
    print(df.nlargest(3, 'Mislabeled Ratio')[['Cluster', 'Mislabeled Ratio', 'Total Images']])
    print("\nCluster size distribution:")
    print(df[['Cluster', 'Cluster Size Ratio']].to_string())

    # Example usage
    if __name__ == "__main__":
        cluster_base_path = r"data\whumix\val\clusters_loss_weight\clusters_3"
        mislabeled_folder = r"data\whumix\val\labeled_val\False"
        true_labeled_folder = r"data\whumix\val\labeled_val\True"

        result_df = analyze_clusters(cluster_base_path, mislabeled_folder, true_labeled_folder)
        print("\nCluster Analysis Summary:")
        print("-" * 50)
        print(f"Total number of clusters: {len(result_df)}")
        print(f"Average cluster size: {result_df['Total Images'].mean():.2f} images")
        print(f"Average mislabeled ratio: {result_df['Mislabeled Ratio'].mean():.2%}")
        print("\nTop 3 clusters with highest mislabeled ratio:")
        print(result_df.nlargest(3, 'Mislabeled Ratio')[['Cluster', 'Mislabeled Ratio', 'Total Images']])
        print("\nCluster size distribution:")
        print(result_df[['Cluster', 'Cluster Size Ratio']].to_string())