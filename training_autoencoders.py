from tabular_autoencoder import TabularAutoencoder
from conv_autoencoder import ConvAutoencoder
import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Autoencoder parameters
TABULAR_ENCODING_DIM = 3
CONV_ENCODING_DIM = 64 

# Training parameters
EPOCHS = 50
LR = 0.01


def train_autoencoder(model, data_loader, epochs, lr, device):
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
    for epoch in tqdm(range(epochs)):
        model.to(device)
        model.train()
        epoch_loss = 0
        for data in data_loader:
            inputs = data.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {epoch_loss/len(data_loader):.4f}')
        scheduler.step()
    return model

def get_embeddings(model, data_loader, device):
    embeddings = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            inputs = data.to(device)
            encoded = model.encode(inputs)
            embeddings.append(encoded.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def combine_embeddings(metric_path,loss_path,tabular_latent_path,conv_feature_path,conv_latent_path,compress_tabular=False,compress_mask=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    metric_df = pd.read_csv(metric_path)
    metric_df["image_id"] = [str(img_id).zfill(12) for img_id in metric_df["image_id"]]
    loss_df = pd.read_csv(loss_path)
    loss_df["image_id"] = [str(img_id).zfill(12) for img_id in loss_df["image_id"]]
    
    tabular_df = pd.merge(loss_df, metric_df,on="image_id",how="left")
    tabular_df.to_csv(tabular_latent_path.split(".")[0]+".csv",index=False)

    if compress_tabular:
        # --- Tabular Autoencoder --- 
        print("\n--- Training Tabular Autoencoder ---")
        tabular_ae = TabularAutoencoder(input_dim=len(tabular_df.drop("image_id",axis=1).columns), encoding_dim=TABULAR_ENCODING_DIM)
        print("tabular data loaded.")
        try:
            tabular_data = tabular_ae.load_data(metric_df = tabular_df.drop("image_id",axis=1))
            
            assert not np.isnan(tabular_data).any()
            assert not np.isinf(tabular_data).any()
            tabular_loader = torch.utils.data.DataLoader(tabular_data, batch_size=32, shuffle=False)
            if not os.path.exists(tabular_latent_path):
                print("training autoencoder model...")
                trained_tabular_ae = train_autoencoder(tabular_ae, tabular_loader, 200, LR, device)
                torch.save(trained_tabular_ae,tabular_latent_path)
            else:
                print("trained model loading...")
                trained_tabular_ae = torch.load(tabular_latent_path)
            
            tabular_embeddings = get_embeddings(trained_tabular_ae, tabular_loader, device)
            tabular_embeddings = pd.DataFrame(tabular_embeddings,columns=["tabular_1","tabular_2","tabular_3"])
            tabular_embeddings["image_id"] = tabular_df["image_id"]
            np.save('outputs/tabular_embeddings.npy', tabular_embeddings)

        except NotImplementedError as e:
            print(f"Skipping Tabular Autoencoder: {e}")
            tabular_embeddings = None
        except FileNotFoundError:
            print(f"Skipping Tabular Autoencoder: {metric_path} not found.")
            tabular_embeddings = None
    else:
        tabular_embeddings = tabular_df

    # --- Convolutional Autoencoder ---
    if compress_mask:
        conv_ae = ConvAutoencoder() # 
        try:
            conv_features = conv_ae.load_data(conv_feature_path) 
            conv_loader = torch.utils.data.DataLoader(conv_features, batch_size=64, shuffle=False)
            if not os.path.exists(conv_latent_path):
                print("\n--- Training Convolutional Autoencoder ---")
                trained_conv_ae = train_autoencoder(conv_ae, conv_loader, EPOCHS, LR, device)
                torch.save(trained_conv_ae,conv_latent_path)
            else:
                print("\n--- Loading Trained Convolutional Latent Features ---")
                trained_conv_ae = torch.load(conv_latent_path)
            
            conv_embeddings = get_embeddings(trained_conv_ae, conv_loader, device)
            conv_embeddings = conv_embeddings.reshape(-1, 3*8*8)

            np.save('outputs/conv_embeddings.npy', conv_embeddings)
        except NotImplementedError as e:
            print(f"Skipping Convolutional Autoencoder: {e}")
            conv_embeddings = None
        except FileNotFoundError:
            print(f"Skipping Convolutional Autoencoder: {conv_latent_path} not found.")
            conv_embeddings = None
    else:
        conv_embeddings = "empty"

    return tabular_embeddings,conv_embeddings