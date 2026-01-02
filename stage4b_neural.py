"""
Stage 4B: Neural Towers Training
Three-tower neural network for recommendation

This stage performs:
1. Define Three-Tower architecture (User, Item, Image towers)
2. Train neural network with MPS/CUDA acceleration
3. Evaluate with MAP@12
4. Save trained model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import gc

from config import NeuralTowerConfig
from utils import print_section, force_garbage_collection, print_memory
from metrics import evaluate_map_at_12


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss for class imbalance"""
    
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        if self.pos_weight is not None:
            if isinstance(self.pos_weight, (int, float)):
                pos_weight_val = float(self.pos_weight)
            else:
                pos_weight_val = float(self.pos_weight.item()) if hasattr(self.pos_weight, 'item') else float(self.pos_weight)
            
            weights = torch.where(
                targets == 1,
                torch.tensor(pos_weight_val, device=inputs.device, dtype=torch.float32),
                torch.tensor(1.0, device=inputs.device, dtype=torch.float32)
            )
            weighted_loss = bce_loss * weights
            return weighted_loss.mean()
        else:
            return bce_loss.mean()


class ThreeTowerModel(nn.Module):
    """
    Three-tower neural network for recommendation:
    - User Tower: User features -> User embedding
    - Item Tower: Item features -> Item embedding
    - Image Tower: Image embeddings -> Image embedding
    - Fusion: Concatenated embeddings -> Final prediction
    """
    
    def __init__(self,
                 user_feature_dim,
                 item_feature_dim,
                 image_feature_dim,
                 user_embedding_dim=128,
                 item_embedding_dim=64,
                 image_embedding_dim=128,
                 fusion_hidden_dims=[256, 128, 64],
                 dropout_rate=0.3):
        super(ThreeTowerModel, self).__init__()
        
        # User Tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, user_embedding_dim),
            nn.BatchNorm1d(user_embedding_dim),
            nn.ReLU()
        )
        
        # Item Tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, item_embedding_dim),
            nn.BatchNorm1d(item_embedding_dim),
            nn.ReLU()
        )
        
        # Image Tower
        self.image_tower = nn.Sequential(
            nn.Linear(image_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, image_embedding_dim),
            nn.BatchNorm1d(image_embedding_dim),
            nn.ReLU()
        )
        
        # Fusion Layer
        fusion_input_dim = user_embedding_dim + item_embedding_dim + image_embedding_dim
        fusion_layers = []
        
        prev_dim = fusion_input_dim
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        fusion_layers.append(nn.Linear(prev_dim, 1))
        fusion_layers.append(nn.Sigmoid())
        
        self.fusion = nn.Sequential(*fusion_layers)
    
    def forward(self, user_features, item_features, image_features):
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        image_emb = self.image_tower(image_features)
        
        fused = torch.cat([user_emb, item_emb, image_emb], dim=1)
        output = self.fusion(fused)
        
        return output.squeeze()


class RecommendationDataset(Dataset):
    """Dataset for recommendation training"""
    
    def __init__(self, df, user_features, item_features, image_features, labels=None):
        self.df = df.reset_index(drop=True)
        self.user_features = user_features.values.astype(np.float32)
        self.item_features = item_features.values.astype(np.float32)
        self.image_features = image_features.values.astype(np.float32)
        self.labels = labels.values.astype(np.float32) if labels is not None else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        user_feat = torch.FloatTensor(self.user_features[idx])
        item_feat = torch.FloatTensor(self.item_features[idx])
        image_feat = torch.FloatTensor(self.image_features[idx])
        
        if self.labels is not None:
            label = torch.FloatTensor([self.labels[idx]])
            return user_feat, item_feat, image_feat, label
        else:
            return user_feat, item_feat, image_feat


def identify_feature_groups(feature_cols):
    """Identify user, item, and image features"""
    user_feature_cols = []
    item_feature_cols = []
    image_feature_cols = []
    
    for col in feature_cols:
        col_lower = col.lower()
        
        if any(prefix in col_lower for prefix in NeuralTowerConfig.IMAGE_FEATURE_PREFIXES):
            image_feature_cols.append(col)
        elif any(prefix in col_lower for prefix in NeuralTowerConfig.USER_FEATURE_PREFIXES):
            user_feature_cols.append(col)
        elif any(prefix in col_lower for prefix in NeuralTowerConfig.ITEM_FEATURE_PREFIXES):
            item_feature_cols.append(col)
        else:
            # Default to item features
            item_feature_cols.append(col)
    
    # Ensure minimum features in each group
    if len(user_feature_cols) == 0:
        user_feature_cols = item_feature_cols[:10]
    if len(image_feature_cols) == 0:
        image_feature_cols = item_feature_cols[:5]
    
    return user_feature_cols, item_feature_cols, image_feature_cols


def load_and_prepare_data():
    """Load and prepare data for neural training"""
    print_section("LOADING DATA FOR NEURAL TOWERS")
    
    train_data = pd.read_parquet(NeuralTowerConfig.MODEL_PATH / 'train_data.parquet')
    print(f"Loaded {len(train_data):,} training samples")
    
    val_data = pd.read_parquet(NeuralTowerConfig.MODEL_PATH / 'val_data.parquet')
    print(f"Loaded {len(val_data):,} validation samples")
    
    # Identify feature columns
    exclude_cols = ['customer_id', 'article_id', 'label', 'user_type', 'train_label', 'val_label']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    # Keep only numeric features
    numeric_features = train_data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Identify feature groups
    user_feature_cols, item_feature_cols, image_feature_cols = identify_feature_groups(numeric_features)
    
    print(f"\nFeature groups:")
    print(f"  User features: {len(user_feature_cols)}")
    print(f"  Item features: {len(item_feature_cols)}")
    print(f"  Image features: {len(image_feature_cols)}")
    
    # Handle missing values
    train_data[numeric_features] = train_data[numeric_features].fillna(0)
    val_data[numeric_features] = val_data[numeric_features].fillna(0)
    
    # Scale features
    scaler_user = StandardScaler()
    scaler_item = StandardScaler()
    scaler_image = StandardScaler()
    
    X_train_user = pd.DataFrame(
        scaler_user.fit_transform(train_data[user_feature_cols]),
        columns=user_feature_cols
    )
    X_train_item = pd.DataFrame(
        scaler_item.fit_transform(train_data[item_feature_cols]),
        columns=item_feature_cols
    )
    X_train_image = pd.DataFrame(
        scaler_image.fit_transform(train_data[image_feature_cols]),
        columns=image_feature_cols
    )
    
    X_val_user = pd.DataFrame(
        scaler_user.transform(val_data[user_feature_cols]),
        columns=user_feature_cols
    )
    X_val_item = pd.DataFrame(
        scaler_item.transform(val_data[item_feature_cols]),
        columns=item_feature_cols
    )
    X_val_image = pd.DataFrame(
        scaler_image.transform(val_data[image_feature_cols]),
        columns=image_feature_cols
    )
    
    y_train = train_data['label']
    y_val = val_data['label']
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'X_train_user': X_train_user,
        'X_train_item': X_train_item,
        'X_train_image': X_train_image,
        'X_val_user': X_val_user,
        'X_val_item': X_val_item,
        'X_val_image': X_val_image,
        'y_train': y_train,
        'y_val': y_val,
        'user_feature_cols': user_feature_cols,
        'item_feature_cols': item_feature_cols,
        'image_feature_cols': image_feature_cols,
        'scaler_user': scaler_user,
        'scaler_item': scaler_item,
        'scaler_image': scaler_image,
    }


def train_neural_model(data):
    """Train the Three-Tower neural network"""
    print_section("TRAINING NEURAL TOWER MODEL")
    
    # Get device
    device = NeuralTowerConfig.get_device()
    
    # Create datasets
    train_dataset = RecommendationDataset(
        data['train_data'],
        data['X_train_user'],
        data['X_train_item'],
        data['X_train_image'],
        data['y_train']
    )
    
    val_dataset = RecommendationDataset(
        data['val_data'],
        data['X_val_user'],
        data['X_val_item'],
        data['X_val_image'],
        data['y_val']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=NeuralTowerConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=NeuralTowerConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    model = ThreeTowerModel(
        user_feature_dim=len(data['user_feature_cols']),
        item_feature_dim=len(data['item_feature_cols']),
        image_feature_dim=len(data['image_feature_cols']),
        user_embedding_dim=NeuralTowerConfig.USER_EMBEDDING_DIM,
        item_embedding_dim=NeuralTowerConfig.ITEM_EMBEDDING_DIM,
        image_embedding_dim=NeuralTowerConfig.IMAGE_EMBEDDING_DIM,
        fusion_hidden_dims=NeuralTowerConfig.FUSION_HIDDEN_DIMS,
        dropout_rate=NeuralTowerConfig.DROPOUT_RATE
    )
    
    model = model.to(device)
    print(f"\nModel on device: {device}")
    
    # Loss and optimizer
    pos_weight = (data['y_train'] == 0).sum() / (data['y_train'] == 1).sum()
    criterion = WeightedBCELoss(pos_weight=min(pos_weight, 3.0))
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=NeuralTowerConfig.LEARNING_RATE,
        weight_decay=NeuralTowerConfig.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_map12': []
    }
    
    best_map12 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    checkpoint_dir = NeuralTowerConfig.MODEL_PATH / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(NeuralTowerConfig.N_EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        
        for user_feat, item_feat, image_feat, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            user_feat = user_feat.to(device)
            item_feat = item_feat.to(device)
            image_feat = image_feat.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(user_feat, item_feat, image_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_losses = []
        all_predictions = []
        
        with torch.no_grad():
            for user_feat, item_feat, image_feat, labels in val_loader:
                user_feat = user_feat.to(device)
                item_feat = item_feat.to(device)
                image_feat = image_feat.to(device)
                labels = labels.to(device).squeeze()
                
                outputs = model(user_feat, item_feat, image_feat)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                all_predictions.extend(outputs.cpu().numpy().tolist())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Calculate MAP@12
        map12_score = evaluate_map_at_12(data['val_data'], np.array(all_predictions))
        history['val_map12'].append(map12_score)
        
        scheduler.step(map12_score)
        
        print(f"\nEpoch {epoch+1}/{NeuralTowerConfig.N_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val MAP@12: {map12_score:.6f}")
        
        # Save best model
        if map12_score > best_map12:
            best_map12 = map12_score
            best_epoch = epoch + 1
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map12': map12_score,
                'history': history
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  Saved best model (MAP@12: {map12_score:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{NeuralTowerConfig.EARLY_STOPPING_PATIENCE})")
        
        # Early stopping
        if patience_counter >= NeuralTowerConfig.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        gc.collect()
    
    # Load best model
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nBest Model: Epoch {best_epoch}, MAP@12: {best_map12:.6f}")
    
    return model, history, best_map12, best_epoch


def save_model(model, data, history, best_map12, best_epoch):
    """Save trained model and metadata"""
    print_section("SAVING MODEL")
    
    final_model_path = NeuralTowerConfig.MODEL_PATH / 'neural_tower_model.pt'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'user_feature_dim': len(data['user_feature_cols']),
            'item_feature_dim': len(data['item_feature_cols']),
            'image_feature_dim': len(data['image_feature_cols']),
            'user_embedding_dim': NeuralTowerConfig.USER_EMBEDDING_DIM,
            'item_embedding_dim': NeuralTowerConfig.ITEM_EMBEDDING_DIM,
            'image_embedding_dim': NeuralTowerConfig.IMAGE_EMBEDDING_DIM,
            'fusion_hidden_dims': NeuralTowerConfig.FUSION_HIDDEN_DIMS,
            'dropout_rate': NeuralTowerConfig.DROPOUT_RATE
        },
        'feature_cols': {
            'user': data['user_feature_cols'],
            'item': data['item_feature_cols'],
            'image': data['image_feature_cols']
        },
        'scalers': {
            'user': data['scaler_user'],
            'item': data['scaler_item'],
            'image': data['scaler_image']
        },
        'best_map12': best_map12,
        'best_epoch': best_epoch,
        'history': history
    }, final_model_path)
    
    print(f"Saved model to {final_model_path}")
    
    # Save training history
    history_path = NeuralTowerConfig.MODEL_PATH / 'neural_tower_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Saved history to {history_path}")


def run_stage4b():
    """Run the complete Stage 4B pipeline"""
    print_section("STAGE 4B: NEURAL TOWERS TRAINING")
    
    device = NeuralTowerConfig.get_device()
    print(f"Device: {device}")
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Train model
    model, history, best_map12, best_epoch = train_neural_model(data)
    
    # Save model
    save_model(model, data, history, best_map12, best_epoch)
    
    print_section("STAGE 4B COMPLETE")
    print(f"Best Model: Epoch {best_epoch}")
    print(f"Best MAP@12: {best_map12:.6f}")
    
    force_garbage_collection()
    
    return model, history


if __name__ == "__main__":
    run_stage4b()

