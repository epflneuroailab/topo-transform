import config

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import wandb
from torch.utils.data import DataLoader
import numpy as np

from models import VJEPA, UniFormer, MViTV1
from topo import TopoTransformedVJEPA, SpatialCorrelationLoss
from data import SmthSmthV2, Kinetics400, ImageNetVid
from validate import load_transformed_model, validate_autocorr, validate_floc


"""
Notes on training:
1. we could incorporate more datasets (e.g., Kinetics-400) for better statistics
"""


if hasattr(config, 'WANDB_API_KEY'):
    os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Lambda(lambda img: img/255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_config_id(model_name, data_name, lr, batch_size=32):
    """Generate unique config identifier"""
    lr_str = f"lr{lr}".replace('.', 'p') if lr >= 0.001 else f"lr{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
    return f"{model_name}_{data_name}_{lr_str}_bs{batch_size}"


def train_model(model, train_loader, val_loader, criterion, config_id, storage, figure_dir,
                device='cuda', num_epochs=50, lr=1e-3, resume=True, use_wandb=False, 
                wandb_project='topo-transform', wandb_run_name=None):
    """Train transformed model with checkpointing and W&B logging"""
    
    checkpoint_dir = config.CACHE_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'best_transformed_model_{config_id}.pt'
    
    # Initialize W&B
    wandb_run_id = None
    if use_wandb:
        if resume and checkpoint_path.exists():
            wandb_run_id = torch.load(checkpoint_path, map_location=device).get('wandb_run_id')
        
        wandb.init(project=wandb_project, name=wandb_run_name or config_id, id=wandb_run_id,
                   resume="allow" if wandb_run_id else None,
                   config={'config_id': config_id, 'num_epochs': num_epochs, 'lr': lr,
                          'batch_size': train_loader.batch_size, 'device': device})
        wandb_run_id = wandb.run.id
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Load checkpoint
    start_epoch, best_val_loss, train_losses, val_losses = 0, float('inf'), [], []
    if resume and checkpoint_path.exists():
        print(f"\n{'='*70}\nResuming from: {checkpoint_path}\n{'='*70}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['transformed_model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            for _ in range(ckpt['epoch'] + 1): 
                scheduler.step()
        
        start_epoch, best_val_loss = ckpt['epoch'] + 1, ckpt['val_loss']
        
        results_path = storage / f"{config_id}_results.pkl"
        if results_path.exists():
            results = torch.load(results_path)
            train_losses, val_losses = results.get('train_losses', []), results.get('val_losses', [])
        
        print(f"Resuming from epoch {start_epoch + 1}/{num_epochs}, Best val loss: {best_val_loss:.6f}\n{'='*70}\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # # Train
        # model.train()
        # train_loss = 0.0
        # pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        # for batch_idx, batch in enumerate(pbar):
        #     videos = batch[0].to(device, non_blocking=True)
        #     optimizer.zero_grad()
        #     loss = criterion(*model(videos))
        #     loss.backward()
        #     optimizer.step()
        #     train_loss += loss.item()
        #     pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
        #     if use_wandb:
        #         wandb.log({'batch_loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']},
        #                  step=epoch * len(train_loader) + batch_idx)
        
        # train_loss /= len(train_loader)
        # train_losses.append(train_loss)
        
        # # Validate
        # model.eval()
        # val_loss = 0.0
        # val_features = []
        # with torch.no_grad():
        #     for batch in tqdm(val_loader, desc='Validation'):
        #         videos = batch[0].to(device, non_blocking=True)
        #         val_feature, layer_positions = model(videos)
        #         loss = criterion(val_feature, layer_positions)
        #         val_features.append([v.cpu() for v in val_feature])
        #         val_loss += loss.item()
        
        # val_loss /= len(val_loader)
        # val_losses.append(val_loss)
        
        # print(f'\nEpoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # # Visualization
        # validate_autocorr(val_features, layer_positions, figure_dir, epoch=epoch)

        with model.smoothing_enabled(fwhm_mm=0.0, resolution_mm=1.0):
            validate_floc(
                model, 
                vit_transform, 
                dataset_names=[
                    "biomotion", 
                    "vpnl", 
                    "kanwisher", 
                    "pitzalis",
                    "motion", 
                    "temporal",
                    # "pitcher",
                ],
                viz_dir=figure_dir,
                device=device,
                plot_individual=True,
                plot_aggregate=True,
                epoch=epoch
            )
        
        # W&B logging
        if use_wandb:
            wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                      'best_val_loss': best_val_loss, 'lr': optimizer.param_groups[0]['lr']},
                     step=(epoch + 1) * len(train_loader))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'config_id': config_id,
                'transformed_model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss,
                'layer_dims': getattr(model, 'layer_dims', None),
                'layer_names': getattr(model, 'layer_names', None),
                'wandb_run_id': wandb_run_id
            }, checkpoint_path)
            print(f'  → Saved best model (val_loss: {best_val_loss:.6f})')
            if use_wandb: 
                wandb.log({'best_model_epoch': epoch + 1})
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch, 'config_id': config_id, 
                'transformed_model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss, 'val_loss': val_loss,
                'wandb_run_id': wandb_run_id
            }, checkpoint_dir / f'checkpoint_transformed_model_{config_id}_epoch_{epoch+1}.pt')
            print(f'  → Saved checkpoint at epoch {epoch+1}')
        
        scheduler.step()
        
        # Save results
        torch.save({
            'train_losses': train_losses, 'val_losses': val_losses,
            'best_val_loss': best_val_loss, 'config_id': config_id,
            'layer_configs': getattr(model, 'layer_configs', None)
        }, storage / f"{config_id}_results.pkl")
        print()
    
    if use_wandb: 
        wandb.finish()
    
    return model, train_losses, val_losses


if __name__ == '__main__':
    # Config
    model_name = 'vjepa'
    data_name = 'smthsmthv2'  # 'smthsmthv2', 'kinetics400', 'imagenet'
    layer_indices = [14, 18, 22] 
    batch_size = 32
    lr = 1e-4
    num_epochs = 5
    neighborhoods_per_batch = 128
    exponentially_interpolate = False
    constant_rf_overlap = False
    large_neighborhood = False
    inf_neighborhood = True
    single_sheet = True
    use_wandb, wandb_project = False, 'tdann-transform'
    resume_training = True
    seed = 42

    # seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Data
    if data_name == 'smthsmthv2':
        data = SmthSmthV2(train_transforms=vit_transform, test_transforms=vit_transform)
    elif data_name == 'kinetics400':
        data = Kinetics400(train_transforms=vit_transform, test_transforms=vit_transform)
    elif data_name == 'imagenet':
        data = ImageNetVid(train_transforms=vit_transform, test_transforms=vit_transform)
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    train_loader = DataLoader(data.trainset, batch_size=batch_size, shuffle=True, 
                             num_workers=int(batch_size/1.5), pin_memory=True)
    val_loader = DataLoader(data.valset, batch_size=batch_size, shuffle=False, 
                           num_workers=int(batch_size/1.5), pin_memory=True)
    
    # Model setup
    model = TopoTransformedVJEPA(layer_indices=layer_indices, exponentially_interpolate=exponentially_interpolate, constant_rf_overlap=constant_rf_overlap, 
                                 single_sheet=single_sheet, large_neighborhood=large_neighborhood, inf_neighborhood=inf_neighborhood, seed=seed)
    config_id = get_config_id(model.name, data_name, lr, batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neighborhoods_per_batch = neighborhoods_per_batch if not single_sheet else neighborhoods_per_batch * len(layer_indices)
    criterion = SpatialCorrelationLoss(model.num_layers, neighborhoods_per_batch=neighborhoods_per_batch, single_sheet=single_sheet)
    
    # Directories
    storage_path = config.CACHE_DIR / "train_topo" / config_id
    storage_path.mkdir(parents=True, exist_ok=True)
    figure_dir = config.CACHE_DIR / "figures" / config_id
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}\nTraining {model_name} | Config: {config_id} | Device: {device}\n{'='*70}")
    
    # Train
    trained_model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, config_id, storage_path,
        figure_dir, device, num_epochs=num_epochs, lr=lr, resume=resume_training,
        use_wandb=use_wandb, wandb_project=wandb_project, wandb_run_name=config_id)
    
    # Plot & save
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'TopoTransform Training - {config_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figure_dir / f'topo_training_{config_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save final results
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': min(val_losses),
        'config_id': config_id,
        'layer_configs': getattr(model, 'layer_dims', None)
    }, storage_path / f"{config_id}_results.pkl")
    
    print(f"\n{'='*70}\nTraining Complete! Best Val Loss: {min(val_losses):.6f}\n{'='*70}")