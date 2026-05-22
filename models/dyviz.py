import sys
import torch
import torch.nn as nn
import yaml
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from contextlib import contextmanager


# ============================================================================
# Path Management Context Manager
# ============================================================================

@contextmanager
def add_to_path(path: Union[str, Path, List[Union[str, Path]]], position: int = 0):
    """
    Context manager to temporarily add folder(s) to sys.path for package imports.
    
    Args:
        path: Single path or list of paths to add to sys.path
        position: Position to insert path(s) in sys.path (0 = beginning, -1 = end)
        
    Yields:
        None
        
    Example:
        with add_to_path('/path/to/models'):
            from models.VJEPA.TJEPA import TJEPA
    """
    # Convert to list if single path
    if not isinstance(path, list):
        paths = [path]
    else:
        paths = path
    
    # Convert to strings and resolve
    paths = [str(Path(p).resolve()) for p in paths]
    
    # Store original sys.path
    original_path = sys.path.copy()
    
    try:
        # Add paths
        for p in reversed(paths):  # Reverse to maintain order when inserting at position
            if p not in sys.path:
                if position == -1:
                    sys.path.append(p)
                else:
                    sys.path.insert(position, p)
                    
        yield
        
    finally:
        # Restore original sys.path
        sys.path = original_path


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str) -> Any:
    """
    Load config from YAML file and convert to namespace for attribute access.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        Config object with attribute-style access
    """
    import importlib
    import config
    importlib.reload(config)
    from config import cfg

    cfg.WARMUP_GAMMA = 1  # Disable warmup by default
    cfg.TARGET_GAMMA = 1.0  # Disable target gamma by default

    cfg.merge_from_file(config_path)
    cfg.WANDB_MODE = "offline"
    if cfg.DDP:
        cfg.SOLVER.VIDEOS_PER_BATCH = cfg.SOLVER.VIDEOS_PER_BATCH // cfg.WORLD_SIZE # Int. div. to ensure batch size divisible by #(GPU)
    
    return cfg


def parse_ds_classes(cfg) -> Dict[str, int]:
    """
    Parse dataset classes from config.
    
    Args:
        cfg: Config object
        
    Returns:
        Dictionary mapping dataset names to number of classes
    """
    ds_classes = {}
    
    # Try to get from config
    for name in cfg.DATASETS.TRAINING.NAMES:
        if name == 'kinetics400':
            ds_classes[name] = 400
        elif name == 'smthsmthv2':
            ds_classes[name] = 174
        elif name == 'diving48':
            ds_classes[name] = 48
        elif name == 'imagenet':
            ds_classes[name] = 1000
        
    return ds_classes


# ============================================================================
# Model Conversion
# ============================================================================

def convert_tjepa_to_pytorch(
    tjepa_model,
    mode: str = 'single',
    classifier_name: Optional[str] = None,
    return_features: bool = False
) -> nn.Module:
    """
    Convert TJEPA model to a standard PyTorch model.
    
    Args:
        tjepa_model: The TJEPA model instance to convert
        mode: Conversion mode - 'single' for single classifier, 'multi' for all classifiers
        classifier_name: Name of the classifier to use in 'single' mode (e.g., 'kinetics400'). 
                        If None in single mode, returns features only.
        return_features: If True in 'single' mode, returns pooled features instead of logits
    
    Returns:
        A standard nn.Module with a simple forward() method
    """
    
    class SimpleTJEPA(nn.Module):
        def __init__(
            self, 
            backbone, 
            head_pooler, 
            classifier=None,
            remap_indices=None
        ):
            super(SimpleTJEPA, self).__init__()
            self.backbone = backbone
            self.head_pooler = head_pooler
            self.classifier = classifier
            self.remap_indices = remap_indices
            
        def forward(self, x):
            x = x.permute(0, 2, 1, 3, 4)  # Change to (B, C, T, H, W)
            # Extract features from backbone
            features = self.backbone(x)
            
            # Pool features
            pooled = self.head_pooler(features)
            pooled_features = pooled[:, 0]  # Take CLS token
            
            # Optionally classify
            if self.classifier is not None:
                logits = self.classifier(pooled_features)
                
                # Apply remapping if needed (for kinetics400)
                if self.remap_indices is not None:
                    logits = logits[:, self.remap_indices]
                    
                return logits
            else:
                return pooled_features
    
    class MultiHeadTJEPA(nn.Module):
        def __init__(self, backbone, head_pooler, classifiers, ds_classes):
            super(MultiHeadTJEPA, self).__init__()
            self.backbone = backbone
            self.head_pooler = head_pooler
            self.classifiers = classifiers
            self.ds_classes = ds_classes
            
        def forward(self, x, dataset_names: Optional[List[str]] = None):
            """
            Args:
                x: Input tensor
                dataset_names: List of dataset names to use. If None, uses all.
            
            Returns:
                Dict mapping dataset name to logits
            """
            # Extract and pool features
            features = self.backbone(x)
            pooled = self.head_pooler(features)
            pooled_features = pooled[:, 0]
            
            # Classify with each head
            if dataset_names is None:
                dataset_names = self.classifiers.keys()
                
            outputs = {}
            for name in dataset_names:
                if name in self.classifiers:
                    outputs[name] = self.classifiers[name](pooled_features)
                    
            return outputs
    
    # Validate mode
    if mode not in ['single', 'multi']:
        raise ValueError(f"mode must be 'single' or 'multi', got '{mode}'")
    
    # Extract common components
    backbone = tjepa_model.backbone
    head_pooler = tjepa_model.head_without_classif
    
    if mode == 'single':
        # Handle single-head conversion
        classifier = None
        remap_indices = None
        
        if classifier_name is not None and not return_features:
            if classifier_name not in tjepa_model.classifiers:
                raise ValueError(
                    f"Classifier '{classifier_name}' not found. "
                    f"Available: {list(tjepa_model.classifiers.keys())}"
                )
            classifier = tjepa_model.classifiers[classifier_name]
            
            # Handle kinetics400 remapping
            if classifier_name == "kinetics400":
                from models.VJEPA.src.utils.remap import VJEPA_REVERSE_MAPPING
                remap_indices = [VJEPA_REVERSE_MAPPING[i] 
                               for i in range(len(VJEPA_REVERSE_MAPPING))]
        
        # Create single-head model
        simple_model = SimpleTJEPA(
            backbone=backbone,
            head_pooler=head_pooler,
            classifier=classifier,
            remap_indices=remap_indices
        )
        simple_model.eval()
        return simple_model
        
    else:  # mode == 'multi'
        # Create multi-head model
        multi_model = MultiHeadTJEPA(
            backbone=backbone,
            head_pooler=head_pooler,
            classifiers=tjepa_model.classifiers,
            ds_classes=tjepa_model.ds_classes
        )
        multi_model.eval()
        return multi_model


# ============================================================================
# Model Loading
# ============================================================================

def wrap_tjepa_forward(model):
    """
    Wraps TJEPA's forward method to add custom behavior while preserving original functionality.
    
    Args:
        model: TJEPA model instance
        
    Example:
        tjepa = TJEPA()
        wrap_tjepa_forward(tjepa)
        output = tjepa(x)  # Now uses wrapped version
    """
    # Save reference to original forward
    original_forward = model.forward
    
    def new_forward(self, x):

        # Call original forward
        x = x.permute(0, 2, 1, 3, 4)  # Change to (B, C, T, H, W)
        result = original_forward(x)
        
        return result
    
    # Replace the forward method
    import types
    model.forward = types.MethodType(new_forward, model)
    
    return model

def load_tjepa(
    config_path: str,
    models_root: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    convert_to_pytorch: bool = False,
    conversion_mode: str = 'single',
    classifier_name: Optional[str] = None,
    return_features: bool = False,
    device: str = 'cuda'
) -> nn.Module:
    """
    Load TJEPA model from config file.
    
    Args:
        config_path: Path to the config YAML file
        models_root: Root directory containing the models package. If None, tries to infer.
        checkpoint_path: Optional path to checkpoint file (.pth or .pt)
        convert_to_pytorch: If True, convert to simple PyTorch model
        conversion_mode: 'single' or 'multi' (only used if convert_to_pytorch=True)
        classifier_name: Classifier name for single mode conversion
        return_features: Return features instead of logits (only for single mode)
        device: Device to load model on ('cpu', 'cuda', etc.)
        
    Returns:
        TJEPA model or converted PyTorch model
        
    Example:
        # Load original TJEPA model
        model = load_tjepa('configs/tjepa_large.yaml')
        
        # Load with explicit models root
        model = load_tjepa(
            'configs/tjepa_large.yaml',
            models_root='/path/to/project',
            checkpoint_path='checkpoints/best_model.pth'
        )
        
        # Load and convert to simple PyTorch model
        model = load_tjepa(
            'configs/tjepa_large.yaml',
            convert_to_pytorch=True,
            conversion_mode='single',
            classifier_name='kinetics400',
            device='cuda'
        )
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('TJEPA_Loader')
    
    # Determine models_root
    if models_root is None:
        # Try to infer from config_path or current directory
        models_root = '/mnt/scratch/ytang/DyViz'
    
    logger.info(f'Using models root: {models_root}')
    
    # Use context manager to temporarily add path
    with add_to_path(models_root):
        # Load config
        logger.info(f'Loading config from {config_path}')
        cfg = load_config(config_path)
        
        # Parse dataset classes
        ds_classes = parse_ds_classes(cfg)
        logger.info(f'Dataset classes: {ds_classes}')

        # Import TJEPA
        from tests.tjepa.VJEPA.VJEPA import TJEPA
        
        # Create model
        logger.info('Creating TJEPA model...')
        model = TJEPA(cfg, ds_classes)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
                
            logger.info(f'Loading checkpoint from {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            logger.info('Checkpoint loaded successfully')
        
        # Move to device
        model = model.to(device)
        model.eval()
        wrap_tjepa_forward(model)
        
        # Convert if requested
        if convert_to_pytorch:
            logger.info(f'Converting to PyTorch model (mode={conversion_mode})')
            model = convert_tjepa_to_pytorch(
                model,
                mode=conversion_mode,
                classifier_name=classifier_name,
                return_features=return_features
            )
        
        logger.info('Model loaded successfully')
        return model


def save_converted_model(
    model: nn.Module,
    save_path: str,
    include_optimizer: bool = False
):
    """
    Save converted PyTorch model.
    
    Args:
        model: The converted PyTorch model
        save_path: Path to save the model
        include_optimizer: Whether to include optimizer state (not applicable for converted models)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__,
    }, save_path)
    
    print(f'Model saved to {save_path}')


# ============================================================================
# Usage Examples
# ============================================================================

"""
# Example 1: Basic usage with automatic path detection
model = load_tjepa(
    config_path='configs/tjepa_large.yaml',
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda'
)

# Example 2: Explicit models root
model = load_tjepa(
    config_path='configs/tjepa_large.yaml',
    models_root='/home/user/project',
    checkpoint_path='checkpoints/best_model.pth',
    device='cuda'
)

# Example 3: Load and convert to simple single-task model
simple_model = load_tjepa(
    config_path='configs/tjepa_large.yaml',
    checkpoint_path='checkpoints/best_model.pth',
    convert_to_pytorch=True,
    conversion_mode='single',
    classifier_name='kinetics400',
    device='cuda'
)

# Example 4: Load as feature extractor
feature_extractor = load_tjepa(
    config_path='configs/tjepa_large.yaml',
    checkpoint_path='checkpoints/best_model.pth',
    convert_to_pytorch=True,
    conversion_mode='single',
    return_features=True,
    device='cuda'
)

# Example 5: Load with all classifier heads
multi_model = load_tjepa(
    config_path='configs/tjepa_large.yaml',
    checkpoint_path='checkpoints/best_model.pth',
    convert_to_pytorch=True,
    conversion_mode='multi',
    device='cuda'
)

# Example 6: Use the models
with torch.no_grad():
    # Single classifier
    output = simple_model(video_tensor)  # Shape: [batch, num_classes]
    
    # Feature extractor
    features = feature_extractor(video_tensor)  # Shape: [batch, embed_dim]
    
    # Multi-head
    all_outputs = multi_model(video_tensor)  # Dict: {dataset_name: logits}
    k400_output = multi_model(video_tensor, dataset_names=['kinetics400'])

# Example 7: Save converted model
save_converted_model(simple_model, 'converted_models/tjepa_kinetics400.pth')

# Example 8: Load saved converted model later
saved_checkpoint = torch.load('converted_models/tjepa_kinetics400.pth')
new_model = SimpleTJEPA(...)  # Need to initialize with same architecture
new_model.load_state_dict(saved_checkpoint['model_state_dict'])

# Example 9: Using add_to_path directly for custom imports
with add_to_path('/custom/path/to/models'):
    from models.VJEPA.TJEPA import TJEPA
    from models.custom_module import CustomModule
    # Your code here

# Example 10: Multiple paths
with add_to_path(['/path/to/models', '/path/to/utils']):
    from models import something
    from utils import something_else
"""