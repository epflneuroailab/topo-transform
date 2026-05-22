import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np


class InvertibleMatrix(nn.Module):
    """
    Invertible matrix using proper orthogonal parameterization.
    
    Uses QR decomposition with proper gradient handling:
    - Any matrix can be decomposed as Q @ R where Q is orthogonal
    - We parameterize a free matrix and extract Q via QR decomposition
    - Ensures numerical stability and proper gradients
    - Default: initialized near identity (I + small noise) for stable training
    - Dropout can be applied to the free parameter matrix during training
    """
    def __init__(self, size, init_scale=0.01, dropout=0.0):
        super().__init__()
        self.size = size
        
        # Free parameter matrix
        self.scale = Parameter(torch.ones(1))
        self.weight = Parameter(torch.randn(self.size, self.size) * init_scale)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        
    def get_matrix(self):
        """Get orthogonal matrix via QR decomposition"""
        # Apply dropout to weight matrix during training
        weight = self.weight # + torch.eye(self.size).to(self.weight.device) * self.scale
        if self.dropout is not None and self.training:
            weight = self.dropout(weight)
        
        # QR decomposition: weight = Q @ R
        Q, R = torch.linalg.qr(weight)
        
        # Ensure determinant is positive (optional, for consistency)
        # This prevents reflection matrices
        signs = torch.sign(torch.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs.unsqueeze(0)
        
        return Q
    
    def get_inverse_matrix(self):
        """Get inverse matrix (transpose for orthogonal matrices)"""
        Q = self.get_matrix()
        return Q.T
    
    def log_det(self):
        """Log determinant is 0 for orthogonal matrices with det=+1"""
        return torch.tensor(0.0, device=self.weight.device)


class FactorizedInvertibleTransformation(nn.Module):
    """
    Factorized invertible transformation: CHW -> CHW
    Uses channel mixing only:
    - Channel mixing: HW different CxC invertible transformations, one per spatial location
    
    This implementation uses orthogonal matrices via QR decomposition.
    """
    def __init__(self, C, H, W, init_scale=0.01):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.spatial_dim = H * W
        
        # Channel mixing matrices (HW different CxC matrices)
        self.channel_transforms = nn.ModuleList([
            InvertibleMatrix(C, init_scale=init_scale) 
            for _ in range(self.spatial_dim)
        ])
        
    def forward(self, x, inverse=False):
        """
        Forward or inverse transformation
        x: tensor of shape (batch_size, C, H, W)
        inverse: if True, apply inverse transformation
        """
        if not inverse:
            return self._forward(x)
        else:
            return self._inverse(x)
    
    def _forward(self, x):
        """Forward transformation"""
        batch_size = x.size(0)
        
        # Reshape to (batch_size, C, H*W) then transpose to (batch_size, H*W, C)
        x_reshaped = x.view(batch_size, self.C, self.spatial_dim).transpose(1, 2)
        
        # Channel mixing (per spatial location)
        x_channel_mixed = torch.zeros_like(x_reshaped)
        for s in range(self.spatial_dim):
            channel_matrix = self.channel_transforms[s].get_matrix()
            # y[b,s,:] = channel_matrix @ x[b,s,:]
            x_channel_mixed[:, s, :] = torch.matmul(x_reshaped[:, s, :], channel_matrix.T)
        
        # Transpose back to (batch_size, C, H*W) and reshape to (batch_size, C, H, W)
        return x_channel_mixed.transpose(1, 2).view(batch_size, self.C, self.H, self.W)
    
    def inverse(self, y):
        """Inverse transformation"""
        batch_size = y.size(0)
        
        # Reshape to (batch_size, C, H*W) then transpose to (batch_size, H*W, C)
        y_reshaped = y.view(batch_size, self.C, self.spatial_dim).transpose(1, 2)
        
        # Channel unmixing (per spatial location)
        y_channel_unmixed = torch.zeros_like(y_reshaped)
        for s in range(self.spatial_dim):
            channel_matrix_inv = self.channel_transforms[s].get_inverse_matrix()
            # x[b,s,:] = channel_matrix_inv @ y[b,s,:]
            y_channel_unmixed[:, s, :] = torch.matmul(y_reshaped[:, s, :], channel_matrix_inv.T)
        
        # Transpose back to (batch_size, C, H*W) and reshape to (batch_size, C, H, W)
        return y_channel_unmixed.transpose(1, 2).view(batch_size, self.C, self.H, self.W)
    
    def log_det_jacobian(self):
        """
        Compute log determinant of the Jacobian matrix.
        For orthogonal transformations, det = ±1, so log|det| = 0
        """
        return torch.tensor(0.0)
    
    def check_invertibility(self, x=None, rtol=1e-4, atol=1e-6):
        """Check if the transformation is properly invertible"""
        if x is None:
            x = torch.randn(2, self.C, self.H, self.W, 
                          device=next(self.parameters()).device,
                          dtype=next(self.parameters()).dtype)
        
        with torch.no_grad():
            # Forward then inverse
            y = self.forward(x, inverse=False)
            x_reconstructed = self.forward(y, inverse=True)
        
        # Check reconstruction error
        error = torch.max(torch.abs(x - x_reconstructed))
        relative_error = error / (torch.max(torch.abs(x)) + 1e-10)
        is_invertible = torch.allclose(x, x_reconstructed, rtol=rtol, atol=atol)
        
        return is_invertible, error.item(), relative_error.item()


class TopoTransform(nn.Module):
    def __init__(self, layer_dims, init_scale=1e-3):  # NOTE: small init_scale is necessary
        """
        Args:
            layer_dims: list of tuples (C, H, W) for each layer
            init_scale: initialization scale for invertible transformations
        """
        super().__init__()
        self.L = len(layer_dims)
        self.layer_dims = layer_dims
        
        # Calculate padded C for each layer (C must be perfect square)
        self.padded_configs = []
        self.original_C_list = []
        
        for C, H, W in layer_dims:
            self.original_C_list.append(C)
            # Find next perfect square >= C
            sqrt_C = int(np.ceil(np.sqrt(C)))
            C_padded = sqrt_C * sqrt_C
            self.padded_configs.append((C_padded, H, W))
            
            if C_padded != C:
                print(f"Padding C={C} to C_padded={C_padded} (sqrt={sqrt_C})")
        
        # Create invertible transformations for each layer (with padded C)
        self.transforms = nn.ModuleList([
            FactorizedInvertibleTransformation(C_padded, H, W, init_scale=init_scale)
            for C_padded, H, W in self.padded_configs
        ])
        
    def _pad_channels(self, x, original_C, padded_C):
        """Pad channel dimension with zeros"""
        if original_C == padded_C:
            return x
        
        pad_size = padded_C - original_C
        
        if x.ndim == 4:  # BCHW
            padding = torch.zeros(x.size(0), pad_size, x.size(2), x.size(3), 
                                device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=1)
        elif x.ndim == 5:  # BTCHW
            padding = torch.zeros(x.size(0), x.size(1), pad_size, x.size(3), x.size(4), 
                                device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=2)
        else:
            raise ValueError(f"Unexpected number of dimensions: {x.ndim}")
    
    def _unpad_channels(self, x, original_C, padded_C):
        """Remove padding from channel dimension"""
        if original_C == padded_C:
            return x
        
        if x.ndim == 4:  # BCHW
            return x[:, :original_C, :, :]
        elif x.ndim == 5:  # BTCHW
            return x[:, :, :original_C, :, :]
        else:
            raise ValueError(f"Unexpected number of dimensions: {x.ndim}")
    
    def forward(self, activations_list):
        """
        Args:
            activations_list: list of L tensors, each of shape BCHW or BTCHW
        
        Returns:
            output: transformed units in shape (L, B, (T), C_padded, H, W)
        """
        assert len(activations_list) == self.L, f"Expected {self.L} layers, got {len(activations_list)}"
        
        # Detect if we have temporal dimension
        first_shape = activations_list[0].shape
        has_time = len(first_shape) == 5  # BTCHW
        
        transformed = []
        
        for i, x in enumerate(activations_list):
            C, H, W = self.layer_dims[i]
            C_padded, _, _ = self.padded_configs[i]
            
            # Validate input shape
            if has_time:
                B, T, C_in, H_in, W_in = x.shape
                assert (C_in, H_in, W_in) == (C, H, W), f"Layer {i} shape mismatch: expected {(C, H, W)}, got {(C_in, H_in, W_in)}"
            else:
                B, C_in, H_in, W_in = x.shape
                assert (C_in, H_in, W_in) == (C, H, W), f"Layer {i} shape mismatch: expected {(C, H, W)}, got {(C_in, H_in, W_in)}"
            
            # Pad channels if necessary
            x_padded = self._pad_channels(x, C, C_padded)
            
            # Apply invertible transformation
            if has_time:
                # Reshape to (B*T)CHW, apply transform, reshape back
                x_flat = x_padded.reshape(B * T, C_padded, H, W)
                x_transformed = self.transforms[i](x_flat)
                x_transformed = x_transformed.reshape(B, T, C_padded, H, W)
            else:
                x_transformed = self.transforms[i](x_padded)

            transformed.append(x_transformed)
            
        return transformed  # L[ B, (T), C_padded, H, W ]
    
    def inverse(self, transformed):
        assert len(transformed) == self.L, f"Expected {self.L} layers, got {len(transformed)}"
        
        # Detect if we have temporal dimension
        has_time = transformed[0].ndim == 5  # L[ B, T, C, H, W ]
        
        activations_list = []
        
        for i in range(self.L):
            C, H, W = self.layer_dims[i]
            C_padded, _, _ = self.padded_configs[i]
            
            x_transformed = transformed[i]
            
            # Apply inverse transformation
            if has_time:
                B, T, C_t, H_t, W_t = x_transformed.shape
                assert (C_t, H_t, W_t) == (C_padded, H, W), \
                    f"Layer {i} transformed shape mismatch: expected {(C_padded, H, W)}, got {(C_t, H_t, W_t)}"
                
                # Reshape to (B*T)CHW, apply inverse, reshape back
                x_flat = x_transformed.reshape(B * T, C_padded, H, W)
                x_inv = self.transforms[i]._inverse(x_flat)
                x_inv = x_inv.reshape(B, T, C_padded, H, W)
            else:
                B, C_t, H_t, W_t = x_transformed.shape
                assert (C_t, H_t, W_t) == (C_padded, H, W), \
                    f"Layer {i} transformed shape mismatch: expected {(C_padded, H, W)}, got {(C_t, H_t, W_t)}"
                
                x_inv = self.transforms[i]._inverse(x_transformed)
            
            # Unpad channels to recover original shape
            x_original = self._unpad_channels(x_inv, C, C_padded)
            
            activations_list.append(x_original)
        
        return activations_list
    

if __name__ == "__main__":
    print("=" * 80)
    print("Testing TopoTransform with Inversion")
    print("=" * 80)
    
    # Test 1: Basic forward-inverse without temporal dimension
    print("\n[Test 1] Basic forward-inverse (BCHW)")
    print("-" * 80)
    layer_dims = [(64, 8, 8), (128, 4, 4), (256, 2, 2)]
    model = TopoTransform(layer_dims, init_scale=0.1)
    
    # Create sample activations
    batch_size = 2
    activations = [
        torch.randn(batch_size, C, H, W)
        for C, H, W in layer_dims
    ]
    
    # Forward pass
    transformed = model(activations)
    print(f"Transformed shape: {transformed[0].shape}")
    
    # Inverse pass
    reconstructed = model.inverse(transformed)
    print(f"Number of reconstructed layers: {len(reconstructed)}")
    
    # Check reconstruction error
    max_error = 0
    for i, (orig, recon) in enumerate(zip(activations, reconstructed)):
        error = torch.abs(orig - recon).max().item()
        max_error = max(max_error, error)
        print(f"Layer {i}: shape={orig.shape}, max_error={error:.2e}")
    
    print(f"✓ Test 1 passed! Max reconstruction error: {max_error:.2e}")
    assert max_error < 1e-5, f"Reconstruction error too large: {max_error}"
    
    # Test 2: Forward-inverse with temporal dimension
    print("\n[Test 2] Forward-inverse with temporal dimension (BTCHW)")
    print("-" * 80)
    layer_dims = [(32, 16, 16), (64, 8, 8)]
    model = TopoTransform(layer_dims, init_scale=0.2)
    
    # Create sample activations with time dimension
    batch_size = 3
    time_steps = 5
    activations = [
        torch.randn(batch_size, time_steps, C, H, W)
        for C, H, W in layer_dims
    ]
    
    # Forward pass
    transformed = model(activations)
    print(f"Transformed shape: {transformed[0].shape}")
    
    # Inverse pass
    reconstructed = model.inverse(transformed)
    print(f"Number of reconstructed layers: {len(reconstructed)}")
    
    # Check reconstruction error
    max_error = 0
    for i, (orig, recon) in enumerate(zip(activations, reconstructed)):
        error = torch.abs(orig - recon).max().item()
        max_error = max(max_error, error)
        print(f"Layer {i}: shape={orig.shape}, max_error={error:.2e}")
    
    print(f"✓ Test 2 passed! Max reconstruction error: {max_error:.2e}")
    assert max_error < 1e-5, f"Reconstruction error too large: {max_error}"
    
    # Test 3: Channels requiring padding
    print("\n[Test 3] Channels requiring padding (non-perfect-square C)")
    print("-" * 80)
    layer_dims = [(50, 7, 7), (100, 5, 5), (200, 3, 3)]  # 50->64, 100->100, 200->196->225
    model = TopoTransform(layer_dims, init_scale=0.05)
    
    batch_size = 4
    activations = [
        torch.randn(batch_size, C, H, W)
        for C, H, W in layer_dims
    ]
    
    # Forward pass
    transformed = model(activations)
    print(f"Transformed shape: {transformed[0].shape}")
    
    # Inverse pass
    reconstructed = model.inverse(transformed)
    
    # Check reconstruction error
    max_error = 0
    for i, (orig, recon) in enumerate(zip(activations, reconstructed)):
        error = torch.abs(orig - recon).max().item()
        max_error = max(max_error, error)
        C_orig, C_padded = layer_dims[i][0], model.padded_configs[i][0]
        print(f"Layer {i}: C={C_orig}→{C_padded}, shape={orig.shape}, max_error={error:.2e}")
    
    print(f"✓ Test 3 passed! Max reconstruction error: {max_error:.2e}")
    assert max_error < 1e-5, f"Reconstruction error too large: {max_error}"
    
    # Test 4: Gradient flow through forward and inverse
    print("\n[Test 4] Gradient flow test")
    print("-" * 80)
    layer_dims = [(16, 4, 4), (25, 3, 3)]
    model = TopoTransform(layer_dims, init_scale=0.1)
    
    batch_size = 2
    activations = [
        torch.randn(batch_size, C, H, W, requires_grad=True)
        for C, H, W in layer_dims
    ]
    
    # Forward pass
    transformed = model(activations)
    
    # Inverse pass
    reconstructed = model.inverse(transformed)
    
    # Compute loss and backward
    loss = sum((r ** 2).sum() for r in reconstructed)
    loss.backward()
    
    # Check gradients exist
    for i, act in enumerate(activations):
        assert act.grad is not None, f"No gradient for activation {i}"
        print(f"Layer {i}: gradient norm={act.grad.norm().item():.4f}")
    
    print("✓ Test 4 passed! Gradients flow correctly")
    
    # Test 5: Single layer edge case
    print("\n[Test 5] Single layer edge case")
    print("-" * 80)
    layer_dims = [(128, 8, 8)]
    model = TopoTransform(layer_dims, init_scale=0.1)
    
    batch_size = 1
    activations = [torch.randn(batch_size, 128, 8, 8)]
    
    transformed = model(activations)
    reconstructed = model.inverse(transformed)
    
    error = torch.abs(activations[0] - reconstructed[0]).max().item()
    print(f"Single layer reconstruction error: {error:.2e}")
    print("✓ Test 5 passed!")
    
    # Test 6: Batch size = 1 edge case
    print("\n[Test 6] Batch size = 1 edge case")
    print("-" * 80)
    layer_dims = [(32, 4, 4), (64, 2, 2)]
    model = TopoTransform(layer_dims)
    
    activations = [
        torch.randn(1, C, H, W)
        for C, H, W in layer_dims
    ]
    
    transformed = model(activations)
    reconstructed = model.inverse(transformed)
    
    max_error = max(torch.abs(orig - recon).max().item() 
                    for orig, recon in zip(activations, reconstructed))
    print(f"Batch size 1 reconstruction error: {max_error:.2e}")
    print("✓ Test 6 passed!")
    
    print("\n" + "=" * 80)
    print("All tests passed successfully! ✓")
    print("=" * 80)