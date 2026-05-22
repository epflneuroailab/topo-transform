"""
improved_flow_single_pitzalis.py

Generates flow-field videos matching Pitzalis-style stimuli.
- Alpha parameter (0-1) controls motion from dilation -> outward spiral -> rotation -> inward spiral -> contraction
- Scrambled mode preserves speed gradient but randomizes trajectory directions
- Center jittering and speed variation across trials

Example:
    python improved_flow_single_pitzalis.py
"""

import numpy as np
import cv2
import math
from pathlib import Path


def _ou_noise(prev, mu, sigma, tau, dt=1.0):
    """
    Ornstein-Uhlenbeck update: returns next noise vector given previous.
    
    prev: (N,2) previous noise
    mu: scalar or (N,2)
    sigma: noise amplitude
    tau: correlation time (larger -> smoother)
    dt: timestep (frames)
    """
    if tau <= 0:
        return np.random.normal(loc=mu, scale=sigma, size=prev.shape)
    alpha = dt / (tau + 1e-12)
    noise = prev + alpha * (mu - prev) + np.sqrt(2 * alpha) * np.random.normal(scale=sigma, size=prev.shape)
    return noise


def generate_flow_single(
    mode="coherent",  # 'coherent' or 'scrambled'
    alpha=0.5,  # 0=dilation, 0.25=outward spiral, 0.5=rotation, 0.75=inward spiral, 1.0=contraction
    out_path="flow_{mode}_alpha{alpha:.2f}.mp4",
    duration=2.0,
    fps=30,
    size=224,
    n_dots=700,
    dot_lifetime_frames=30,
    dot_size=1,
    speed=2.5,
    center=None,
    center_jitter=0.0,  # max jitter in pixels from image center
    speed_variation=0.0,  # fractional variation in speed (e.g., 0.1 = ±10%)
    return_frames=False,
    seed=123
):
    """
    Generate flow field video with alpha-controlled motion pattern.
    
    Alpha mapping:
        0.0  -> pure dilation (outward radial)
        0.25 -> outward spiral
        0.5  -> pure rotation
        0.75 -> inward spiral
        1.0  -> pure contraction (inward radial)
    
    Mode:
        'coherent': dots follow coherent flow pattern
        'scrambled': each dot trajectory is rotated by random angle around center,
                     preserving speed gradient but destroying coherence
    
    Returns path string or (path, frames) if return_frames True.
    """
    assert mode in ("coherent", "scrambled")
    assert 0.0 <= alpha <= 1.0
    
    np.random.seed(seed)
    frames_count = max(1, int(round(duration * fps)))
    w = h = size
    
    # Apply center jitter
    if center is None:
        base_cx, base_cy = (w - 1) / 2.0, (h - 1) / 2.0
    else:
        base_cx, base_cy = center
    
    if center_jitter > 0:
        cx = base_cx + np.random.uniform(-center_jitter, center_jitter)
        cy = base_cy + np.random.uniform(-center_jitter, center_jitter)
    else:
        cx, cy = base_cx, base_cy
    
    # Apply speed variation
    if speed_variation > 0:
        speed_factor = 1.0 + np.random.uniform(-speed_variation, speed_variation)
        speed = speed * speed_factor
    
    # Initialize dot positions uniformly
    pos = np.random.rand(n_dots, 2) * np.array([w, h])
    
    # Ages and lifetimes
    ages = np.random.randint(0, dot_lifetime_frames, size=(n_dots,))
    lifetimes = np.full(n_dots, dot_lifetime_frames, dtype=int)
    
    # For scrambled mode: generate random rotation angles per dot
    if mode == "scrambled":
        scramble_angles = np.random.uniform(0, 2*np.pi, size=n_dots)
    
    # Prepare video writer
    out_path = out_path.format(mode=mode, alpha=alpha)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)
    frames_list = [] if return_frames else None
    
    max_stroke = max(2.0, dot_size * 4.0)
    
    for f in range(frames_count):
        t = f / fps
        frame_gray = np.zeros((h, w), dtype=np.uint8)
        
        # Compute relative positions from center
        rel = pos - np.array([cx, cy])  # (N,2)
        dist = np.linalg.norm(rel, axis=1) + 1e-8
        
        # Compute base velocity components
        # Radial direction (normalized)
        dir_rad = rel / dist[:, None]
        
        # Tangential direction (rotation, perpendicular to radial)
        dir_rot = np.stack([-rel[:, 1], rel[:, 0]], axis=1) / dist[:, None]
        
        # Map alpha to motion pattern:
        # alpha in [0, 0.5]: interpolate from outward radial to rotation
        # alpha in [0.5, 1]: interpolate from rotation to inward radial
        
        if alpha <= 0.5:
            # Blend from outward radial (alpha=0) to rotation (alpha=0.5)
            # At alpha=0: pure radial outward
            # At alpha=0.5: pure rotation
            blend = alpha * 2.0  # maps [0,0.5] to [0,1]
            radial_component = (1.0 - blend)  # 1 at alpha=0, 0 at alpha=0.5
            rotation_component = blend  # 0 at alpha=0, 1 at alpha=0.5
            radial_sign = 1.0  # outward
        else:
            # Blend from rotation (alpha=0.5) to inward radial (alpha=1.0)
            # At alpha=0.5: pure rotation
            # At alpha=1.0: pure radial inward
            blend = (alpha - 0.5) * 2.0  # maps [0.5,1] to [0,1]
            rotation_component = (1.0 - blend)  # 1 at alpha=0.5, 0 at alpha=1
            radial_component = blend  # 0 at alpha=0.5, 1 at alpha=1
            radial_sign = -1.0  # inward
        
        # Combine directional components
        dir_combined = rotation_component * dir_rot + radial_component * radial_sign * dir_rad
        dir_combined = dir_combined / (np.linalg.norm(dir_combined, axis=1)[:, None] + 1e-8)
        
        # Speed gradient: slower at center, faster at periphery
        # This is preserved in both coherent and scrambled modes
        max_dist = 0.5 * w
        speed_profile = 0.5 + 1.5 * (dist / max_dist)  # increases with eccentricity
        speed_profile = np.clip(speed_profile, 0.3, 2.0)  # reasonable bounds
        
        vel = dir_combined * (speed_profile[:, None] * speed)
        
        # Apply scrambling if in scrambled mode
        if mode == "scrambled":
            # Rotate each velocity vector by its random angle around the origin
            cos_a = np.cos(scramble_angles)
            sin_a = np.sin(scramble_angles)
            vel_x = vel[:, 0] * cos_a - vel[:, 1] * sin_a
            vel_y = vel[:, 0] * sin_a + vel[:, 1] * cos_a
            vel = np.stack([vel_x, vel_y], axis=1)
        
        # Update positions
        pos += vel / fps * 30.0
        
        # Age dots and respawn those beyond lifetime or out of bounds
        ages += 1
        out_of_bounds = (pos[:, 0] < -2) | (pos[:, 0] >= w+2) | (pos[:, 1] < -2) | (pos[:, 1] >= h+2)
        
        # Recompute distance after position update to check for center accumulation
        rel_new = pos - np.array([cx, cy])
        dist_new = np.linalg.norm(rel_new, axis=1)
        too_close_to_center = dist_new < 5.0  # within 5 pixels of center
        
        dead = (ages >= lifetimes) | out_of_bounds | too_close_to_center
        
        if dead.any():
            k = np.where(dead)[0]
            pos[k] = np.random.rand(len(k), 2) * np.array([w, h])
            ages[k] = 0
            lifetimes[k] = dot_lifetime_frames
            
            # Reassign scramble angles for respawned dots
            if mode == "scrambled":
                scramble_angles[k] = np.random.uniform(0, 2*np.pi, size=len(k))
        
        # Render dots as short strokes oriented along velocity
        vnorm = np.linalg.norm(vel, axis=1) + 1e-8
        lengths = np.clip(0.6 + 2.0 * (vnorm / (np.max(vnorm)+1e-8)), 0.8, max_stroke)
        
        # Compute endpoints
        p1 = (pos - (vel / vnorm[:, None]) * (lengths[:, None]/2.0)).astype(int)
        p2 = (pos + (vel / vnorm[:, None]) * (lengths[:, None]/2.0)).astype(int)
        
        # Clip coordinates
        p1[:, 0] = np.clip(p1[:, 0], 0, w-1)
        p1[:, 1] = np.clip(p1[:, 1], 0, h-1)
        p2[:, 0] = np.clip(p2[:, 0], 0, w-1)
        p2[:, 1] = np.clip(p2[:, 1], 0, h-1)
        
        # Draw lines
        for (x1, y1), (x2, y2) in zip(p1, p2):
            cv2.line(frame_gray, (int(x1), int(y1)), (int(x2), int(y2)), 
                    color=255, thickness=1, lineType=cv2.LINE_AA)
        
        # Optional border
        cv2.rectangle(frame_gray, (0, 0), (w-1, h-1), 255, thickness=1)
        
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        writer.write(frame_bgr)
        if return_frames:
            frames_list.append(frame_bgr)
    
    writer.release()
    out_path_resolved = Path(out_path).resolve().as_posix()
    print(f"Saved {mode} (alpha={alpha:.2f}) -> {out_path_resolved}")
    
    if return_frames:
        return out_path_resolved, np.stack(frames_list, axis=0)
    return out_path_resolved


if __name__ == "__main__":
    presentation = "preview"  # change to "widefield" for bigger stimuli
    
    if presentation == "preview":
        SIZE = 224
        NDOTS = 700
    else:
        SIZE = 800
        NDOTS = 3500
    
    FPS = 30
    DURATION = 2.0
    DOT_LIFETIME_SEC = 0.5
    DOT_LIFE_FRAMES = max(1, int(round(DOT_LIFETIME_SEC * FPS)))
    SPEED = 3.2
    SEED = 42
    NUM_TRIALS = 128
    
    # Center jitter range (in pixels)
    CENTER_JITTER = 20.0
    # Speed variation (fractional, e.g., 0.1 = ±10%)
    SPEED_VARIATION = 0.1
    
    root_out_dir = Path("/mnt/scratch/ytang/datasets/flow_fields")
    root_out_dir.mkdir(parents=True, exist_ok=True)
    
    for mode in ("coherent", "scrambled"):
        out_dir = root_out_dir / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for trial in range(NUM_TRIALS):
            # Randomly select alpha for each scrambled trial
            alpha_random = np.random.uniform(0.0, 1.0)

            out_path = out_dir / f"{mode}_trial{trial:02d}_alpha{alpha_random:.2f}.avi"

            generate_flow_single(
                mode=mode,
                alpha=alpha_random,
                out_path=out_path.as_posix(),
                duration=DURATION,
                fps=FPS,
                size=SIZE,
                n_dots=NDOTS,
                dot_lifetime_frames=DOT_LIFE_FRAMES,
                dot_size=2,
                speed=SPEED,
                center_jitter=CENTER_JITTER,
                speed_variation=SPEED_VARIATION,
                seed=SEED + trial + 10000  # different seed space for scrambled
            )