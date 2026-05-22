import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path


def load_motion_data(txt_file_path):
    """Load biological motion data from UTF-16 text file."""
    with open(txt_file_path, 'r', encoding='utf-16') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split()
    n_frames, n_markers, hz = int(header[2]), int(header[5]), int(header[8])
    
    data = [float(v) for line in lines[1:] for v in line.strip().split() if line.strip()]
    return np.array(data).reshape(n_frames, n_markers, 3), hz


def scramble_motion(data, seed=None):
    """Randomly assign starting positions while preserving local trajectories."""
    if seed is not None:
        np.random.seed(seed)
    
    n_frames, n_markers = data.shape[:2]
    scrambled = np.zeros_like(data)
    
    # Random starting positions sampled independently from the original markers
    random_starts = np.random.uniform(
        low=np.min(data[0], axis=0), 
        high=np.max(data[0], axis=0), 
        size=(n_markers, data.shape[2])
    )
    
    for i in range(n_markers):
        trajectory = data[:, i, :].copy()
        local_offsets = trajectory - trajectory[0, :]  # preserve relative motion
        scrambled[:, i, :] = random_starts[i] + local_offsets
    
    return scrambled


def create_visualization(data, hz, angles, output_path, frame_size=(224, 224),
                         dpi=100, video_fps=30, video_duration=2.0):
    """Create and save dynamic biological motion video as AVI."""
    n_frames, n_views = data.shape[0], len(angles)
    width_inches = (frame_size[0] * n_views) / dpi
    height_inches = frame_size[1] / dpi
    
    # Repeat frames to fill full duration
    n_video_frames = int(video_fps * video_duration)
    frame_indices = np.arange(n_video_frames) % n_frames
    
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi, facecolor='black')
    
    # Calculate bounds
    all_data = data.reshape(-1, 3)
    max_range = np.array([
        all_data[:, i].max() - all_data[:, i].min() for i in range(3)
    ]).max() / 2.0
    mid = np.array([(all_data[:, i].max() + all_data[:, i].min()) * 0.5 for i in range(3)])
    
    scatters = []
    for idx, angle in enumerate(angles):
        ax = fig.add_subplot(1, n_views, idx + 1, projection='3d')
        scatter = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2],
                             c='white', s=25, alpha=1.0)
        scatters.append(scatter)
        
        ax.view_init(elev=0, azim=angle)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_axis_off()
        ax.set_facecolor('black')
        
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('black')
        ax.grid(False)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    def update(i):
        frame = frame_indices[i]
        for scatter in scatters:
            scatter._offsets3d = (data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
        return scatters

    writer = FFMpegWriter(fps=video_fps, codec='libx264', bitrate=1800)
    anim = FuncAnimation(fig, update, frames=n_video_frames, interval=1000/video_fps, blit=False)
    anim.save(output_path, writer=writer, dpi=dpi, savefig_kwargs={'facecolor': 'black'})
    plt.close()


def create_static_video(data, angles, output_path, frame_size=(224, 224), dpi=100,
                       video_fps=30, video_duration=2.0):
    """Create static AVI video from one frame repeated over time."""
    n_views = len(angles)
    width_inches = (frame_size[0] * n_views) / dpi
    height_inches = frame_size[1] / dpi
    
    n_video_frames = int(video_fps * video_duration)
    
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi, facecolor='black')
    
    # Calculate bounds
    all_data = data.reshape(-1, 3)
    max_range = np.array([
        all_data[:, i].max() - all_data[:, i].min() for i in range(3)
    ]).max() / 2.0
    mid = np.array([(all_data[:, i].max() + all_data[:, i].min()) * 0.5 for i in range(3)])
    
    scatters = []
    for idx, angle in enumerate(angles):
        ax = fig.add_subplot(1, n_views, idx + 1, projection='3d')
        scatter = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2],
                             c='white', s=50, alpha=1.0)
        scatters.append(scatter)
        
        ax.view_init(elev=0, azim=angle)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_axis_off()
        ax.set_facecolor('black')
        
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('black')
        ax.grid(False)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    def update(frame):
        return scatters

    writer = FFMpegWriter(fps=video_fps, codec='libx264', bitrate=1800)
    anim = FuncAnimation(fig, update, frames=n_video_frames, interval=1000/video_fps, blit=False)
    anim.save(output_path, writer=writer, dpi=dpi, savefig_kwargs={'facecolor': 'black'})
    plt.close()


def generate_biological_motion_videos(txt_file_paths, angles=[90], scrambled=True, 
                                       static=True, frame_size=(224, 224), dpi=100,
                                       seed=None, static_frame_idx=None, 
                                       output_base_dir='/mnt/user-data/outputs',
                                       video_fps=30, video_duration=2.0):
    """
    Generate biological motion videos from motion capture data.
    All outputs saved as .avi (dynamic and static).
    """
    if isinstance(txt_file_paths, str):
        txt_file_paths = [txt_file_paths]
    
    output_info = {}
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    for txt_file_path in txt_file_paths:
        txt_file_path = Path(txt_file_path)
        if not txt_file_path.exists():
            print(f"Warning: {txt_file_path} not found. Skipping.")
            continue
        
        print(f"\nProcessing: {txt_file_path.name}")
        data, hz = load_motion_data(txt_file_path)
        print(f"Loaded: {data.shape[0]} frames, {data.shape[1]} markers, {hz} Hz")
        
        base_name = txt_file_path.stem
        angle_suffix = '_'.join(map(str, angles))
        
        output_info[str(txt_file_path)] = {
            'normal_static': [], 'normal_dynamic': [],
            'scrambled_static': [], 'scrambled_dynamic': []
        }
        
        # Normal motion
        print("Generating normal motion...")
        if static:
            if seed is not None:
                np.random.seed(seed)
            frame_idx = static_frame_idx if static_frame_idx is not None else np.random.randint(0, data.shape[0])
        
            static_video_dir = output_base_dir / "normal_static"
            static_video_dir.mkdir(exist_ok=True)
            output_path_video = static_video_dir / f"{base_name}_angle_{angle_suffix}.avi"
            
            print(f"  Static video {frame_idx} ({video_duration}s @ {video_fps}fps): {output_path_video.name}")
            create_static_video(data[frame_idx:frame_idx+1], angles, str(output_path_video),
                                frame_size, dpi, video_fps, video_duration)
            output_info[str(txt_file_path)]['normal_static'].append(str(output_path_video))
        
        dynamic_dir = output_base_dir / "normal_dynamic"
        dynamic_dir.mkdir(exist_ok=True)
        output_path = dynamic_dir / f"{base_name}_angle_{angle_suffix}.avi"
        
        print(f"  Dynamic ({video_duration}s @ {video_fps}fps): {output_path.name}")
        create_visualization(data, hz, angles, str(output_path),
                             frame_size, dpi, video_fps, video_duration)
        output_info[str(txt_file_path)]['normal_dynamic'].append(str(output_path))
        
        # Scrambled motion
        if scrambled:
            print("Generating scrambled motion...")
            scrambled_data = scramble_motion(data, seed)
            
            if static:
                if seed is not None:
                    np.random.seed(seed)
                frame_idx = static_frame_idx if static_frame_idx is not None else np.random.randint(0, data.shape[0])
                
                static_video_dir = output_base_dir / "scrambled_static"
                static_video_dir.mkdir(exist_ok=True)
                output_path_video = static_video_dir / f"{base_name}_angle_{angle_suffix}.avi"
                
                print(f"  Static video {frame_idx} ({video_duration}s @ {video_fps}fps): {output_path_video.name}")
                create_static_video(scrambled_data[frame_idx:frame_idx+1], angles, str(output_path_video),
                                    frame_size, dpi, video_fps, video_duration)
                output_info[str(txt_file_path)]['scrambled_static'].append(str(output_path_video))
            
            dynamic_dir = output_base_dir / "scrambled_dynamic"
            dynamic_dir.mkdir(exist_ok=True)
            output_path = dynamic_dir / f"{base_name}_angle_{angle_suffix}.avi"
            
            print(f"  Dynamic ({video_duration}s @ {video_fps}fps): {output_path.name}")
            create_visualization(scrambled_data, hz, angles, str(output_path),
                                 frame_size, dpi, video_fps, video_duration)
            output_info[str(txt_file_path)]['scrambled_dynamic'].append(str(output_path))
        
        print(f"Completed: {txt_file_path.name}")
    
    return output_info


if __name__ == "__main__":
    txt_file_dir = '/mnt/scratch/ytang/datasets/Vanrie-BRMIC-2004-txt'
    txt_file_paths = list(Path(txt_file_dir).glob('*.txt'))

    for scrambled in [True, False]:
        for static in [True, False]:
            for angle in [0, 45, 90, 135]:
                print("\n" + "="*60)
                print(f"SCRAMBLED: {scrambled}, STATIC: {static}")
                print(f"ANGLE: {angle} degrees")
                print("="*60)

                try:
                    output_info = generate_biological_motion_videos(
                        txt_file_paths=txt_file_paths,
                        output_base_dir='/mnt/scratch/ytang/datasets/biological-motion',
                        angles=[angle],
                        scrambled=scrambled,
                        static=static,
                        frame_size=(224, 224),
                        seed=None,
                        video_fps=30,
                        video_duration=2.0
                    )
                except Exception as e:
                    print(f"Error processing files: {e}")
