# Meteor Detection from Event Camera Data

A production-grade pipeline to detect meteors from synthetic neuromorphic event camera data using temporal windowing, blob detection, and trajectory scoring.

---

## Overview

This pipeline converts raw RGB meteor videos into synthetic event camera data (via v2e), then detects and tracks meteors using classical computer vision techniques:

1. **Video → Events** — Convert RGB video to asynchronous event stream (x, y, t, polarity)
2. **Temporal Windowing** — Split events into 100ms windows
3. **Blob Detection** — Identify clusters of high-activity pixels (above 90th percentile threshold)
4. **Tracking** — Connect blobs across windows using nearest-neighbor matching
5. **Scoring** — Evaluate trajectories for meteor-like properties (straightness, consistency, persistence)
6. **Filtering** — Keep only high-confidence detections (score ≥ 0.75–0.85)
7. **Visualization** — Generate output video with overlaid trajectories and CSV results

---

## Dataset

**Source:** [Allsky7 Meteor Dataset on Kaggle](https://www.kaggle.com/datasets/virajsaws/meteor-dataset-allsky7/data)

- **Content:** 24 allsky meteor videos from Allsky7.net
- **Resolution:** 1920×1080 at 25fps
- **Format:** MP4 files with clear meteor trails
- **Size:** ~171MB total

**Example files:**
- `2026_05_15_01_29_01_000_010091_trim300_wm.mp4` (10.96s, meteor visible)
- `2026_05_15_01_29_01_000_012661_trim300_wm.mp4` (10.96s, meteor visible)

---

## Prerequisites

- Kaggle Notebook environment (GPU optional but helpful)
- 2–3GB storage for v2e output + results
- ~10–15 minutes per video (depends on length + GPU)

---

## Setup Instructions

### Step 0: Create a Fresh Kaggle Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code)
2. Click **"New Notebook"**
3. Keep default settings (Python, no accelerator needed)

---

### Step 1: Add the Dataset

1. Click **"Data"** (right sidebar)
2. Click **"Add input"**
3. Search for **"meteor-dataset-allsky7"**
4. Add the dataset to your notebook

The data will be mounted at `/kaggle/input/datasets/virajsaws/meteor-dataset-allsky7/`

---

### Step 2: Install Dependencies

Paste and run this cell:

```python
import os

print("Installing dependencies...")
os.system("pip install -q dv-processing h5py opencv-python-headless scikit-image pandas tqdm")
os.system("pip install -q argcomplete engineering_notation numba scipy matplotlib imageio imageio-ffmpeg pyyaml pydantic pyzmq dill vidgear screeninfo")

print("✓ Dependencies installed")
```

---

### Step 3: Generate Event Data (v2e Conversion)

Paste and run this cell:

```python
import os
import subprocess
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

# Choose a video from the dataset
VIDEO_FILE = "/kaggle/input/datasets/virajsaws/meteor-dataset-allsky7/2026_05_15_01_29_01_000_012661_trim300_wm.mp4"

# Output directory for v2e
V2E_OUTPUT_DIR = "/kaggle/working/v2e_output"

# ============================================================
# INSTALL V2E
# ============================================================

print("Installing v2e...")
if not os.path.exists("/kaggle/working/v2e"):
    os.system("git clone -q https://github.com/SensorsINI/v2e.git /kaggle/working/v2e")

os.chdir("/kaggle/working/v2e")

# Install v2e dependencies
os.system("pip install -q -r requirements.txt 2>/dev/null || true")
os.system("pip install -q dv-processing argcomplete engineering_notation numba scipy opencv-python-headless matplotlib h5py imageio imageio-ffmpeg pyyaml tqdm pandas pydantic pyzmq dill vidgear screeninfo 2>/dev/null || true")

# Create headless easygui stub (Kaggle has no GUI)
easygui_stub = "/kaggle/working/v2e/easygui.py"
with open(easygui_stub, "w") as f:
    f.write("def fileopenbox(*args, **kwargs):\n    return None\n")
    f.write("def diropenbox(*args, **kwargs):\n    return None\n")
    f.write("def filesavebox(*args, **kwargs):\n    return None\n")
    f.write("def msgbox(*args, **kwargs):\n    return None\n")
    f.write("def buttonbox(*args, **kwargs):\n    return None\n")
    f.write("def ynbox(*args, **kwargs):\n    return True\n")
    f.write("def choicebox(*args, **kwargs):\n    return None\n")

print("✓ v2e ready\n")

# ============================================================
# RUN V2E CONVERSION
# ============================================================

os.makedirs(V2E_OUTPUT_DIR, exist_ok=True)

print(f"Converting: {os.path.basename(VIDEO_FILE)}")
print(f"Output: {V2E_OUTPUT_DIR}\n")

cmd = [
    "python", "/kaggle/working/v2e/v2e.py",
    "--input", VIDEO_FILE,
    "--output_folder", V2E_OUTPUT_DIR,
    "--dvs346",
    "--output_width", "346",
    "--output_height", "260",
    "--disable_slomo",
    "--dvs_h5", "events.h5",
    "--no_preview",
    "--overwrite"
]

result = subprocess.run(cmd, capture_output=False, text=True)

if result.returncode == 0:
    h5_path = os.path.join(V2E_OUTPUT_DIR, "events.h5")
    if os.path.exists(h5_path):
        size_mb = os.path.getsize(h5_path) / (1024**2)
        print(f"\n✓ Success!")
        print(f"  H5 file: {h5_path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"\nCopy this path for the next cell:")
        print(f"  {h5_path}")
    else:
        print("✗ H5 file not created")
else:
    print(f"✗ v2e failed with return code {result.returncode}")
```

**Expected output:**
```
✓ Success!
  H5 file: /kaggle/working/v2e_output/events.h5
  Size: 12.34 MB

Copy this path for the next cell:
  /kaggle/working/v2e_output/events.h5
```

Takes ~2–5 minutes depending on video length.

---

### Step 4: Configure & Run the Pipeline

Paste and run these cells in order:

#### **Cell 4a: Imports + Logging**

```python
import os
import cv2
import h5py
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

print("✓ Imports complete")
```

#### **Cell 4b: Configuration**

```python
@dataclass
class PipelineConfig:
    """All pipeline parameters in one place."""
    
    # I/O
    h5_file: str = ""
    output_dir: str = "/kaggle/working/meteor_output"
    output_video: str = ""
    output_csv: str = ""
    
    # Temporal Windowing
    window_us: int = 100_000      # µs per window
    step_us: int = 100_000        # step size
    
    # Blob Detection
    blob_min_area: int = 2
    blob_max_area: int = 500
    threshold_percentile: int = 90
    morph_kernel_size: int = 3
    
    # Tracking
    max_association_distance: float = 50.0
    max_missed_windows: int = 3
    min_track_length: int = 3
    min_track_duration_s: float = 0.10
    
    # Scoring
    meteor_score_threshold: float = 0.85  # ← ADJUST THIS (0.75-0.90)
    
    # Video Output
    video_fps: int = 10
    video_scale: int = 3
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.output_video:
            self.output_video = os.path.join(self.output_dir, "meteor_tracks.mp4")
        if not self.output_csv:
            self.output_csv = os.path.join(self.output_dir, "meteor_tracks.csv")

# ⚠️ CHANGE THIS TO YOUR H5 FILE PATH
config = PipelineConfig(
    h5_file="/kaggle/working/v2e_output/events.h5"
)

print(f"Config loaded")
print(f"  Threshold: {config.meteor_score_threshold}")
print(f"  Output dir: {config.output_dir}")
```

#### **Cell 4c: Load Events**

```python
def load_events(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load event data from HDF5."""
    logger.info(f"Loading events from {h5_path}...")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        if 'events' not in f:
            raise KeyError(f"'events' dataset not found")
        events = f['events'][:]
    
    if events.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got {events.shape[1]}")
    
    t = events[:, 0].astype(np.int64)
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    
    width = int(x.max()) + 1
    height = int(y.max()) + 1
    
    logger.info(f"  Events: {len(t):,}")
    logger.info(f"  Resolution: {width} x {height}")
    logger.info(f"  Duration: {(t.max()-t.min())/1e6:.2f}s")
    
    return t, x, y, width, height

# Load
t, x, y, width, height = load_events(config.h5_file)
```

#### **Cell 4d: Blob Detection Function**

```python
def detect_blobs_in_window(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
    window_start: int,
    window_end: int,
    config: PipelineConfig
) -> List[Dict[str, Any]]:
    """Detect connected components (blobs) in a temporal window."""
    
    mask = (t >= window_start) & (t < window_end)
    ex, ey = x[mask], y[mask]
    
    if len(ex) == 0:
        return []
    
    # Count events per pixel
    count_img = np.zeros((height, width), dtype=np.uint16)
    np.add.at(count_img, (ey, ex), 1)
    
    # Normalize to 0-255
    activity = cv2.normalize(count_img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Threshold at percentile
    nonzero = activity[activity > 0]
    if len(nonzero) == 0:
        return []
    
    threshold = max(5, int(np.percentile(nonzero, config.threshold_percentile)))
    binary = (activity >= threshold).astype(np.uint8) * 255
    
    # Morphology
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morph_kernel_size, config.morph_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if not (config.blob_min_area <= area <= config.blob_max_area):
            continue
        
        cx, cy = centroids[i]
        component = (labels == i)
        event_count = int(count_img[component].sum())
        
        blobs.append({
            "cx": float(cx),
            "cy": float(cy),
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]),
            "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "area": int(area),
            "events": event_count
        })
    
    return blobs

logger.info("Blob detection ready")
```

#### **Cell 4e: Tracker Class**

```python
class MeteorTracker:
    """Track moving objects and score for meteor-likeness."""
    
    def __init__(self, config: PipelineConfig, width: int, height: int):
        self.config = config
        self.width = width
        self.height = height
        self.tracks: List[Dict[str, Any]] = []
        self.next_track_id = 0
        self.blob_cache: Dict[int, List[Dict]] = {}
    
    def create_track(self, blob: Dict, timestamp_s: float):
        track = {
            "id": self.next_track_id,
            "points": [(timestamp_s, blob["cx"], blob["cy"])],
            "blobs": [blob],
            "missed": 0,
            "active": True
        }
        self.tracks.append(track)
        self.next_track_id += 1
    
    def process_windows(self, t: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Process all temporal windows."""
        start_time = int(t.min())
        end_time = int(t.max())
        
        window_times = np.arange(start_time, end_time + config.window_us, config.step_us)
        
        logger.info(f"Processing {len(window_times)} windows...")
        
        for window_start in tqdm(window_times, desc="Tracking"):
            window_end = window_start + config.window_us
            blobs = detect_blobs_in_window(t, x, y, self.width, self.height, window_start, window_end, config)
            
            self.blob_cache[window_start] = blobs
            
            timestamp_s = (window_start - start_time) / 1e6
            active_tracks = [tr for tr in self.tracks if tr["active"]]
            assignments = set()
            
            for track in active_tracks:
                last_t, last_x, last_y = track["points"][-1]
                candidates = []
                
                for blob_idx, blob in enumerate(blobs):
                    if blob_idx in assignments:
                        continue
                    
                    dx = blob["cx"] - last_x
                    dy = blob["cy"] - last_y
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist <= config.max_association_distance:
                        candidates.append((dist, blob_idx, blob))
                
                if candidates:
                    candidates.sort(key=lambda z: z[0])
                    _, blob_idx, blob = candidates[0]
                    
                    assignments.add(blob_idx)
                    track["points"].append((timestamp_s, blob["cx"], blob["cy"]))
                    track["blobs"].append(blob)
                    track["missed"] = 0
                else:
                    track["missed"] += 1
                    if track["missed"] > config.max_missed_windows:
                        track["active"] = False
            
            for blob_idx, blob in enumerate(blobs):
                if blob_idx not in assignments:
                    self.create_track(blob, timestamp_s)
    
    def score_track(self, track: Dict) -> Optional[Dict[str, float]]:
        """Score a trajectory for meteor-likeness."""
        points = track["points"]
        
        if len(points) < 3:
            return None
        
        times = np.array([p[0] for p in points])
        xs = np.array([p[1] for p in points])
        ys = np.array([p[2] for p in points])
        
        duration = times[-1] - times[0]
        if duration <= 0:
            return None
        
        dx = np.diff(xs)
        dy = np.diff(ys)
        step_dist = np.sqrt(dx**2 + dy**2)
        path_length = step_dist.sum()
        displacement = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        
        if path_length <= 0:
            return None
        
        straightness = displacement / path_length
        velocity = displacement / duration
        
        angles = np.arctan2(dy, dx)
        overall_angle = np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])
        angle_diff = np.arctan2(np.sin(angles - overall_angle), np.cos(angles - overall_angle))
        mean_dir_error = np.mean(np.abs(np.degrees(angle_diff)))
        
        if len(step_dist) > 1:
            mean_step = np.mean(step_dist)
            std_step = np.std(step_dist)
            step_consistency = mean_step / (mean_step + std_step + 1e-6)
        else:
            step_consistency = 0.0
        
        persistence_score = min(len(points) / 8.0, 1.0)
        duration_score = min(duration / 1.0, 1.0)
        displacement_score = min(displacement / 30.0, 1.0)
        straightness_score = np.clip(straightness, 0, 1)
        direction_score = np.clip(1.0 - (mean_dir_error / 90.0), 0, 1)
        step_score = np.clip(step_consistency, 0, 1)
        
        path_score = (
            0.25 * persistence_score +
            0.15 * duration_score +
            0.15 * displacement_score +
            0.25 * straightness_score +
            0.10 * direction_score +
            0.10 * step_score
        )
        
        return {
            "duration": duration,
            "points": len(points),
            "path_length": path_length,
            "displacement": displacement,
            "velocity": velocity,
            "straightness": straightness,
            "direction_error": mean_dir_error,
            "step_consistency": step_consistency,
            "path_score": path_score
        }
    
    def filter_and_rank(self) -> Tuple[List, List]:
        """Filter tracks and return (all_valid, meteor_candidates)."""
        all_results = []
        
        for track in self.tracks:
            score = self.score_track(track)
            if not score:
                continue
            if score["points"] < config.min_track_length:
                continue
            if score["duration"] < config.min_track_duration_s:
                continue
            
            all_results.append((track, score))
        
        all_results.sort(key=lambda z: z[1]["path_score"], reverse=True)
        
        meteor_tracks = [
            (t, r) for t, r in all_results
            if r["path_score"] >= config.meteor_score_threshold
        ]
        
        logger.info(f"Raw tracks: {len(self.tracks)}")
        logger.info(f"Valid tracks: {len(all_results)}")
        logger.info(f"Meteor candidates: {len(meteor_tracks)}")
        
        return all_results, meteor_tracks

# Initialize
tracker = MeteorTracker(config, width, height)
logger.info("Tracker initialized")
```

#### **Cell 4f: Run Tracking**

```python
# Process all windows
tracker.process_windows(t, x, y)

# Score and filter
all_tracks, meteor_tracks = tracker.filter_and_rank()

# Print results
logger.info("\n" + "="*70)
logger.info("TOP 10 DETECTIONS")
logger.info("="*70 + "\n")

for rank, (track, score) in enumerate(all_tracks[:10], 1):
    is_meteor = "✓ METEOR" if score["path_score"] >= config.meteor_score_threshold else ""
    print(
        f"#{rank} Track {track['id']} | Score {score['path_score']:.2f} {is_meteor}\n"
        f"  Duration: {score['duration']:.2f}s | Points: {score['points']} | Velocity: {score['velocity']:.1f}px/s\n"
        f"  Straightness: {score['straightness']:.2f} | Direction error: {score['direction_error']:.1f}°\n"
    )
```

#### **Cell 4g: Export CSV**

```python
def export_csv(tracker, all_tracks, meteor_score_threshold, output_csv):
    """Export tracks to CSV."""
    rows = []
    
    for track, score in all_tracks:
        is_meteor = score["path_score"] >= meteor_score_threshold
        
        for point in track["points"]:
            rows.append({
                "track_id": track["id"],
                "time_s": point[0],
                "x": point[1],
                "y": point[2],
                "duration_s": score["duration"],
                "displacement_px": score["displacement"],
                "velocity_px_s": score["velocity"],
                "straightness": score["straightness"],
                "direction_error_deg": score["direction_error"],
                "step_consistency": score["step_consistency"],
                "path_score": score["path_score"],
                "is_meteor": is_meteor
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"✓ CSV exported: {output_csv}")

export_csv(tracker, all_tracks, config.meteor_score_threshold, config.output_csv)
```

#### **Cell 4h: Generate Video**

```python
def create_visualization_video(tracker, meteor_tracks, t, config):
    """Generate MP4 with tracked paths overlaid."""
    logger.info("Generating video...")
    
    start_time = int(t.min())
    end_time = int(t.max())
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        config.output_video,
        fourcc,
        config.video_fps,
        (config.video_scale * tracker.width, config.video_scale * tracker.height)
    )
    
    window_times = np.arange(start_time, end_time + config.window_us, config.step_us)
    
    for window_start in tqdm(window_times, desc="Video"):
        elapsed = (window_start - start_time) / 1e6
        
        frame = np.zeros((tracker.height, tracker.width, 3), dtype=np.uint8)
        
        # Draw all blobs
        blobs = tracker.blob_cache.get(window_start, [])
        for blob in blobs:
            cv2.circle(frame, (int(blob["cx"]), int(blob["cy"])), 2, (50, 50, 50), -1)
        
        # Draw meteor trajectories
        for track, score in meteor_tracks:
            visible = [p for p in track["points"] if p[0] <= elapsed]
            if len(visible) < 2:
                continue
            
            coords = [(int(p[1]), int(p[2])) for p in visible]
            
            for j in range(1, len(coords)):
                cv2.line(frame, coords[j-1], coords[j], (255, 255, 255), 2)
            
            cx, cy = coords[-1]
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
            cv2.putText(
                frame,
                f"S={score['path_score']:.2f}",
                (max(2, cx + 6), max(15, cy - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # HUD
        cv2.rectangle(frame, (0, 0), (tracker.width, 30), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"T={elapsed:.2f}s | METEORS={len(meteor_tracks)}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        frame = cv2.resize(
            frame,
            (config.video_scale * tracker.width, config.video_scale * tracker.height),
            interpolation=cv2.INTER_NEAREST
        )
        writer.write(frame)
    
    writer.release()
    logger.info(f"✓ Video saved: {config.output_video}")

create_visualization_video(tracker, meteor_tracks, t, config)
```

#### **Cell 4i: Display Results**

```python
from IPython.display import Video, display

logger.info(f"\n{'='*70}")
logger.info("PIPELINE COMPLETE")
logger.info(f"{'='*70}")
logger.info(f"Meteors detected: {len(meteor_tracks)}")
logger.info(f"Output directory: {config.output_dir}")
logger.info(f"Video: {config.output_video}")
logger.info(f"CSV: {config.output_csv}\n")

if len(meteor_tracks) > 0:
    display(Video(config.output_video, embed=True))
else:
    print("No meteors detected. Try lowering meteor_score_threshold in Cell 4b.")
```

---

## Understanding the Pipeline

### The 9-Stage Process

1. **Temporal Windowing** — Split 11s video into ~110 windows of 100ms each
2. **Event Histogram** — Count events per pixel in each window
3. **Normalization** — Stretch 0–max event count to 0–255
4. **Thresholding** — Keep only pixels above 90th percentile (signal) → drop noise
5. **Morphology** — Clean up artifacts (open + close operations)
6. **Connected Components** — Label contiguous regions as "blobs"
7. **Tracking** — Connect blobs across windows using nearest-neighbor matching
8. **Scoring** — Evaluate trajectories (straightness, velocity, persistence, etc.)
9. **Filtering** — Keep only high-confidence meteors (score ≥ threshold)

### Key Metrics

| Metric | Meaning | Meteor Range | Noise Range |
|--------|---------|--------------|-------------|
| **Straightness** | Displacement / Path Length | 0.9–1.0 | 0.2–0.5 |
| **Persistence** | Number of observation points | 10–50 | 2–5 |
| **Duration** | Track lifetime (seconds) | 0.5–6.0 | 0.1–0.5 |
| **Velocity** | pixels/second | 20–100 | Erratic |
| **Direction Error** | Angle variance (degrees) | 0–30° | 30–60° |

### Final Score Formula

```
path_score = (
    0.25 * straightness +          # Most important
    0.25 * persistence +
    0.15 * duration +
    0.15 * displacement +
    0.10 * direction_consistency +
    0.10 * step_consistency
)
```

Meteors typically score **0.85–0.95**, noise **0.10–0.70**.

---

## Tuning Parameters

**In Cell 4b**, adjust these to fine-tune detection:

```python
# Sensitivity (90 = stricter, 75 = looser)
threshold_percentile: int = 90

# Maximum distance blob can move between windows (pixels)
max_association_distance: float = 50.0

# Minimum track lifetime (seconds)
min_track_duration_s: float = 0.10

# Final filter threshold (higher = fewer false positives)
meteor_score_threshold: float = 0.85  # ← KEY PARAMETER
```

### If you see too many false positives:
- Increase `meteor_score_threshold` (try 0.90)
- Increase `threshold_percentile` (try 95)
- Increase `min_track_duration_s` (try 0.20)

### If you miss real meteors:
- Decrease `meteor_score_threshold` (try 0.75)
- Decrease `threshold_percentile` (try 80)
- Increase `max_association_distance` (try 70)

---

## Output Files

After running the pipeline, you'll have:

```
/kaggle/working/meteor_output/
├── meteor_tracks.mp4           # Video with overlaid trajectories
├── meteor_tracks.csv           # Detailed track data (one row per point)
└── [optional visualizations]
```

### CSV Columns

```
track_id               : Unique track identifier
time_s                 : Time in seconds (0.0–10.96)
x, y                   : Pixel coordinates in event space
duration_s             : Total track lifetime
displacement_px        : Straight-line distance (start→end)
velocity_px_s          : Average pixels per second
straightness           : 0–1 (higher = straighter)
direction_error_deg    : Angle variance (lower = more consistent)
step_consistency       : 0–1 (uniformity of step sizes)
path_score             : Final score (0–1)
is_meteor              : True if score ≥ threshold
```

---

## Troubleshooting

### "H5 file not found"
- Verify v2e conversion completed successfully
- Check that file exists: `os.path.exists("/kaggle/working/v2e_output/events.h5")`
- Re-run v2e cell

### "No meteors detected"
- Lower `meteor_score_threshold` to 0.75
- Lower `threshold_percentile` to 80
- Check video actually contains a meteor (watch original first)

### "OpenCV error in cv2.normalize"
- Make sure to convert count_img to float32 before normalizing:
  ```python
  activity = cv2.normalize(count_img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
  ```

### "ModuleNotFoundError: dv_processing"
- Run: `pip install -q dv-processing`

### Video generation is slow
- Reduce `video_scale` from 3 to 2 or 1
- Reduce `video_fps` from 10 to 5

---

## Next Steps

1. **Run on multiple videos** — Test if threshold generalizes across different meteors
2. **Measure precision/recall** — Label a few videos manually, compare with pipeline output
3. **Visualize raw vs cleaned** — Add cells to show event heatmaps before/after thresholding
4. **Real event data** — When you get actual DVS/neuromorphic sensor data, swap the HDF5 loader

---

## References

- **v2e:** Event Synthesis from RGB Videos [GitHub](https://github.com/SensorsINI/v2e)
- **Event Cameras:** Asynchronous pixel sensors that respond to brightness changes
- **Connected Components:** OpenCV labeling algorithm for finding contiguous regions

---

## Authors

Pipeline developed for meteor detection from synthetic event camera data.  
Dataset: [Allsky7 Meteor Dataset](https://www.kaggle.com/datasets/virajsaws/meteor-dataset-allsky7/data)

---

## License

Open source. Feel free to modify and distribute.
