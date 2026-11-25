# Building the dataset with Roboflow

Classes: **aircraft**, **drone**, **vehicle**. Roboflow numbers them alphabetically: `0 aircraft, 1 drone, 2 vehicle`.
(Roboflow's UI changes from time to time. If a button has moved, the concepts below still apply.)

## 1. Annotation rules (agree on these BEFORE labelling)

Consistency matters more than volume. Put these rules in your report as the "labelling protocol".

| Class | Includes | Excludes / edge cases |
|---|---|---|
| `drone` | multirotors, fixed-wing UAVs, hobby quadcopters | birds, kites, balloons (leave unlabelled; they are useful hard negatives) |
| `aircraft` | airliners, light aircraft, fighter jets, **helicopters** | anything clearly a UAV goes to `drone` |
| `vehicle` | cars, vans, trucks, buses, motorbikes | bicycles, parked trailers without a cab (pick one rule and keep it) |

- **Tight boxes:** box edges touch the outermost visible pixels. Include rotors/propellers, but not shadows or contrails.
- **Label every instance**, including tiny ones. If you can tell it is a drone (≥ ~5 px), box it.
  An unlabelled real drone teaches the model that drones are background.
- **Occluded / truncated:** box only the visible part. Objects cut off at the image border still get a box.
- **Unsure?** Mark the image with a tag (`unsure`) and decide later. Don't guess different ways on different days.
- **Background images:** keep 5-10% images with no objects (empty sky, birds, clouds, empty roads) and mark them **Null**.

## 2. Annotate in Roboflow

1. **Create project:** app.roboflow.com → *New Project* → type **Object Detection**, name `aerial-objects`,
   annotation group `aerial-object`. Pick the license that matches your sources.
2. **Upload:** drag in images or videos.
   - For **videos**, choose a frame sampling rate of **1-2 fps**. At 30 fps, neighbouring frames are near-duplicates that cost labelling time without adding information.
   - Name video files distinctly (`clip07_sky_dusk.mp4`) so frame names like `clip07_..._0123` keep the source.
   - Already have labels (YOLO/VOC/COCO)? Upload them with the images and Roboflow imports them.
3. **Create classes first** (*Classes* tab: `aircraft`, `drone`, `vehicle`) to prevent typos like `Drone`/`drones`.
4. **Assign & annotate** (*Annotate* tab → assign batch → open an image):
   - `B` = bounding-box tool, draw, pick the class. Use arrow keys / `→` for next image.
   - Mark empty images as **Null** instead of skipping them.
   - **Speed-up:** after ~150-200 hand-labelled images, use **Label Assist** or **Auto Label** (prompt with
     "drone", "airplane", "car"). Treat the result as a draft and **review every box**, especially tiny drones and birds.
   - Polygons (Smart Polygon) are fine: `prepare_dataset.py` converts them to boxes.
5. **Review:** approve the batch → *Add Approved to Dataset*.
6. **Health Check** (*Analytics/Health Check*): look at class balance, the object-size heatmap and annotation counts.
   Screenshot it for your report.

## 3. Generate a version and export (YOLOv8 format)

*Versions → Create New Version*:

| Step | Setting | Why |
|---|---|---|
| Train/Valid/Test split | anything (e.g. 70/20/10) | we **re-split locally** by video, see step 4 |
| Preprocessing: Auto-Orient | **ON** | fixes EXIF-rotated phone photos |
| Preprocessing: Resize | **OFF** (or *Fit within* 1280) | **never "Stretch to 640"**: it distorts aspect ratio, shrinks tiny drones further and stops you training at 960 |
| Preprocessing: Tile | only for 4K frames with tiny drones | if you tile, you must also tile at inference (SAHI) |
| Augmentation | **None** (recommended) | Ultralytics augments online with fresh variants every epoch, see section 5 |

Then *Download Dataset* → format **YOLOv8** → either:
- **Download zip**: unzip to `data/raw/roboflow/`, or
- **Code snippet**: use our script (keeps the API key out of your code):
  ```bash
  export ROBOFLOW_API_KEY=...        # Settings → API Keys. Never commit this.
  python data/download_roboflow.py --workspace <ws> --project aerial-objects --version 1
  ```

Export layout:
```
data/raw/roboflow/
├── data.yaml                 # names: ['aircraft', 'drone', 'vehicle']
├── train/images/  clip07_0123_jpg.rf.3f9a....jpg
├── train/labels/  clip07_0123_jpg.rf.3f9a....txt    # "cls cx cy w h", normalised 0-1
├── valid/...
└── test/...
```

## 4. Validate and re-split locally

Roboflow splits images **at random**. If your images are video frames, frame 120 can be in train and frame 121 in
test. The model then sees near-copies of its test images, and mAP comes out unrealistically high.
`prepare_dataset.py` strips the `_jpg.rf.<hash>` suffix and keeps each video in a single split:

```bash
python data/prepare_dataset.py --roboflow data/raw/roboflow --dry-run                 # stats only
python data/prepare_dataset.py --roboflow data/raw/roboflow --group-pattern "^(.*)_\d+$"
```
`--group-pattern` takes the part before the trailing frame number as the video id (`clip07_0123` → `clip07`).
Adjust it to your naming. The script writes `data/dataset/` + `data/dataset.yaml`.

**Using Roboflow's split as-is** (fine if images are independent photos, not video frames):
```yaml
# data/dataset.yaml
path: /abs/path/to/data/raw/roboflow
train: train/images
val: valid/images
test: test/images
names: {0: aircraft, 1: drone, 2: vehicle}
```
(Roboflow's own `data.yaml` uses `../train/images`-style paths that often break with Ultralytics. Use the version above.)

## 5. Augmentation

Augmentation happens **during training** (`train/hyp_*.yaml`), not in Roboflow:

```bash
python train/train.py --hyp train/hyp_surveillance.yaml ...   # ground camera looking at sky / road (default)
python train/train.py --hyp train/hyp_topdown.yaml ...        # camera ON a drone looking down
```

| Parameter | Surveillance | Top-down | Reasoning |
|---|---|---|---|
| `hsv_v` / `hsv_s` | 0.5 / 0.6 | 0.4 / 0.7 | exposure varies a lot: backlit aircraft, haze, dusk |
| `hsv_h` | 0.015 | 0.015 | colour is a real cue, so keep hue shifts tiny |
| `degrees` | 5 | 10 | large rotations make axis-aligned boxes loose |
| `scale` | 0.4 | 0.5 | > 0.5 shrinks small drones to a few pixels |
| `flipud` | 0.0 | 0.5 | an upside-down sky never occurs, but top-down views have no fixed "up" |
| `fliplr` | 0.5 | 0.5 | always plausible |
| `perspective` | 0 | 0.0005 | oblique drone camera |
| `mosaic` / `close_mosaic` | 1.0 / 15 | 1.0 / 10 | more small objects per batch; last epochs on realistic frames |
| `mixup` | 0.05 | 0.1 | heavy mixup creates ghost objects that resemble birds |

`pip install albumentations` also lets Ultralytics add low-probability blur, median blur, CLAHE and grayscale,
which roughly simulates motion blur and IR/greyscale cameras.

**Exception: when offline (Roboflow) augmentation does help:** a small dataset (< 1,000 images) or a known
deployment condition your data lacks. In that case add only **Blur ≤ 1.5 px**, **Noise ≤ 1%**,
**Brightness ±20%**, **Exposure ±10%** and output **2x**. Roboflow only augments the train split, and the
re-split in step 4 keeps augmented copies with their source image.
