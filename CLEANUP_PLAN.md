# Cleanup and Fine-Tuning Plan

Notes on what needs to be done to clean up the codebase and prepare for hyperparameter sweeps.

---

## Code Cleanup

### Imports (done)
Fixed the relative imports in the API files - they were using `import api_helper` which only works when running from that directory. Changed to absolute imports so they work from anywhere.

### Debug prints to remove
There are some leftover DEBUG print statements in `api_helper.py` (around lines 134 and 235) that should be removed before any production use.

### Unused code?
`utils/metrics.py` has `cosine_similarity()` and `rank1_accuracy()` functions that aren't used anywhere right now. Might want to keep these for evaluation later, or just delete them. Not urgent.

### Dataset classes - keep all three
- `TripletDataset` - returns (anchor, positive, negative) tuples. Will want this for trying different sampling strategies.
- `LabeledImageDataset` - what we're using now with the PK sampler
- `SingleImageDataset` - used for gallery building and inference

---

## Image Preprocessing Options

Things we can try to normalize images and potentially improve re-identification accuracy:

### Rotation and cropping by heading (implemented)
The `rotate_by_direction` flag rotates images so all vessels face the same direction (north/up). After rotation, crops to the inscribed square to avoid black corners. This should help the model since it doesn't have to learn rotational invariance.

Controlled by `rotate_by_direction` in `configs/shared.yaml`.

### Background normalization (not implemented yet)
Satellite images have varying background colors depending on:
- Water conditions (deep blue, greenish coastal, murky)
- Weather and lighting
- Sensor differences between Sentinel-2 and Landsat

Ideas to try:
- **Background color standardization** - detect water pixels and normalize to a consistent color
- **Background removal/masking** - black out everything except the vessel
- **Histogram equalization** - normalize the overall color distribution
- **Grayscale conversion** - remove color entirely, focus on shape

### Vessel centering
Right now we assume vessels are roughly centered. Could add:
- Auto-centering based on detected vessel pixels
- Tighter cropping around the vessel

### Scale normalization
We have vessel length estimates. Could use this to:
- Resize all vessels to appear the same physical size
- Or normalize by pixel-to-meter ratio

---

## Hyperparameter Sweep Ideas

Things worth trying:

**Model stuff**
- Different backbones: resnet18/34/50, maybe efficientnet
- Embedding dimension: 128, 256, 512, 1024
- Whether using length actually helps

**Loss functions**
- Just triplet loss
- Just arcface
- Combined (what we have now)
- Different margins (0.1 to 0.5)
- Cosine vs euclidean distance

**Training**
- Learning rate: probably sweep from 1e-5 to 5e-4
- Batch size: 64, 128, 256 depending on GPU memory
- More or fewer epochs

**PK sampler settings**
- P (identities per batch): 8, 16, 32
- K (samples per identity): 2, 4, 8

**Image preprocessing**
- With and without rotation by heading
- With and without background normalization
- Different image sizes (128, 224, 384)
- Grayscale vs color

**Data augmentation**
- Current augmentation vs none
- Aggressive vs light augmentation
- Which specific augmentations help (blur, noise, color jitter, etc.)

### How to approach it
1. Start with loss function comparison - this has the biggest impact
2. Try image preprocessing options (rotation, background normalization)
3. Then try different backbones/embedding sizes
4. Then tune learning rate and batch size
5. Finally ablate the data augmentation choices

---

## Future Refactoring

Not urgent but would be nice:

**API helper duplication** - The two functions for fetching vessel events are almost identical. Could consolidate into one function with an optional MMSI filter.

**Dataset image loading** - All three dataset classes have basically the same image loading code copy-pasted. Should extract into a shared helper.

**Experiment tracking** - Would be nice to add wandb or mlflow for logging sweeps, but can do this later.

---

## What to do first

1. Remove the debug prints
2. Implement background normalization option
3. Figure out experiment tracking setup for sweeps
4. Run initial sweep on loss functions and preprocessing options
