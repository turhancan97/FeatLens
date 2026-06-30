"""V-JEPA 2.1 temporal feature maps over a real video clip -> filmstrip + GIF.

Unlike a 2D backbone (which `featlens.video` runs frame-by-frame), V-JEPA is a *temporal*
model: the whole clip is fed once and its spatiotemporal token sequence is split back into
per-time-step grids (`featlens.tokens.tokens_to_spatiotemporal`). With `method="pca"` one PCA
basis is shared across all frames, so the colors stay consistent over time and you can watch the
subject move — here the cockatoo (the central region) bobs against a fixed perch/background.

Run:  pip install "featlens[video]" && python examples/vjepa_video.py
Needs network on first run (the V-JEPA weights download via torch.hub) and is CPU-heavy
(~30s/clip on CPU at 384px; a GPU is much faster). Uses the bundled examples/videos/cockatoo.mp4.
Outputs are written next to this file (the committed gallery the README displays).
"""

from pathlib import Path

import featlens as ll

HERE = Path(__file__).parent
CLIP = HERE / "videos" / "cockatoo.mp4"

# vjepa2_1_vitb16 is the temporal V-JEPA 2.1 base model (384px native). 16 sampled frames
# collapse to 8 spatiotemporal steps (tubelet_size=2); the shared-PCA filmstrip is the row.
ll.video("vjepa2_1_vitb16", CLIP, layers=[-1], n_frames=16, method="pca", img_size=384,
         out=HERE / "vjepa_video.png")

print(f"Wrote vjepa_video.png + vjepa_video.gif to {HERE}")
