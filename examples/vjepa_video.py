"""V-JEPA 2.1 temporal feature maps over a real video clip -> filmstrip + GIFs.

Unlike a 2D backbone (which `featlens.video` runs frame-by-frame), V-JEPA is a *temporal*
model: the whole clip is fed once and its spatiotemporal token sequence is split back into
per-time-step grids (`featlens.tokens.tokens_to_spatiotemporal`). With `method="pca"` one PCA
basis is shared across all frames, so the colors stay consistent over time and you can watch the
subject move — here the cockatoo (the central region) bobs against a fixed perch/background.

Run:  pip install "featlens[video]" && python examples/vjepa_video.py
Needs network on first run (the V-JEPA weights download via torch.hub) and is CPU-heavy
(~30s/clip on CPU at 384px; a GPU is much faster). Uses the bundled examples/videos/cockatoo.mp4.
Outputs are written next to this file (the committed gallery the README displays):
  - vjepa_video.png          the filmstrip (8 time-steps across the last layer)
  - vjepa_video.gif          those feature maps animated
  - vjepa_video_compare.gif  the input clip beside its feature map, side by side
"""

from pathlib import Path

import numpy as np
from PIL import Image

import featlens as ll
from featlens.video import _load_frames

HERE = Path(__file__).parent
CLIP = HERE / "videos" / "cockatoo.mp4"
N_FRAMES, FPS, DISP = 16, 4, 224

# vjepa2_1_vitb16 is the temporal V-JEPA 2.1 base model (384px native). 16 sampled frames
# collapse to 8 spatiotemporal steps (tubelet_size=2); the shared-PCA filmstrip is the row.
res = ll.video("vjepa2_1_vitb16", CLIP, layers=[-1], n_frames=N_FRAMES, method="pca", img_size=384,
               out=HERE / "vjepa_video.png", return_data=True)

# Side-by-side GIF: the real input frame next to its V-JEPA feature map. The 16 sampled frames
# collapse to 8 spatiotemporal steps (tubelet_size=2), so step t lines up with input frame 2t.
feat = np.asarray(res["frames_rgb"])[-1]            # [steps, DISP, DISP, 3] in [0, 1]
inputs = _load_frames(CLIP, N_FRAMES)               # the same uniformly sampled frames
gap = np.ones((DISP, 6, 3), np.float32)
combo = []
for t in range(feat.shape[0]):
    src = inputs[min(2 * t, len(inputs) - 1)].resize((DISP, DISP), Image.LANCZOS)
    pair = np.concatenate([np.asarray(src, np.float32) / 255.0, gap, feat[t]], axis=1)
    combo.append(Image.fromarray((np.clip(pair, 0, 1) * 255).astype(np.uint8)))
combo[0].save(HERE / "vjepa_video_compare.gif", save_all=True, append_images=combo[1:],
              duration=max(1, int(1000 / FPS)), loop=0, optimize=True)

print(f"Wrote vjepa_video.png, vjepa_video.gif and vjepa_video_compare.gif to {HERE}")
