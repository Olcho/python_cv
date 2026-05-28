import time
from pathlib import Path

import cv2 as cv
import numpy as np
from skimage import graph, segmentation

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / 'clownfish.png'
OUT_DIR = BASE_DIR / 'practice12_results'
OUT_DIR.mkdir(exist_ok=True)

img_bgr = cv.imread(str(INPUT_PATH))
if img_bgr is None:
    raise FileNotFoundError(f'Cannot read image: {INPUT_PATH}')
img_rgb_original = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
# Resize for practical execution speed and document-friendly output size.
img_rgb = cv.resize(img_rgb_original, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)


def save_rgb(path, rgb):
    cv.imwrite(str(path), cv.cvtColor(rgb, cv.COLOR_RGB2BGR))


def add_label(rgb, text):
    out = rgb.copy()
    cv.rectangle(out, (8, 8), (245, 48), (255, 255, 255), -1)
    cv.rectangle(out, (8, 8), (245, 48), (0, 0, 0), 2)
    cv.putText(out, text, (17, 35), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv.LINE_AA)
    return out


# Practice 1: compare SLIC compactness and n_segments.
settings = [(5, 100), (50, 100), (5, 500), (50, 500)]
marked_images = []
actual_counts = []

for compactness, n_segments in settings:
    print(f'Running SLIC compactness={compactness}, n_segments={n_segments}...', flush=True)
    labels = segmentation.slic(
        img_rgb,
        compactness=compactness,
        n_segments=n_segments,
        start_label=1,
        channel_axis=-1,
    )
    actual_counts.append(len(np.unique(labels)))

    marked = segmentation.mark_boundaries(img_rgb, labels, color=(0, 1, 1))
    marked = np.uint8(marked * 255.0)
    marked = add_label(marked, f'Comp.={compactness}, nseg={n_segments}')

    out_name = f'task1_slic_comp{compactness}_nseg{n_segments}.png'
    save_rgb(OUT_DIR / out_name, marked)
    marked_images.append(marked)

row1 = np.hstack([marked_images[0], marked_images[1]])
row2 = np.hstack([marked_images[2], marked_images[3]])
task1_result = np.vstack([row1, row2])
save_rgb(OUT_DIR / 'task1_slic_comparison.png', task1_result)

# Practice 2: normalized cut using superpixels.
ncut_input_rgb = cv.resize(img_rgb_original, None, fx=0.18, fy=0.18, interpolation=cv.INTER_AREA)

print('Running SLIC for normalized cut...', flush=True)
start = time.time()
slic = segmentation.slic(
    ncut_input_rgb,
    compactness=5,
    n_segments=500,
    start_label=1,
    channel_axis=-1,
)
print('Building RAG from superpixels...', flush=True)
g = graph.rag_mean_color(ncut_input_rgb, slic, mode='similarity')
print('Running normalized cut...', flush=True)
ncut = graph.cut_normalized(slic, g, thresh=0.05, num_cuts=1, rng=1)
elapsed = time.time() - start

# Draw thicker yellow boundaries for better visibility in the report.
boundary = np.zeros(ncut.shape, dtype=np.uint8)
boundary[:, 1:] |= (ncut[:, 1:] != ncut[:, :-1])
boundary[1:, :] |= (ncut[1:, :] != ncut[:-1, :])
boundary = cv.dilate(boundary, np.ones((3, 3), np.uint8), iterations=1)
ncut_marked = ncut_input_rgb.copy()
ncut_marked[boundary > 0] = (255, 255, 0)
ncut_marked = add_label(ncut_marked, 'Normalized Cut')
save_rgb(OUT_DIR / 'task2_normalized_cut.png', ncut_marked)

input_labeled = add_label(ncut_input_rgb, 'Input: clownfish')
side_by_side = np.hstack([input_labeled, ncut_marked])
save_rgb(OUT_DIR / 'task2_input_and_ncut.png', side_by_side)

print('Practice 1 actual SLIC label counts')
for (compactness, n_segments), count in zip(settings, actual_counts):
    print(f'  compactness={compactness}, n_segments={n_segments} -> labels={count}')
print('Practice 2')
print(f'  SLIC labels before ncut: {len(np.unique(slic))}')
print(f'  normalized cut labels: {len(np.unique(ncut))}')
print(f'  execution time: {elapsed:.2f} sec')
print(f'  results saved to: {OUT_DIR}')
