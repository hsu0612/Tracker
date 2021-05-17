import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

img = np.array(cv2.imread('./img14_6.jpg'))
img = img_as_float(img)

segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=50, compactness=10, sigma=1,
                     start_label=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=50, compactness=0.001)

def sort_superpixel(x):
    return x[1]

superpixel_mask = np.zeros((128, 128), dtype=int)
superpixel_mask = superpixel_mask-1
superpixel_mask[32:96, 32:96] = segments_slic[32:96, 32:96].copy()

superpixel_mask_2 = segments_slic.copy()
superpixel_mask_2[32:96, 32:96] = 0.0
print(superpixel_mask_2.shape)

final_result = np.zeros((128, 128), dtype=int)

superpixel_list = []
for i in range(0, segments_slic.max() + 1, 1):
    if np.count_nonzero(superpixel_mask_2 == i) != 0:
        superpixel_mask[superpixel_mask == i] = 0
    # error_map_copy = error_map[0].detach().cpu().numpy().copy()
    # error_map_copy = np.where(superpixel_mask == i, error_map_copy, 0)
    # error_map_copy = (error_map_copy.sum() / num_ele)
    # final_mask = np.where(superpixel_mask == i, error_map_copy, final_mask)

superpixel_mask[superpixel_mask > 0] = 255

cv2.imwrite("./a.jpg", superpixel_mask)

print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()