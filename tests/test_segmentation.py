from pathlib import Path

import numpy as np
from PIL import Image

from kmeans.segmentation import segment_img
# Assumes your are in the top-level directory of package
indir = Path("images")
outdir = Path("tests/output")

with Image.open(indir / "img00.png") as pil:
    img = np.asarray(pil.convert('RGB'))
num_groups=4
seg_img = segment_img(img, groups=num_groups)
seg_img2 = segment_img(img, groups=num_groups, random_colors=True)

out1 = Image.fromarray(np.concatenate((img, seg_img), axis=1))
out2 = Image.fromarray(np.concatenate((img, seg_img2), axis=1))

outfile1 = outdir / f"seg_groups{num_groups:02d}.jpg"
outfile2 = outdir / f"seg_rand_groups{num_groups:02d}.jpg"

out1.save(outfile1, "JPEG")
out2.save(outfile2, "JPEG")
