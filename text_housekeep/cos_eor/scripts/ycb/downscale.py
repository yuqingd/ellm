from PIL import Image
import glob
import os
import os.path as osp
from tqdm import tqdm

DOWNSCALE_DIM = 64

base_dir = 'data/objects/ycb'
files = glob.glob(base_dir + '/**/*.png', recursive=True)
for f in tqdm(files):
    if '_opt' in f:
        continue
    im = Image.open(f)
    #opt_f = f.split('.')[0]+'.png'
    im = im.resize((DOWNSCALE_DIM,DOWNSCALE_DIM),Image.ANTIALIAS)
    im.save(f, quality=25)
