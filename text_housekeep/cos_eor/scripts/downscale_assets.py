from PIL import Image
import glob
import os
import os.path as osp
from tqdm import tqdm
from shutil import copyfile

# replace original images with downsampled versions
DOWNSCALE_DIM = 64
base_dirs = [
    'data/objects/ycb',
    'data/google_object_dataset',
    'data/replica-cad',
    # 'iGibson/gibson2/data/ig_dataset/objects'
    # 'data/scene_datasets/igibson/assets_assemble/Rs_int',
]

files = []
# iterate over all files and downscale
for dir in base_dirs:
    _files = glob.glob(dir + '/**', recursive=True)
    files.extend(_files)
files = [f for f in files if f.endswith(".png")]


import pdb
pdb.set_trace()

for f in tqdm(files):
    img = Image.open(f)

    if img.height != img.width:
        print(F"skipping: ", f)

    # copy original file
    dir, fname = os.path.split(f)
    org_fname = fname.split(".")[0] + "_original.png"
    org_fname = os.path.join(dir, org_fname)
    try:
        assert not os.path.exists(org_fname)
    except:
        import pdb
        pdb.set_trace()

    # copyfile(f, org_fname)
    # try:
    #     assert os.path.exists(org_fname)
    # except:
    #     import pdb
    #     pdb.set_trace()
    # dump new file
    img = img.resize((DOWNSCALE_DIM,DOWNSCALE_DIM),Image.ANTIALIAS)
    img.save(f, quality=25)

