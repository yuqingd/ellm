## Data preparation and directory structure


We now provide the instructions to populate the datasets under `data/`. Make sure that all data is arranged using the provided directory structure.

```
data
├── amazon-berkeley
│   └── 3dmodels
│       └── original
├── datasets
│   └── cos_eor_v11_pruned
│       ├── train
│       ├── val
│       └── test
├── default.physics_config.json  
├── google_object_dataset
│       ├── 11pro_SL_TRX_FG
│       └── ...
├── objects
│   ├── configs_convex
│   ├── ycb
│   ├── 002_heavy_master_chef_can.object_config.json
│   └── ...
├── replica-cad
│   ├── configs
│   ├── objects
│   └── ...
├── scene_datasets
│   └── igibson
│       ├── assets
│       ├── assets_assemble
│       └── scenes
└── urdf_obj_cache

```
#### Amazon Berkeley Objects 
- Download and extract the [Amazon Berkeley](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) 3D models dataset (abo-3dmodels.tar) inside `data/amazon-berkeley`.
- Run `python cos_eor/scripts/ab_objects_viz.py` to generate config files for the downloaded objects

#### Google Scanned Objects
- Run the script `python cos_eor/scripts/gso_downloader.py` to download the [Google Scanned Objects](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects) and generate config files.

#### YCB objects
- Run the script `./cos_eor/scripts/ycb/full_setup.sh`. This script will download YCB objects under `data/objects`, generate physics config files and downscale the assets.

#### Replica-CAD dataset
- Download [ReplicaCAD Interactive objects](https://aihabitat.org/datasets/replica_cad/) dataset under data/replica-cad and extract it to `data/replica-cad`.

#### Downscaling the assets
- Use the script `python cos_eor/scripts/downscale_assets.py` to downscale the downloaded assets.
