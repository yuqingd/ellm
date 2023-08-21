import numpy as np
import yaml


# load LM data
model_name = "roberta-large"
dump_path = f"cos_eor/utils/{model_name}_scores.npy"
lm_dump = np.load(dump_path, allow_pickle=True).item()

# load human annotations
