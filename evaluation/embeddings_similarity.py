# encodings are generated with cd gestaltmatcher -> python predict.py --data_dir ../../data/crops_256_12 --num_classes 5

import pandas as pd
from numpy import dot, argmax
from numpy.linalg import norm
import os


def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


encodings_dir = "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/encodings/"
file_name_orig = "encodings_gmdb_3.csv"
file_name_synth = "encodings_selection_3.csv"

f_orig = os.path.join(encodings_dir, file_name_orig)
f_synth = os.path.join(encodings_dir, file_name_synth)

encodings_orig = pd.read_csv(f_orig, delimiter=";", converters={'representations': pd.eval})
encodings_synth = pd.read_csv(f_synth, delimiter=";", converters={'representations': pd.eval})

similarities = {'orig_file': [], 'synth_file': [], 'similarity': []}

for ix1, row_orig in encodings_orig.iterrows():
    for ix2, row_synth in encodings_synth.iterrows():
        sim = cosine_similarity(row_orig['representations'], row_synth['representations'])
        similarities['orig_file'].append(row_orig[0])
        similarities['synth_file'].append(row_synth[0])
        similarities['similarity'].append(sim)

similarities = pd.DataFrame(similarities)
similarities.sort_values(by='similarity', ascending=False, inplace=True)
out_path = "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/similarities/12/"
similarities.to_csv(os.path.join(out_path, 'similarities.csv'))
