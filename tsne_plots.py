# t-SNE plot for the feature embeddings of real and synthesized portraits of all syndromes
import pandas as pd
import os
from sklearn.manifold import TSNE
import seaborn as sns

encodings_dir = "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/encodings/"
file_names_orig = ["encodings_gmdb_0.csv", "encodings_gmdb_1.csv", "encodings_gmdb_2.csv", "encodings_gmdb_3.csv",
                   "encodings_gmdb_12.csv"]
file_names_synth = ["encodings_1_2_0_justweightedsynth.csv", "encodings_1_2_1_justweightedsynth.csv",
                    "encodings_1_2_2_justweightedsynth.csv", "encodings_1_2_3_justweightedsynth.csv",
                    "encodings_1_2_12_justweightedsynth.csv"]

df = pd.DataFrame()
num_images_orig = []
for file_name in file_names_orig:
    f = os.path.join(encodings_dir, file_name)
    encodings = pd.read_csv(f, delimiter=";", converters={'representations': pd.eval})
    num_images_orig.append(len(encodings))
    df = pd.concat([df, encodings], ignore_index=True)

num_images_synth = []
for file_name in file_names_synth:
    f = os.path.join(encodings_dir, file_name)
    encodings = pd.read_csv(f, delimiter=";", converters={'representations': pd.eval})
    num_images_synth.append(len(encodings))
    df = pd.concat([df, encodings], ignore_index=True)

matrix_data = pd.DataFrame(df['representations'].values.tolist())

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(matrix_data)

tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
tsne_df['method'] = ['Original'] * sum(num_images_orig) + ['Synthetic'] * sum(num_images_synth)
tsne_df['syndrome'] = ['C.d.L.'] * num_images_orig[0] + ['Williams-B.'] * num_images_orig[1] + ['Kabuki'] * num_images_orig[
    2] + ['Angelman'] * num_images_orig[3] + ['HPMRS'] * num_images_orig[4] + ['C.d.L.'] * num_images_synth[0] + [
    'Williams-B.'] * num_images_synth[1] + ['Kabuki'] * num_images_synth[2] + ['Angelman'] * num_images_synth[3] + [
    'HPMRS'] * num_images_synth[4]

# Plot the t-SNE results
sns.scatterplot(data=tsne_df, x='x', y='y', hue='syndrome', style='method')

