# t-SNE plot for the feature embeddings of real and synthesized portraits for an individual syndrome
import pandas as pd
import os
from sklearn.manifold import TSNE
import seaborn as sns

encodings_dir = "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/encodings/"
file_name_orig = "encodings_gmdb_0.csv"
file_name_synth = "encodings_1_2_0_justweightedsynth.csv"

f_orig = os.path.join(encodings_dir, file_name_orig)
f_synth = os.path.join(encodings_dir, file_name_synth)

encodings_orig = pd.read_csv(f_orig, delimiter=";", converters={'representations': pd.eval})
encodings_synth = pd.read_csv(f_synth, delimiter=";", converters={'representations': pd.eval})

df = pd.concat([encodings_orig, encodings_synth], ignore_index=True)

matrix_data = pd.DataFrame(df['representations'].values.tolist())

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(matrix_data)

tsne_df = pd.DataFrame(tsne_result, columns=['x', 'y'])
tsne_df['method'] = ['Original'] * len(encodings_orig) + ['Synthetic'] * len(encodings_synth)

# Plot the t-SNE results
plt.scatter(tsne_df['x'], tsne_df['y'], c=tsne_df['method'])
sns.scatterplot(data=tsne_df, x='x', y='y', hue='method')
plt.show()
