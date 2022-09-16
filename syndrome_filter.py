# filter unsorted patient .jpg images in a folder per syndrome as given in the metadata file
import pandas as pd
import os
import shutil

syndromes = pd.read_table('data/v1_0_2/metadata/gmdb_syndromes_v1.0.2.tsv')

# turn image_ids string to list of image_ids
syndromes['image_ids'] = syndromes['image_ids'].apply(lambda x: x.replace(x[0], ''))
syndromes['image_ids'] = syndromes['image_ids'].apply(lambda x: x.replace(x[-1], ''))
syndromes['image_ids'] = syndromes['image_ids'].apply(lambda x: x.replace(' ', ''))
syndromes['image_ids'] = syndromes['image_ids'].apply(lambda x: x.split(','))

# get image IDs for syndromes in list
# 0: Cornelia de Lange
# 1: Williams Beuren
# 2: Kabuki
# 3: Angelman
# 12: HPMRS
syndrome_list = [0, 1, 2, 3, 12]
syndrome_files = []

for item in syndrome_list:
    syndrome_files.append([x + '.jpg' for x in syndromes.loc[item, 'image_ids']])

# copy syndrome images to separate dir
for i, syndrome in enumerate(syndrome_files):
    target_path = f'data/crops_256_{syndrome_list[i]}'
    print(f'Next images copied to: {target_path}')
#   os.mkdir(f'data/crops_256_{syndrome_list[i]}')  # somehow breaks the code sometimes. just create the dirs manually
    for file in syndrome:
        target_image = os.path.join('data/crops_256', file)
        print(f'Copying {target_image}')
        shutil.copy(target_image, target_path)

