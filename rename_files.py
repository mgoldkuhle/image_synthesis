# to rename all .jpg images in a folder with an ascending trailing index

import os
path = 'data/crops_256_cond/ffhq/'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, 'renamed', ''.join(['ffhq', str(index), '.jpg'])))
