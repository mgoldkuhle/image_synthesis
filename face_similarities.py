# reference: https://github.com/serengil/deepface

import argparse
import pandas as pd
from deepface import DeepFace
from numpy import dot
from numpy.linalg import norm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path1', default='./data/dir1', help='Path for the images to be compared to. Can be directory of images or single image file')
parser.add_argument('--path2', default='./data/dir2', help='Directory for the images to compare. Can be directory of images or single image file')
parser.add_argument('--out', default='./output', help='Directory for the output files')
parser.add_argument('--model', default='ArcFace', help='Model for the face embedding, see https://github.com/serengil/deepface')
args = parser.parse_args()


def cosine_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


if __name__ == '__main__':
    image_path1 = args.path1  # "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/CdL_256_paper256_2000kimg/samples/seed0627.png"
    image_path2 = args.path2  # "C:/Users/Manu/ownCloud/IGSB/thesis/data/crops_256_cdl/4904.jpg"
    model_name = args.model
    out_path = args.out

    if os.path.isfile(image_path1) and os.path.isfile(image_path2):
        embedding = DeepFace.represent(image_path1, model_name, enforce_detection=False)
        embedding2 = DeepFace.represent(image_path2, model_name, enforce_detection=False)

        similarity = cosine_similarity(embedding, embedding2)
        print(f'The cosine similarity between both images is {round(similarity, 4)}')

    elif os.path.isdir(image_path1) and os.path.isfile(image_path2):

        embedding2 = DeepFace.represent(image_path2, model_name, enforce_detection=False)

        similarities = {'file': [], 'similarity': []}

        for file_name in os.listdir(image_path1):
            f = os.path.join(image_path1, file_name)
            embedding = DeepFace.represent(f, model_name, enforce_detection=False)
            sim = cosine_similarity(embedding2, embedding)
            similarities['file'].append(file_name)
            similarities['similarity'].append(sim)

        similarities = pd.DataFrame(similarities)
        similarities.sort_values(by='similarity', ascending=False, inplace=True)
        print(similarities)
        similarities.to_csv(os.path.join(out_path, 'similarities.csv'))

    elif os.path.isdir(image_path1) and os.path.isdir(image_path2):
        similarities = {'file1': [], 'file2': [], 'similarity': []}

        embeddings1 = []
        embeddings2 = []

        for file_name1 in os.listdir(image_path1):
            f1 = os.path.join(image_path1, file_name1)
            print(f'Embedding path1/{file_name1}')
            embeddings1.append(DeepFace.represent(f1, model_name, enforce_detection=False))

        for file_name2 in os.listdir(image_path2):
            f2 = os.path.join(image_path2, file_name2)
            print(f'Embedding path2/{file_name2}')
            embeddings2.append(DeepFace.represent(f2, model_name, enforce_detection=False))

        for ix1, file_name1 in enumerate(os.listdir(image_path1)):
            for ix2, file_name2 in enumerate(os.listdir(image_path2)):

                sim = cosine_similarity(embeddings1[ix1], embeddings2[ix2])
                similarities['file1'].append(file_name1)
                similarities['file2'].append(file_name2)
                similarities['similarity'].append(sim)

        similarities = pd.DataFrame(similarities)
        similarities.sort_values(by='similarity', ascending=False, inplace=True)
        print(similarities)
        similarities.to_csv(os.path.join(out_path, 'similarities.csv'))

    else:

        print('Check paths')
