# reference: https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

import os
from skimage.metrics import structural_similarity
import cv2
import numpy as np

# image path
image_path = 'D:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/CdL_1000kimg/'
target_path = os.path.join(image_path, 'target2.png')
embedding_path = os.path.join(image_path, 'tzung_presentation.png')

# Load images
target = cv2.imread(target_path)
embedding = cv2.imread(embedding_path)

# Convert images to grayscale
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
embedding_gray = cv2.cvtColor(embedding, cv2.COLOR_BGR2GRAY)

# Compute SSIM between the two images
(score, diff) = structural_similarity(target_gray, embedding_gray, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] target we can use it with OpenCV
diff = (diff * 255).astype("uint8")
diff_box = cv2.merge([diff, diff, diff])

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(target.shape, dtype='uint8')
filled_embedding = embedding.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(target, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(embedding, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (255,255,255), -1)
        cv2.drawContours(filled_embedding, [c], 0, (0,255,0), -1)

cv2.imshow('target', target)
cv2.imshow('embedding', embedding)
cv2.imshow('diff', diff)
cv2.imshow('diff_box', diff_box)
cv2.imshow('mask', mask)
cv2.imshow('filled embedding', filled_embedding)
cv2.waitKey()
