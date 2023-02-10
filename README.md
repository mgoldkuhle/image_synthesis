# image_synthesis
Generating portrait images with genetically dysmorphic features and evaluate them with GestaltMatcher

StyleGAN2-ADA repository: https://github.com/NVlabs/stylegan2-ada-pytorch

see stylegan2_prerequisites.txt on the prerequisites to run stylegan2-ada-pytorch on a local Windows machine or a linux server

- generate_images.py mass generates samples for multiple model snapshots at once in an ordered fashion

## GestaltMatcher
The slightly modified GestaltMatcher repository is used for feature embeddings and evaluation of the synthetic images for augmentation purposes

Original repository: https://github.com/igsb/GestaltMatcher

I have mainly added functions for easier data loading and evaluation metrics for classification with few classes

## data wrangling

- syndrome_filter.py extracts images from a directory that belong to a certain syndrome as indicated by a metadata file

- patient_split.py splits data patient-wise into k cross-validation folds
 
- cross_val_dirs.py to copy split patient .jpg images created by patient_split.py in a directory structure to be used in stylegan2 training

- rename_files.py renames files in a directory with an ascending trailing index

- make_json_labels.py automatically creates the .json label file needed for conditional training of styleGAN2-ada


## evaluation

- embeddings_similarity.py for pairwise similarity between feature embeddings of two image datasets

- test_set_evaluation.py takes gestaltmatcher predictions and evaluates confusion matrices and metrics

- tsne_plots.py creates 2d visualizations for data embedded in the gestaltmatcher feature space to compare the distributions of real and synthetic data

- image_comparison.py to highlights the pixel-wise differences between two images (not used in the thesis)

- Scripts in the R folder for data exploration, styleGAN training run summaries and statistical testing

    - data_summary.R for a GMDB data overview
    - runs.R to evaluate trained StyleGAN2 models
    - styleGAN_training_progress.R for KID per Epoch in StyleGAN2-ADA training for multiple models
    - evaluation_curves.R to compare training and validation curves of GestaltMatcher
    - gestaltmatcher_comparison.R to compare test set metrics for different setups
    - compare_means.R to create exemplary plots for linear models