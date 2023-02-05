# python detect_pipe.py --images_dir ../../data/v1_0_2/images/ --save_dir ../../data/crops_256 --crop_size 256

# python projector.py --outdir=../training/ --target=../../data/healthy/manuel.png --network=../training/00012-crops_256_cdl-mirror-auto2-kimg3000-resumeffhq256/network-snapshot-002900.pkl

# create efficient dataset from source folder
# python dataset_tool.py --source ../../data/crops_256_cond.zip --dest ../../data/cond.zip

# python run_training.py --data-dir=../../data/datasets --dataset=crops128 --config=config-f --total-kimg 1 --resume-pkl=./stylegan2-ffhq-config-f.pkl

