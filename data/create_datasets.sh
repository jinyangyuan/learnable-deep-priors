#!/bin/bash

folder_downloads='downloads'
folder_src='src'
if [ ! -d $folder_downloads ]; then
    mkdir $folder_downloads
fi

# Download the Shapes Dataset used in "Tagger: Deep Unsupervised Perceptual Grouping"
# The url is described in https://github.com/CuriousAI/tagger/blob/master/install.sh
url_shapes_20='http://cdn.cai.fi/datasets/shapes50k_20x20_compressed_v2.h5'
file_shapes_20='shapes_20x20.h5'
if [ ! -f $folder_downloads/$file_shapes_20 ]; then
    wget $url_shapes_20 -O $folder_downloads/$file_shapes_20
fi

# Download the Static Shapes Dataset used in "Neural Expectation Maximization"
# The Dropbox url is described in https://github.com/sjoerdvansteenkiste/Neural-EM/blob/master/README.md
url_shapes_28='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AABZBL6D1KrCF8CPe-an5psoa/shapes.h5?dl=1'
file_shapes_28='shapes_28x28.h5'
if [ ! -f $folder_downloads/$file_shapes_28 ]; then
    wget $url_shapes_28 -O $folder_downloads/$file_shapes_28
fi

# Download the MNIST Dataset
# This python script is the modified version of https://github.com/IDSIA/brainstorm/blob/master/data/create_mnist.py
python $folder_src/download_mnist.py --folder_downloads $folder_downloads

# Convert the Shapes Dataset used in "Tagger: Deep Unsupervised Perceptual Grouping"
python $folder_src/convert_multi_shapes_20x20.py --folder_downloads $folder_downloads

# Convert the Static Shapes Dataset used in "Neural Expectation Maximization"
python $folder_src/convert_multi_shapes_28x28.py --folder_downloads $folder_downloads

# Create the Multi Shapes Dataset derived from "Binding via Reconstruction Clustering"
# This python script is the modified version of https://github.com/Qwlouse/Binding/blob/master/Datasets/Shapes.ipynb
for num_objects in 2 4; do
    python $folder_src/create_multi_shapes_28x28.py --num_objects $num_objects
done

# Create the Multi MNIST Dataset derived from "Binding via Reconstruction Clustering"
# This python script is the modified version of https://github.com/Qwlouse/Binding/blob/master/Datasets/Multi-MNIST.ipynb
python $folder_src/create_multi_mnist.py --folder_downloads $folder_downloads
