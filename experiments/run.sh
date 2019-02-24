#!/bin/bash

run_file='../src/main.py'
folder_data='../data'
gpu='0'
use_pretrained=1
if (( $use_pretrained == 0 )); then
    train_list='1 0'
else
    train_list='0'
fi

# Multi-Shapes 20x20
dataset='shapes_20x20'
folder_base=$dataset
if [ ! -d $folder_base ]; then
    mkdir $folder_base
fi
path_data=$folder_data'/'$dataset'_data.h5'
state_size=16
updater_size=32
folder=$folder_base'/'$state_size'_'$updater_size
for train in $train_list; do
    python $run_file \
        --gpu $gpu \
        --dataset $dataset \
        --path_data $path_data \
        --folder $folder \
        --train $train \
        --state_size $state_size \
        --updater_size $updater_size
done

# Multi-Shapes 28x28
dataset='shapes_28x28'
path_data=$folder_data'/'$dataset'_3_data.h5'
for binary_image in 1 0; do
    folder_base=$dataset'_'$binary_image
    if [ ! -d $folder_base ]; then
        mkdir $folder_base
    fi
    for params in '16 16' '16 32' '32 32' '32 64' '64 64'; do
        IFS=' ' params=( $params )
        state_size=${params[0]}
        updater_size=${params[1]}
        folder=$folder_base'/'$state_size'_'$updater_size
        for train in $train_list; do
            python $run_file \
                --gpu $gpu \
                --dataset $dataset \
                --path_data $path_data \
                --folder $folder \
                --train $train \
                --state_size $state_size \
                --updater_size $updater_size \
                --binary_image $binary_image
        done
    done
done

# Multi-MNIST
dataset='mnist'
for folder_base in 'mnist_20' 'mnist_500' 'mnist_all'; do
    if [ ! -d $folder_base ]; then
        mkdir $folder_base
    fi
    path_data=$folder_data'/'$folder_base'_data.h5'
    for params in '16 32' '32 32' '32 64' '64 64'; do
        IFS=' ' params=( $params )
        state_size=${params[0]}
        updater_size=${params[1]}
        folder=$folder_base'/'$state_size'_'$updater_size
        for train in $train_list; do
            python $run_file \
                --gpu $gpu \
                --dataset $dataset \
                --path_data $path_data \
                --folder $folder \
                --train $train \
                --state_size $state_size \
                --updater_size $updater_size
        done
    done
done

# Generalization
dataset='shapes_28x28'
folder_base=$dataset'_1'
train=0
state_size=64
updater_size=64
folder=$folder_base'/'$state_size'_'$updater_size
for num_objects in 2 4; do
    path_data=$folder_data'/'$dataset'_'$num_objects'_data.h5'
    file_result_base='general_'$num_objects'_result_{}.h5'
    python $run_file \
        --gpu $gpu \
        --dataset $dataset \
        --path_data $path_data \
        --folder $folder \
        --train $train \
        --state_size $state_size \
        --updater_size $updater_size \
        --file_result_base $file_result_base \
        --num_objects $num_objects
done
