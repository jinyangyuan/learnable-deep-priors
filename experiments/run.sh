#!/bin/bash

function func_basic {
    dataset=$1
    path_config=$2
    run_file=$3
    folder_data=$4

    path_data=$folder_data'/'$dataset'_data.h5'
    folder=$dataset
    for train in 1 0; do
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder $folder \
            --train $train
    done
}

function func_general {
    dataset=$1
    path_config=$2
    run_file=$3
    folder_data=$4

    path_data=$folder_data'/'$dataset'_3_data.h5'
    folder=$dataset
    for train in 1 0; do
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder $folder \
            --train $train
    done
    for max_objects in 2 4; do
        path_data=$folder_data'/'$dataset'_'$max_objects'_data.h5'
        file_result_base='general_'$max_objects'_result_{}.h5'
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder $folder \
            --file_result_base $file_result_base \
            --train 0 \
            --max_objects $max_objects
    done
}

export CUDA_VISIBLE_DEVICES='0'
run_file='../src/main.py'
folder_data='../data'

path_config='config_shapes.yaml'
func_general 'shapes_28x28' $path_config $run_file $folder_data
func_basic 'shapes_20x20' $path_config $run_file $folder_data

path_config='config_mnist.yaml'
for dataset in 'mnist_20' 'mnist_500' 'mnist_all'; do
    func_basic $dataset $path_config $run_file $folder_data
done
