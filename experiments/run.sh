#!/bin/bash

function func_basic {
    name=$1
    path_data=$folder_data'/'$name'.h5'
    folder_log='logs/'$name
    folder_out=$name
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
}

function func_general {
    name=$1
    path_data=$folder_data'/'$name'_3.h5'
    folder_log='logs/'$name
    folder_out=$name
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
    for obj_slots in 2 4; do
        path_data=$folder_data'/'$name'_'$obj_slots'.h5'
        file_result='general_'$obj_slots'.h5'
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder_log $folder_log \
            --folder_out $folder_out \
            --num_slots $(( obj_slots + 1 )) \
            --file_result $file_result
    done
}

export CUDA_VISIBLE_DEVICES='0'
run_file='../src/main.py'
folder_data='../data'

path_config='config_shapes.yaml'
func_general 'shapes_28x28'
func_basic 'shapes_20x20'

path_config='config_mnist.yaml'
for name in 'mnist_20' 'mnist_500' 'mnist_all'; do
    func_basic $name
done
