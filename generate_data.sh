#!/bin/bash
cd data
#生成训练数据
../训练程序/generate_training_samples -i trainlist.txt -o train
mv dataset_nums.dat trainset_nums.dat
#生成测试数据
../训练程序/generate_training_samples -i testlist.txt -o test
mv dataset_nums.dat testset_nums.dat
cd ..
#生成solver
训练程序/generate_solvers --train data/trainset_nums.dat --test data/testset_nums.dat -o solvers -b 64
