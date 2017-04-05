# Tutorial on deeplandmark
### Introduction

This code is an implement of the algorithm introduced in paper [Deep Convolutional Network Cascade for Facial Point Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Sun_Deep_Convolutional_Network_2013_CVPR_paper.pdf) . The network models are borrowed from project [deep-landmark](https://github.com/luoyetx/deep-landmark) .

###Project Structure

训练程序：code for generating lmdb
data：lmdb files
deploy_model：deploy models for facial landmarker
train_model：training models for facial landmarker
solvers：training parameters
model_values：trained model files
src：facial landmarkers

###Building

make -C 训练程序 -j9 && make -j9

###Training

You can skip the training if you just want to detect facial landmarks with this project because all pretrained caffemodel files are given in model_values directory
1. download [Large-scale CelebFaces Attributes](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
uncompress the dataset, and prepare two list files which are compose of lines in the following format

<absolute path to image>  <left>  <right>  <top>  <bottom coordinate of the facial bounding box>  <x of 1st landmark>  <y of 1st landmark>  ...   <x of 5th landmark>  <y of 5th landmark>

one list file for training images(trainlist.txt), and another for testing ones(testlist.txt).

2. generate LMDB files and solver files

in project root directory
```Shell
./generate_data.sh
```
edit solver files as needed

3. train models

in train_model directory
train every model with caffe
for example:
```Shell
caffe train -solver ../solvers/1_EN_solver.prototxt -gpu all
```
the trained model files are located in corresponding directories in model_values directory

4. move trained model files to model_values directory

in 训练程序 directory
```Shell
./move_training_results -i ../model_values
```

###Run

execute landmarker on a computer with a webcam. 

