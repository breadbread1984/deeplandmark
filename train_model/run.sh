#!/bin/bash
screen -dmS landmarker
id = 0
screen -S landmarker -p ${id} -X "htop"
for f in *_train.prototxt
do
id = ${id} + 1
screen -S landmarker -X screen ${id}
screen -S landmarker -p ${id} -X "caffe train -solver=../solvers/`basename "$f" _train.prototxt`_solver.prototxt -gpu all"
done
