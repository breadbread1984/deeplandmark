#!/bin/bash
screen -dmS landmarker
screen -S landmarker -p 0 -X exec htop
cd train_model
for f in *.prototxt
do
	screen -S landmarker -X screen caffe train -solver="../solvers/`basename $f _train.prototxt`_solver.prototxt" -gpu all
done

