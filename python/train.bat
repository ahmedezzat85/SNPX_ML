@echo off

set PYTHON=python
set BE=mxnet
set DATASET=CIFAR-10
set NUM_EPOCH=200
set BATCH_SZ=200
set FP16=0
set FMT=NCHW
set model=resnet-18
set LR=0.001
set DATA_AUG=1

%PYTHON% snpx_train_classifier.py ^
		--backend %BE% ^
		--model-name %model%  ^
		--target-dataset %DATASET% ^
		--data-format %FMT% ^
		--num-epoch %NUM_EPOCH% ^
		--batch-size %BATCH_SZ% ^
		--use-fp16 %FP16%       ^
		--lr %LR%  ^
		--data-aug %DATA_AUG%