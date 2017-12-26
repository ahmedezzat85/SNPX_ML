@echo off

set PYTHON=python3
set BE=tensorflow
set DATASET=CIFAR-10
set NUM_EPOCH=200
set BATCH_SZ=128
set FP16=0
set FMT=NCHW
set model=resnet
set LR=0.1
set DATA_AUG=1
set MODEL_LOG_DIR=resnet-tf-aug
set OPT=sgd
set L2_REG=0
set EPOCH=0

%PYTHON% snpx_train_classifier.py ^
		--backend %BE% ^
		--model-name %model%  ^
		--target-dataset %DATASET% ^
		--data-format %FMT% ^
		--num-epoch %NUM_EPOCH% ^
		--batch-size %BATCH_SZ% ^
		--use-fp16 %FP16%       ^
		--optimizer %OPT% ^
		--lr %LR%  ^
		--l2-reg %L2_REG% ^
		--logs-subdir %MODEL_LOG_DIR% ^
		--data-aug %DATA_AUG% ^
		--begin-epoch %EPOCH%