@echo off

set PYTHON=python3
set BE=tensorflow
set DATASET=CIFAR-10
set NUM_EPOCH=1
set BATCH_SZ=256
set FP16=0
set FMT=NCHW
set model=resnet-2
set MODEL_LOG_DIR=adam-001
set LR=0.001
set L2_REG=0.0001
set DATA_AUG=1
set OPT=adam
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