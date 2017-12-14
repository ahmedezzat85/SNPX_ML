@echo off

set PYTHON=python3
set BE=tensorflow
set DATASET=CIFAR-10
set NUM_EPOCH=200
set BATCH_SZ=100
set L2_REG=0.00001
set FP16=0
set FMT=NCHW
set model=mobile_net


%PYTHON% snpx_train_classifier.py ^
		--backend %BE% ^
		--model-name %model%  ^
		--target-dataset %DATASET% ^
		--data-format %FMT% ^
		--num-epoch %NUM_EPOCH% ^
		--batch-size %BATCH_SZ% ^
		--use-fp16 %FP16%