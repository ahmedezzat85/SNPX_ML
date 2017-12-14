@echo off

set PYTHON=python3
set BE=tensorflow
set DATASET=CIFAR-10
set BATCH_SZ=200
set model=mini_vgg_bn


%PYTHON% snpx_test_classifier.py ^
		--backend %BE% ^
		--model-name %model%  ^
		--target-dataset %DATASET% ^
		--batch-size %BATCH_SZ%