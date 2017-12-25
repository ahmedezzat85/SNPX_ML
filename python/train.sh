PYTHON=python2
BE=mxnet
DATASET=CIFAR-10
NUM_EPOCH=100
BATCH_SZ=100
FP16=0
FMT=NCHW
model=resnet
LR=0.01
DATA_AUG=1
MODEL_LOG_DIR=resnet-sgd-opt
OPT=sgd
L2_REG=0.0001
EPOCH=8

$PYTHON snpx_train_classifier.py --backend $BE --model-name $model --target-data $DATASET \
	--data-format $FMT --num-epoch $NUM_EPOCH --batch-size $BATCH_SZ --use-fp16 $FP16     \
	--optimizer $OPT --lr $LR --l2-reg $L2_REG --logs-subdir $MODEL_LOG_DIR --data-aug $DATA_AUG \
	--begin-epoch $EPOCH