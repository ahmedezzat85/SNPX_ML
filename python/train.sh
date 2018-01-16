PYTHON=python3
BE=tensorflow
DATASET=CIFAR-10
NUM_EPOCH=5
BATCH_SZ=128
FP16=0
FMT=NCHW
model=resnet-2
MODEL_LOG_DIR=adam-001
LR=0.001
L2_REG=0.0001
DATA_AUG=1
OPT=adam
EPOCH=0

$PYTHON snpx_train_classifier.py --backend $BE --model-name $model --target-data $DATASET \
	--data-format $FMT --num-epoch $NUM_EPOCH --batch-size $BATCH_SZ --use-fp16 $FP16     \
	--optimizer $OPT --lr $LR --l2-reg $L2_REG --logs-subdir $MODEL_LOG_DIR --data-aug $DATA_AUG \
	--begin-epoch $EPOCH