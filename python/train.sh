PYTHON=python3
BE=tensorflow
DATASET=CIFAR-10
NUM_EPOCH=200
BATCH_SZ=100
L2_REG=0.00001
FP16=0
FMT=NHWC
model=snpx_net

$PYTHON snpx_train_classifier.py --backend $BE --model-name $model --target-data $DATASET --data-format $FMT \
	--num-epoch $NUM_EPOCH --batch-size $BATCH_SZ --use-fp16 $FP16