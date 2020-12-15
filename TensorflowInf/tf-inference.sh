set -e

MODEL_NETWORK="SqueezeNet"
NUM_THREADS=1
IMG_FILE="SqueezeNet/SampleImages/n02109961_36.JPEG"

print_usage() {
    echo usage
}

while getopts "m:n:i:" flag; do
    case "$flag" in
        m)  MODEL_NETWORK=$OPTARG;;
        n)  NUM_THREADS=$OPTARG;;
        i)  IMG_FILE=$OPTARG;;
        *)  print_usage
            exit 1;;
    esac
done

./requirements.sh || exit 1

if [ ${IMG_FILE:0:1} == "/" ]; then
    IMG_FILE_ABS_PATH="$IMG_FILE"
else
    IMG_FILE_ABS_PATH="${PWD}/${IMG_FILE}"
fi

case "$MODEL_NETWORK" in
    SqueezeNet) PRE_TRAINED_MODEL_LINK="https://github.com/avoroshilov/tf-squeezenet/raw/master/sqz_full.mat"
                PRE_TRAINED_MODEL_FILE="sqz_full.mat"
                SCRIPT="squeezenet_main.py"
                EXTRACT=0;;

    ResNet)     PRE_TRAINED_MODEL_LINK="http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz"
                PRE_TRAINED_MODEL_FILE="resnet_v2_fp32_savedmodel_NHWC.tar.gz"
                SCRIPT="ResNet_main.py"
                EXTRACT=1;;

    *)          print_usage
                exit 1;;
esac

SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
cd $SCRIPT_DIR

cd $MODEL_NETWORK
if [[ ! -f "PreTrainedModel/$PRE_TRAINED_MODEL_FILE" ]]; then
    echo "Downloading pretrained $MODEL_NETWORK model from $PRE_TRAINED_MODEL_LINK"
    mkdir -p PreTrainedModel
    axel -a -n 5 -c --output ./PreTrainedModel $PRE_TRAINED_MODEL_LINK
fi

if [ $EXTRACT -eq 1 ]; then
    cd PreTrainedModel
    tar -xvzf $PRE_TRAINED_MODEL_FILE
    cd ..
fi

python3 $SCRIPT --img $IMG_FILE_ABS_PATH

cd ../..
Scripts/fixed-rep-to-float.py TensorflowInf/${MODEL_NETWORK}/${MODEL_NETWORK}_img_input.inp
python3 compile.py -R 64 tf TensorflowInf/${MODEL_NETWORK}/graphDef.bin ${NUM_THREADS} trunc_pr split 
Scripts/ring.sh tf-TensorflowInf_${MODEL_NETWORK}_graphDef.bin-${NUM_THREADS}-trunc_pr-split

set +e
