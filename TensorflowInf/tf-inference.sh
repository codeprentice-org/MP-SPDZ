set -e

verify_python() {
    echo -e Checking for Python3...
    PYTHON_CMD=${PYTHON_CMD:-`which python3`}
    if [ -z "$PYTHON_CMD" ]; then
        echo -e Error: Python3 is not installed
        echo -e Please install Python 3 first
        echo -e Quitting...\n
        exit 1
    else
        echo -e Python3 is installed\n
    fi

    echo Checking for Python version 3.$1 - 3.$2...
    if [[ "`$PYTHON_CMD --version`" =~ ^Python[[:space:]]*(3\.[$1-$2].*)$ ]]; then
        PYTHON_VERSION=${BASH_REMATCH[1]}
        echo -e Python version=$PYTHON_VERSION\n
    else
        echo -e Error: Python installation must be version 3.$1-3.$2
        echo -e Quitting...\n
        exit 2
    fi
}

print_usage() {
    echo -e "\nUsage:\n\t$0 [-m <MODEL_NETWORK>] [-i <INPUT_IMAGE_FILE>] [-n <NUMBER_OF_THREADS>]\n"
}

NUM_THREADS=1
while getopts "m:n:i:h" flag; do
    case "$flag" in
        m)  MODEL_NETWORK=$OPTARG;;
        i)  IMG_FILE=$OPTARG;;
        n)  NUM_THREADS=$OPTARG;;
        h)  print_usage
            exit 0;;
        *)  print_usage
            exit 4;;
    esac
done

verify_python 5 7

MODEL_NETWORK=$(echo $MODEL_NETWORK | sed 's:/*$::')
MODEL_NETWORK=${MODEL_NETWORK,,}
case "$MODEL_NETWORK" in
    squeezenet) MODEL_NETWORK=SqueezeNet
                PRE_TRAINED_MODEL_LINK="https://github.com/avoroshilov/tf-squeezenet/raw/master/sqz_full.mat"
                PRE_TRAINED_MODEL_FILE="sqz_full.mat"
                EXTRACT=0;;

    resnet)     MODEL_NETWORK=ResNet
                PRE_TRAINED_MODEL_LINK="http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz"
                PRE_TRAINED_MODEL_FILE="resnet_v2_fp32_savedmodel_NHWC.tar.gz"
                EXTRACT=1;;

    *)          echo -e Error: Please select a valid model network\n
                exit 6;;
esac

if [ -z "$IMG_FILE" ]; then
    echo -e Error: Please select an input image\n
    exit 5
fi

if [ ${IMG_FILE:0:1} == "/" ]; then
    IMG_FILE_ABS_PATH="$IMG_FILE"
else
    IMG_FILE_ABS_PATH="${PWD}/${IMG_FILE}"
fi

SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
cd $SCRIPT_DIR

cd $MODEL_NETWORK
if [[ ! -f "PreTrainedModel/$PRE_TRAINED_MODEL_FILE" ]]; then
    echo -e "Downloading pretrained $MODEL_NETWORK model from $PRE_TRAINED_MODEL_LINK"
    mkdir -p PreTrainedModel
    axel -a -n 5 -c --output ./PreTrainedModel $PRE_TRAINED_MODEL_LINK
fi

if [ $EXTRACT -eq 1 ]; then
    cd PreTrainedModel
    tar -xvzf $PRE_TRAINED_MODEL_FILE
    cd ..
fi

python3 $MODEL_NETWORK.py --img $IMG_FILE_ABS_PATH

cd ../..
Scripts/fixed-rep-to-float.py TensorflowInf/${MODEL_NETWORK}/${MODEL_NETWORK}_img_input.inp
python3 compile.py -R 64 tf TensorflowInf/${MODEL_NETWORK}/graphDef.bin ${NUM_THREADS} trunc_pr split 
Scripts/ring.sh tf-TensorflowInf_${MODEL_NETWORK}_graphDef.bin-${NUM_THREADS}-trunc_pr-split

set +e
