# ----------------------------------------------------------------------
# Configurations

NUM_THREADS=1
IMG_FILE="SampleImages/n02109961_36.JPEG"
PRE_TRAINED_MODEL_LINK="http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz"
EXTRACT=1
MODEL_NETWORK="ResNet"
COMPILE=1

# ----------------------------------------------------------------------

set -e

# Print usage
print_usage() {
    echo -e "\nUsage:\n\t$0 [-i <INPUT_IMAGE_FILE>] [-n <NUMBER_OF_THREADS>] [-c]\n"
}

# Parse options and arguments
while getopts "n:i:ch" flag; do
    case "$flag" in
        i)  IMG_FILE=$OPTARG;;
        n)  NUM_THREADS=$OPTARG;;
        c)  COMPILE=0;;
        h)  print_usage
            exit 0;;
        *)  print_usage
            exit 1;;
    esac
done

# Get absolute path to image file
if [ ${IMG_FILE:0:1} == "/" ]; then
    IMG_FILE_ABS_PATH="$IMG_FILE"
else
    IMG_FILE_ABS_PATH="${PWD}/${IMG_FILE}"
fi

# Navigate to script directory
SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
cd $SCRIPT_DIR

# Get Pre-Trained Model Filename
PRE_TRAINED_MODEL_LINK=${PRE_TRAINED_MODEL_LINK%/}
PRE_TRAINED_MODEL_FILE=`echo $PRE_TRAINED_MODEL_LINK | grep -oP "([^/]+$)"`

# Download Pre-Trained Model
if [[ ! -f "PreTrainedModel/$PRE_TRAINED_MODEL_FILE" ]]; then
    echo -e "Downloading pretrained $MODEL_NETWORK model from $PRE_TRAINED_MODEL_LINK"
    mkdir -p PreTrainedModel && cd PreTrainedModel
    curl -L -o $PRE_TRAINED_MODEL_FILE $PRE_TRAINED_MODEL_LINK
    cd ..
fi

if [ $EXTRACT -eq 1 ]; then
    cd PreTrainedModel
    tar -xvzf $PRE_TRAINED_MODEL_FILE
    cd ..
fi

python3 $MODEL_NETWORK.py --img $IMG_FILE_ABS_PATH

cd ../..
if [ $COMPILE -eq 1 ]; then
    python3 compile.py -C -R 64 tf TensorflowInf/${MODEL_NETWORK}/graphDef.bin ${NUM_THREADS} trunc_pr split
fi

cp TensorflowInf/${MODEL_NETWORK}/${MODEL_NETWORK}_img_input.inp Player-Data/Input-P0-0
Scripts/emulate.sh tf-TensorflowInf_${MODEL_NETWORK}_graphDef.bin-${NUM_THREADS}-trunc_pr-split
