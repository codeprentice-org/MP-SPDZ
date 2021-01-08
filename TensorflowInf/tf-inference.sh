set -e

# Print usage
# TODO: Also show valid options
print_usage() {
    echo -e "\nUsage:\n\t$0 [-m <MODEL_NETWORK>] [-i <INPUT_IMAGE_FILE>] [-n <NUMBER_OF_THREADS>]\n"
}

NUM_THREADS=1
# Parse options and arguments
while getopts "m:n:i:h" flag; do
    case "$flag" in
        m)  MODEL_NETWORK=$OPTARG;;
        i)  IMG_FILE=$OPTARG;;
        n)  NUM_THREADS=$OPTARG;;
        h)  print_usage
            exit 0;;
        *)  print_usage
            exit 1;;
    esac
done

# Categorize selected model network
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

    *)          echo -e "Error: Please select a model network from the following list: squeezenet, resnet\n"
                exit 1;;
esac

# Check for empty image file path
if [ -z "$IMG_FILE" ]; then
    echo -e "Error: Please select an input image\n"
    exit 1
fi

# Get absolute path to image file
if [ ${IMG_FILE:0:1} == "/" ]; then
    IMG_FILE_ABS_PATH="$IMG_FILE"
else
    IMG_FILE_ABS_PATH="${PWD}/${IMG_FILE}"
fi

# Navigate to script directory
SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
cd $SCRIPT_DIR

# Download Pre-Trained Model
cd $MODEL_NETWORK
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
Scripts/fixed-rep-to-float.py TensorflowInf/${MODEL_NETWORK}/${MODEL_NETWORK}_img_input.inp
python3 compile.py -R 64 tf TensorflowInf/${MODEL_NETWORK}/graphDef.bin ${NUM_THREADS} trunc_pr split
Scripts/ring.sh tf-TensorflowInf_${MODEL_NETWORK}_graphDef.bin-${NUM_THREADS}-trunc_pr-split

set +e
