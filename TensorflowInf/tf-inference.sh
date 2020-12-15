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

MODEL_NETWORK_NO_CAPS="${MODEL_NETWORK,,}"
case "$MODEL_NETWORK_NO_CAPS" in
    squeezenet) PRE_TRAINED_MODEL_LINK="https://github.com/avoroshilov/tf-squeezenet/raw/master/sqz_full.mat"
                PRE_TRAINED_MODEL_FILE="sqz_full.mat"
                SCRIPT="squeezenet_main.py"
                EXTRACT=0;;

    resnet)     PRE_TRAINED_MODEL_LINK="http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz"
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

if [ EXTRACT -eq 1 ] then;
    cd PreTrainedModel
    tar -xvzf $PRE_TRAINED_MODEL_FILE
    cd ..
fi

# python3 $SCRIPT .....
# make both resnet not need scalingfac and saveimgandwtdata
# handle issue with relative image path
# cd ../..
# Scripts/fixed-rep-to-float.py TensorflowInf/$MODEL_NETWORK/inp_thing
# python3 compile.py .....
# Scripts/ring.sh tf-......

set +e
