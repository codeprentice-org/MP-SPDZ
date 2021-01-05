# Set script to exit on error
set -e

# Set default number of threads to use
NUM_THREADS=1
# Set default image file
IMG_FILE="SampleImages/n02109961_36.JPEG"

# Print script usage
print_usage() {
    echo "Usage: ./tf-inference.sh <Image File> [options]"
    echo " -n	    Number of threads to use. Value defaults to 1"
}

# Parse options
while getopts 'n:F:' flag; do
    case "${flag}" in
        n) NUM_THREADS=${OPTARG};;
        F) IMG_FILE=${OPTARG};;
        *) print_usage
        exit 1 ;;
    esac
done

.././requirements.sh || exit 1

# Navigate to script directory
SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
cd ${SCRIPT_DIR}

if [[ ! -f "PreTrainedModel/resnet_v2_fp32_savedmodel_NHWC.tar.gz" ]]; then
    echo "Downloading pretrained ResNet model from http://download.tensorflow.org"
    mkdir -p PreTrainedModel
    axel -a -n 5 -c --output ./PreTrainedModel http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz
fi

# Extract pretrained model
cd PreTrainedModel
tar -xvzf resnet_v2_fp32_savedmodel_NHWC.tar.gz

# Run ResNet script
cd ..
python3 ResNet.py --img $IMG_FILE

# Run Mp-SPDZ compilation
cd ../..
./Scripts/fixed-rep-to-float.py TensorflowInf/ResNet/ResNet_img_input.inp
python3 compile.py -R 64 tf TensorflowInf/ResNet/graphDef.bin ${NUM_THREADS} trunc_pr split
./Scripts/ring.sh tf-TensorflowInf_ResNet_graphDef.bin-${NUM_THREADS}-trunc_pr-split

set +e
