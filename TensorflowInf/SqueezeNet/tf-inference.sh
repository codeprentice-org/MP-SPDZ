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

# Download pre-trained model
if [[ ! -f "PreTrainedModel/sqz_full.mat" ]]; then
    echo "Downloading pretrained SqueezeNet model from https://github.com/avoroshilov/tf-squeezenet"
    mkdir -p PreTrainedModel
    axel -a -n 5 -c --output ./PreTrainedModel https://github.com/avoroshilov/tf-squeezenet/raw/master/sqz_full.mat
fi

# Run SqueezeNet script
python3 SqueezeNet.py --img $IMG_FILE

# Run MP-SPDZ compilation
cd ../../
Scripts/fixed-rep-to-float.py TensorflowInf/SqueezeNet/SqueezeNet_img_input.inp

# TODO: abstract script so that the protocol being used can be changed at runtime
# See https://github.com/data61/MP-SPDZ#tensorflow-inference for details
python3 compile.py -R 64 tf TensorflowInf/SqueezeNet/graphDef.bin ${NUM_THREADS} trunc_pr split 
Scripts/ring.sh tf-TensorflowInf_SqueezeNet_graphDef.bin-${NUM_THREADS}-trunc_pr-split

set +e
