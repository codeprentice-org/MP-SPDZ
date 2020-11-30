set -e

NUM_THREADS=1
PARENT_DIR=`dirname ${BASH_SOURCE[0]}`

print_usage() {
    echo "Usage: ./tf-inference.sh <Image File> [options]"
    echo " -n	    Number of threads to use. Value defaults to 1"
}

while getopts 'n:' flag; do
    case "${flag}" in
	n) NUM_THREADS="${OPTARG}" ;;
	*) print_usage
	exit 1 ;;
    esac
done

if [ -z "$1" ]; then
    IMG_FILE="SampleImages/n02109961_36.JPEG"
else
    IMG_FILE=$1
fi

echo Checking for Python3...
PYTHON_CMD=${PYTHON_CMD:-`which python3`}
if [ -z "$PYTHON_CMD" ]; then
    echo Error: Python3 is not installed
    echo Please install Python 3.5-3.7 first
    echo Quitting...
    exit 1
else
    echo Python3 is installed
fi

echo Checking for Python version...
if [[ "`$PYTHON_CMD --version`" =~ ^Python[[:space:]]*(3\.[5-7].*)$ ]]; then
    PYTHON_VERSION=${BASH_REMATCH[1]}
    echo Python version=$PYTHON_VERSION
else
    echo Error: Python installation must be version 3.5-3.7
    echo Quitting...
    exit 2
fi

cd ${PARENT_DIR}
if [[ ! -f "PreTrainedModel/sqz_full.mat" ]]; then
    echo "Downloading pretrained SqueezeNet model from https://github.com/avoroshilov/tf-squeezenet"
    mkdir -p PreTrainedModel
    axel -a -n 5 -c --output ./PreTrainedModel https://github.com/avoroshilov/tf-squeezenet/raw/master/sqz_full.mat
fi

python3 process-model.py --in $IMG_FILE

cd ../../
Scripts/fixed-rep-to-float.py TensorflowInf/SqueezeNet/SqNetImgNet_img_input.inp

# TODO: abstract script so that the protocol being used can be changed at runtime
# See https://github.com/data61/MP-SPDZ#tensorflow-inference for details
./compile.py -R 64 tf TensorflowInf/SqueezeNet/graphDef.bin ${NUM_THREADS} trunc_pr split 
Scripts/ring.sh tf-TensorflowInf_SqueezeNet_graphDef.bin-1-trunc_pr-split

set +e
