set -e

# Check for Python 3 version
function python_check() {
    echo -e "Checking for Python3..."

    PYTHON_CMD=${PYTHON_CMD:-`which python3`}
    if [ -z "$PYTHON_CMD" ]; then
        echo -e "Error: Python 3 is not installed"
        echo -e "Please install Python 3 first"
        echo -e "Quitting...\n"
        exit 1
    else
        echo -e "Python 3 is installed\n"
    fi

    echo "Checking for Python version 3.$1 - 3.$2..."

    if [[ "`$PYTHON_CMD --version`" =~ ^Python[[:space:]]*(3\.[$1-$2].*)$ ]]; then
        PYTHON_VERSION=${BASH_REMATCH[1]}
        echo -e "Python version=$PYTHON_VERSION\n"
    else
        echo -e "Error: Python installation must be version 3.$1-3.$2"
        echo -e "Quitting...\n"
        exit 2
    fi
}

# Check for pip3
function pip_check() {
    echo -e "Checking for pip3..."

    PIP_CMD=`which pip3`
    if [ -z "$PIP_CMD" ]; then
        echo -e "pip3 is not installed"
        echo -e "Would you like to install pip? [y/n]"
        read input
        if [[ ${input,,} == "y" ]]; then
            sudo apt-get install -y python3-pip
        else
            echo -e "Error: pip3 is not installed"
            echo -e "Quitting...\n"
            exit 1
        fi
    else
        PIP_VERSION=`$PIP_CMD --version`
        echo -e "pip3 version=$PIP_VERSION\n"
    fi
}

# Check for Python library
function pip_check_library() {
    name=$1
    if [ ! -z $2 ]; then
        install_name=$2
    else
        install_name=$name
    fi

    echo -e "Checking for $name library..."

    version=`${PYTHON_CMD} -c "import ${name}; print(${name}.__version__)"`
    if [ ! -z $version ]; then
        echo -e "$name version $version\n"
    else
        echo -e "$name is not installed"
        echo -e "Would you like to install $name? [y/n]"
        read input
        if [[ ${input,,} == "y" ]]; then
            $PIP_CMD install $install_name
        else
            echo -e "Error: $name is not installed"
            echo -e "Quitting...\n"
            exit 1
        fi
    fi
}

# Check for Python library version
function pip_check_library_version() {
    name=$1
    version_required=$2
    if [ ! -z $3 ]; then
        install_name=$3
    else
        install_name=$name
    fi

    echo -e "Checking for $name library version $version_required..."

    version=`${PYTHON_CMD} -c "import ${name}; print(${name}.__version__)"`
    if [[ $version == $version_required ]]; then
        echo -e "$name version $version\n"
    elif [ ! -z $version ]; then
        echo -e "$name version $version"
        echo -e "require version $version_required"
        echo -e "Would you like to uninstall the current version and install $name version $version_required? [y/n]"
        read input
        if [[ ${input,,} == "y" ]]; then
            $PIP_CMD uninstall -y $install_name
            $PIP_CMD install $install_name==$version_required
        else
            echo -e "Quitting...\n"
            exit 1
        fi
    else
        echo -e "$name is not installed"
        echo -e "Would you like to install $name? [y/n]"
        read input
        if [[ ${input,,} == "y" ]]; then
            $PIP_CMD install $install_name==$version_required
        else
            echo -e "Quitting...\n"
            exit 2
        fi
    fi
}

# Print usage
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

# Check for Python version 3.5 - 3.7
python_check 5 7
# Check for pip3
pip_check
# Check for SciPy
pip_check_library scipy
# Check for Pillow
pip_check_library PIL pillow
# Check for Tensorflow 1.14.0
pip_check_library_version tensorflow 1.14.0
# Check for numpy 1.16.4
pip_check_library_version numpy 1.16.4

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

    *)          echo -e Error: Please select a valid model network\n
                exit 2;;
esac

# Check for empty image file path
if [ -z "$IMG_FILE" ]; then
    echo -e "Error: Please select an input image\n"
    exit 3
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
    mkdir -p PreTrainedModel
    cd PreTrainedModel
    curl -L -o $PRE_TRAINED_MODEL_FILE $PRE_TRAINED_MODEL_LINK
    cd ..
fi

if [ $EXTRACT -eq 1 ]; then
    cd PreTrainedModel
    tar -xvzf $PRE_TRAINED_MODEL_FILE
    cd ..
fi

$PYTHON_CMD $MODEL_NETWORK.py --img $IMG_FILE_ABS_PATH

cd ../..
Scripts/fixed-rep-to-float.py TensorflowInf/${MODEL_NETWORK}/${MODEL_NETWORK}_img_input.inp
$PYTHON_CMD compile.py -R 64 tf TensorflowInf/${MODEL_NETWORK}/graphDef.bin ${NUM_THREADS} trunc_pr split 
Scripts/ring.sh tf-TensorflowInf_${MODEL_NETWORK}_graphDef.bin-${NUM_THREADS}-trunc_pr-split

set +e
