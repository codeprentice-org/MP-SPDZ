set -e

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

cd TensorflowInf/SqueezeNet
if ! [[ -f "PreTrainedModel/sqz_full.mat" ]]; then
    mkdir -p PreTrainedModel
    ./download-pretrained-model.sh
fi

$PYTHON_CMD process-model.py --in ./SampleImages/n02109961_36.JPEG

cd ../..
Scripts/fixed-rep-to-float.py TensorflowInf/SqueezeNet/SqNetImgNet_img_input.inp

./compile.py -R 64 new-tf TensorflowInf/SqueezeNet/graphDef.bin 1 trunc_pr split

Scripts/ring.sh new-tf-TensorflowInf_SqueezeNet_graphDef.bin-1-trunc_pr-split

set +e
