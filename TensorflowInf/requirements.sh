# Check for Python 3
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

# Check Python version
echo Checking Python version...
if [[ "`$PYTHON_CMD --version`" =~ ^Python[[:space:]]*(3\.[5-7].*)$ ]]; then
    PYTHON_VERSION=${BASH_REMATCH[1]}
    echo Python version=$PYTHON_VERSION
else
    echo Error: Python installation must be version 3.5-3.7
    echo Quitting...
    exit 2
fi
