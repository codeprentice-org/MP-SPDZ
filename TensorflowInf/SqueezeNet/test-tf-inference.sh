# Test pass count
PASSES=0
# Test failure count
FAILURES=0
# Test count
TESTS=0

# Run Tensorflow inference on image and compare output
test_tf_inference_file() {
    TESTS=$((TESTS+1))
    filename=$1
    compile=$2
    echo Testing MP-SPDZ Tensorflow Inference on $filename...

    # Get image reference code from filename
    if [[ "$filename" =~ ^.*\/(n[[:digit:]]+)_[[:digit:]]+\..*$ ]]; then
        imgCodeExpected=${BASH_REMATCH[1]}
    else
        echo Error: Unable to parse image reference code from file name
        echo Test FAILED for $filename
        FAILURES=$((FAILURES+1))
        return
    fi

    if [ ! -z "$compile" ]; then
        no_compile_option="-c"
    fi

    # Run Tensorflow inference and log output
    ./tf-inference.sh -n 1 -i $filename $no_compile_option &> test.log

    # Get line number of image from log
    grep_output=`grep guess test.log`
    if [[ "$grep_output" =~ ^guess[[:space:]]+([[:digit:]]+)$ ]]; then
        line_number=$(( ${BASH_REMATCH[1]} + 1 ))
    else
        echo Error: Unable to find result from test.log
        echo Test FAILED for $filename
        FAILURES=$((FAILURES+1))
        return
    fi

    # Read and parse text at line number
    lineEntry=`sed "${line_number}q;d" ../synset_words.txt`
    if [[ "$lineEntry" =~ ^[[:space:]]*(n[[:digit:]]+)[[:space:]]+.*$ ]]; then
        imgCodeFound=${BASH_REMATCH[1]}
    else
        echo Error: Unable to look up image reference in ../synset_words.txt
        echo Test FAILED for $filename
        FAILURES=$((FAILURES+1))
        return
    fi

    # Compare found and expected image reference codes
    if [[ $imgCodeFound == $imgCodeExpected ]]; then
        echo Test PASSED for $filename
        PASSES=$((PASSES+1))
    else
        echo Error: Incorrect image reference code found
        echo Expected $imgCodeExpected, found $imgCodeFound
        echo Test FAILED for $filename
        FAILURES=$((FAILURES+1)) 
        return
    fi
}

# Navigate to script directory
SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
cd $SCRIPT_DIR

# Run test on every image in SampleImages
for FILE in SampleImages/*
    do test_tf_inference_file $FILE $no_compile
    no_compile=1
    echo ''
done
# Print test results
echo $TESTS tests executed
echo $PASSES tests PASSED
echo $FAILURES tests FAILED
