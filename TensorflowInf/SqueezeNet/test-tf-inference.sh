PASSES=0
FAILURES=0
TESTS=0

function test_tf_inference_file() {
    TESTS=$((TESTS+1))
    filename=$1
    echo Testing MP-SPDZ Tensorflow Inference on $filename...
    
    if [[ "$filename" =~ ^.*\/(n[[:digit:]]+)_[[:digit:]]+\..*$ ]]; then
        imgCodeExpected=${BASH_REMATCH[1]}
    else
        echo Error: Unable to parse image reference code from file name
        echo Test FAILED for $filename
        FAILURES=$((FAILURES+1))
        return
    fi

    ./tf-inference.sh $filename &> test.log

    grep_output=`grep guess test.log`
    if [[ "$grep_output" =~ ^guess[[:space:]]+([[:digit:]]+)$ ]]; then
        line_number=$(( ${BASH_REMATCH[1]} + 1 ))
    else
        echo Error: Unable to find result from test.log
        echo Test FAILED for $filename
        FAILURES=$((FAILURES+1))
        return
    fi

    lineEntry=`sed "${line_number}q;d" ../synset_words.txt`
    if [[ "$lineEntry" =~ ^[[:space:]]*(n[[:digit:]]+)[[:space:]]+.*$ ]]; then
        imgCodeFound=${BASH_REMATCH[1]}
    else
        echo Error: Unable to look up image reference in ../synset_words.txt
        echo Test FAILED for $filename
        FAILURES=$((FAILURES+1))
        return
    fi

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

PARENT_DIR=`dirname ${BASH_SOURCE[0]}`
cd $PARENT_DIR
for FILE in SampleImages/*; do test_tf_inference_file $FILE; echo ''; done
echo $TESTS tests executed
echo $PASSES tests PASSED
echo $FAILURES tests FAILED
