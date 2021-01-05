SCRIPT_DIR=`dirname ${BASH_SOURCE[0]}`
cd $SCRIPT_DIR
IMG_CODE="n02109961"
IMG_DOWNLOADS_DIR="./"
IMG_NET_URL="http://59.36.11.51/dataset/workspace/mindspore_dataset/imagenet/imagenet_original/train/"
SYNSET_FILE="synset_words.txt"

function downloadFromDirectory() {
    mkdir -p $IMG_DOWNLOADS_DIR
    cd $IMG_DOWNLOADS_DIR

    DIR_NAME=$1
    mkdir -p $DIR_NAME
    cd $DIR_NAME

    PAGE_SRC=`wget -O - ${IMG_NET_URL}${DIR_NAME}`
    HREF_LIST=`echo $PAGE_SRC | grep -oP "href\s*=\s*\"n\d+_\d+\.JPEG\""`
    while read -r line
    do
        echo Downloading ${line}...
        wget -O ${line} ${IMG_NET_URL}${DIR_NAME}/${line}
    done < <(echo $HREF_LIST | grep -oP "n\d+_\d+\.JPEG")
    cd ..
}

function downloadAll() {
    mkdir -p $IMG_DOWNLOADS_DIR
    cd $IMG_DOWNLOADS_DIR

    while IFS= read -r line
    do
        if [[ $line =~ ^[[:space:]]*(n[[:digit:]]+)[[:space:]]+.*$ ]]; then
            echo Downloading images under reference code ${BASH_REMATCH[1]}
            cd ..
            downloadFromDirectory ${BASH_REMATCH[1]}
            cd $IMG_DOWNLOADS_DIR
        fi
    done < "../$SYNSET_FILE"
    cd ..
}

while getopts "an:" opt; do
    case ${opt} in
        a)  DOWNLOAD_ALL="TRUE";;
        n)  IMG_CODE=$OPTARG;;
    esac
done

if [[ -z $DOWNLOAD_ALL ]]; then
    downloadFromDirectory $IMG_CODE
else
    downloadAll
fi
