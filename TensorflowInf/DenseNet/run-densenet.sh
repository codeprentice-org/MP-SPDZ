# Note: The pretrained model must be downloaded from the link below and put into PreTrainedMode/
# Link: https://drive.google.com/file/d/0B_fUSpodN0t0eW1sVk1aeWREaDA/view
cd PreTrainedModel && tar -xvzf tf-densenet121.tar.gz && cd -
python3 DenseNet_main.py --runPrediction True --scalingFac 12 --saveImgAndWtData True

cd ../../
Scripts/fixed-rep-to-float.py TensorflowInf/DenseNet/SqNetImgNet_img_input.inp

# TODO: abstract script so that the protocol being used can be changed at runtime
# See https://github.com/data61/MP-SPDZ#tensorflow-inference for details
./compile.py -R 64 tf TensorflowInf/DenseNet/graphDef.bin 1 trunc_pr split 
Scripts/ring.sh tf-TensorflowInf_DenseNet_graphDef.bin-1-trunc_pr-split