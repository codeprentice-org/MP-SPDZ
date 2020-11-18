cd TensorflowInf/SqueezeNet
rm -f PreTrainedModel
axel -a -n 5 -c --output ./PreTrainedModel https://github.com/avoroshilov/tf-squeezenet/raw/master/sqz_full.mat
python3 process-model.py --in ./SampleImages/n02109961_36.JPEG
cd ../..
Scripts/fixed-rep-to-float.py TensorflowInf/SqueezeNet/SqNetImgNet_img_input.inp
./compile.py -R 64 new-tf TensorflowInf/SqueezeNet/graphDef.bin 1 trunc_pr split
Scripts/ring.sh new-tf-TensorflowInf_SqueezeNet_graphDef.bin-1-trunc_pr-split
