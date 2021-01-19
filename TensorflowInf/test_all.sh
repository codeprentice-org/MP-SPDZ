echo -e "-------------"
echo -e "DENSENET TEST"
echo -e "-------------\n"
cd DenseNet/
./test-tf-inference.sh
cd ..

echo -e "-----------"
echo -e "RESNET TEST"
echo -e "-----------\n"
cd ResNet/
./test-tf-inference.sh
cd ..

echo -e "---------------"
echo -e "SQUEEZENET TEST"
echo -e "---------------\n"
cd SqueezeNet/
./test-tf-inference.sh
cd ..
