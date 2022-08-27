# chmod 755 linear.sh
# nohup ./linear.sh > linear.out &

python linear.py \
--distortion supcon \
--dataset cifar10 \
--test_dataset cifar10 \
--model resnet50 \
--device cuda:0 \
--test_epoch 1 \
--epochs 100 \
--cosine \
--size 128

python linear.py \
--distortion supcon \
--dataset cifar10 \
--test_dataset cifar10 \
--model resnet50 \
--device cuda:0 \
--test_epoch 5 \
--epochs 100 \
--cosine \
--size 128

