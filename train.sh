# chmod 755 train.sh
# nohup ./train.sh > train.out &

python train.py \
--device cuda:1 \
--epochs 1200 \
--warm \
--cosine \
--model resnet50 \
--learning_rate 0.003 \
--size 128 \
--batch_size 256
