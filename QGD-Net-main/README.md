# QGD-Net
## 训练:
- python train.py -d isic2017 -e 40 -g 16 -m qgd_net
## 预测
- python predict.py -m qgd_net -p ./checkpoints/qgd_net/isic2017/num_groups_16/isic2017/checkpoint_epoch4.pth -g 16 -i ./data/isic2017/imgs/ISIC_0000025.jpg -o output.jpg