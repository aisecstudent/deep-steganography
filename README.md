# deep-steganography
深度隐写术: 基于深度学习和数据增强的图像隐写术，可将一张全彩色图像隐藏在另一个相同大小的图像中，并且任何一个图像的解码质量损失都最小。详情参见论文：《Hiding Images within Images》。

## 执行环境

Python >= 3.6.1，Pytorch >= 1.4

## 使用说明
在`main.py`所在的目录下新建一个data目录，将图片放入此目录中，执行`python mail.py`便可启动训练。
