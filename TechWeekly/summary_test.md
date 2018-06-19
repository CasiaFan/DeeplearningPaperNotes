Google Learn2Compress service in ML Kit

([Blog](https://ai.googleblog.com/2018/05/custom-on-device-ml-models.html)) ([Apply Link](https://docs.google.com/forms/d/e/1FAIpQLSd7Uzx6eepXeF5osByifFsBT_L3BJOymIEjG9uz1wa51Fl9dA/formResponse))

**核心技术**： Learn2Compress主要包含**模型剪枝，量化，联合训练和知识蒸馏**方法，支持将**TF Lite**的模型进行内存优化和加速。

![image2-1](https://1.bp.blogspot.com/-rLAjT1bpCKk/WvN5OhmkkcI/AAAAAAAACuA/-N1hoYLhDdcjk1qZy279lontG7dshf9QgCLcBGAs/s1600/f1.png)

- 剪枝： 能过压缩2x的模型尺寸，并且保留97%的精度

- 量化：8-bit fixed point 量化

- 联合训练和知识蒸馏： teacher-student learning strategy： 用大的teacher网络训练一个紧凑的student网络并减少精度损失。Teacher网络可以被固定（distillation的时候）或者进行联合优化，而且一般会同时训练多个不同大小和inference效率的student网络提供选择。

  ![image2-2](https://3.bp.blogspot.com/-eD3Mc4FLsvA/WvN6BweGY_I/AAAAAAAACuU/OZqGR1UUvL05Ctr1b8JD3SaKlCNCZVMdACLcBGAs/s1600/f2.png)

目前该技术支持的模型包括**MobileNet， NASNet， inception，[ProjectionNet](https://arxiv.org/pdf/1708.00630.pdf)**的**分类**任务；由于功能仍然在测试阶段，因此需要申请。

**实验结果：**

**推测精度：**

![image2-3](https://4.bp.blogspot.com/-vhIB65lfBbo/WvN5iGy2HkI/AAAAAAAACuI/0fT8SIYfZaEGG3CyLPbE3jVK7BGNMjD1wCLcBGAs/s640/f3.png)

**推测速度**：

![image2-4](https://3.bp.blogspot.com/-nrWYQszTHrA/WvN5yN02aCI/AAAAAAAACuQ/x8FpayO0_kIFJwLGg5EaQR4_qpMD5JK4QCLcBGAs/s640/f4.png)



### Face Recognition

1. **Pose-Robust Face Recognition via Deep Residual Equivariant Mapping**

([Paper](https://arxiv.org/abs/1803.00839)) ([Code](https://github.com/penincillin/DREAM))

**核心技术**：DREAM模块（深度残差等变映射）：首先假设在深度**特征空间**中，侧脸区域的特征和正脸区域的特征相关，即输入任意姿势的图像，可以通过添加残差映射函数将特征映射到正脸的特征空间上（特征等变性理论）。

![image3-1](http://5b0988e595225.cdn.sohucs.com/images/20180313/98cd6481724d4668a7188fd30ced19bf.jpeg)

DREAM模块：引入soft-gate控制机制来自适应控制残差量从而可以控制不同人脸姿势的中引入的残差量（正脸引入少而极侧脸引入较多）；此模块可以直接加入到现有的网络框架中，与主干CNN进行联合端到端的训练。

![image3-2](http://5b0988e595225.cdn.sohucs.com/images/20180313/1895fc8e8b62445c961f464cced6bfb2.jpeg)

**优势**：

1. 实施简单，DREAM模块可以直接拼接到现有的模型，无需改变现有的脸部特征的维度，并且可以直接BP端到端的训练
2. 轻量，添加的参数很少，不会对模型性能造成较大的影响（ResNet18中参数量只增加0.3%，时间成本增加1.6%）。而之前常用的正脸化方法包括**3D人脸归一化**到正脸，**GAN**网络生成正脸的方法开销比较大，而且极侧脸的转化还是很困难。
3. 在不影响正脸识别效果的基础上提高极侧脸的识别率。不需要更详细的人脸数据。

**讨论：**本文认为目前的人脸识别技术在侧脸表现差的原因主要在于训练的时候**正侧脸数据不均衡**导致的。

**实验结果**：

侧脸转换成正脸的效果：

![image3-3](http://5b0988e595225.cdn.sohucs.com/images/20180313/2ae94b78504e45bfaf2312870d47a9bc.jpeg)

错误率：

![image3-4](http://5b0988e595225.cdn.sohucs.com/images/20180313/4af247108bd846a481ed6fb1ed624cab.jpeg)

对侧脸假阳性和假阴性样本的效果：

![image3-5](http://5b0988e595225.cdn.sohucs.com/images/20180313/fa311b3b79164e3d8b5d55856f2c72a2.jpeg)

AutoTVM (2018.2)

（[Learning to Optimize Tensor Programs](https://arxiv.org/abs/1805.08166)）( [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799))

（[消息连接](https://mp.weixin.qq.com/s/YVIvdMznb3oatIXqD5a5_A)）

核心：通过深度学习编译器（TVM）自动优化深度学习算子性能，实现高效优化硬件代码。思路为：1.建立足够大的搜索空间，确保人工手写的优化全部包含在内；2.快速搜索这个空间，获取最优的实现。前者需要不断总结抽象手工优化的规律；后者需要利用机器学习来学习程序空间的代价估价函数然后利用这个函数来进行检索。



CliqueNet: Convolutional Neural Networks with Alternately Updated Clique (2018.5， CVPR2018)

([Paper](https://arxiv.org/abs/1802.10419)) ([Code](https://github.com/iboing/CliqueNet))

核心：Clique Block。与DenseNet相比，CliqueNet的每个Chique Block只有固定通道数的特征图会传送到下一个Clique Block，从而避免DenseNet中由于复用不同层级特征图随着深度的增加导致的密集型连接路径线性增加导致的参数量剧增。

Cliquet Block的计算一般分为两个二段（更高阶的计算成本太高），第一阶段的传播通DenseNet，视为block的初始化；第二阶段的每一个卷积输入不仅包括前面层的特征图，还包括后面层级的特征图。这一部分通过循环反馈结构利用更高级视觉信息精炼出前面层级的卷集核，实现空间注意力的效果。由于每一个block只有第二阶段会作为下一个block的输入，因此Block的特征图维度不会超越线性增加。

此外还通过多尺度特征策略来避免参数的快速增长：将每个Block的输入和输出特征图拼接到一起，然后global pooling的到一个向量。用所有block的最后pooling出来的这个向量进行预测。



ProjectionNet: learning efficient on-device deep networks using neural projections (2017.8)

([paper](https://arxiv.org/abs/1708.00630)) ([code](https://github.com/akosichesca/projectionnet))

核心：同时训练两个不同的网络：full trainer NN（现有的深度网络）和**projection network**，后者利用**随机投射**将输入和中间值转换成**bits**，因此其计算和存储所需的内存很小。两者通过end-to-end bp进行训练，projection net通过**学徒学习**的方式从full network中进行学习，训练完成后可以直接用projection net进行inference。



**projection network**

