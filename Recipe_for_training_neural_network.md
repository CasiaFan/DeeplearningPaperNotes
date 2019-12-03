## Recipe for training neural network

([Link](https://karpathy.github.io/2019/04/25/recipe/))

### 数据

浏览分析至少几百上千张样本图片，了解其分布和寻找pattern。可以通过对标签的类型，标注的大小，数量等信息进行搜索，过滤，排序来可视化数据的分布，从中发现一些数据质量和预处理时可能的一些问题。关注点包括：

- 是否有重复样本
- 图像/表情是否损坏
- 数据是否均衡
- 直观看局部特征是否足够，还是需要加入全局上下文
- 图像形式变化有多少，这种变化是否是有害的，能否通过预处理去掉
- 空间信息是否有用，是否可以用pool；细节信息是否重要，为保留细节信息可以支撑多少次下采样
- 标签的噪声有多少



### 构建端到端训练/评估骨架

采用最简单的模型来分析loss，metrics，预测和进行ablation分析确保评估方法没有错误。具体调试的trick包括：

- 固定random seed：保证两次运行的结果是一致的
- 简化：开始阶段可以不要加入数据增强，可能是bug来源
- 增加评估用的有用指标：不要光依赖每个batch的loss，加入跟模型目标相关的指标，比如正确率
- 通过初始化确认loss：通过人工假数据输入loss函数确认loss函数是正常工作的
- 初始化：分类时**最后一层**的初始化很重要，由于是避免样本不均衡的时候初始权重会导致预测结果一开始就有严重偏差
- 人类基准：test集打标两次，一次作为预测，一次作为gt，看人类的判断准确率是多少，可以用来作为基准。
- 输入不相关的基准：将输入全部归零，以此为输入进行训练，此时的模型表现是应该比实际数据时要差的
- 对单个batch overfit：对少量样本进行过拟合实验。增加模型的能力知道能够获得可能的最低loss（接近0）；此时图和标签能够很好的一一对应。如果不能达到这个，说明可能哪里有问题。
- 确认训练loss下降：相比小网络，使用大网络的时候训练loss应该会更低。
- 对输入网络之前的数据可视化：确认输入网络的数据是否符合预期
- 使用backprop验证依赖关系是否正确：有时候可能会不小心将batch维度上的信息进行了混合或者错误使用了transpose函数，此时网络一般还是可以正常训练。处理这种bug可以设计loss使输出i只在i输入上产生非0梯度。
- 泛化特殊case：直接从头写通用的函数可能会出错，可以先写特殊case的函数，确认正常工作后在进行泛化



### 过拟合

到这一步我们已经对上面说到的数据，训练评估流程，模型大致表现已经有很好的了解了，下一步就是进行迭代优化获得更好的模型。一般套路是先训练一个足够大的模型可以很好过拟合数据，然后再添加正则化进行调整，获得满足需要的模型。

- 选择初始模型：抄一个跟目标最接近的文章的网络作为基础得到baseline，然后进行优化。一般来说，选择resnet50骨架总是最安全的。
- adam is safe：虽然adam表现可能不及SGD，但是对问题的容忍能力更强，在初始阶段选用adam with lr=3e-4总是安全的
- 一次实验只优化一个点：确保加入的trick是有效的
- 不要轻信默认的 lr衰减配置：默认的lr衰减配置是基于imagenet的（epoch 30乘以0.1），但是实际情况下数据量与ImageNet是不同的，需要进行调整，否则过快的衰减会导致学习不充分不能收敛。



### 正则化

在获得在训练集上loss可以降到最小的模型后，用正则化来提升验证集上的精度

- 输入更多数据

- 数据增强：可以用激进一些的增强方法，比如domain randomization， clever hybrid, use of simulation， GAN。

- 预训练

- 坚持监督学习不动摇：这个阶段的无监督学习还是渣渣

- 更小的输入维度：抑制虚假信息，避免过拟合

- 更小的模型尺寸：比如pooling代替FC

- 减小batch size： 更小的batch对应了更强的正则化

- 添加dropout：但是又BN的时候要小心

- 增加权重衰减惩罚

- early stopping： 在模型似乎要过拟合的时候停止训练

- 尝试更大的模型，然后用ealy stopping

  

### Tune

这时就是调参工程师反复调参的时候了。

- 随机网格搜索超参：其实随机搜索是最好的，因为网络对某些参数更加敏感
- 超参优化：比如贝叶斯超参优化工具箱



### Last Tricks

- ensemble： 使用ensemble一般可以保证2个点的精度提升，如果计算资源不过，试试蒸馏的黑魔法。
- 没事接着训：玄学



  