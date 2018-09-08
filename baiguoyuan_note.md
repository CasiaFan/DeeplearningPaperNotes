### 序列模型输入数据预处理方案（百果园方案）

1. **输入数据维度**

  一般来说，支持矩阵输入的模型（比如xgboost和RNN深度模型）的数据输入维度一般是3D：分别为**样本数，时间步长，和特征**。

比如：

```
[
[[x11, y11, z11],
 [x12, y12, z12],
  ...
 [x1n, y2n, z3n],
 [y21, y22, y23]],
[[x21, y21, z21],
 [x22, y22, z22],
 ...
 [x2n, y2n, z2n]]
]
```

就是输入2个样本，时间步长为n，特征为3个（x, y, z）

2. **时序步长**

  时序一般采用最小时间单位间隔的连续序列，也可以结合特征选择周期性的时间最为间隔单位。比如，连续10天，或者连续10个周五。如果采用后面的时序，那么在模型的输入数据化中就可以去掉指示日期周期性的特征。

3. **特征**

a). 特征的选取

特征选取就是选择可能会影响输出的因素（这里我们指销量）。根据模型本身的目的和特性需要进行选择的特征也是不同个的。

对于单店：可选取的特征包括

- 销量
- 价格
- 天气
- 日期
- 活动
- 舆论

对于连锁店：还可以添加：

- 城市
- 地段

b). 特征的量化

对于不同的特征，需要进行量化的方式也不同，一般来说，从经验判断影响因素比较大的并且本身就是数字类型的参数可以直接作为输入；而比较重要但是本身不是数字类型的参数或者本身是数字类型的参数但是可能结果对其的变化不敏感的参数，但是并可以进行分段量化，分成几个等级；而无法判断的是否重要的非数字型参数可以直接用0/1编码或one-hot编码。

以**大类**为例：

**销量：**

销量特征包含1. 该大类的总销量（原始值）； 2.竞品大类的销量（原始值/分段量化）3. 是否销售完(0/1编码)或者销量/备货比例（分段量化）

对于竞品，可以按两种方式考虑：1.价格接近的；2. 种类接近的：比如都是瓜类，柑橘类，葡萄类等

**价格**：

包括：1. 大类本身的平均价格（加权平均）（分段量化）；2. 竞品平均价格

价格可以计算价格指数进行分段量化，统计改品类历史价格区间，判断其当前价格处于哪一档；或者简单的根据与前N天比涨，平，跌设置成-1， 0，1

**天气：**

包括：1. 是否影响出门（针对到店销量）（0/1编码）

主要看是否是极端天气

**日期：**

包括： 1. 工作日/休息日/节假日(0/1编码)； 2. 品类是否上市/上市初期/大量上市期/上市末期（0/1编码或者分段量化）

对于某些特定的节假日可以调高量化权重，比如圣诞节前的苹果，春节前的橘子之类

**活动：**

 包括：1. 是否该品类有促销/促销力度（分段分类或0/1分类）；2. 是否竞品品类有促销（0/1分类）；3（？？？）. 是否周边水果店该品类有促销（数据无法获取）

**舆论**：

包括：1. 社交媒体（微博，新闻等）关于品类的舆情：正向/负向/未知/中立（分段量化或0/1量化）

**城市**

城市属性比较复杂，这里可以先抽象为1. 城市消费水平（1/2/3先城市等）（分段量化）；2. 城市位置（华南/华东/…）=> 城市获取该品类商品的难易程度（分段量化）

**地段：**

这个属性也比较复杂，可以抽象为1. 客流量（住宅区/商业区/交通枢纽等）（分段量化）；2. 消费水平=>可以用租金作为参考（分段量化） 3. 竞品商家数量（？？？，需要地图数据）



对于子类，在上述特征的基础上将相应的特征进行细化，需要注意的是，此时子类内部将**互为竞品**。



**思考：**

1. 对于单店来说，固定的变量可以不用考虑，比如城市和地段；但是如果将这些参数作为模型的一个维度的特征输入，那么可以将所有店的数据都纳入那一个模型进行训练，这种模型相对来说鲁棒性会高一些。

2. 会员购买历史与会员定制化推送/预订
