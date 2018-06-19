### 海康网络摄像头测试

#### **前提**

1. 采用opencv读取USB webcam的方式获取图像
2. 测试包括两种摄像头：**test0**为视角130；**test1**为视角160.
3. 同一层样品的**高度类似**
4. 模拟冰箱单层容积（LxWxH）为52cm x 34cm x 30cm; 摄像头高度约29cm。

#### **成像效果**

a). **测试样本可乐（24.5cm）**

1. **单摄像头**

   最左端

   ![image-20180531120506784](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531120506784.png)

   最左侧+内移6cm

   ![image-20180531162401994](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531162401994.png)

   最外侧

   ![image-20180531120915700](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531120915700.png)

   最内侧

   ![image-20180531135903959](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531135903959.png)

   最内侧+内移3cm

   ![image-20180531162522242](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531162522242.png)

   最右侧

   ![image-20180531140012962](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531140012962.png)

   最内侧+高度上提5cm

   ![image-20180531163104050](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531163104050.png)

2. **双摄像头**

   最左侧

   ![image-20180531140654531](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531140654531.png)

   ![image-20180531141856652](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531141856652.png)

   最右侧

   ![image-20180531141006236](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531141006236.png)

   ![image-20180531142116284](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531142116284.png)

b). 测试样本美汁源（21cm）

1. **单摄像头**

   最左侧

   ![image-20180531153305509](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531153305509.png)

   最外侧

   ![image-20180531153449376](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531153449376.png)

   最内侧

   ![image-20180531153608439](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531153608439.png)

   最右侧

   ![image-20180531153715472](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531153715472.png)

2. **双摄像头**

   最左侧

   ![image-20180531154411935](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531154411935.png)

   最右侧

   ![image-20180531154553995](/var/folders/7z/16pdt9_538vf868lt3_g17nr0000gn/T/abnerworks.Typora/image-20180531154553995.png)

**结论：**

1. 海康的USB摄像头为USB2.0端口。~~采用USB hub的情况下单机最高可以支持**4**个摄像头~~。目前测试USB3.0hub可同时稳定支持**6**个，超过7个以上会出现获取到空白帧，但流不会丢，重新截取有效。（USB流传输限制？）USB摄像头长时间工作稳定性测试中。
2. 采用视场160度的摄像头可以单个就能覆盖所有区域，但对于可乐（24cm）在摄像头高度为29cm时在避免遮挡的情况下有效区域比托盘需要在长度上缩小6x2cm，宽度上缩小3x2cm；而在摄像头高度提高5cm在34cm时则基本能避免遮挡，但是相应的样本成像会变小。对于美汁源（21cm）则基本能够避免遮挡。
3. 在使用双摄像头的时候视场130的摄像头无法检测完全最内侧和最外侧的商品，有效区域宽度减少2x2cm，对于最左侧和最右侧物体的检测效果会比较好。



USB hub支持多个USB摄像头同时读流的前提：

1. Ubuntu （测试为16.04，arch linux显示USB设备报错`v4l2-ctl -D --list-devices`）
2. 海康USB（罗技c930e无法再同一个hub）
3. opencv installed from pip or source works