### Firefly-RK3399开发板TF模型测试结果

### 1. 系统： 安卓7.1.1

#### a) 测试框架TF lite

|         model          | Model size  | input size |        inference time         |
| :--------------------: | ----------- | :--------: | :---------------------------: |
| ssd_mobilenet_v1_quant | 4M (int8)   |    300     |             ~90ms             |
|    ssd_mobilenet_v2    | 67M(float)  |    300     |            ~900ms             |
| ssd_mobilenet_v2_quant | 17M(int8)   |    300     |            ~500ms             |
|   *ssd_resnet50_fpn    | 127M(float) |    640     | Fail (Op not fully supported) |



#### b) 测试框架 TF mobile

| model             | model size  | Input size | inference time |
| ----------------- | ----------- | ---------- | -------------- |
| ssd_mobilenet_v2  | 67M(float)  | 300        | ~90ms          |
| ssd_mobilenet_fpn | 49M(float)  | 640        | ~8s            |
| *ssd_resnet50_fpn | 129M(float) | 640        | ~12s           |



### 2. 系统： Ubuntu16.04

#### a) 测试框架 Tensorflow aarch 64

| model             | model size | input size | inference time |
| ----------------- | ---------- | ---------- | -------------- |
| ssd_mobilenet_v2  | 67M        | 480        | 1.8s           |
| *ssd_resnet50_fpn | 129M       | 480        | Fail（OOM）    |

#### b) 测试框架 ncnn

| model            | model size | input size | inference time |
| ---------------- | ---------- | ---------- | -------------- |
| ssd_mobilenet_v2 | 69M        | 460        | 1.5s           |

#### c) 测试框架 opencv dl

| model  | model size | input size | inference time |
| ------ | ---------- | ---------- | -------------- |
| YOLOv3 | 240M       | 608        | 14s            |

**注：** \* 标注的是目前所用的模型