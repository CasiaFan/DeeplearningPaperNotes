## Startdt Algorithm API Introduction
API repository address: http://api.startdt.net/dashboard/#!/project/qd3JbNavJ
### Part 1. Face Service (人脸服务)

#### 1. 人脸检测（包含人脸质量检测)
sample1: ![image1](pic/test7.jpg)

sample2: ![image2](pic/test_quality14.jpg)

sample3: ![image3](pic/test_quality18.jpg)

a). 检测相对正脸API：**人脸检测** `/face_detector_api` <br>
320x320图片：耗时 **150ms**<br>
**sample 1 result:** :`{"box": [], "flag": 0, "success": true}`  <br>
侧脸未检测到

**sample 2 result:** :`{"box": [[128, 104, 196, 174]], "flag": 1, "success": true}`<br>
检测到正脸，但人脸质量较差（模糊）

**sample 3 result:** :`{"box": [[128, 104, 196, 174]], "flag": 2, "success": true}`<br>
检测到正脸，但人脸质量合格

b). 检测各种姿势人脸：**人脸姿势检测** `/face_pose_api`<br>
320x320图片：耗时 **160ms**<br>
**sample 1 result**: 返回信息: `{"box": [84, 148, 158, 242], "flag": 1, "success": true}` <br>
检测到人脸，但人脸质量较差（侧脸）

**sample 2 result**: 返回信息: `{"box": [132, 103, 190, 180], "flag": 1, "success": true}` <br>
检测到人脸，但人脸质量较差（正脸，但是太模糊）

**sample 3 result**: 返回信息: `{"box": [126, 102, 177, 173], "flag": 2, "success": true}` <br>
检测到人脸，但人脸质量合格

在之后的版本中人脸质量将会从人脸检测中独立，包含以下类型：**正脸，侧脸，遮挡，模糊，噪声**

#### 2. 人脸特征
**特征服务2**: `/v3/face_register` <br>
320x320图片耗时：**1200ms**
提取人脸特征并上传oss，返回faceid和face feature

#### 3. 人脸属性
**人脸属性**: `/face_attributes` <br>
320x320图片耗时: **900ms**
**sample 3 result**: 返回信息: `{"info": [{"gender": "female", "age": 47}], "msg": "", "success": true}` <br>

Other attr under development: **表情（笑）**， **饰品（眼镜）**， ...

#### 4. 活体检测
a). **基于人脸的活体检测**: `/face_liveness_detect` <br>
320x320图片耗时：**200ms**

### Part 2. Apparel Service （服装服务）
#### 1. 服装检测
**服装检测**: `/clothing_detect` <br>
sample4: ![image4](pic/android_test4.jpg)

300x480图片耗时: **1500ms**<br>
**result:** `{"info": {"category": ["Up"], "bbox": [[0, 116, 282, 423]]}, "flag": 1, "success": 1}`
检测到一件上装

#### 2. 服装分类
**服装属性:** /clothing_attributes
300x480图片耗时: **300ms**<br>
result: `{"info": {"category": ["T恤"], "box": [0, 0, 282, 470], "flag": 1}, "success": 1}`
默认将当前图片全部区域作为服装分类的输入。一般需要先调用服装检测接口。

#### 3. 相似服装推荐
**服装推荐:** /clothing_recommend
300x480图片耗时: **3600ms**

### Part 3. Pedestrian Service (行人服务)
#### 1. 行人检测
**行人检测：** /person_detect_api
300x480图片耗时： **1200ms** <br>

#### 2. 行人特征
**person_register:** /person_register

### part 4. 综合API
**1. 人脸服务**： /face_service_api
耗时由选择的服务所决定

**2. Magic Mirror API**
**魔镜API** /magic_mirror_server_api
300x400图片耗时: **4500ms**
