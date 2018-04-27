## 1. Face Datasets
### 1). Face Recognition Datasets
- **Celeba** ([official link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))<br>
  Local path: `/startdt_data/face_datasets/DeepFashion/CelebA` <br>
  Raw image directory: `Img/img_celeba.7z/img_celeba` <br>
  Clean image directory: `deep_clean1_112_96` (cropped in size 112x96)

- **CASIA** ([official link](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)) <br>
  Local path: `/startdt_data/face_datasets/casia-maxpy-clean` <br>

- **LFW** ([official link](http://vis-www.cs.umass.edu/lfw/))<br>
  Local path: `/startdt_data/face_datasets/lfw`

- **MegaFace** ([official link](http://megaface.cs.washington.edu/)) <br>
  Local path: `/startdt_data/face_datasets/MegaFace`

- **MS-Celeb-1M** ([official link](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)) <br>
  Local path: `/startdt_data/face_datasets/ms_celeb_v1`

- **VGG Face** ([official link](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/)) <br>
  Local path: `/startdt_data/face_datasets/VGG_face`

- **Large-scale Labeled Face (LSLF) dataset** ([official link](http://discovery.cs.wayne.edu/lab_website/index.php/lsdl/)) <br>
  Local path: `/startdt_data/face_datasets/LSLF`

### 2). Face Attribute Datasets
- **OUI adience** ([official link](https://www.openu.ac.il/home/hassner/Adience/data.html)) <br>
  Doc: wild face **age** and **gender** calssfication <br>
  Local path: `/startdt_data/face_datasets/OUI_adience`

- **AgeDB** ([official link](https://ibug.doc.ic.ac.uk/resources/agedb/)) <br>
  Doc: wild image **age** database <br>
  Local path: `/startdt_data/face_datasets/AgeDB`

- **CAS-PEAL Face Database** ([official link](http://www.jdl.ac.cn/peal/index.html)) <br>
  Doc: indoor **Pose, Expression, Accessories, and Lighting** <br>
  Local path: `/startdt_data/face_datasets/CAS-PEAL-Face-Database`

- **IMDB-WIKI** ([official link](http://www.jdl.ac.cn/peal/index.html)) <br>
  Doc: celeb face **age** and **gender** <br>
  Local path: `/startdt_data/face_datasets/IMDB-WIKI-Face`

- **SCUT-FBP5500** ([official link](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release))
  Doc: face **beauty** <br>
  Local path: `/startdt_data/face_datasets/SCUT-FBP`

- **Celeba Attribut self-made Dataset** <br>
  Doc: **Smile, glasses, sunglasses, focus blur, motion blur, noise, backlighting**
  Local path: `/startdt_data/face_datasets/DeepFashion/CelebA/training_groups`

### 3). Face Quality Assessment Datasets
- **Specs on Faces(SoF)** ([official link](https://sites.google.com/view/sof-dataset)) <br>
  Doc: face in **backlighting**, **sunglasses occluded** <br>
  Local path: `/startdt_data/face_datasets/SoF`

- **CrowedFaces dataset** ([official link](http://discovery.cs.wayne.edu/lab_website/index.php/lsdl/)) <br>
  Doc: small occluded in crowd
  Local path: `/startdt_data/face_datasets/CrowdFace`

### 4). Anti-Spoof Face Datasets
  - Self-made <br>
    Raw images: <br> `/startdt_data/face_datasets/face_anti_spoof_datasets/magic_mirror_face_dataset/image_magic_mirror_raw` <br>
    `/startdt_data/face_datasets/face_anti_spoof_datasets/drama_dataset/frame` <br>

    Pos: <br>
    `/startdt_data/face_datasets/face_anti_spoof_datasets/drama_dataset/face/real_face` <br>
    `/startdt_data/face_datasets/face_anti_spoof_datasets/magic_mirror_face_dataset/magic_mirror_face`

    Neg: <br>
    `/startdt_data/face_datasets/face_anti_spoof_datasets/magic_mirror_face_dataset/magic_mirror_face_fake_*` <br>
    `/startdt_data/face_datasets/face_anti_spoof_datasets/drama_dataset/face/fake_face`

  - Unknown source <br>
    Local path: `/startdt_data/face_datasets/face_anti_spoof_datasets/Detectedface`

## 2. Pedestrian Datasets
- **Caltech Pedestrian Detection Benchmark** ([official link](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)) <br>
  Local path: `/startdt_data/caltech_pedestrian_dataset`

- **GM-ATCI Rear-view pedestrians dataset** ([official link](https://sites.google.com/site/rearviewpeds1/))  <br>
  Local path: `/startdt_data/GM_ACTI_Pedestrian_dataset`

- **INRIA Person Dataset** ([official link](http://pascal.inrialpes.fr/data/human/)) <br>
  Local path: `/startdt_data/INRIAPerson`

- **HDA Dataset** ([official link](http://vislab.isr.ist.utl.pt/hda-dataset/)) <br>
  Doc: for **person Re-id** <br>
  Local path: `/startdt_data/HDA_Dataset_V1.3`

## 3. Hand Datasets
- **EgoHand dataset** ([official link](http://vision.soic.indiana.edu/projects/egohands/)) <br>
  Doc: hand **detection** dataset <br>
  Local path: `/startdt_data/hand_gesture/egohands`

- **CVRR dataset** ([official link](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/)) <br>
  Doc: hand **detection** dataset when **driving** <br>
  Local path: `/startdt_data/hand_gesture/CVRR`

- **HGR dataset** ([official link](http://sun.aei.polsl.pl/~mkawulok/gestures/)) <br>
  Doc: **3D** hand **gesture recognition** <br>
  Local path: `/startdt_data/hand_gesture/HGR`

- **NYU hand pose dataset** ([official link](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm)) <br>
  Doc: **3D** hand **pose recognition** <br>
  Local path: `/startdt_data/hand_gesture/NYU_hand`

- **Oxford hand dataset** ([official link](http://www.robots.ox.ac.uk/~vgg/data/hands/)) <br>
  Doc: hand **detection** dataset <br>
  Local path: `/startdt_data/hand_gesture/oxford`

- **20BN-JESTER DATASET V1** ([official link](https://twentybn.com/datasets/jester/v1)) <br>
  Doc: **hand gesture short video** <br>
  Local path: `/startdt_data/hand_gesture/20bn`

- **RoViT dataset** ([official link](http://www.rovit.ua.es/dataset/mhpdataset/)) <br>
  Doc: **3D hand pose gesture** <br>
  Local path: `/startdt_data/hand_gesture/RoViT`

## 4. Apparel Datasets
- **DeepFashion dataset** ([official link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html))<br>
  Doc: clothing **detection, attributes, landmarks** <br>
  Local path: `/startdt_data/clothing_data`

- **Fashionista dataset** ([official link](http://vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/)) <br>
  Doc: **pose estimation, clothing parsing** <br>
  Local path: `/startdt_data/clothing_data/apparel_dataset`

- **天池服装数据集** ([official link](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.3ccbd780LGXBuA&raceId=231648)) <br>
  Doc: **clothing attributes and keypoints** <br>
  Local path: `/startdt_data/tianchi_apparel_dataset`

## 5. Head detection Dataset
- **HollywoodHeads Dataset** ([official link](http://www.di.ens.fr/willow/research/headdetection/)) <br>
  Doc: head detection <br>
  Local path: `/startdt_data/HollywoodHeads`

## 6. Object Detection Dataset
- **COCO** ([official link](http://cocodataset.org/)) <br>
  Doc: **object detection** and **person landmarks detection** <br>
  Local path: `/startdt_data/COCO/dataset`

## 7. Classification Dataset
- **ImageNet** ([official link](www.image-net.org)) <br>
  Doc: *images not downloaded yet* <br>
  Local path: `/startdt_data/ImageNet`

- **CIFAR10** ([official link](https://www.cs.toronto.edu/~kriz/cifar.html)) <br>
  Local path: `/startdt_data/cifar10`

## 8. others
- **Indoor Scene Dataset** ([official link](http://web.mit.edu/torralba/www/indoor.html)) <br>
  Doc: indoor **scene** Classification <br>
  Local path: `/startdt_data/indoor_scene_data`

- **Blur Image Dataset**
Local path: `/startdt_data/blur_image` <br>
  - **CERTH** ([official link](http://mklab.iti.gr/project/imageblur))
  - **Blur Detection Dataset** ([official link](http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/dataset.html))
  - **Blur-Noise Trade Off Dataset** ([official link](http://home.deib.polimi.it/boracchi/Projects/BlurNoiseTradeOff_DataSet.html))
