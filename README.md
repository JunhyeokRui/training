
# Training: Parking Slot Detection and Tracking

## 1. Technical Details

### 1-1. requirements

tested in ubuntu 18.04

#### pip

1. Any python version over 3.6+ would be fine for modern deep learning experiments.

2. Install all the python packages I used using ```requirements.txt``` file.

   ```
   pip install -r requirements.txt
   ```


## 2. Dataset

### 2-1. get image file from bag file

Image-type dataset is first generated from raw bagfiles recorded during actual parking experiments. (Run ``rosrun yhpark_psd image_saver.py`` to save imagefiles from existing bagfiles.)



https://user-images.githubusercontent.com/86763564/159829379-eeb51e79-4839-4107-b3a1-ba22477e9df6.mp4



Change image saving path in file ``image_saver.py`` line 73.

Total 6 parking experiments on different parking spots are executed, and we name those experiments as trial A~F. Network training is usually done with trial A,B,C and remaining trials are used for testing. 

Dataset is (and should be) stored right outside this repository, using following folder structure. 

```
..
├── AVM_center_data_4class(image set name)             
│   ├── Images
│   │   ├── trial_A
│   │   │   ├── image_001.jpeg
│   │   │   ├── image_002.jpeg
│   │   │   └── ...
│   │   ├── trial_B
│   │   ├── trial_C
│   │   └── ...
│   ├── Labels
│   │   ├── trial_A
│   │   │   ├── image_001.txt
│   │   │   ├── image_002.txt
│   │   │   └── ...
│   │   ├── trial_B
│   │   ├── trial_C
│   │   └── ...
└── └── README.md

```
``../AVM_center_400/Labels/trial_*/image_*.txt`` is the corresponding YOLO label for the image  ``../AVM_center_400/Images/trial_*/image_*.jpeg``.


## 3. Marking Point Detection

### 3-1. Annotation 

Following python code is written to generate labels following the format that YOLOv5 requires for training. 

default image size is 400, 400 you can change this in annotation_tool_for_avm.py line 12

```
python annotation_tool_for_avm.py --dataset path_of_your_data --trial your_trial --bb_size bounding_box_size
```
Before clicking on the marking point, you should specify the type of the point by keyboard input. 

```
class 0 : outer-side corner  (Key = A)
class 1 : inner-side corner  (Key = S)
class 2 : outer-side auxiliary point  (Key = D)
class 3 : outer-side auxiliary point  (Key = F)
```

Checkout the demo to see how this annotation tool works. Pay attention to the keyboard inputs. 



https://user-images.githubusercontent.com/68195716/132684641-eb6b5b43-c573-4825-8aac-03e874531b3c.mp4




### 3-2. Training

To create train / test / validation split based on parking episodes, you must run the following code. 

```
python split_generator.py --train A,B,C --valid D,E --test F --dataset AVM_center_data --run split_ABC_DE_F
```

This code automatically creates ```./data/split_ABC_DE_F.yaml``` file that is required for YOLO training. You can pass this split YAML file through python argument during training. 

```
python train.py --batch-size 128 --data data/split_ABC_DE_F.yaml --name EXPERIMENT_NAME_OF_YOUR_CHOICE
```

Running this saves a trained weight under ```./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt```.

**Note:** parser ``--data`` indicates the path of the yaml file telling the path of the train, test, validation image sets

**Note:** YOLO, like any other deep learning models, requires large amount of data, and large batch size. In that sense, RTX2060 Super with 8GB VRAM is not optimal for YOLO training (although it can technically work). I used my personal deep learning GPU servers each equipped with Tesla-T4 (15GB VRAM)and RTX3080TI (12GB VRAM). If retraining is required in some time in the future, I recommend using RTX3090 computer with 24GB VRAM (if possible) with maximum batchsize that the GPU allows. Fortunately, YOLOv5 code is quite intuitively written and is highly customizable. If enough GPU power is ready, training will not be a problem. I've made some changes to the code that better suits the point detection task. 


### 3-3. Detection on ROS

```
python ROS_detect_marking_points.py --yolo_weight ./runs/train/EXPERIMENT_NAME_OF_YOUR_CHOICE/weights/last.pt --view-img
```

You should use ```--view-img``` flag to open up an opencv window that shows the live detection results. 

#### Marking Point Detection WorkFlow


https://user-images.githubusercontent.com/68195716/132668170-fafce57b-352e-41f5-8c2d-db991a7137a4.mp4



## 4. Marking Point Tracking

### 4-1. Run with pretrained weight

Change the ```REID_CKPT``` path in ```deep_sort.yaml``` to your path

Example

```
REID_CKPT:"/home/dyros/JH/downloads/parking_spot_corner_detector/deep_sort_pytorch/deep_sort/deep/resnet18_margin_0.3_more_data_epoch_300.pth" 
```

Even with a pretrained feature extractor network, DeepSORT can quite robustly track multiple marking points. You can check the result with pretrained weights using the code below:

```
python ROS_track_marking_points.py --yolo_weight ./runs/train/11_16_parking2/weights/best.pt --view-img
```

### 4-2. Create dataset for feature extractor

But as you can see, pretrained weight trained on Market1501 dataset (suited for human tracking) induces a lot of label switching. It might be appropriate to train a custom feature extractor network to increase the robustness of DeepSORT. What we should first do is to **create a new dataset** for tracking. The dataset should contain folders of diferent marking points, captured from different instances. The dataset has the following structure. 

```
..
├── dyros_deepsort_dataset           
│   ├── cropped_30
│   │   ├── trial_A
│   │   │   ├── point_0
│   │   │   │   ├── 1.png
│   │   │   │   ├── 2.png
│   │   │   │   ├── 3.png
│   │   │   │   └── ...
│   │   │   ├── point_1
│   │   │   │   ├── 1.png
│   │   │   │   ├── 2.png
│   │   │   │   ├── 3.png
│   │   │   │   └── ...
│   │   │   ├── point_2
│   │   │   ├── point_3
│   │   │   ├── ...
│   │   │   └── ...
│   │   └──
│   ├── pixel_labels
│   │   ├── trial_A
│   │   │   ├── point_0
│   │   │   │   ├── 1.txt
│   │   │   │   ├── 2.txt
│   │   │   │   ├── 3.txt
│   │   │   ├── ...
│   │   │   ├── point_1
│   │   │   │   ├── 1.txt
│   │   │   │   ├── 2.txt
│   │   │   │   ├── 3.txt
│   │   │   │   └── ...
│   │   │   ├── point_2
│   │   │   ├── point_3
│   │   │   ├── ...
│   │   │   └── ...
│   │   └──
└── └── README.md
```

I've done this step point by point

Note, ```point_a``` and ```point_b``` are physically differernt marking points, while ```x.png``` and ```y.png``` are the same physical marking points, captured from different frames. I created this dataset using the following annotation tool:

```
python annotation_tool_for_avm_deepsort.py --trial --dataset your_dataset --trial trial_A --bb_size 20
```

If you want to generate a new cropped dataset with different ```bb_size```, use:

```
python annotation_tool_for_avm_deepsort.py --dataset your_dataset --trial trial_A --bb_size 40
```
``--crop-only`` parser for cropping corner points only(no label created)

### 4-3. Training siamese network with Triplet loss

I used the famous [Pytorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning) library to train the feature extractor network.  Check out ```siaemese_training.py``` code that I used for training. It uses ResNet as a CNN architecture. 

before starting the training, check the path for the dataset 

```
python siamese_training.py --crop-size 30 --margin 0.3 --epoch 300 --name TEST --resize 20
```

Running this code saves its trained weight in the following directory: ```./deep_sort_pytorch/deep_sort/deep/TEST_last.pth```.

You can perform hyperparameter tuning adjusting the following arguments; 

```
python siamese_training.py --help

>  --crop-size # bounding box size around the marking point, default = 30
>  --margin # minimum distance between the embeddings of different marking points, default = 0.3
>  --batch-size  # default = 64
>  --rot_aug # default = True
>  --gaussian_aug # default = False
>  --resize # default = 0 (Resize the image to the given size.)
```
