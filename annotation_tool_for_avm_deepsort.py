import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
from natsort import natsorted


parser = argparse.ArgumentParser(description='corner_annotation')
parser.add_argument('--trial', required=True)
parser.add_argument('--dataset', default='AVM_center_data_4class_test')
parser.add_argument('--img_size', default=400, type=int)
parser.add_argument('--bb_size', default=20,type=int)
parser.add_argument('--crop-only', action='store_true')


args = parser.parse_args()

bbsize = args.bb_size
def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        c.append(0)
        cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)
        cv2.rectangle(img, (x-bbsize, y-bbsize),(x+bbsize,y+bbsize),  (0, 0, 255), 2)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        print(x,y)

#option
dataset = args.dataset
crop_only = args.crop_only
path = '../{}/'.format(dataset)
trial = args.trial


#parser = argparse.ArgumentParser(description='corner_annotation')
#parser.add_argument('--trial', default=None)
#args = parser.parse_args()



print(path)
cnt = 0
init=0
while cnt == 0:
    if os.path.isfile(os.path.join(path, 'cropped_{}'.format(bbsize), trial,'point_{}'.format(str(init)),"0.png")):
        init+=1
        print(init)
        continue
    else:
        cnt = 1
        print("baa")
        break
        
n_instance=init
n_img =0




if crop_only:
    while True:
        os.makedirs(os.path.join(path, 'cropped_{}'.format(bbsize)),exist_ok = True)
        os.makedirs(os.path.join(path, 'pixel_labels'),exist_ok = True)
        for images in tqdm(natsorted(os.listdir(os.path.join(path,'images',trial)))):
            if 'jpeg' in images:

                print('{}'.format(images))
                img = cv2.imread(os.path.join(path,'images',trial,images))
                a = []
                b = []
                c = []
                raw_img = img.copy()

                cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow("image", 400,400)
                cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN)
                img = cv2.resize(img, (400, 400))                    # Resize image
                raw_img = cv2.resize(raw_img, (400, 400))                    # Resize image
                cv2.imshow("image", img)

                cv2.waitKey(0)
                print(a)
                print(b)
                
                os.makedirs(os.path.join(path, 'cropped_{}'.format(bbsize), trial, 'point_{}'.format(str(n_instance))),exist_ok = True)
                os.makedirs(os.path.join(path, 'pixel_labels', trial, 'point_{}'.format(str(n_instance))),exist_ok = True)
                for i in range(len(a)):
                    crop_img = raw_img[b[0]-bbsize:b[0]+bbsize, a[0]-bbsize:a[0]+bbsize]
                    cv2.imwrite(os.path.join(path, 'cropped_{}'.format(bbsize), trial,'point_{}'.format(str(n_instance)), "{}.png".format(n_img)), crop_img)
                    
                    n_img +=1

else:
    while True:
        
        os.makedirs(os.path.join(path, 'cropped_{}'.format(bbsize)),exist_ok = True)
        os.makedirs(os.path.join(path, 'pixel_labels'),exist_ok = True)
        for images in tqdm(natsorted(os.listdir(os.path.join(path,'images',trial)))):
            if 'jpeg' in images:

                print('{}'.format(images))
                img = cv2.imread(os.path.join(path,'images',trial,images))
                a = []
                b = []
                c = []
                raw_img = img.copy()

                cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
                cv2.resizeWindow("image", 400,400)
                cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN)
                
                img = cv2.resize(img, (400, 400))                    # Resize image
                raw_img = cv2.resize(raw_img, (400, 400))                    # Resize image
                cv2.imshow("image", img)

                cv2.waitKey(0)
                print(a)
                print(b)
               
                os.makedirs(os.path.join(path, 'cropped_{}'.format(bbsize), trial, 'point_{}'.format(str(n_instance))),exist_ok = True)
                os.makedirs(os.path.join(path, 'pixel_labels', trial, 'point_{}'.format(str(n_instance))),exist_ok = True)
                for i in range(len(a)):
                    crop_img = raw_img[b[0]-bbsize:b[0]+bbsize, a[0]-bbsize:a[0]+bbsize]                    
                    cv2.imwrite(os.path.join(path, 'cropped_{}'.format(bbsize), trial,'point_{}'.format(str(n_instance)), "{}.png".format(n_img)), crop_img)
                    np.savetxt(os.path.join(path, 'pixel_labels', trial,'point_{}'.format(str(n_instance)), "{}.txt".format(n_img)),np.array([b[i],a[i]]))
                    n_img +=1