import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE=224
CLASSES={"NORMAL":0,"PNEUMONIA":1}

def load_and_proccess_images(folder_path):
    images=[]
    labels=[]

    for label,class_id in CLASSES.items():
        class_path=os.path.join(folder_path,label)
        for img_name in os.listdir(class_path):
            img_path=os.path.join(class_path,img_name)
            image=cv2.imread(img_path)
            if image is not None:
              image=cv2.resize(image ,(IMG_SIZE,IMG_SIZE))
              images.append(image)
              labels.append(class_id)

            #convert to numby

    images=np.array(images ,dtype="float32") /255.0
    labels=np.array(labels)

    return images ,labels


if __name__=="__main__":
    train_path="dataset/chest_xray/train"
    X_train,y_train=load_and_proccess_images(train_path)
    print(f"loaded {len(X_train)} training images .")

    for i in range(5):
        plt.imshow(X_train[i])
        plt.title(f"class : {'NORMAL' if y_train[i]==0 else'PNEUMONIA'}")
        plt.axis("off")
        plt.show()

    X_train, X_val,y_train,y_val = train_test_split(X_train,y_train , test_size=0.2,random_state=42)
    print(f"Training set size: {len(X_train)} images.")
    print(f"Validation set size: {len(X_val)} images.")