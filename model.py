import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from main import CLASSES,load_and_proccess_images,IMG_SIZE

BATCH_SIZE= 32
EPOCHS=10





def build_model(input_shape):
    model = Sequential()

    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    
    model.add(Flatten())

    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    
    model.add(Dense(len(CLASSES), activation='softmax'))

    return model


if __name__ == "__main__":
    train_path="dataset/chest_xray/train"
    X_train,y_train = load_and_proccess_images(train_path)  

    X_train, X_val,y_train,y_val = train_test_split(X_train,y_train , test_size=0.2,random_state=42)

    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # 3 channels for RGB
    model = build_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val), batch_size = BATCH_SIZE, epochs=EPOCHS)
    
    model.save("model.h5")

    print("model training complete model saved as model.h5")
    
    