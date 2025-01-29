from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


IMG_SIZE=224
CLASSES={"NORMAL":0,"PNEUMONIA":1}


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
   
    input_shape = (IMG_SIZE, IMG_SIZE, 3)  # 3 channels for RGB
    model = build_model(input_shape)

    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    
    model.summary()