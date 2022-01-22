#Import required libraries
import numpy as np 
import cv2
import os

import keras
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers.advanced_activations import PReLU
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping






epochs = 25
learning_rate = 0.01
batch_size=32
img_dim = (96,96,3)
size=96

path = r"C:/Users/pulkit_jain/Downloads/Computer Vision Dataset/CV_dataset/"
metadata = pd.read_csv(path+"meta.csv")
metadata=metadata.head(2000)

#metadata.gender.value_counts()

metadata['Image_Matrix']=None
metadata['label']=None


metadata['label']=np.where(metadata['gender']=='male',1,0)

for key,row in metadata.iterrows():
    image = cv2.imread(path+row['fileName'])
    image = cv2.resize(image, (size, size))
    image = img_to_array(image)
    
    metadata['Image_Matrix'][key] = image
    
metadata.drop(columns=['gender','fileName'],inplace=True)
    
# Using numpy's savez function to store our loaded data as NPZ files
# =============================================================================
# np.savez('metadata.npz', np.array(metadata.values))
# 
# metadata = np.load("metadata.npz",allow_pickle=True)
# metadata = metadata['arr_0']
# metadata = pd.DataFrame(metadata,columns=['Image_Matrix', 'label'])
# 
# =============================================================================


images = metadata['Image_Matrix'].values.tolist()
images = np.array(images,dtype='float')/255.0

labels = metadata['label'].values.tolist()
labels = np.array(labels)
train_x,test_x,train_y,test_y = train_test_split(images,labels,test_size=0.2,random_state=42)


#Normalize the dataset

# Reshaping our data from (x,) to (x,1)
#train_y=to_categorical(train_y,num_classes=2)
#test_y=to_categorical(test_y,num_classes=2)

train_y = train_y.reshape(train_y.shape[0], 1)
test_y = test_y.reshape(test_y.shape[0], 1)
# x_train = y_train.reshape(x_train.shape[0], size,size,1)
# x_test = y_test.reshape(x_test.shape[0], size,size, 1)

# Change our image type to float32 data type


# Normalize our data by changing the range from (0 to 255) to (0 to 1)



#creating model
def build_model(width,height,depth,classes):
    input_shape = (height, width, depth)
    chanDim = -1
    
    if K.image_data_format() == 'channels_first':
        input_shape = (depth,height,width)
        chanDim = 1
    
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3),padding="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(64, (3, 3),padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    
    model.add(Conv2D(64, (3, 3),padding="same", input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
# =============================================================================
#     model.add(Conv2D(128, (3, 3),padding="same"))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization(axis=chanDim))
#     
#     model.add(Conv2D(128, (3, 3),padding="same", input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
# =============================================================================
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(1,activation='sigmoid'))
    model.add(Activation('sigmoid'))
    
    return model

model = build_model(width = 96, height=96, depth = 3, classes =2)
optimizer = Adam(lr=learning_rate,decay = learning_rate/epochs)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001,
                               patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    


history = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,callbacks = [early_stopping])


scores = model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('gender_detection_model.model')

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('accuracy_plot.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.savefig('loss_plot.png')

