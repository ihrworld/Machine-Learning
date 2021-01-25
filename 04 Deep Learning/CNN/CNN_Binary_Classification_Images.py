############# CONVOLUTIONAL NEURAL NETWORKS (CNN) #############

## Import Required Packages 

#importing tensorflow and keras libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#importing numpy for vectors and arrays operations 
import numpy as np
#importing preprocessing package to apply feature scaling
from sklearn.preprocessing import LabelEncoder
#importing packages to work with images
from keras.preprocessing import image
import PIL
from PIL import Image
from IPython.display import Image


## 1 Data Preprocessing 
### 1.1 Preprocessing the training set 
train_datagen = ImageDataGenerator(rescale = 1./255, #feature scaling or normalization
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#connecting the image augmentation tool to the training set directory
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


### 1.2 Preprocessing the test set
#creating an image generator object, but without transforming the test set
test_datagen = ImageDataGenerator(rescale = 1./255) #feature scaling or normalization
#connecting the image augmentation tool to the test set directory
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


## 2 Building the CNN 

### 2.1. Initialising the CNN 
cnn = tf.keras.models.Sequential()

### 2.2. Step 1 - Convolution 
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

### 2.3. Step 2 - Pooling  
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### 2.4. Step 3 - Adding a second convolutional layer  
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

### 2.5. Step 4 - Flattening  
cnn.add(tf.keras.layers.Flatten())

### 2.6. Step 5 - Full Connection   
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

### 2.7. Step 6 - Output Layer   
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


## 3 Training the CNN 

### 3.1. Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### 3.2. Training the CNN
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

## 4 Making a single prediction 

###first test
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction1 = 'dog'
else:
  prediction1 = 'cat'

print(prediction1) #displaying the image

###second test
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction2 = 'dog'
else:
  prediction2 = 'cat'

print(prediction2)  #displaying the image


###third test
test_image = image.load_img('dataset/single_prediction/test.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction3 = 'dog'
else:
  prediction3 = 'cat'

print(prediction3) #displaying the image

