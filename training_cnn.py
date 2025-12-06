from matplotlib import plt as plt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import plot_model

# get datasets for training and test
train_ds=keras.utils.image_dataset_from_directory(
    directory="resources/dataset/train",
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)

val_ds=keras.utils.image_dataset_from_directory(
    directory="resources/dataset/val",
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)

# normalization
def process(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(process)
val_ds = val_ds.map(process)

# architecture
model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu', input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(train_ds, validation_data=val_ds, epochs=30)

# plotting training history
def plot_history(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

# inference
import cv2
def inference(model,test_img_path):
    test_img=cv2.imread(test_img_path)
    test_img=plt.imshow(test_img)
    test_img=cv2.resize(test_img,(256,256))
    test_input=test_img.reshape(1,256,256,3)
    pred=model.predict(test_input)
    if int(pred[0][0])==0:
        print("Pomme")
    else:
        print("Tomato")
inference(model,"resources/dataset/test/cat/cat.4000.jpg")


# Decontruction of the model
model = VGG16()
model.summary()
plot_model(model)
model.layers
# check filters and biases of all the conv layers
for i in range(len(model.layers)):
    layer=model.layers[i]
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(f"Layer {layer.name}:")
        print(f"Filters shape: {filters.shape}") # 3x3xnumber of input channels x number of filters
        print(f"Biases shape: {biases.shape}") 

# retrieve weights from the second hidden layer
filters,bias=model.layers[1].get_weights()
print(filters.shape)
print(bias.shape)

# normalize filter values to 0 to 1 to visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters=6
ix=1
fig=plt.figure(figsize=(15,10))
for i in range(n_filters):
    # get the filter
    f=filters[:,:,:,i]
    # plot each channel separately
    for j in range(3):
        ax=plt.subplot(n_filters,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:,:,j], cmap='gray')
        ix+=1
plt.show()



model=Model(inputs=model.inputs, outputs=model.layers[1].output)

image=load_img("resources/dataset/test/cat/cat.4000.jpg", target_size=(224,224))
image=img_to_array(image)
image=np.expand_dims(image, axis=0)
image=preprocess_input(image)
feature_maps=model.predict(image)
fig=plt.figure(figsize=(15,10))
for i in range(1,feature_maps.shape[-1]+1):
    ax=plt.subplot(8,8,i)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_maps[0,:,:,i-1], cmap='gray')

model2=VGG16()
layer_index=[2,5,9,13,17]
output=[model2.layers[i].output for i in layer_index]
model=Model(inputs=model2.inputs, outputs=output) # create new model that will return these outputs, given the model input
feature_maps=model.predict(image)
for i,fmap in zip(layer_index,feature_maps):
    fig=plt.figure(figsize=(20,15))
    fig.suptitle(f"Feature maps at layer {i}", fontsize=16)
    for i in range(1,fmap.shape[-1]+1):
        ax=plt.subplot(8,8,i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(fmap[0,:,:,i-1], cmap='gray') # to see how the images evolve after each conv layer

# data augmentation
img = image.load_img('/content/apple.jpeg', target_size=(200, 200))
plt.imshow(img)
datagen=ImageDataGenerator(rotation_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,width_shift_range=0.2,
                           height_shift_range=0.2)
img=image.img_to_array(img)
input_batch=img.reshape(1,200,200,3)
i=0
for batch in datagen.flow(input_batch, batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%5==0:
        break

# now we can apply the date augmentation in the training phase using the flow_from_directory method
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('resources/dataset/train',target_size=(224,224),
                                             batch_size=32,class_mode='binary')
test_set=test_datagen.flow_from_directory('resources/dataset/val',target_size=(224,224),
                                            batch_size=32,class_mode='binary')
history=model.fit(training_set,steps_per_epoch=len(training_set),epochs=20,validation_data=test_set,validation_steps=len(test_set))
plot_history(history)
model.evaluate(test_set)
model.save('cnn_model.h5')
model=load_model('cnn_model.h5')
inference(model,"resources/dataset/test/cat/cat.4000.jpg")

# transfer learning with pretrained models
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model=ResNet50(weights='imagenet')
img_path='resources/lena.png'
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
preds=model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
print("predictions completed",decode_predictions(preds, top=3)[0])


