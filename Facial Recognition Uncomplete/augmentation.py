
from __future__ import annotations
from msilib.schema import BBControl
import albumentations as alb
import cv2
import tensorflow as tf
from keras.applications import VGG16
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, GlobalMaxPooling2D, Dense, GlobalMaxPooling2D
 

augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2), alb.RandomGamma(p=0.2),
                          alb.RGBShift(p=0.2), alb.VerticalFlip(p=0.5)],
                            bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

img = cv2.imread(os.path.join('data', 'train', 'images','fa376321-03f1-11ed-8c59-cde093f4fcc6.jpg'))
with open(os.path.join('data','train','labels','fa376321-03f1-11ed-8c59-cde093f4fcc6.json'),'r') as f:
    label = json.load(f)

coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords = list(np.divide(coords, [640,480,640,480]))

augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
 
cv2.rectangle(augmented['image'], tuple(np.multiply(augmented['bboxes'][0][:2], [450, 450]).astype(int)), 
                            tuple(np.multiply(augmented['bboxes'][0][2:], [450, 450]).astype(int)), (255,0,0), 2)
plt.imshow(augmented['image'])

for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
      img  = cv2.imread(os.path.join('data', partition, 'images', image))

      coords = [0,0,0.00001,0.00001]
      label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
      if os.path.exists(label_path):
        with open(label_path, 'r') as f:
          label = json.load(f)
        
        coords[0] = label['shapes'][0]['points'][0][0]
        coords[1] = label['shapes'][0]['points'][0][1]
        coords[2] = label['shapes'][0]['points'][1][0]
        coords[3] = label['shapes'][0]['points'][1][1]
        coords = list(np.divide(coords, [640,480,640,480]))


      try:
        for x in range(75):
          augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
          cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

          annotation = {}
          annotation['image'] = image

          if os.path.exists(label_path):
            if len(augmented['bboxes']) ==0:
              annotation['bbox'] = [0,0,0,0]
              annotation['class'] = 0

            else:
              annotation['bbox'] = augmented['bboxes'][0]
              annotation['class'] = 0

            with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
              json.dump(annotation, f)
      
      except Exception as e:
        print(e)

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images= train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images= train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images= test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images= test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images= val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images= val_images.map(lambda x: x/255)



def load_labels(label_path):
  with open(label_path.numpy(), 'r', encoding= "utf-8") as f:
    label = json.load(f)
  return [label['class']], label['bbox']


train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle = False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle = False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle = False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))



train = tf.data.Dataset.zip((train_images,train_labels))
train = train.shuffle(5000)
train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = test.shuffle(1000)
val.batch(8)
val = test.prefetch(4)

train.as_numpy_iterator().next()[1]



data_sample = train.as_numpy_iterator()

res = data_sample.next()

vgg = VGG16(include_top= False)


# building Neural Network
def build_model():
  input_layer = Input(shape=(120,120,3))

  vgg = VGG16(include_top=False)(input_layer)

  #Classification Model
  f1 = GlobalMaxPooling2D()(vgg)
  class1 = Dense(2048, activation='relu')(f1)
  class2 = Dense(1,activation='sigmoid')(class1)
  
  #Regression Model (Bounding Box Model)
  f2 = GlobalMaxPooling2D()(vgg)
  regress1 = Dense(2048,activation='relu')(f2)
  regress2 = Dense(4, activation='sigmoid')(regress1)

  facetracker = Model(inputs = input_layer, outputs=[class2, regress2])
  return facetracker


facetracker = build_model()

X, y = res  

X = np.expand_dims(X, axis= 0)

classes, coords = facetracker.predict(X)

#Defining Losses and Optimizers

batches_per_epoch = len(train)
lr_decay = (1./0.75 - 1)/batches_per_epoch

opt = tf.keras.optimizers.Adam(learning_rate = 0.0001, decay = lr_decay)

def localization_loss(y_true, yhat):
  delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))

  print(y_true.shape)

  h_true = y_true[:,3] - y_true[:,1]
  w_true = y_true[:,2] - y_true[:,0]

  h_pred = yhat[:,3] - yhat[:,1]
  w_pred = yhat[:,2] - yhat[:,0]

  delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))

  return delta_coord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

#Train Neural Network

class FaceTracker(Model):
  def __init__(self, facetracker, **kwargs):
    super().__init__( **kwargs)
    self.model = facetracker

  def compile(self, opt, classloss, localizationloss, **kwargs):
    super().compile(**kwargs)
    self.closs = classloss
    self.lloss = localizationloss
    self.opt = opt

  def train_step(self, batch, **kwargs):
    X, y = batch

    X = tf.reshape(X, [120,120,3])
    X = tf.expand_dims(X, axis=0)

    with tf.GradientTape() as tape:
      classes, coords = self.model(X, training = True)
      
      batch_classloss= self.closs(y[0], classes)
      batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

      total_loss = batch_localizationloss + 0.5*batch_classloss

      grad = tape.gradient(total_loss, self.model.trainable_variables)

    opt.apply_gradients(zip(grad, self.model.trainable_variables))

    return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
  
  def test_step(self,batch, **kwargs):
    X, y = batch

    classes, coords = self.model(X, training =False)

    batch_classloss = self.closs(y[0], classes)
    batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
    total_loss = batch_localizationloss + 0.5*batch_classloss

    return{"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}

  def call(self, X, **kwargs):
    return self.model(X, **kwargs)

model = FaceTracker(facetracker)

model.compile(opt, classloss, regressloss)

logdir = 'logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=40, validation_data = val, callbacks= [tensorboard_callback])




