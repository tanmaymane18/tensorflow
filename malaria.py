#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = "E:\Machine Learning\kaggle\malaria\cell-images-for-detecting-malaria\cell_images"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=128,
    class_mode='binary'
)


# In[5]:


test_datagen = ImageDataGenerator(rescale=1./255)
validation_dir = r'E:\Machine Learning\kaggle\malaria\cell-images-for-detecting-malaria\test'
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)


# In[6]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), input_shape=(100, 100, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[7]:


model.summary()


# In[8]:


from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])


# In[9]:


model.fit(train_generator, epochs=10, batch_size=32)


# In[9]:


model.save("malaria")


# In[1]:


print(model.evaluate(validation_generator))


# In[7]:


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')


# In[ ]:





# In[ ]:




