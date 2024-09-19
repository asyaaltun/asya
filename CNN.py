import tensorflow as tf
import numpy as np
import keras.utils as image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, #0 ile 255 arrasındakileri 0 ile 1 aralıgına ceker
                                   shear_range=0.2, # görüntülere rastgele kesme dönüşümler uygular
                                   zoom_range=0.2, #görüntülere rastgele yakınlaştırma uygular
                                   horizontal_flip= True # görüntüyü yatayda çevirir
                                   )

train_dataset = train_datagen.flow_from_directory("dataset_beyin/train", # yol
                                                   target_size=(64,64), # resimlerin boyutu
                                                   batch_size=32,  #tek seferde işlenecek resim miktarı
                                                   class_mode="binary" #ikili sınıflandırma
                                                     )

test_datagen = ImageDataGenerator(rescale=1./255)
test_dataset = test_datagen.flow_from_directory("dataset_beyin/test",
                                                target_size=(64, 64),  # resimelrin boyutu
                                                batch_size=32,  # tek seferde işlenecek reism miktarı
                                                class_mode="binary"
                                                )

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, # bu filtreler goruntude farklı ozelliği öğrenir
                                 kernel_size=3,  # filtrelerin boyutu
                                 activation="relu", #aktivasyon kodu
                                 input_shape=[64,64,3])) #girdi resmin boyutları

model.add(tf.keras.layers.MaxPool2D(pool_size=2, # havuzlama pencere boyutu
                                    strides=2)) #havuzlama penceresinin kayma adımı

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Flatten()) #resmi tek boyut haline geitrir

model.add(tf.keras.layers.Dense(units=128,activation="relu"))
model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
print( model.summary())

model.fit(x=train_dataset, validation_data=test_dataset,epochs=200)
loss, acc = model.evaluate(test_dataset)
print("acc:",acc,"loss:",loss)

test_data = image.load_img("dataset_beyin/deneme/no9.png",target_size=(64,64)) #resmi yukler
test_data = image.img_to_array(test_data) #resmi arraye cevir
test_data = np.expand_dims(test_data,axis=0)

output = model.predict(test_data)
train_dataset.class_indices # train içindeki sınıf indexlerini verir

if output[0][0]==1:
    print("tümör var")
else:
    print("tümör yok")

#yapılan modeli kaydetme
import time
model.save("model_cnn.h5")
model.save("model_cnn"+time.strftime("%d.%m.%Y-- %H.%M.%S")+ ".h5")

