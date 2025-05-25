# Enhancing-lumpy-skin-disease-through-deep-learning-algorithms
#Importing Libraries
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout,GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Base Path for all files
data_dir = 'drive/MyDrive/ds1/archive/Lumpy Skin Images Dataset'
###### Using ImageDataGenerator to load the Images for Training and Testing the CNN Model
datagenerator = {
 "train": ImageDataGenerator(horizontal_flip=True,
 vertical_flip=True,
 rescale=1. / 255,
 validation_split=0.1,
 shear_range=0.1,
 zoom_range=0.1,
 width_shift_range=0.1,
height_shift_range=0.1,
 rotation_range=30,
 ).flow_from_directory(directory=data_dir,
 target_size=(256, 256),
 subset='training',
 ),
 "valid": ImageDataGenerator(rescale=1 / 255,
 validation_split=0.1,
 ).flow_from_directory(directory=data_dir,
 target_size=(256, 256),
 subset='validation',
 ),
}
base_model = DenseNet121(weights=None, include_top=False, input_shape=(256, 256, 3))
# Load Weights for the DenseNet121 Model
base_model.load_weights('drive/MyDrive/ds1/DenseNet-BC-121-32-no-top.h5')
# Setting the Training of all layers of InceptionV3 model to false
base_model.trainable = False
# Adding some more layers at the end of the Model as per our requirement
model = Sequential([
 base_model,
 GlobalAveragePooling2D(),
 Dropout(0.15),
 Dense(1024, activation='relu'),
 Dense(2, activation='softmax') # 10 Output Neurons for 10 Classes
])

# Using the Adam Optimizer to set the learning rate of our final model
opt = optimizers.Adam(learning_rate=0.0001)
# Compiling and setting the parameters we want our model to use
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True)
# Setting variables for the model
batch_size = 32
epochs = 10
# Seperating Training and Testing Data
train_generator = datagenerator["train"]
valid_generator = datagenerator["valid"]
# Calculating variables for the model
steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size
# File Path to store the trained models
filepath = "drive/MyDrive/ds1/model_{epoch:02d}-{val_accuracy:.2f}.h5"
# Using the ModelCheckpoint function to train and store all the best models
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint1]
# Training the Model
history = model.fit_generator(generator=train_generator, epochs=epochs,
steps_per_epoch=steps_per_epoch,validation_data=valid_generator, validation_steps=validation_steps,callbacks=callbacks_list)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
# ________________ Graph 1 -------------------------
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
# ________________ Graph 2 -------------------------
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()
 
# Calculate the Loss and Accuracy on the Validation Data

test_loss, test_acc = model.evaluate(valid_generator)
print('test accuracy : ', test_acc)
User Interface:
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
# Load the pre-trained model
loaded_best_model = tf.keras.models.load_model("model_08-0.95.h5")
# Streamlit app
def main():
 st.title("Lumpy Skin Disease Prediction")
 # File uploader widget
 file = st.file_uploader("Upload Image", type=["jpg", "png"])
 if file is not None:a
 # Display the uploaded image
 img = image.load_img(file, target_size=(256, 256))
 img_array = image.img_to_array(img) / 255.0
 st.image(img_array, caption="Uploaded Image", use_column_width=True)
 # Make predictions when the "Predict" button is clicked
 if st.button("Predict"):
 # Reshape the image array
 img_array = np.expand_dims(img_array, axis=0)
 # Get predictions from the model
 predictions = loaded_best_model.predict(img_array)
 # Map numerical class to labels
 class_labels = {0: 'Lumpy Skin', 1: 'Normal Skin'}
 
 # Display the maximum probability and classified label
 predicted_class_label = class_labels[np.argmax(predictions[0])]
 st.write(f"Maximum Probability: {np.max(predictions[0], axis=-1)}")
 st.write(f"Classified: {predicted_class_label}")
 # Display individual class probabilities in a bar chart
 class_probabilities = [round(prob * 100, 2) for prob in predictions[0]]
 st.write("\n-------------------Individual Probability--------------------------------\n")
 for label, prob in zip(class_labels.values(), class_probabilities):
 st.write(f"{label.upper()}: {prob}%")
 # Plot bar chart with class labels on x-axis
 st.bar_chart({label: prob for label, prob in zip(class_labels.values(), class_probabilities)})
if __name__ == "__main__":
main()
