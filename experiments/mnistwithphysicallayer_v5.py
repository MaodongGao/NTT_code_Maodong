# Added Physical Layer
# Added Split & Solve Algorithm


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns  # For a nicer confusion matrix visualization
#from google.colab import files
#from google.colab import drive
#drive.mount('/content/gdrive')
from config.config_manager import ConfigManager
from systemcontroller import SystemController
from matrix_multiplier import OpticalMVM


# Load default configuration
default_sysconfig = ConfigManager.load_default_config()


sysconfig = default_sysconfig # basic system configuration

grid_config_slm1 = {
        "matrixsize_0": 10,
        "matrixsize_1": 8,
        "elem_width": 18,
        "elem_height": 17,
        "topleft_x": 900,
        "topleft_y": 540,
        "gap_x": 2,
        "gap_y": 3
    }
# 1280, 1920
grid_config_slm2 = {
        "matrixsize_0": 10,
        "matrixsize_1": 10,
        "elem_width": 18,
        "elem_height": 21,
        "topleft_x": 880,
        "topleft_y": 480,
        "gap_x": 2,
        "gap_y": 1
    }


grid_config_img = {
        "matrixsize_0": 10,
        "matrixsize_1": 8,
        "elem_width": 11,
        "elem_height": 15,
        "topleft_x": 550,
        "topleft_y": 325,
        "gap_x": 2,
        "gap_y": 7 }


cam_config = {
        "bitDepth": 14,
        "ExposureTime": 4000,
        "Analog_gain": "High",
        "FPS": "30",
        "width": "1280",
        "height": "1024",
        "offset_x": "0",
        "offset_y": "0",
        "NUC_path": "C:\\Users\\NTTRi LAB\\optical_matrix_multiplier\\hardware\\NUCs\\NUC_4000us.yml"
    }

SC = SystemController(default_sysconfig)

omvm = OpticalMVM(SC, cam_config, grid_config_slm1, grid_config_slm2, grid_config_img)







class NoisyLayer(layers.Layer):
    def __init__(self, layer, noise_stddev, **kwargs):
        super(NoisyLayer, self).__init__(**kwargs)
        self.layer = layer
        self.noise_stddev = noise_stddev

    def build(self, input_shape):
        self.layer.build(input_shape)

    def call(self, inputs, training=None):
        output = self.layer(inputs, training=training)
        if not training:  # Only add noise during inference (testing)
            noise = tf.random.normal(shape=tf.shape(output), mean=0., stddev=self.noise_stddev)
            output = output + noise
        return output

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

class SimulatedPhysicalLayer(tf.keras.layers.Layer):
    def __init__(self, units=10, activation=None, **kwargs):
        super(SimulatedPhysicalLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Initialize the weights
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='kernel')
        # Initialize the biases
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='bias')
        super().build(input_shape)

    def call(self, inputs):
        # Perform the matrix-vector multiplication
        z = tf.matmul(inputs, self.w) + self.b
        # Apply the activation function (if any)
        return self.activation(z)

    # Corrected compute_output_shape method
    def compute_output_shape(self, input_shape):
        # Assuming input_shape is a tuple representing (batch_size, ..., input_dim),
        # and the output shape should be (batch_size, ..., units)
        return input_shape[:-1] + (self.units,)

#class PhysicalLayer(tf.keras.layers.Layer):
#    def __init__(self, weight_matrix, bias_vector, **kwargs):
#        super(PhysicalLayer, self).__init__(**kwargs)
#        self.weight_matrix = tf.constant(weight_matrix)
#        self.bias_vector = tf.constant(bias_vector)
#
#    def call(self, inputs):
#        x = tf.matmul(inputs, self.weight_matrix) + self.bias_vector
#        # matrix-matrix multiplication using experimental setup: multiply(self, matrix1, matrix2)
#        # x = multiply(inputs, self.weight_matrix) + self.bias_vector
#        return tf.nn.relu(x)

class PhysicalLayer_Split_Solve(tf.keras.layers.Layer):
    def __init__(self, weight_matrix, bias_vector, **kwargs):
        super(PhysicalLayer_Split_Solve, self).__init__(**kwargs)
        # Assume weight_matrix and bias_vector are TensorFlow tensors
        self.weight_matrix = weight_matrix
        self.bias_vector = bias_vector

    def call(self, inputs):
        # Maximum size each split can be, based on hardware limitations
        max_size = 10
        num_splits_input = inputs.shape[1] // max_size + (1 if inputs.shape[1] % max_size else 0)
        num_splits_weight = self.weight_matrix.shape[0] // max_size + (1 if self.weight_matrix.shape[0] % max_size else 0)

        # Verify that the number of splits for input and weight matrices match
        assert num_splits_input == num_splits_weight, "Mismatch in the number of splits between input and weight matrices"

        # Splitting the input matrix and weight matrix
        input_splits = tf.split(inputs, num_or_size_splits=num_splits_input, axis=1)
        weight_splits = tf.split(self.weight_matrix, num_or_size_splits=num_splits_weight, axis=0)
        # print(input_splits[1].shape)
        # print(weight_splits[1].shape)
        # Initialize a list to store the results of multiplication
        results = []

        # Iterate over the splits, perform multiplication, and aggregate the results
        for i in range(num_splits_input):
            # result_split = tf.matmul(input_splits[i], weight_splits[i])
            # matrix-matrix multiplication using experimental setup: multiply(self, matrix1, matrix2)
            # result_split = multiply(input_splits[i], weight_splits[i])
            
            result_split = omvm.multiply(input_splits[i], weight_splits[i])
            results.append(result_split)

        # Sum the results of the multiplication
        result = tf.add_n(results) #+ self.bias_vector

        return tf.nn.relu(result)


######################
###### TRAINING ######
######################


# Define a function to create and train a model with a given level of noise
def train_model(noise_stddev):
    # Build the LeNet-5 model with noise
    model = models.Sequential()
    model.add(NoisyLayer(layers.Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)), noise_stddev=noise_stddev))
    model.add(NoisyLayer(layers.AveragePooling2D(), noise_stddev=noise_stddev))
    model.add(NoisyLayer(layers.Conv2D(16, (5, 5), activation='relu'), noise_stddev=noise_stddev))
    model.add(NoisyLayer(layers.AveragePooling2D(), noise_stddev=noise_stddev))
    model.add(NoisyLayer(layers.Flatten(), noise_stddev=noise_stddev))
    model.add(NoisyLayer(layers.Dense(120, activation='relu'), noise_stddev=noise_stddev))
    model.add(NoisyLayer(layers.Dense(80, activation='relu'), noise_stddev=noise_stddev))

    # Add the physical layer
    model.add(SimulatedPhysicalLayer(10, activation='relu')) # a new Dense layer with 32 units and ReLU activation

    # Continue with the final output layer
    model.add(NoisyLayer(layers.Dense(10, activation='softmax'), noise_stddev=noise_stddev))  # 10 classes for MNIST (0-9 digits)

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=1,
                        validation_data=(test_images, test_labels))

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    return model, test_acc




# Load and prepare the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# Add a channel dimension to the images
#train_images = train_images[..., tf.newaxis]
#test_images = test_images[..., tf.newaxis]

# Downsample the images from 28x28 to 10x10
#train_images = tf.image.resize(train_images, [14, 14], method='bilinear')
#test_images = tf.image.resize(test_images, [14, 14], method='bilinear')

# Reshape the data to include the channel dimension
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Noise levels to test
#noise_levels = np.linspace(0, 0.3, 31)  # 0%, 1%, ..., 30%
noise_levels = np.linspace(0, 0.05, 2)  # 0%, 5%

# Initialize lists to store models and their accuracies
models_with_noise = []
acc_with_noise = []

# Train and test models with noise, storing both models and accuracies
for noise_stddev in noise_levels:
    model, accuracy = train_model(noise_stddev)
    models_with_noise.append(model)
    acc_with_noise.append(accuracy)

# Save the results to a DataFrame
df = pd.DataFrame({'Noise Level': noise_levels, 'Accuracy': acc_with_noise})

# Define your Google Drive path
#drive_path = '/content/gdrive/My Drive/'

# Save the DataFrame to a CSV file on your Google Drive
#df.to_csv(drive_path + 'accuracy_results.csv', index=False)


# Save the results to a CSV file
#df = pd.DataFrame({'Noise Level': noise_levels, 'Accuracy': acc_with_noise})
#df.to_csv('accuracy_results.csv', index=False)
#files.download(drive_path + 'accuracy_results.csv')

# Read the data from the .csv file
#df = pd.read_csv(drive_path + 'accuracy_results.csv')

# Extract the noise levels and accuracies
#noise_levels = df['Noise Level']
#acc_with_noise = df['Accuracy']

# Plot the results
#plt.figure(figsize=(10, 6))
#index = np.arange(len(noise_levels))

#rects = plt.bar(index, acc_with_noise, color='grey', label='With Noise')

#plt.xlabel('Noise Level')
#plt.ylabel('Accuracy')
#plt.title('Accuracy with noise at different noise levels')
#plt.xticks(index, noise_levels)

#plt.tight_layout()


######################
###### Inference #####
######################


# Assuming test_images, test_labels are already defined and model is trained

chosen_model = models_with_noise[0] # Choose models trained with different noise level

# Assuming test_images, test_labels are already defined and model is trained
# Pick N images
N = 8 # Inference batch size
start_index = 0 # Starting index of the images you want to predict
images_to_predict = test_images[start_index:start_index+N]
true_labels = test_labels[start_index:start_index+N]

# The images are already in batch form if you slice like this, no need to add an extra dimension

# Find index of the PhysicalDenseLayer
for i, layer in enumerate(chosen_model.layers):
    if isinstance(layer, SimulatedPhysicalLayer):
        index_physical_dense_layer = i
        break

# Create a new model that goes up to and includes the PhysicalDenseLayer
model_up_to_physical_layer = tf.keras.Model(inputs=chosen_model.input,
                                            outputs=chosen_model.layers[index_physical_dense_layer-1].output)

model_after_physical_layer = tf.keras.Model(inputs=chosen_model.layers[index_physical_dense_layer].output,
                                            outputs=chosen_model.output)


physical_dense_layer_weights = chosen_model.layers[index_physical_dense_layer].get_weights()
weight_matrix = physical_dense_layer_weights[0]  # Weight matrix
bias_vector = physical_dense_layer_weights[1]    # Bias vector
print("Shape weight_matrix in PhysicalDenseLayer:", weight_matrix.shape)

# Predict with the modified model
input_of_physical_dense_layer = model_up_to_physical_layer.predict(images_to_predict)

#output_of_physical_dense_layer = PhysicalDenseLayer(units=15, activation='relu')(input_of_physical_dense_layer)
output_of_physical_dense_layer = tf.nn.relu( tf.matmul(input_of_physical_dense_layer,weight_matrix)+bias_vector )

final_predictions = model_after_physical_layer.predict(output_of_physical_dense_layer)

# Final prediction with the original model
#final_predictions = chosen_model.predict(images_to_predict)
predicted_labels = np.argmax(final_predictions, axis=1)

# Output shapes and content
print("###############################################################")
print("Physical Layer Input Shape:", input_of_physical_dense_layer.shape)
print("Physical Layer Output Shape:", output_of_physical_dense_layer.shape)
print("Final Output Shape:", final_predictions.shape)
print("###############################################################")


# Combine the layers and build a new model

inputs = tf.keras.Input(shape=chosen_model.input_shape[1:])

# Assume 'chosen_model' is the trained model, and we're recreating it
x = inputs
for layer in chosen_model.layers[:index_physical_dense_layer]:  # Up to and including PhysicalDenseLayer
    x = layer(x)


new_model_before = tf.keras.Model(inputs=inputs, outputs=x)
output_new_model_before = new_model_before.predict(images_to_predict)

print(weight_matrix)
print(bias_vector)
print(output_new_model_before)

####################################################
# Insert the physical layer processing matrix-matrix multiplication: [8 x 84] x [84 x 10] = [8 x 10]
#x = PhysicalLayer(weight_matrix, bias_vector)(x)
output_physical_layer = PhysicalLayer_Split_Solve(np.clip(weight_matrix, 0, 1), bias_vector)(np.clip(output_new_model_before, 0, 1))
####################################################

#x = inputs.numpy()
print("Shape:", output_physical_layer.shape)

output_physical_layer = tf.reshape(output_physical_layer, (-1, 8, 10))

#inputs = tf.keras.Input(shape=chosen_model.input_shape[index_physical_dense_layer+1:])
#inputs = tf.keras.Input(shape=(10,8))
inputs2 = tf.keras.Input(shape = output_physical_layer.shape)
inputs2 = tf.squeeze(inputs2, axis=1)
print("Shape:", inputs2.shape)
x = inputs2
# Continue with layers after the PhysicalDenseLayer
for layer in chosen_model.layers[index_physical_dense_layer+1:]:
    x = layer(x)

# Finalize the model
new_model = tf.keras.Model(inputs=inputs2, outputs=x)

#output_physical_layer_reshaped = tf.expand_dims(output_physical_layer, axis=0)

final_predictions = new_model.predict(output_physical_layer)
predicted_labels = np.argmax(final_predictions, axis=1)
predicted_labels = np.squeeze(predicted_labels)
print(predicted_labels)
print(predicted_labels.shape)

# Display the images with true and predicted labels
plt.figure(figsize=(15,5))
for i in range(N):
    plt.subplot(2,5,i+1)
    plt.imshow(images_to_predict[i].reshape(28, 28), cmap='gray')
    #plt.imshow(images_to_predict[i], cmap='gray')
    plt.title(f"True: {true_labels[i]}, Pred: {predicted_labels[i]}")
    plt.axis('off')
plt.tight_layout()
print("BBBBBBBBBBBBBBBBBBBBBBBBB")
plt.show()


# output_new_model_before = new_model_before.predict(test_images)
# output_physical_layer = PhysicalLayer_Split_Solve(weight_matrix, bias_vector)(output_new_model_before)
# predictions = new_model.predict(output_physical_layer)

# predicted_labels = np.argmax(predictions, axis=1) # Convert predictions to class labels

# Compute confusion matrix
#cm = confusion_matrix(true_labels, predicted_labels)
#cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100 # Normalize the confusion matrix to percentages

# Plot confusion matrix as percentages
#plt.figure(figsize=(10, 7))
#sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
#plt.xlabel('Predicted Label')
#plt.ylabel('True Label')
#plt.title('Confusion Matrix (%)')
#plt.show()



