import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import os

# Define the f1_score function for loading the model
def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_val

# Function to compute saliency map
def compute_saliency_map(model, img_array):
    img_array = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array)
        top_pred_index = tf.argmax(predictions[0])
        top_class = predictions[:, top_pred_index]
    grads = tape.gradient(top_class, img_array)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    return dgrad_max_

# Load the model
model_path = f'results/sessions-10/session_20240623-200825-0/model.keras'
model = load_model(model_path, custom_objects={'f1_score': f1_score})

# Load and preprocess the image
image_path = input("enter image path")
img = load_img(image_path, target_size=(48, 48))
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array_expanded = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Compute the saliency map
saliency_map = compute_saliency_map(model, img_array_expanded)

# Apply Gaussian smoothing to the saliency map
saliency_map_smoothed = gaussian_filter(saliency_map, sigma=1)

# Enhance the contrast of the saliency map
saliency_map_smoothed = (saliency_map_smoothed - saliency_map_smoothed.min()) / (saliency_map_smoothed.max() - saliency_map_smoothed.min())

# Plot the saliency map
plt.figure(figsize=(6, 6))
plt.imshow(img_array.squeeze(), cmap='gray')  # Display the original image in gray scale
plt.imshow(saliency_map_smoothed, cmap='rainbow', alpha=0.4)  # Overlay the saliency map with the rainbow colormap and higher transparency
plt.colorbar()
plt.title('Saliency Map')

# Save the saliency map
result_dir = f'results/sessions-10/session_20240623-200825-0/maps/'
os.makedirs(result_dir, exist_ok=True)
saliency_map_path = os.path.join(result_dir, 'saliency_map.png')
plt.savefig(saliency_map_path)
plt.show()

print(f'Saliency map saved to {saliency_map_path}')
