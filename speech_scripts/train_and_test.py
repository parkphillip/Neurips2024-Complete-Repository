import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from dataset import load_data
from model import get_model
from utils import plot_roc_curve, plot_confusion_matrix, save_logs, create_session_dirs
from sklearn.metrics import roc_auc_score

# Load data
X_train, X_test, y_train, y_test = load_data()

# Reshape data for the model
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define model
input_shape = (X_train.shape[1], 1)
model = get_model(input_shape)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
test_auc = roc_auc_score(y_test, y_pred, average='macro')

print(f'Test accuracy: {test_acc}')
print(f'Test AUC: {test_auc}')

# Create session-specific directories
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
session_dir = f'results/session_{timestamp}'
create_session_dirs(session_dir)

# Save model
model.save(os.path.join(session_dir, 'models', 'emotion_recognition_model.h5'))

# Save training logs
history.history['test_accuracy'] = [test_acc]  # Ensure test accuracy is saved as a list
history.history['test_auc'] = [test_auc]  # Ensure test AUC is saved as a list
save_logs(history, os.path.join(session_dir, 'logs', 'training_logs.json'))

# Save y_test and y_pred for aggregation
np.save(os.path.join(session_dir, 'logs', 'y_test.npy'), y_test)
np.save(os.path.join(session_dir, 'logs', 'y_pred.npy'), y_pred)

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(session_dir, 'visualizations', 'accuracy', 'accuracy.png'))
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(session_dir, 'visualizations', 'loss', 'loss.png'))
plt.show()

# Generate and save ROC curve
plot_roc_curve(y_test, y_pred, os.path.join(session_dir, 'visualizations', 'roc_curve', 'roc_curve.png'))

# Generate and save confusion matrix
plot_confusion_matrix(y_test, y_pred, os.path.join(session_dir, 'visualizations', 'confusion_matrix', 'confusion_matrix.png'))
