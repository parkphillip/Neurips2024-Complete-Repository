import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_curve, auc
from datetime import datetime

# Function to validate images
def validate_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except (IOError, SyntaxError, UnidentifiedImageError):
        return False

# Function to get valid image paths and labels
def get_valid_image_paths_and_labels(directory):
    valid_image_paths = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if validate_image(file_path):
                valid_image_paths.append(file_path)
                labels.append(os.path.basename(root))
            else:
                print(f"Invalid image file: {file_path}")
    return valid_image_paths, labels

# Directories
trainDir = './data/fer2013/train/'
testDir = './data/fer2013/test/'

# Function to train and evaluate model
def train_and_evaluate(run_id, class_label, train_df, test_df):
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create binary labels for the current class
    train_df['binary_class'] = (train_df['class'] == class_label).astype(int).astype(str)
    test_df['binary_class'] = (test_df['class'] == class_label).astype(int).astype(str)

    # Create data generators
    train_dataset = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='binary_class',
        target_size=(48, 48),
        class_mode='binary',
        subset='training',
        batch_size=64,
        shuffle=True)

    valid_dataset = valid_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='binary_class',
        target_size=(48, 48),
        class_mode='binary',
        subset='validation',
        batch_size=64,
        shuffle=True)

    test_dataset = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='binary_class',
        target_size=(48, 48),
        class_mode='binary',
        batch_size=64,
        shuffle=False)

    # Build the model with VGG16 as the base
    base_model = VGG16(input_shape=(48, 48, 3), include_top=False, weights="imagenet")

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    # Unfreeze the last few layers of VGG16
    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True

    model.summary()

    # Custom F1 score metric
    def f1_score(y_true, y_pred):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    # Metrics and callbacks
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        f1_score,
    ]

    # Callbacks
    result_dir = f'results/sessions/session_{run_id}/class_{class_label}'
    os.makedirs(result_dir, exist_ok=True)
    lrd = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.50, min_lr=1e-10)
    mcp = ModelCheckpoint(f'{result_dir}/model.keras', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=METRICS)

    # Train the model
    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs=50,
                        verbose=1,
                        callbacks=[lrd, mcp, es])

    # Evaluate on test dataset and save the labels and predictions
    y_test = np.concatenate([test_dataset[i][1] for i in range(len(test_dataset))])
    y_pred_proba = model.predict(test_dataset)

    # Save the test dataset labels and predictions
    np.save(f'{result_dir}/y_test.npy', y_test)
    np.save(f'{result_dir}/y_pred_proba.npy', y_pred_proba)

    # Evaluate on test dataset
    test_loss, test_acc, test_precision, test_recall, test_auc, test_f1 = model.evaluate(test_dataset)
    print(f'Test Loss for class {class_label}: {test_loss}')
    print(f'Test Accuracy for class {class_label}: {test_acc}')
    print(f'Test Precision for class {class_label}: {test_precision}')
    print(f'Test Recall for class {class_label}: {test_recall}')
    print(f'Test AUC for class {class_label}: {test_auc}')
    print(f'Test F1 Score for class {class_label}: {test_f1}')

    # Save evaluation results
    evaluation_results = {
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Test Precision': test_precision,
        'Test Recall': test_recall,
        'Test AUC': test_auc,
        'Test F1 Score': test_f1
    }
    with open(f'{result_dir}/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test.astype(float), y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Save ROC AUC results
    roc_auc_df = pd.DataFrame({'Class': [class_label], 'ROC AUC': [roc_auc]})
    roc_auc_df.to_csv(f'{result_dir}/roc_auc.csv', index=False)

    # Print the run ID
    print(f'Run ID: {run_id}, Class: {class_label}')

    return test_acc

# Run multiple iterations
num_iterations = int(input("Enter number of iterations: "))
train_image_paths, train_labels = get_valid_image_paths_and_labels(trainDir)
test_image_paths, test_labels = get_valid_image_paths_and_labels(testDir)

# Convert to DataFrame for easier handling
train_df = pd.DataFrame({'filename': train_image_paths, 'class': train_labels})
test_df = pd.DataFrame({'filename': test_image_paths, 'class': test_labels})

for i in range(num_iterations):
    run_id = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}-{i}'
    unique_classes = train_df['class'].unique()
    accuracies = []

    for class_label in unique_classes:
        acc = train_and_evaluate(run_id, class_label, train_df, test_df)
        accuracies.append(acc)

    # Calculate aggregate metrics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # Save aggregate metrics
    aggregate_results = {
        'Mean Test Accuracy': mean_accuracy,
        'Standard Deviation': std_accuracy
    }
    aggregate_result_dir = f'results/sessions/session_{run_id}'
    os.makedirs(aggregate_result_dir, exist_ok=True)
    with open(f'{aggregate_result_dir}/aggregate_evaluation_results.json', 'w') as f:
        json.dump(aggregate_results, f)

    print(f'Mean Test Accuracy for run {run_id}: {mean_accuracy}')
    print(f'Standard Deviation for run {run_id}: {std_accuracy}')
