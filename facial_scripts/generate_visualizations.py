import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
import tensorflow as tf
from datetime import datetime

def load_evaluation_results(session_dirs):
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    results = {class_label: {"y_test": [], "y_pred_proba": []} for class_label in class_labels}

    for session_dir in session_dirs:
        class_dirs = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
        for class_dir in class_dirs:
            y_test_path = os.path.join(session_dir, class_dir, 'y_test.npy')
            y_pred_proba_path = os.path.join(session_dir, class_dir, 'y_pred_proba.npy')
            
            if os.path.exists(y_test_path) and os.path.exists(y_pred_proba_path):
                y_test = np.load(y_test_path)
                y_pred_proba = np.load(y_pred_proba_path)
                class_label = class_dir.split('_')[-1]
                results[class_label]["y_test"].append(y_test)
                results[class_label]["y_pred_proba"].append(y_pred_proba)
    
    return results

def plot_roc_curves(results, result_dir):
    plt.figure(figsize=(10, 10))  # Make the figure square by setting both dimensions equal
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'purple']
    
    for class_label, color in zip(results.keys(), colors):
        y_test_concat = np.concatenate(results[class_label]["y_test"])
        y_pred_proba_concat = np.concatenate(results[class_label]["y_pred_proba"])
        fpr, tpr, _ = roc_curve(y_test_concat, y_pred_proba_concat)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'Class {class_label} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc='lower right', fontsize=18)
    plt.grid(True)
    plt.savefig(f'{result_dir}/roc_curve.png')
    plt.show()

    print(f'ROC curve saved to {result_dir}/roc_curve.png')

def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_val

def main():
    results_root = 'results/sessions'
    session_dirs = [os.path.join(results_root, d) for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]

    if not session_dirs:
        print("No session directories found.")
        return

    results = load_evaluation_results(session_dirs)

    result_dir = f'results/conclusions/mean_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(result_dir, exist_ok=True)

    # Plot ROC curves
    plot_roc_curves(results, result_dir)

if __name__ == "__main__":
    main()
