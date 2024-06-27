# computing mean test accuracy/auc and std for both

import os
import json
import numpy as np
import pandas as pd

def compute_statistics():
    session_dirs = [d for d in os.listdir('results/sessions') if os.path.isdir(os.path.join('results/sessions', d))]
    per_run_accuracies = []
    per_run_aucs = []

    for session_dir in session_dirs:
        accuracies = []
        aucs = []
        class_dirs = [d for d in os.listdir(os.path.join('results/sessions', session_dir)) if os.path.isdir(os.path.join('results/sessions', session_dir, d))]
        
        for class_dir in class_dirs:
            evaluation_results_path = os.path.join('results/sessions', session_dir, class_dir, 'evaluation_results.json')
            if os.path.exists(evaluation_results_path):
                with open(evaluation_results_path, 'r') as f:
                    evaluation_results = json.load(f)
                    accuracies.append(evaluation_results['Test Accuracy'])
                    aucs.append(evaluation_results['Test AUC'])
        
        if accuracies:
            per_run_accuracies.append(np.mean(accuracies))
        if aucs:
            per_run_aucs.append(np.mean(aucs))

    if per_run_accuracies:
        mean_accuracy = np.mean(per_run_accuracies)
        std_accuracy = np.std(per_run_accuracies)
        print(f'Mean Binary Test Accuracy: {mean_accuracy}')
        print(f'Standard Deviation of Binary Test Accuracy: {std_accuracy}')
    else:
        print("No evaluation results found for accuracy.")

    if per_run_aucs:
        mean_auc = np.mean(per_run_aucs)
        std_auc = np.std(per_run_aucs)
        print(f'Mean Test AUC: {mean_auc}')
        print(f'Standard Deviation of Test AUC: {std_auc}')
    else:
        print("No evaluation results found for AUC.")

compute_statistics()
