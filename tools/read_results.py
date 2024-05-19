import os
import pandas as pd

path = 'results'

for root, dnames, fnames in os.walk(path):
    for dname in dnames:
        if dname[0:5] == 'TCGA_':
            
            file_path = os.path.join(root, dname, 'train_test_summary/LeNet5/just_Student/')
            test_file_name = 'test_confidence_intervals.csv'
            val_file_name = 'val_confidence_intervals.csv'
            
            if not os.path.exists(os.path.join(file_path, test_file_name)):
                continue

            f = pd.read_csv(os.path.join(file_path, test_file_name))

            auc = f['#AUC']
            upper = f['#upper']
            lower = f['#lower']

            print(auc, upper, lower)
