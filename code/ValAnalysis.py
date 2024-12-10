import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

def val_analysis(raw_file, out_dir='./', best=False, pLumA_file='../data/pLumA_values.csv'):
    """Analyze validation results and compute metrics."""
    
    # Control print statements based on 'best' flag
    if best:
        enablePrint()
    else:
        blockPrint()

    # Define class labels
    classes = ['basal', 'her2', 'luma', 'lumb']
    lines = list(open(raw_file))
    
    patch_list = []
    patient_list = []
    label_list = []
    Slabel_list = []
    preds_list = []
    max_prob_list = []

    # Process each line in the raw file
    for line in lines:
        line = line.strip()
        if line == "": continue
        if 'TCGA' in line:
            t = line.split("', '")
            for item in t:
                for char in "()',": 
                    item = item.replace(char, "")
                patch = item.split('/')[-1]
                patient = patch[:12]
                slabel = item.split('/')[-3]
                label1 = classes.index(slabel)
                label = 1 if label1 == 2 else 0
                patch_list.append(patch)
                patient_list.append(patient)
                label_list.append(label)
                Slabel_list.append(slabel)
        else:
            remove_str = [
                "tensor([[", 
                "]], device='cuda:0')", 
                "]], device='cuda:1')",
                "]], device='cuda:2')",
                "]], device='cuda:3')"
            ]
            for str_ in remove_str: 
                line = line.replace(str_, "")
            for char in "[],": 
                line = line.replace(char, "")
            preds = [float(s) for s in line.split(' ')]
            max_prob = preds.index(max(preds))
            preds_list.append(preds)
            max_prob_list.append(max_prob)
    
    dest_name = raw_file.split('/')[-1]        
    csv_file = open(os.path.join(out_dir, f'{dest_name}.csv'), 'w')
    csv_file.write("patch,patient,slabel,label,p0,p1,pred,ans\n")
    
    # Write results to CSV
    for i in range(len(patch_list)):
        row_ = str((
            patch_list[i], 
            patient_list[i], 
            Slabel_list[i],
            label_list[i], 
            preds_list[i], 
            max_prob_list[i], 
            int(max_prob_list[i] == label_list[i])
        ))

        for char in "()'[]": 
            row_ = row_.replace(char, "")
        csv_file.write(row_.replace(" ", "") + '\n')
    csv_file.close()
    
    df = pd.read_csv(os.path.join(out_dir, f'{dest_name}.csv'))
    unique_patients = df.patient.unique()

    pLumA_df = pd.read_csv(pLumA_file)
    
    patient_iluma = {}
    
    # Calculate iluma for each patient
    for patient in unique_patients:
        p_frame = df[df['patient'] == patient]
        positive_count = p_frame['pred'].sum()
        total_count = len(p_frame)
        iluma = positive_count / total_count
        patient_iluma[patient] = iluma
    
    pLumA_values = []
    iluma_values = []
    
    # Correlate pLumA values with iluma values
    for patient in unique_patients:
        if patient in pLumA_df['patient'].values:
            pLumA_values.append(pLumA_df[pLumA_df['patient'] == patient]['pLumA'].values[0])
            iluma_values.append(patient_iluma[patient])
    
    correlation, _ = pearsonr(iluma_values, pLumA_values)
    
    patch_cm = str(confusion_matrix(df['label'], df['pred']))
    patch_acc = str(np.mean(df.ans))

    # Print results
    print(f'Best Patch level accuracy = {patch_acc}\n\n' 
           + f'Best Patch Confusion Matrix = \n{patch_cm}\n\n'
           + f'Correlation between iluma and pLumA = {correlation}\n\n')

    return correlation

