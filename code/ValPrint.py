import sys
import os
from ValAnalysis import val_analysis
import glob
from tqdm import tqdm
import shutil

# Initialize the best patient correlation
best_patient_correlation = -1.0  

# Path from the command-line argument
path = sys.argv[1]

# Get all raw files from the specified path
full_raw = glob.glob(path + '/*')

# Destination path for the processed CSV files
dest = os.path.join(path.rsplit('/', 2)[0], 'raw_to_CSV', path.rsplit('/', 2)[2])

# Create destination directory or clear it if it exists
if not os.path.exists(dest):
    os.makedirs(dest)
else:
    shutil.rmtree(dest)
    os.makedirs(dest)

# Process each file and find the best patient correlation
for f in tqdm(full_raw):
    patient_correlation = val_analysis(f, out_dir=dest)
    if float(patient_correlation) > best_patient_correlation:
        best_itr = f
        best_patient_correlation = patient_correlation

# Define the paths for the best results and model
dest_best_result = os.path.join(best_itr.rsplit('/', 3)[0], 'best_results', best_itr.rsplit('/', 3)[2])
src_best = os.path.join(best_itr.rsplit('/', 3)[0], 'raw_to_CSV', best_itr.rsplit('/', 3)[2], best_itr.rsplit('/', 3)[3])
src_model = os.path.join(best_itr.rsplit('/', 3)[0], 'models', best_itr.rsplit('/', 3)[2], 'epoch_' + best_itr.rsplit('/', 3)[3])

# Create the best result directory if it doesn't exist
if not os.path.exists(dest_best_result):
    os.makedirs(dest_best_result)

# Copy the best result files to the destination
shutil.copy(src_best + '.csv', dest_best_result)
shutil.copy(best_itr, dest_best_result)
shutil.copy(src_model, dest_best_result)

# Re-evaluate the best iteration with detailed analysis
best_patient_correlation = val_analysis(best_itr, out_dir=dest, best=True)

print(f'Best Patch level Correlation = {best_patient_correlation}')
