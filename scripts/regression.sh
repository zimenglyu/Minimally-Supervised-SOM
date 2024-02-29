#!/bin/bash
# ----------------------------------------------
#  This script is used to train SOM for regression 
#  trained SOM will be saved as a pickle file
# ----------------------------------------------

# Define variables
train_spectra=""
lab_result_spectra=""
lab_result_label=""
test_data=""
data_set="flight"
method="weighted"
epoch=20000
neighbor=7
norm_method="minmax"
n=5

# Determine folder name for output
folder_name="result_regression/"  # Specify the folder path here

# Check if the directory exists, if not, create it
if [ ! -d "$folder_name" ]; then
    echo "Creating folder: $folder_name"
    mkdir -p "$folder_name"
else
    echo "Folder already exists: $folder_name"
fi


# Execute Python script with constructed parameters
python MS-SOM/SOM_Predictor.py \
    --train_spectra "$train_spectra" \
    --lab_result_spectra "$lab_result_spectra" \
    --lab_result_label "$lab_result_label" \
    --estimate_method "weighted" \
    --testing_path $test_data \
    --norm_method $norm_method \
    --num_neighbors $neighbor \
    --n_columns $n --n_rows $n --n_epochs $epoch \
    --output_path $folder_name \
    --fold 0 \
    --plot_umatrix True --plot_hist True \
    --data_set ${data_set} \
    --load_som 0 \
    --shuffle_data 1 \
    --som_path $som_path \
    --plot_umatrix True 

