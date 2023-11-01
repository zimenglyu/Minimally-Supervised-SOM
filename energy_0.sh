data_set="energy"
epoch=20000
dataset_path='/Users/zimenglyu/Documents/datasets/regression'
train_spectra='/Users/zimenglyu/Documents/datasets/regression/energydata_train.csv'
file_size=200

# for file in 0 1 2 3 4
for file in 0 1 2 3 4 5 6 7 8 9
do
    for n in 30
    do
        result_folder_name="SOM_energy_FINAL/filesize_$file_size/som_$n/$file"  # Specify the folder path here
        if [ ! -d "$result_folder_name" ]; then
        echo "Creating folder: $result_folder_name"
        mkdir -p "$result_folder_name"
        else
        echo "Folder already exists: $result_folder_name"
        fi

        lab_result_spectra="$dataset_path/$file_size/energydata_$file.csv"
        lab_result_label="$dataset_path/$file_size/energydata_${file}_label.csv"
        test_data="$dataset_path/$file_size/energydata_$file.csv"
        
        for neighbor in 7
        do
            for norm_method in "minmax" 
            do
                for fold in 2 
                do
                    python SOM_Predictor.py --train_spectra $train_spectra \
                    --lab_result_spectra $lab_result_spectra \
                    --lab_result_label $lab_result_label \
                    --estimate_method "weighted" \
                    --testing_path $test_data \
                    --norm_method $norm_method \
                    --num_neighbors $neighbor \
                    --n_columns $n --n_rows $n --n_epochs $epoch \
                    --output_path $result_folder_name \
                    --fold $fold \
                    --plot_umatrix True --plot_hist True \
                    --data_set ${data_set} \
                    --load_som 0 \
                    --plot_umatrix True 
                done
            done
        done
    done
done