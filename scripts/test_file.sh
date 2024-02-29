
label=5
# train_spectra=/Users/zimenglyu/Documents/datasets/SOM_flight/flight_35_36_38.csv
# lab_result_spectra=/Users/zimenglyu/Documents/datasets/SOM_flight/flight_35_36_38_${label}.csv
# lab_result_label=/Users/zimenglyu/Documents/datasets/SOM_flight/flight_35_36_38_${label}_label.csv
# test_data=/Users/zimenglyu/Documents/datasets/SOM_flight/flight_35_36_38.csv
train_spectra="/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438.csv"
lab_result_spectra=/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_${label}.csv
lab_result_label=/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_${label}_label.csv
# test_data=/Users/zimenglyu/Documents/datasets/SOM_flight/flight_all_3.csv
test_data="/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/Cyclone_10_March_Sampling_Spectra.csv"

data_set="flight"

fold=0
method=weighted
epoch=20000
som_path=/Users/zimenglyu/Documents/code/git/susi/SOM_Flight_38/SOM_Flight_${label}
neighbor=15
norm_method="standard"
fold=0

for epoch in 20000 
do
    for n in 5 7 10 
    do
        folder_name="testing_result_38/label_$label/$n"  # Specify the folder path here
        if [ ! -d "$folder_name" ]; then
        echo "Creating folder: $folder_name"
        mkdir -p "$folder_name"
        else
        echo "Folder already exists: $folder_name"
        fi

        for neighbor in 3 5 7 
        do 
            for norm in  "minmax" "standard" "robust"
            do
                for fold in 0 1 2 3 4 
                do
                    som_name=$som_path/flight/$n/flight_${n}_${epoch}_neighbor_${neighbor}_${norm}_${fold}.pickle
                    echo "som_name: $som_name"
                    python SOM_Predictor.py --train_spectra $train_spectra \
                    --lab_result_spectra $lab_result_spectra \
                    --lab_result_label $lab_result_label \
                    --estimate_method  "vote"  \
                    --testing_path $test_data \
                    --norm_method $norm \
                    --num_neighbors $neighbor \
                    --n_columns $n --n_rows $n --n_epochs $epoch \
                    --output_path $folder_name \
                    --fold $fold \
                    --plot_umatrix False --plot_hist False \
                    --data_set ${data_set} \
                    --load_som 1 \
                    --shuffle_data 1 \
                    --som_path $som_name 
                done
            done        
        done
    done
done
