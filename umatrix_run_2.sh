train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209.csv"
lab_result_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
lab_result_label="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/Cyclone_10_March_Sampling_Spectra.csv"

train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"
lab_result_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/may_2021_cyclone_10_lab_results_spectra.csv"
lab_result_label="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/may_2021_cyclone_10_lab_results.csv"
test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"

data_set="May_2021"
fold=0
method=weighted
epoch=10000
som_path="/Users/zimenglyu/Documents/code/git/susi/som_results/combined_5_5_10000_neighbor_3.pickle"

folder_name="som_results/"  # Specify the folder path here
if [ ! -d "$folder_name" ]; then
echo "Creating folder: $folder_name"
mkdir -p "$folder_name"
else
echo "Folder already exists: $folder_name"
fi

for n in 3 4 6 7
do
    for neighbor in 3
    do
        for norm_method in "standard" 
        do
            python SOM_Predictor.py --train_spectra $train_spectra \
            --lab_result_spectra $lab_result_spectra \
            --lab_result_label $lab_result_label \
            --testing_path $test_data \
            --norm_method $norm_method \
            --num_neighbors $neighbor \
            --n_columns $n --n_rows $n --n_epochs $epoch \
            --output_path $folder_name \
            --fold $fold \
            --plot_umatrix True --plot_hist True \
            --data_set ${data_set} \
            --plot_umatrix True 

        done
    done
done
            # --som_path $som_path \
            # --estimate_method $method \
#         --do_prediction $test_data \
# python plot_predictions.py

# data_set="Sep_2022_cyc_10_5_2"
# for n in 5
# do
#     for neighbor in 2
#     do

#         python main.py --train_spectra $train_spectra \
#         --test_spectra $test_spectra \
#         --test_lab_result $lab_result \
#         --n_columns $n --n_rows $n --n_epochs 10000 \
#         --plot_umatrix True --plot_hist True \
#         --data_set $data_set \
#         --plot_umatrix True \
#         --do_prediction $test_data \
#         --num_neighbors $neighbor
#     done
# done

# python plot_predictions.py

    # --do_prediction $test_data \
            # --do_prediction $test_data \