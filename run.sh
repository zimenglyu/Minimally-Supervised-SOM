# train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_clean.csv"
# test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_clean.csv"
# lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_lab_results.csv"

# train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean.csv"
# test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean.csv"
# lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_lab_results.csv"
# data_set="Sep_2022_cyc_10"

# train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_spectra.csv"
# test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_spectra.csv"
# lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_lab_results.csv"
# # test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_crop_rollavg_10_spectra.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/Cyclone_10_March_Sampling_Spectra.csv"

# data_set="Cyc_10_sample"

# train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"
# test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"
# lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_lab_results.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"
# data_set="May_2021_cyc_10_7_3"

# train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/spectra_March_2023_10.csv"
# test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/spectra_March_2023_10.csv"
# lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Lab_Data_March_2023_cyclone_10.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/spectra_March_2023_10.csv"
# # test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_crop_rollavg_10_spectra.csv"
# # test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/Cyclone_10_March_Sampling_Spectra.csv"

# data_set="Mar_2023_cyc_10"

train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209.csv"
test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209.csv"
lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-04/2023_04_cyc_10_spectra.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/Cyclone_10_March_Sampling_Spectra.csv"

data_set="2023-04"

# train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"
# test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"
# lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/may_2021_cyclone_10_lab_results.csv"
# test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2021-05/FT_2021_10_spectra.csv"
# # test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/Cyclone_10_March_Sampling_Spectra.csv"

# data_set="202105_cyc_10"

for n in 10
do
    for neighbor in 5 
    do
        for norm_method in "standard"
        do
            for fold in 0 
            do
                python main.py --train_spectra $train_spectra \
                --test_spectra $test_spectra \
                --test_lab_result $lab_result \
                --norm_method $norm_method \
                --n_columns $n --n_rows $n --n_epochs 50000 \
                --plot_umatrix True --plot_hist True \
                --data_set ${data_set}_${n}_${neighbor} \
                --plot_umatrix True \
                --do_pca True \
                --fold $fold \
                --num_neighbors $neighbor
            done
        done
    done
done
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