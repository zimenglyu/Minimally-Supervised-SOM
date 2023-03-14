# train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_clean.csv"
# test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_clean.csv"
# lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_lab_results.csv"

train_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_spectra.csv"
test_spectra="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_spectra.csv"
lab_result="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_lab_results.csv"
test_data="/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_clean_spectra_roll_10.csv"
data_set="Sep_2022_cyc_10_to_3"
for n in 10
do
    for neighbor in 7
    do

        python main.py --train_spectra $train_spectra \
        --test_spectra $test_spectra \
        --test_lab_result $lab_result \
        --n_columns $n --n_rows $n --n_epochs 10000 \
        --plot_umatrix True --plot_hist True \
        --data_set $data_set \
        --do_prediction $test_data \
        --num_neighbors $neighbor
    done
done

# python plot_predictions.py

    # --do_prediction $test_data \
            # --do_prediction $test_data \