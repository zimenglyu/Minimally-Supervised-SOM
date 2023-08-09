import numpy as np
import pandas as pd


# clean data and drop na values
# datadir = '/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_clean.csv'
# data = pd.read_csv(datadir)
# # data = data.loc[data['CycloneNumber']==10]
# data = pd.read_csv(datadir, parse_dates=["DateTime"], index_col="DateTime")
# # data = data[(data['CycloneNumber']==10)]

# data.dropna(inplace=True)
# data.to_csv("/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_3_2022_09_clean.csv")

# merge testing lab results
test_spectra_path = '/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Cyclone_3_March_Sampling_Spectra.csv'
test_lab_path = '/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Lab_Data_March_2023.csv'
test_spectra = pd.read_csv(test_spectra_path, parse_dates=["DateTime"], index_col="DateTime")
test_lab = pd.read_csv(test_lab_path, parse_dates=["DateTime"], index_col="DateTime")
lab_columns = test_lab.columns
merged_data = pd.merge(test_lab, test_spectra, how="inner",  on='DateTime')
rslt_df = merged_data[merged_data['Cyclone'] == 3]
# rslt_df = rslt_df.drop(columns=lab_columns)
print(rslt_df)
rslt_df.to_csv("/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Lab_Data_March_2023_3.csv")

# input_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_crop.csv"
# output_path = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2022-09/MGA_10_2022_09_clean_crop_rollavg_10.csv"
# data = pd.read_csv(input_path, parse_dates=["DateTime"], index_col="DateTime")
# # rolling average of 10 minutes
# # data['DateTime']= pd.to_datetime(data['DateTime'])
# data = data.rolling(window=10).mean()

# # data = data.iloc[::10]
# data = data.apply (pd.to_numeric, errors='coerce')
# data = data.dropna()
# data.to_csv(output_path)