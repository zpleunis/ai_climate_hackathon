import warnings
warnings.filterwarnings("ignore")

import os
import glob
import pandas as pd

# change to directory 
# os.chdir('/Users/ldang/Desktop/GitHub/ai_climate_hackathon/data/climate_data')
os.chdir('/data/climate_data')

# get .csv filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv("climate_daily_combined.csv", index=False, encoding='utf-8-sig')