import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt

dir_name = "t40_jan_sound_model/"

result_files = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]

for csv_file in result_files:
    data = pd.read_csv(csv_file)
    #arr = (csv_file,
    #    delimiter=",", dtype=str)
    plt.plot(data["Recall"],data["Precision"],'r+')
    plt.savefig(csv_file.replace("csv","jpg"))
    
    import pdb; pdb.set_trace()
