from GaussianMixtureModel import SourceData
from GaussianMixtureModel import GMMs
from datetime import timedelta
from datetime import datetime
from sklearn import mixture 
import numpy as np
import pandas as pd

# settings
ip_str = '42.219.158.226'
ip_str = 'sampled_10IPs'
output_str = '10IP'

starttime = datetime.now()
# init data
np.random.seed(131)
source_data = './../data/baseline1&2/cleaned_data/expanded_day_1_%s.csv' % ip_str
all_record = pd.read_csv(source_data)

byt_train = np.reshape(all_record['byt'].values, (-1, 1))
byt_log_train = np.log(byt_train)

time_delta_train = np.reshape(all_record['teDelta'].values, (-1, 1))
sip = all_record['sa'].values[0]
dip_train = np.ravel(all_record['da'].values)

# init models
byt_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(byt_log_train)

teDelta_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(time_delta_train)
start_date_time_str = '2016-04-11 00:00:00'  
start_date_time_obj = datetime.strptime(start_date_time_str, '%Y-%m-%d %H:%M:%S')

def choice_from_dip_pool():
    return np.random.choice(dip_train, 1)

dip_model = choice_from_dip_pool

# gen data
row_number = 637608
byt_col = byt_model.sample(row_number)
te_col = []
dip_col = []
gen_data = []
for i in range(row_number):
    _, gen_te_delta = teDelta_model.sample()
#     print(gen_te_delta, int(gen_te_delta[0]))
    start_date_time_obj = start_date_time_obj + timedelta(seconds=int(gen_te_delta[0]))
    gen_te = start_date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
#     print(gen_te)
    te_col.append(gen_te)
    dip_col.append(dip_model())
    gen_data.append([te_col[i], sip, dip_col[i][0], int(np.exp(byt_col[0][i][0])), gen_te_delta[0]])
    print(i, '===>', gen_data[-1])

# exit()

# write to a csv file
import csv

with open("./../data/baseline1&2/gen_data/baseline1_%s.csv" % output_str, "w", newline="") as f:
    fieldnames = ['te', 'sa', 'da', 'byt', 'teDelta']
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(gen_data)

endtime = datetime.now()
print('process time', (endtime-starttime).seconds/60, 'mins')