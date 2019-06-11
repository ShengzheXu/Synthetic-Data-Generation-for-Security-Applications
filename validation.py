import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.cluster import normalized_mutual_info_score
import configparser
from utils.config_utils import recieve_cmd_config
from utils.plot_utils import boxplot
from utils.plot_utils import temporal_lineplot
from utils.plot_utils import distribution_lineplot
from utils.plot_utils import distribution_lineplot_in_one
from models.baselines import baseline1
from utils.distribution_utils import get_distribution_with_laplace_smoothing
import glob

def KLc(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def loglikelihood(p):
    meta_model = baseline1()
    meta_model.load_params(meta_model.byt_model)

    logprob, responsibilities = meta_model.byt_model.score_samples(p)
    pdf = np.exp(logprob)
    return sum(logprob)

def cross_validation_preparation(data):
    k_fold = 5
    csvfile = open('import_1458922827.csv', 'r').readlines()
    filename = 1
    len_rows = int(len(csvfile) / float(k_fold) * k_fold)
    block_rows = len_rows / k_fold
    for i in range(k_fold):
        if i % block_rows == 0:
            open(str(filename) + 'import_1458922827.csv', 'w').writelines(csvfile[i:i+block_rows])
            filename += 1
    # build 5 model
    for i in range(k_fold):
        # open test data and test loglikelihood
        test_data = None
        loglikelihood(test_data)

working_folder = ''

def real_vali_KL(rawdata, gendata, bins):
    values1 = list(np.log(rawdata.byt))
    values2 = list(np.log(gendata.byt))

    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    cats2 = pd.cut(values2, bins)
    pr2 = list(cats2.value_counts())

    pk = get_distribution_with_laplace_smoothing(pr1)
    qk = get_distribution_with_laplace_smoothing(pr2)

    # If only probabilities pk are given, the entropy is calculated as S = -sum(pk * log(pk), axis=0).
    # If qk is not None, then compute the Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=0).
    KL = scipy.stats.entropy(pk, qk)

    # myKL = KLc(pk, qk)
    # NMI = normalized_mutual_info_score(pr2,pr2)

    return KL

def real_vali_JS(rawdata, gendata, bins):
    values1 = list(np.log(rawdata.byt))
    values2 = list(np.log(gendata.byt))

    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    cats2 = pd.cut(values2, bins)
    pr2 = list(cats2.value_counts())

    pk = get_distribution_with_laplace_smoothing(pr1)
    qk = get_distribution_with_laplace_smoothing(pr2)
    print(len(pk), len(qk))
    avgk = [(x+y)/2 for x,y in zip(pk, qk)]

    JS = (scipy.stats.entropy(pk, avgk) + scipy.stats.entropy(qk, avgk)) / 2

    return JS

def vali_one(raw_ip_str, gen_ip_str, bins):
    rawdata = pd.read_csv(working_folder + raw_ip_str)
    gendata = pd.read_csv(working_folder + gen_ip_str)

    KL = real_vali_JS(rawdata, gendata, bins)

    print(','.join([raw_ip_str, gen_ip_str, str(KL)]))
    return KL

def show_distribution(target_data, bins):
    values1 = list(np.log(target_data.byt))
    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    
    pk = get_distribution_with_laplace_smoothing(pr1)
    # print('===', len(pk))
    return pk

def show_conditioned_distribution(bins):
    # target_folder = './data/raw_data/'
    target_folder = './data/gen_data/baseline2_1days_folder/'
    # target_folder = './data/gen_data/sample_baseline2_1days_folder/'
    # target_folder = './data/gen_data/argmax_baseline2_1days_folder/'

    target_record = pd.concat([pd.read_csv(f) for f in glob.glob(target_folder+'*.csv')], ignore_index = True)

    x_data = bins[1:]
    y_data = []

    for t_hour in range(24):
        str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        target_chunk = target_record[target_record['te'].str.contains(' '+str_hour+':')]

        y_data.append(show_distribution(target_chunk, bins))
    
    distribution_lineplot(x_data, y_data, x_label='bins', y_label='probability', title='conditioned distribution')
    distribution_lineplot_in_one(x_data, y_data, x_label='bins', y_label='probability', title='conditioned distribution')

def vali_hourly(raw_ip_str, gen_ip_str, bins):
    rawdata = pd.read_csv(working_folder + raw_ip_str)
    gendata = pd.read_csv(working_folder + gen_ip_str)
    
    rtn = []
    for T in range(24):
        hour_str = '00'
        if T<10:
            hour_str = hour_str[:-1] + str(T)[0]
        else:
            hour_str = str(T)
        # print('checking:', T, hour_str)
        raw_chunk = rawdata[rawdata['te'].str.contains(' '+hour_str+':')]
        gen_chunk = gendata[gendata['te'].str.contains(' '+hour_str+':')]
        # KL = real_vali_KL(raw_chunk, gen_chunk, bins)
        JS = real_vali_JS(raw_chunk, gen_chunk, bins)
        print(','.join(['For %d hour' % T, raw_ip_str, gen_ip_str, str(JS)]))
        rtn.append(JS)

    return rtn

def vali_as_a_whole(bins):
    source_folder = './data/raw_data/'
    target1_folder = './data/gen_data/baseline1_1days_folder/'
    target2_folder = './data/gen_data/baseline2_1days_folder/'

    source_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)
    target1_record = pd.concat([pd.read_csv(f) for f in glob.glob(target1_folder+'*.csv')], ignore_index = True)
    target2_record = pd.concat([pd.read_csv(f) for f in glob.glob(target2_folder+'*.csv')], ignore_index = True)

    x_data = ['JS(raw|baseline1)', 'JS(raw|baseline2)', 'JS(baselin1|baseline2)']
    y_data = [[], [], []]

    for t_hour in range(24):   
        str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
        source_chunk = source_record[source_record['te'].str.contains(' '+str_hour+':')]
        target1_chunk = target1_record[target1_record['te'].str.contains(' '+str_hour+':')]
        target2_chunk = target2_record[target2_record['te'].str.contains(' '+str_hour+':')]

        y_data[0].append(real_vali_JS(source_chunk, target1_chunk, bins))
        y_data[1].append(real_vali_JS(source_chunk, target2_chunk, bins))
        y_data[2].append(real_vali_JS(target1_chunk, target2_chunk, bins))
    
    temporal_lineplot(x_data, y_data, x_label='hour', y_label='KL divergency', title='3 pairs KL divergency compare')


if __name__ == "__main__":
    # load in the configs
    config = configparser.ConfigParser()
    config.read('config.ini')
    # override the config with the command line
    recieve_cmd_config(config['DEFAULT'])
    
    user_list = config['DEFAULT']['userlist'].split(',')
    working_folder = config['DEFAULT']['working_folder']
    baseline_choice = config['DEFAULT']['baseline']
    bins = list(map(float, config['DEFAULT']['bins'].split(',')))
    
    test_list = config['VALIDATE']['test_set'].split(',')

    if config['VALIDATE']['raw_compare'] == 'True':
        x_data = []
        y_data = []
        for ip2 in user_list:
            x_data.append(ip2)
            y_data_i = []
            for ip1 in user_list:
                if ip1 == ip2:
                    continue
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'raw_data/day_1_%s.csv' % ip2 
                y_data_i.append(vali_one(src_file, des_file, bins))
            y_data.append(y_data_i)
            
        boxplot(x_data, y_data, title='KL(99 other raw ips || 1 raw_ip)')
    
    if config['VALIDATE']['hour_compare'] == 'True':
        for ip2 in user_list:
            x_data = []
            y_data = []
            for h in range(24):
                x_data.append(h)
                y_data.append([])
            
            for ip1 in user_list: 
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'gen_data/%s_1days_folder/%s_1days_%s.csv' % (baseline_choice, baseline_choice, ip2)
                y_data_i = vali_hourly(src_file, des_file, bins)
                for h in range(24):
                    y_data[h].append(y_data_i[h])
            print(len(y_data), len(y_data[0]))
        
            boxplot(x_data, y_data, title='KL(100 raw ips || %s)' % ip2)

    if config['VALIDATE']['conditioned_whole'] == 'True':
        show_conditioned_distribution(bins)

    if config['VALIDATE']['hour_whole'] == 'True':
        vali_as_a_whole(bins)
