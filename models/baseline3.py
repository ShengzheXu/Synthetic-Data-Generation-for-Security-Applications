from models.baselines import baseline
from models.baselines import cached_model
from sklearn import mixture
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import operator
import configparser
from scipy import integrate
from utils.distribution_utils import log_sum_exp_trick
import os
import random

class baseline3(baseline):
    def __init__(self):
        super().__init__()
        self.byt_model = None
    
    def fit(self, start_date, sip, byt_log_train, time_delta_train, sip_train, dip_train, byt1_log_train, teT_train, the_df, bins):
        super().fit(sip, start_date, time_delta_train, sip_train, dip_train)
        
        self.bins = bins
        self.byt_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(byt_log_train)
        self.cached_byt = cached_model(self.byt_model)
        self.learnt_p_B_gv_T_B1 = self.build_bayesian_dep(the_df, np.ravel(byt_log_train), np.ravel(byt1_log_train), self.byt_model)

        self.cal_likelihood(byt_log_train, byt1_log_train, teT_train)

    def cal_likelihood(self, given_data_byt, given_data_byt_1, given_data_T):
        add_likeli = 0
        user_order = 1
        gmm_pdf = lambda x: self.byt_model.score_samples(np.reshape([x], (-1, 1)))
        cats_byt = pd.cut(given_data_byt.flatten(), self.bins)
        cats_byt1 = pd.cut(given_data_byt_1.flatten(), self.bins)
        
        for id in range(0, len(given_data_byt)):
            # print(id, len(byt_log_train), len(byt_log_train[id]))
            # marginal distribution
            if given_data_byt_1[id][0] == -1:
                print('%dth row, %dth user' % (id, user_order))
                user_order += 1
                id_byt_interval = cats_byt[id] #self.find_bin(given_data_byt[id][0])
                p_b, _est_error = integrate.quad(gmm_pdf, id_byt_interval.left, id_byt_interval.right)
                add_likeli += p_b
                print('marginal likelihood', add_likeli, 'from', given_data_byt[0][0], '=>', id_byt_interval)
            # cal rest of the joint likelihood for learnt_p_B_gv_T_B1[t][interval_1][interval]
            else:
                id_byt = cats_byt[id] # self.find_bin(given_data_byt[id][0])
                id_byt_1 = cats_byt1[id] # self.find_bin(given_data_byt_1[id][0])
                id_t = given_data_T[id][0]
                # print('calcing lileli, %dth row, t:'%id, id_t, given_data_byt[id][0], "=>", id_byt, ";", given_data_byt_1[id][0], "=>", id_byt_1)
                add_likeli += np.log(self.learnt_p_B_gv_T_B1[id_t][id_byt])
        
        log_likeli = add_likeli/len(given_data_byt)
        print('likelihood:', add_likeli, 'average likelihood', log_likeli)
        self.likelihood = log_likeli
    
    def generate_one(self, dep_info):
        gen_date_obj, gen_te, gen_sip, gen_dip, gen_te_delta = self.generate_sup_info()

        [now_t, last_b] = dep_info
        gen_byt = self.select_bayesian_output(now_t, last_b)
        gen_byt = int(np.exp(gen_byt))
        
        return gen_date_obj, gen_te, gen_sip, gen_dip, gen_byt, gen_te_delta

    def find_bin(self, byt_number):
        # print('byt_number', byt_number, self.bins)
        # print(type(self.bins))
        l = 0
        r = len(self.bins_name_list)
        # print(self.bins_name_list)
        while l<r:
            mid = int((l+r)/2)
            mid_bin = self.bins_name_list[mid]
            # print('looking at', mid, mid_bin ,"when l:%d r:%d" % (l, r))
            if mid_bin.left < byt_number <= mid_bin.right:
                return mid_bin
            if byt_number > mid_bin.right:
                l = mid + 1
            else: # byt_number <= mid_bin.left:
                r = mid
        print("????????????????????????")
        # return pd.cut([byt_number], self.bins)[0]

    # def integrate(self, f, a, b, N):
    #     x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
    #     x = np.reshape(x, (-1, 1))
    #     fx = f(x)
    #     area = np.sum(fx)*(b-a)/N
    #     return area

    def build_bayesian_dep(self, all_record, log_byt_col, log_byt_1_col, clf):
        print(self.bins)
        cats = pd.cut(log_byt_col, self.bins)
        print('bytlen', len(log_byt_col))
        print('dfcollen', all_record.shape)

        all_record['bytBins'] = cats

        cats = pd.cut(log_byt_1_col, self.bins)
        all_record['byt-1Bins'] = cats

        pr = all_record['bytBins'].value_counts()/all_record['bytBins'].size
        bins_name = sorted(pr.keys(), key=lambda x: x.left)
        self.bins_name_list = bins_name
        p_B = {}
        p_T_B = {}
        p_B1_B = {}
        for interval in bins_name:
            p_B1_B[interval] = {}
            p_T_B[interval] = {}
        
        from utils.distribution_utils import get_distribution_with_laplace_smoothing

        for interval in bins_name:
            # print('now working on:', interval)
            gmm_pdf = lambda x: clf.score_samples(np.reshape([x], (-1, 1)))
            p_B[interval], _est_error = integrate.quad(gmm_pdf, interval.left, interval.right)
            df = all_record[all_record['bytBins'] == interval]

            t_list = []
            b1_list = []
            for t in range(24):
                df_t = df[df['teT'] == t]
                # print('t', t, df_t.size, df.size)
                # p_T_B[interval][t] = df_t.size / df.size
                t_list.append(df_t.size)

            for interval_1 in bins_name:
                df_b_1 = df[df['byt-1Bins'] == interval_1]
                # p_B1_B[interval][interval_1] = df_b_1.size / df.size
                b1_list.append(df_b_1.size)

            # smooth & normalize them
            smoothed_t_list = np.log(get_distribution_with_laplace_smoothing(t_list))
            for t in range(24):
                p_T_B[interval][t] = smoothed_t_list[t]

            smoothed_b1_list = np.log(get_distribution_with_laplace_smoothing(b1_list))
            ith = 0
            for interval_1 in bins_name:
                p_B1_B[interval][interval_1] = smoothed_b1_list[ith]
                ith += 1
                # print('smoothed', t, interval_1, learnt_p_B_gv_T_B1[t][interval_1])

        print('testing:', p_B)
        learnt_p_B_gv_T_B1 = {}
        for t in range(24):
            print("learning learnt_p_B_gv_T table: t = %d" % t)
            ex_label = 0

            learnt_p_B_gv_T_B1[t] = {}
            the_map_to_list = []
            # calc the relative value
            for interval in bins_name:
                learnt_p_B_gv_T_B1[t][interval] = p_T_B[interval][t] + p_B[interval]
                the_map_to_list.append(learnt_p_B_gv_T_B1[t][interval])
                # print(t, interval_1, interval, p_T_B[interval][t], p_B1_B[interval][interval_1], p_B[interval])
            
            # normalize it
            ln_sum = log_sum_exp_trick(the_map_to_list)
            real_sum = np.exp(ln_sum)

            normed_list = np.exp(the_map_to_list) / real_sum
            non_exp_normed_list = the_map_to_list / ln_sum
            if ex_label == 0:
                print("check_sum", sum(normed_list), sum(np.exp(the_map_to_list)), real_sum)
                print("check non-exp sum", sum(non_exp_normed_list), sum(the_map_to_list), ln_sum)
                ex_label = 1
            ith = 0
            for interval in bins_name:
                learnt_p_B_gv_T_B1[t][interval] = normed_list[ith]
                ith += 1
            # print('normed final', t, interval_1, learnt_p_B_gv_T_B1[t][interval_1])
            
        return learnt_p_B_gv_T_B1

    def select_bayesian_output(self, t, b_1):
        if b_1 == -1:
            # first line
            gen_byt = self.byt_model.sample()
            return gen_byt[0]
        else:
            # print(t,np.log(b_1))
            # b1 = self.find_bin(np.log(b_1))
            # print('==>', t,b_1,b1)
            normed_list = [self.learnt_p_B_gv_T_B1[t][interval] for interval in self.bins_name_list]
            
            # print("check_sum", t, sum(normed_list))
            # print(normed_list)
            selected_B = np.random.choice(self.bins_name_list, p=normed_list)
            # selected_B = max(self.learnt_p_B_gv_T_B1[t][b1].items(), key=operator.itemgetter(1))[0]
            return np.random.uniform(selected_B.left, selected_B.right)
    
    def save_the_model(self):
        import gc
        self.save_common_params()
        choice_config = configparser.ConfigParser()
        choice_config.read(self.params_dir+"model_base_params.ini")
        choice_config['SAVED_MODEL_PARAS']['bins'] = ','.join([str(x) for x in self.bins])
        with open(self.params_dir+"model_base_params.ini", 'w') as configfile:
            choice_config.write(configfile)

        self._save_gmm(self.byt_model, "byt")

        for t in range(24):        
            learnt_nparray = np.array(self.learnt_p_B_gv_T_B1[t])
            if not os.path.exists(self.params_dir+"baseline3table/"):
                os.makedirs(self.params_dir+"baseline3table/")
            np.save(self.params_dir+"baseline3table/table_for_t%d.npy" % t, learnt_nparray)
            del learnt_nparray
            gc.collect()
        print("<=====Baseline3 Parameters Saved")

    def load_the_model(self):
        self.load_common_params()
        self.cached_byt, self.byt_model = self._load_gmm("byt")
        self.learnt_p_B_gv_T_B1 = {}
    
        for t in range(24): 
            self.learnt_p_B_gv_T_B1[t] = np.load(self.params_dir+"baseline3table/table_for_t%d.npy" % t).tolist()
        # print(self.learnt_p_B_gv_T_B1[0])
        # print("above!!!!! t=0")
        choice_config = configparser.ConfigParser()
        choice_config.read(self.params_dir+"model_base_params.ini")
        self.bins = list(map(float, choice_config['SAVED_MODEL_PARAS']['bins'].split(',')))
        self.bins_name_list = list(self.learnt_p_B_gv_T_B1[0].keys())
        self.bins_name_list.sort(key=lambda x: x.left)
        # print(type(self.bins_name_list))
        # print(len(self.bins_name_list))
        # [pd.Interval(left=self.bins[i-1], right=self.bins[i]) for i in range(1, len(self.bins))]
        print("=====>Baseline3 Model Parameters Loaded")