from sklearn import mixture
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import operator
import configparser

np.random.seed(131)

class baseline(object):
    def __init__(self):
        self.te_adj = False
        self.params_dir = "./saved_model/"
        self.for_whom = None
        self.likelihood = None

    def init_sip_dip(self, sip_train, dip_train):
        pr_outgoing = 0
        for ip in self.sip:
            # print(ip, (sip_train == ip).sum())
            pr_outgoing += (sip_train == ip).sum()
        pr_outgoing /= len(sip_train)
        
        rt_sip_pool = [x for x in sip_train if x not in self.sip]
        rt_dip_pool = [x for x in dip_train if x not in self.sip]
        return pr_outgoing, rt_sip_pool, rt_dip_pool

    def fit(self, sip, start_date, time_delta_train, sip_train, dip_train):
        self.sip = sip
        self.teDelta_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(time_delta_train)
        self.start_date_time_str = start_date
        self.start_date_time_obj = datetime.strptime(self.start_date_time_str, '%Y-%m-%d %H:%M:%S')

        print(sip)
        self.outgoing_prob, self.sip_pool, self.dip_pool = self.init_sip_dip(sip_train, dip_train)
        print('outgoing prob', self.outgoing_prob)

        def choice_from_sip_pool():
            return np.random.choice(self.sip_pool, 1)[0]

        def choice_from_dip_pool():
            return np.random.choice(self.dip_pool, 1)[0]

        self.sip_model = choice_from_sip_pool
        self.dip_model = choice_from_dip_pool

    def reset_gen(self, for_whom):
        self.for_whom = for_whom
        self.start_date_time_obj = datetime.strptime(self.start_date_time_str, '%Y-%m-%d %H:%M:%S')

    def generate_sup_info(self):
        gen_te_delta, _ = self.teDelta_model.sample()
        gen_te_delta = self.adjusted_time_delta_for_high_throughput(gen_te_delta)

        self.start_date_time_obj = self.start_date_time_obj + timedelta(seconds=gen_te_delta)
        gen_te = self.start_date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
        gen_sip, gen_dip = self.generate_sd_ip()

        return gen_te, gen_sip, gen_dip, gen_te_delta

    def generate_sd_ip(self):
        import random
        rolling_point = random.random()
        if rolling_point < self.outgoing_prob:
            return self.for_whom, self.dip_model()
        else:
            return self.sip_model(), self.for_whom

    # save and load helpers
    def save_common_params(self):
        self._save_gmm(self.teDelta_model, "time_delta")
        self._save_choice()
    
    def load_common_params(self):
        self.teDelta_model = self._load_gmm("time_delta")
        self._load_choice()
    
    def _save_gmm(self, gmm_model, model_name):
        np.save(self.params_dir+"%s_params_weights.npy" % model_name, gmm_model.weights_)
        np.save(self.params_dir+"%s_params_mu.npy" % model_name, gmm_model.means_)
        np.save(self.params_dir+"%s_params_sigma.npy" % model_name, gmm_model.covariances_)
        print("<===== %s Model Parameters Saved" % model_name)

    def _load_gmm(self, model_name):
        gmm_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(np.random.randn(10, 1))
        weights = np.load(self.params_dir+"%s_params_weights.npy" % model_name)
        mu = np.load(self.params_dir+"%s_params_mu.npy" % model_name)
        sigma = np.load(self.params_dir+"%s_params_sigma.npy" % model_name)
        gmm_model.weights_ = weights   # mixture weights (n_components,) 
        gmm_model.means_ = mu          # mixture means (n_components, 2) 
        gmm_model.covariances_ = sigma  # mixture cov (n_components, 2, 2)
        print("<=====Model Parameters Loaded")
        print("weights", weights)
        print("means", mu)
        print("covariances", sigma)
        print("=====>Model Parameters Loaded")
        return gmm_model
    
    def _save_choice(self):
        choice_config = configparser.ConfigParser()
        
        choice_config['SAVED_MODEL_PARAS'] = {}
        
        choice_config['SAVED_MODEL_PARAS']['outgoing_prob'] = str(self.outgoing_prob)
        choice_config['SAVED_MODEL_PARAS']['sip_pool'] = ','.join(self.sip_pool)
        choice_config['SAVED_MODEL_PARAS']['dip_pool'] = ','.join(self.dip_pool)
        choice_config['SAVED_MODEL_PARAS']['start_date'] = self.start_date_time_str

        with open(self.params_dir+"model_base_params.ini", 'w') as configfile:
            choice_config.write(configfile)
    
    def _load_choice(self):
        choice_config = configparser.ConfigParser()
        choice_config.read(self.params_dir+"model_base_params.ini")
        self.outgoing_prob = float(choice_config['SAVED_MODEL_PARAS']['outgoing_prob'])
        self.sip_pool = choice_config['SAVED_MODEL_PARAS']['sip_pool'].split(',')
        self.dip_pool = choice_config['SAVED_MODEL_PARAS']['dip_pool'].split(',')
        self.start_date_time_str = choice_config['SAVED_MODEL_PARAS']['start_date']
        self.start_date_time_obj = datetime.strptime(self.start_date_time_str, '%Y-%m-%d %H:%M:%S')

        def choice_from_sip_pool():
            return np.random.choice(self.sip_pool, 1)[0]

        def choice_from_dip_pool():
            return np.random.choice(self.dip_pool, 1)[0]

        self.sip_model = choice_from_sip_pool
        self.dip_model = choice_from_dip_pool
    
    def adjusted_time_delta_for_high_throughput(self, te_delta):
        adj_te_delta = int(te_delta[0]) if int(te_delta[0])>=0 else 0
        if self.te_adj:
            time_eps = np.random.randint(6)
            if time_eps == 0:
                adj_te_delta = [adj_te_delta[0]+1]
        return adj_te_delta


class baseline1(baseline):
    def __init__(self):
        super().__init__()
        self.byt_model = None
    
    def fit(self, start_date=None, sip=None, byt_log_train=None, time_delta_train=None, sip_train=None, dip_train=None):
        super().fit(sip, start_date, time_delta_train, sip_train, dip_train)
        
        self.byt_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(byt_log_train)
        self.cal_likelihood(byt_log_train)

    def cal_likelihood(self, given_data):
        log_likeli = self.byt_model.score(given_data)
        logprob = self.byt_model.score_samples(given_data)
        add_likeli = sum(logprob)
        print('likelihood:', add_likeli, add_likeli/len(logprob))
        self.likelihood = log_likeli

    def generate_one(self, dep_info=None):
        gen_te, gen_sip, gen_dip, gen_te_delta = self.generate_sup_info()

        gen_byt, _ = self.byt_model.sample()
        gen_byt = int(np.exp(gen_byt[0]))
        
        return gen_te, gen_sip, gen_dip, gen_byt, gen_te_delta
    
    def save_the_model(self):
        self.save_common_params()
        self._save_gmm(self.byt_model, "byt")

    def load_the_model(self):
        self.load_common_params()
        self.byt_model = self._load_gmm("byt")

class baseline2(baseline):
    def __init__(self):
        super().__init__()
        self.byt_model = None
    
    def fit(self, start_date, sip, byt_log_train, time_delta_train, sip_train, dip_train, byt1_log_train, teT_col, bins):
        super().fit(sip, start_date, time_delta_train, sip_train, dip_train)
        
        self.bins = bins
        self.byt_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(byt_log_train)
        self.learnt_p_B_gv_T_B1 = self.build_bayesian_dep(teT_col, np.ravel(byt_log_train), np.ravel(byt1_log_train), self.byt_model)

        self.cal_likelihood(byt_log_train, byt1_log_train, teT_col)

    def cal_likelihood(self, given_data_byt, given_data_byt_1, given_data_T):
        # learnt_p_B_gv_T_B1[t][interval_1][interval]
        add_likeli = 0
        for id in range(len(given_data_byt)):
            # print(id, len(byt_log_train), len(byt_log_train[id]))
            id_byt = self.find_bin(given_data_byt[id][0])
            id_byt_1 = self.find_bin(given_data_byt_1[id][0])
            id_t = given_data_T[id][0]
            add_likeli += np.log(self.learnt_p_B_gv_T_B1[id_t][id_byt_1][id_byt])
        
        log_likeli = add_likeli/len(given_data_byt)
        print('likelihood:', add_likeli, add_likeli/len(given_data_byt))
        self.likelihood = log_likeli
    
    def generate_one(self, dep_info):
        gen_te, gen_sip, gen_dip, gen_te_delta = self.generate_sup_info()

        [now_t, last_b] = dep_info
        gen_byt = self.select_bayesian_output(now_t, last_b)
        gen_byt = int(np.exp(gen_byt))
        
        return gen_te, gen_sip, gen_dip, gen_byt, gen_te_delta

    def find_bin(self, byt_number):
        # print('byt_number', byt_number, self.bins)
        # print(type(self.bins))
        return pd.cut([byt_number], self.bins)[0]

    def integrate(self, f, a, b, N):
        x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
        x = np.reshape(x, (-1, 1))
        fx = f(x)
        area = np.sum(fx)*(b-a)/N
        return area

    def build_bayesian_dep(self, all_record, log_byt_col, log_byt_1_col, clf):
        print(self.bins)
        cats = pd.cut(log_byt_col, self.bins)
        print('bytlen', len(log_byt_col))
        print('dfcollen', all_record.shape)

        all_record['bytBins'] = cats

        cats = pd.cut(log_byt_1_col, self.bins)
        all_record['byt-1Bins'] = cats

        pr = all_record['bytBins'].value_counts()/all_record['bytBins'].size
        bins_name = pr.keys()
        p_B = {}
        p_T_B = {}
        p_B1_B = {}
        for t in range(24):
            p_T_B[t] = {}

        for interval in bins_name:
            p_B1_B[interval] = {}

        for interval in bins_name:
            print('now working on:', interval)
            p_B[interval] = self.integrate(clf.score_samples, interval.left, interval.right, 100)
            df = all_record[all_record['bytBins'] == interval]

            for t in range(24):
                df_t = df[df['teT'] == t]
                print('t', t, df_t.size, df.size)
                p_T_B[t][interval] = df_t.size / df.size

            for interval_1 in bins_name:
                df_b_1 = df[df['byt-1Bins'] == interval_1]
                p_B1_B[interval_1][interval] = df_b_1.size / df.size
        print('testing:', p_B)
        learnt_p_B_gv_T_B1 = {}
        for t in range(24):
            learnt_p_B_gv_T_B1[t] = {}
            for interval_1 in bins_name:
                learnt_p_B_gv_T_B1[t][interval_1] = {}
                for interval in bins_name:
                    learnt_p_B_gv_T_B1[t][interval_1][interval] = p_T_B[t][interval] * p_B[interval] * p_B1_B[interval_1][interval]

        # do a laplacian smoothing for the learnt table
        from utils.distribution_utils import get_distribution_with_laplace_smoothing
        for t in range(24):
            for interval_1 in bins_name:
                the_map_to_list = []
                # extract first
                for interval in bins_name:
                    the_map_to_list.append(learnt_p_B_gv_T_B1[t][interval_1][interval])
                
                smoothed_list = get_distribution_with_laplace_smoothing(the_map_to_list)
                # recover then
                ith = 0
                for interval in bins_name:
                    learnt_p_B_gv_T_B1[t][interval_1][interval] = smoothed_list[ith]
                    ith += 1
                
        return learnt_p_B_gv_T_B1

    def select_bayesian_output(self, t, b_1):
        if b_1 == -1:
            # first line
            gen_byt, _ = self.byt_model.sample()
            return gen_byt[0]
        else:
            b1 = self.find_bin(np.log(b_1))
            # print(t,b_1,b1)
            maxB = max(self.learnt_p_B_gv_T_B1[t][b1].items(), key=operator.itemgetter(1))[0]
            return np.random.uniform(maxB.left, maxB.right)
    
    def save_the_model(self):
        self.save_common_params()
        choice_config = configparser.ConfigParser()
        choice_config.read(self.params_dir+"model_base_params.ini")
        choice_config['SAVED_MODEL_PARAS']['bins'] = ','.join([str(x) for x in self.bins])
        with open(self.params_dir+"model_base_params.ini", 'w') as configfile:
            choice_config.write(configfile)

        self._save_gmm(self.byt_model, "byt")
        learnt_nparray = np.array(self.learnt_p_B_gv_T_B1)
        np.save(self.params_dir+"baseline2table.npy", learnt_nparray)
        print("<=====Baseline2 Parameters Saved", learnt_nparray.shape)

    def load_the_model(self):
        self.load_common_params()
        self.byt_model = self._load_gmm("byt")
        self.learnt_p_B_gv_T_B1 = np.load(self.params_dir+"baseline2table.npy").tolist()
        choice_config = configparser.ConfigParser()
        choice_config.read(self.params_dir+"model_base_params.ini")
        self.bins = list(map(float, choice_config['SAVED_MODEL_PARAS']['bins'].split(',')))
