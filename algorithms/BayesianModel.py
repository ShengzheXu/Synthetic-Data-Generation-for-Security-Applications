#from IPython.display import Image
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.api as sm
 
# %matplotlib inline
 
plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
 
all_record = pd.read_csv('./../data/day_1_internal_ip_only.csv')

fig = plt.figure(figsize=(11,3))
_ = plt.title('Frequency of records by number of bytes')
_ = plt.xlabel('number of bytes')
_ = plt.ylabel('Number of records')
_ = plt.hist(all_record['byt'].values, 
             range=[0, 1500], bins=60, histtype='stepfilled')

fig.savefig('data_dis')


# frequency approach ==========================================================
y_obs = all_record['byt'].values
 
def poisson_logprob(mu, sign=-1):
    return np.sum(sign*stats.poisson.logpmf(y_obs, mu=mu))
 
freq_results = opt.minimize_scalar(poisson_logprob)
print("The estimated value of mu is: %s" % freq_results['x'])
mu = np.int(freq_results['x'])

fig = plt.figure(figsize=(11,3))
ax = fig.add_subplot(111)
x_st  = 4000
x_lim = 6000

for i in np.arange(x_st, x_lim):
    plt.bar(i, stats.poisson.pmf(mu, i), color=colors[3])
    
_ = ax.set_xlim(x_st, x_lim)
_ = ax.set_ylim(0, 0.1)
_ = ax.set_xlabel('number of bytes')
_ = ax.set_ylabel('Probability mass')
_ = ax.set_title('Estimated Poisson distribution for number of bytes')
_ = plt.legend(['$lambda$ = %s' % mu])

fig.savefig('probability_prediction')


# Bayesian approach ===========================================================
with pm.Model() as model:
    mu = pm.Uniform('mu', lower=0, upper=800)
    likelihood = pm.Poisson('likelihood', mu=mu, observed=all_record['byt'].values)
    
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(200000, step, start=start, progressbar=True)

mu = trace.trace.get_values('mu')[-1]
# print("The estimated value of mu is: %s" % freq_results['x'])

fig = plt.figure(figsize=(11,3))
ax = fig.add_subplot(111)
x_st  = 0
x_lim = 1000

for i in np.arange(x_st, x_lim):
    plt.bar(i, stats.poisson.pmf(mu, i), color=colors[3])
    
_ = ax.set_xlim(x_st, x_lim)
_ = ax.set_ylim(0, 0.1)
_ = ax.set_xlabel('number of bytes')
_ = ax.set_ylabel('Probability mass')
_ = ax.set_title('Estimated Poisson distribution for number of bytes')
_ = plt.legend(['$lambda$ = %s' % mu])

fig.savefig('bayesian_prediction')
