'''
Power analysis
'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotnine as pn
import os

# Theme for matplotlib
colors = {"cdblue": "#286090",
          "brown" :"#7D5D2D",
          "green" :"#233A2D",
          "tan" :"#C5C88F",
          "blue" :"#455778",
          "lightblue" :"#9EACC5",
          "gold" :"#A58E38",
          "cdgrey": "#DDDDDD"}

andy_theme = {'font.sans-serif': 'Verdana',
              'font.family': 'sans-serif',
              'axes.grid': True,
              'grid.linestyle': '--',
              'grid.color': colors['cdgrey'],
              'legend.framealpha': 1,
              'legend.facecolor': 'white',
              'legend.shadow': True,
              'legend.fontsize': 14,
              'legend.title_fontsize': 16,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'axes.labelsize': 16,
              'axes.titlesize': 20,
              'figure.dpi': 200,
              'axes.titlelocation': 'left'}

matplotlib.rcParams.update(andy_theme)

##########################
# PART 1 WITH ORIGINAL DATA
# leaving in so people can see how I prepared the data

if os.path.exists('Entity_data.csv'):
    # read in data
    ent = pd.read_csv('Entity_data.csv')
    ent = ent[ent['incident_no'] != 'REDACTED'].reset_index()
    # These are NIBRS categories, 52 is gun crime
    ent['violent'] = np.floor(ent['ucr']/100).astype(int).isin([9,10,11,12,13,52])
    ent['nonviolent'] = ~ent['violent']
    # roles
    vic = ['VICTIM','JUV-VICTIM','INJURED','KILLED','JUV-INJURED','COMPLAINANT','OWNER',
           'JUV-KILLED']
    off = ['SUSPECT','ARRESTEE','SUBJECT','INVOLVED','JUV-ARRESTE',
           'JUV-SUBJECT','JUV-INVOLVE','JUV-OFF/SUS','VICTIM/ARRE',
           'VICTIM/OFFE','JUV-VIC/OFF','JUV-VIC/ARR']
    ent['vic'] = ent['role'].isin(vic)
    ent['off'] = ent['role'].isin(off)
    # weeks
    ent['occ_date'] = pd.to_datetime(ent['occ_date'])
    begin_time = ent['occ_date'].min()
    days = (ent['occ_date'] - begin_time).dt.days
    ent['weekn'] = np.floor(days/7)
    # save out to zip file that can be shared
    ent_fields = ['weekn','ucr','pin','vic','off','violent','nonviolent']
    ent[ent_fields].to_csv('EntityReduced.zip')
##########################


##########################
# PART 2 THAT CAN BE REPLICATED WITH REDACTED DATA

# save out redacted data, so others can replicate
ent = pd.read_csv('EntityReduced.zip')

# further breakdowns
ent['vic_viol'] = ent['vic']*ent['violent']
ent['vic_nviol'] = ent['vic']*ent['nonviolent']
ent['off_viol'] = ent['off']*ent['violent']
ent['off_nviol'] = ent['off']*ent['nonviolent']
sum_fields = ['off_viol','vic_viol','off_nviol','vic_nviol']

# aggregate to person-weeks, only worry about people with more than 2 incidents
persons = ent.groupby('pin',as_index=False)[sum_fields].sum()
totn = persons[sum_fields].sum(axis=1)
persons = persons[totn > 2]['pin']
weekn = pd.Series(pd.unique(ent['weekn']),name='weekn')
exp_pers = pd.merge(persons,weekn,how='cross')
entgb = ent.groupby(['weekn','pin'],as_index=False)[sum_fields].sum()
exp_pers = pd.merge(exp_pers,entgb,on=['weekn','pin'],how='left').fillna(0)

# calculate cum-sums per person
exp_pers.sort_values(by=['pin','weekn'],inplace=True,ignore_index=True)
cumv = exp_pers.groupby('pin',as_index=False)[sum_fields].cumsum()
cumv['weekn'] = exp_pers['weekn']
exp_pers['pin'] = exp_pers['pin'].astype(str).str.replace('.0','')
cumv['pin'] = exp_pers['pin']

# Start at x weeks in
# Get top 100, randomly select 5
# set future values to 50%
# calculate treated vs control
# keep tabs on people in future treated

treat = []
week_start = 6*4 # start at ~6 months in
topk = 100
week_sel = 5
treat_eff = [0.5,0.2,0.1]
exp_pers['treated'] = 0
exp_pers['control'] = 0


for w in range(week_start,int(weekn.max())+1):
    # get top 100 not in prior treated
    wv = cumv[cumv['weekn'] == w]
    wv = wv[~wv['pin'].isin(treat)]
    wv_new = wv.sort_values(by=sum_fields,ascending=False)
    new_peop = wv_new['pin'].head(topk)
    # randomly assign to treatment/control
    new_treat = np.random.choice(new_peop,size=week_sel)
    treat += list(new_treat)
    control = set(new_peop) - set(new_treat)
    tn = ((exp_pers['pin'].isin(new_treat)) & 
                           (exp_pers['weekn'] > w))
    exp_pers.loc[tn,'treated'] = 1
    exp_pers.loc[tn,'control'] = 0
    fc = ((exp_pers['pin'].isin(control)) & 
          (exp_pers['weekn'] > w))
    exp_pers.loc[fc,'control'] = 1

# estimate GLM effect, controlling for 
any_tc = exp_pers[['treated','control']].sum(axis=1)
samp = exp_pers[any_tc > 0].copy()
exp_pers['weekn'] = exp_pers['weekn'].astype(int)

orig_viol = exp_pers['off_viol']

def get_eff(treat_eff,weeks=173,data=samp):
    d2 = data[['off_viol','pin','weekn','treated']].copy()
    d2 = d2[d2['weekn'] <= weeks].reset_index()
    treat = d2['treated'] == 1
    d2.loc[treat,'off_viol'] = d2.loc[treat,'off_viol']*(1-treat_eff)
    covp = {'groups': d2['pin'],'df_correction':True}
    pmod = smf.glm('off_viol ~ treated + C(weekn)',data=d2,family=sm.families.Poisson())
    fitmod = pmod.fit(maxiter=100000,cov_type='cluster',cov_kwds=covp)
    coefs = fitmod.summary2().tables[1]
    treat_di = coefs.tail(1).T.to_dict()['treated']
    treat_di['true_eff'] = np.log(1-treat_eff)
    treat_di['week'] = weeks
    return treat_di

res_info = []
treat_effects = [0.0,0.1,0.2,0.5]

for w in range(53,173):
    for t in treat_effects:
        es = get_eff(t,w)
        res_info.append(es)

res = pd.DataFrame(res_info)
res['week'] = res['week'] - week_start

res.to_csv('sim_res.csv',index=False)

res = pd.read_csv('sim_res.csv')
res['PercEff'] = np.round(1 - np.exp(res['true_eff']),1)

lab_di = {0.0:'No Effect',
          0.1:'Small [0.1]',
          0.2:'Mid [0.2]',
          0.5:'Large [0.5]'}

res['EffLab'] = res['PercEff'].replace(lab_di)

res['EffLab'] = pd.Categorical(res['EffLab'])
res['EffLab'] = res['EffLab'].cat.reorder_categories(lab_di.values())

plot = (pn.ggplot(res, pn.aes(x='week',y='Coef.',
                  ymin='[0.025',ymax='0.975]'))
        + pn.geom_ribbon(alpha=0.7, fill='#DDDDDD')
        + pn.geom_line(pn.aes(y='true_eff'),color='blue',size=0.8)
        + pn.geom_line(size=1.3)
        + pn.facet_wrap("EffLab")
        + pn.labs(x='Week',y='Estimate',caption='Grey area are 95% confidence intervals. Blue line is the true simulated effect')
        + pn.theme_matplotlib()
       )

plot.save("CoefPlot.png",dpi=500,bbox_inches='tight')


# Below is the work for cumulative totals
# but does not account for clustering at the person level

# For those treated, set crimes to 50%
#orig_tots = exp_pers[sum_fields].copy()
#exp_pers.loc[exp_pers['treated'] == 1,sum_fields] = exp_pers.loc[exp_pers['treated'] == 1,sum_fields]*treat_eff

#stc = []
#
#for s in sum_fields:
#    exp_pers[s + '_treat'] = exp_pers['treated']*exp_pers[s]
#    exp_pers[s + '_cont'] = exp_pers['control']*exp_pers[s]
#    stc += [s + '_treat',s + '_cont']
#
#stc += ['treated','control']
#
## calculate cumulative totals over the study
#weekg = exp_pers.groupby(['weekn'],as_index=False)[stc].sum()
#weekg = weekg[weekg['weekn'] > week_start]
#
#pair_v = [['off_viol_treat','off_viol_cont'],
#          ['vic_viol_treat','vic_viol_cont'],
#          ['off_nviol_treat','off_nviol_cont'],
#          ['vic_nviol_treat','vic_nviol_cont']]
#
#finte = []
#finse = []
#
## For those treatments, set to 50%, then calculate rates, and standard error estimate
#for t,c in pair_v:
#    for i,te in enumerate(treat_eff):
#        rate_treat = (weekg[t]*(1-te)).cumsum()/weekg['treated'].cumsum()
#        rate_cont = weekg[c].cumsum()/weekg['control'].cumsum()
#        treat_ir = np.log(rate_treat/rate_cont)
#        treat_se = np.sqrt(1/weekg['treated'].cumsum() + 1/weekg['control'].cumsum())
#        treat_name = t.replace('_treat','') + "T" + str(i)
#        treat_sename = t.replace('_treat','') + "SE" + str(i)
#        weekg[treat_name] = treat_ir
#        weekg[treat_sename] = treat_se
#        finte.append(treat_name)
#        finse.append(treat_sename)
#
#
## now make some graphs over the period, filter out first year for inf
#weekgplot = weekg[finte + finse + ['weekn']].copy()
#wgp = weekgplot[weekgplot['weekn'] > 52].copy()
#wgp['weekn'] = wgp['weekn'].astype(int)
#
#wgp['Low'] = wgp['off_violT1'] - 2*wgp['off_violSE1']
#wgp['High'] = wgp['off_violT1'] + 2*wgp['off_violSE1']
#
#fig, ax = plt.subplots()
#ax.fill_between(wgp['weekn'], wgp['Low'], wgp['High'],
#                alpha=0.2, color='k')
#ax.plot(wgp['weekn'],wgp['off_violT1'],color='k',linewidth=2)
#ax.set_title('Cumulative Effect on Violent Crimes')
#plt.show()
#
## Select out key weeks and make a nice table
#sel_weeks = wgp['weekn'].isin([53,79,195,130,157])
#wgp[sel_weeks]




