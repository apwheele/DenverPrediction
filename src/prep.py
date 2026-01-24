'''
This preps the data
'''
from ast import comprehension
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(10)

d = './data'
ent_old = f'{d}/entity_data_2019to2022.csv'
ent_new = f'{d}/entity_data_2022topres.csv'
inc_data = f'{d}/incident_data_2019topres.csv'
lookup_path = f'{d}/LookupTable.csv.zip'

t = './train_data'
train_data_path = f'{t}/TrainData.csv.zip'
holdout_data_path = f'{t}/HoldOutData.csv.zip'

# Prepping data
def get_data():
    # Prepping entity data
    e1 = pd.read_csv(ent_old)
    e2 = pd.read_csv(ent_new)
    ent_data = pd.concat([e1,e2],ignore_index=True)
    ent_data['occ_date'] = pd.to_datetime(ent_data['occ_date'])
    # de-identifying starts here
    pin_ref = ent_data[['pin']].drop_duplicates().reset_index(drop=True).copy()
    pin_ref['rand'] = np.random.random(len(pin_ref))
    pin_ref = pin_ref.sort_values('rand').reset_index(drop=True)
    pin_ref['pin_new'] = np.arange(1, len(pin_ref) + 1)
    pin_ref = pin_ref.drop(columns=['rand'])
    # merge in de-identified pins
    ent_data = ent_data.merge(pin_ref, on='pin', how='left')
    ent_data['pin'] = ent_data['pin_new']
    ent_data = ent_data.drop(columns=['pin_new'])
    # save out pin key
    pin_ref.to_csv(lookup_path, index=False)
    # Merging in XY from incident data
    inc_d = pd.read_csv(inc_data)
    # To check nibr codes
    nibr = inc_d.groupby(['ibr_code','offense_desc','ucr'],as_index=False).size()
    # getting centroid if spread out
    inc_d = inc_d.groupby('incident_no',as_index=False)[['x_coordinate','y_coordinate']].mean()
    # This makes the aggregations per area a quarter of a square mile
    size_side = 5280
    inc_d['x_grid'] = np.floor(inc_d['x_coordinate']/(size_side)).fillna(-1).astype(int)
    inc_d['y_grid'] = np.floor(inc_d['y_coordinate']/(size_side)).fillna(-1).astype(int)
    inc_d['gridCount'] = inc_d.groupby(['x_grid','y_grid']).transform('size')
    inc_d[['x_grid','y_grid']].drop_duplicates() # check see how dense, 30 some for each
    ent_data = ent_data.merge(inc_d[['incident_no','x_grid','y_grid', 'gridCount']],on='incident_no',how='left')
    ent_data[['x_grid','y_grid','gridCount']] = ent_data[['x_grid','y_grid','gridCount']].fillna(-1)
    # Getting the dummy variables for the different x/y categories
    x_dummy = pd.get_dummies(ent_data['x_grid'].astype(int))
    x_dummy.columns = ['x' + str(v) for v in list(x_dummy)]
    y_dummy = pd.get_dummies(ent_data['y_grid'].astype(int))
    y_dummy.columns = ['y' + str(v) for v in list(y_dummy)]
    y_dummy = y_dummy.iloc[:,1:] # get rid of missing, missing in X is enough
    ent_data[list(x_dummy)] = x_dummy
    ent_data[list(y_dummy)] = y_dummy
    grid_vars = list(x_dummy) + list(y_dummy)
    # Dummy variables for different categories
    vic = ['VICTIM','JUV-VICTIM','INJURED','KILLED','JUV-INJURED','COMPLAINANT','OWNER',
           'REGIST OWNR','JUV-KILLED','JUV-REGIST','INJURED','JUV-OWNER',]
    off = ['SUSPECT','ARRESTEE','SUBJECT','INVOLVED','JUV-ARRESTE',
           'JUV-SUBJECT','JUV-INVOLVE','JUV-OFF/SUS','VICTIM/ARRE',
           'VICTIM/OFFE','JUV-VIC/OFF','JUV-VIC/ARR','JUV-VICTIM',
           'JUV-COMP','SUMMONS','JUV-WARNED','JUV-DIVERTE','DIVERTED',
           'JUV-SUMMONS']
    inv = ['INVOLVED','WITNESS','PASSENGER','JUV-PASSENG','JUV-INVOLVE',
           'JUV-WITNESS','JUV-DRIVER','DRIVER','PEDESTRIAN','FINDER',
           'JUV-PEDESTR','WARNED',]
    killed = ['KILLED','JUV-KILLED']
    officer = ['OFFICER OIS','OFFICER VICTIM']
    # Lets just get rid of officers in entire dataset
    officer_data = ent_data['role'].isin(officer)
    ent_data = ent_data[~officer_data].reset_index(drop=True)
    no_keep = killed # will get rid of people murdered later
    ent_data['vic'] = ent_data['role'].isin(vic)
    ent_data['off'] = ent_data['role'].isin(off)
    ent_data['inv'] = ent_data['role'].isin(inv)
    ent_data['no_keep'] = ent_data['role'].isin(no_keep)
    inv_vars = ['vic','off','inv']
    # Make sure no categories are missing
    tot = vic + off + inv + killed + officer
    cats = ent_data[~ent_data['role'].isin(tot)]['role'].value_counts()
    if cats.shape[0] > 0:
        print('There are some missing roles')
        print(cats)
    
    # Murder 09
    # Rape/Kidnap 11, 10, 64
    # Agg Assault ( 1315, 1314, 1310)
    # Simp Assault (1313, 1316)
    # MV Theft 24
    # Weapon 52
    # Theft 23, 21, 25, 27, 28, 51, 26
    # Burglary 22
    # Robbery 12
    # Mischief/Arson 20, 29, 72, 52, 40, 41, 57, 73, 62, 70
    # Drugs 35
    # Other 54, 100, 30, 37, 39, 64, 53, 36, 38, 50, 48, 55
    
    er = np.floor(ent_data['ucr']/100)
    
    ent_data['Murder'] = 1*(er == 9)
    ent_data['Rape'] = 1*(er.isin([11,10,64])) # check 64
    ent_data['AggAssault'] = 1*(ent_data['ucr'].isin([1315,1314,1310,1311]))
    ent_data['SimpAssault'] = 1*(ent_data['ucr'].isin([1313,1316]))
    ent_data['MVTheft'] = 1*(er.isin([24]))
    ent_data['Weapon'] = 1*(er.isin([52])) # check this
    ent_data['Theft'] = 1*(er.isin([23, 21, 25, 27, 28, 51, 26, 58, 71, 63]))
    ent_data['Burglary'] = 1*(er.isin([22]))
    ent_data['Robbery'] = 1*(er.isin([12]))
    ent_data['Mischief'] = 1*(er.isin([20, 29, 72, 40, 41, 57, 73, 62, 70]))
    ent_data['Drugs'] = 1*(er.isin([35]))
    ent_data['Other'] = 1*(er.isin([54, 100, 30, 37, 39, 53, 36, 38, 50, 48, 55, 99, 93, 90, 91, 49, ]))
    
    # I need to take out simple assault 13B
    ent_data['violent_off'] = (np.floor(ent_data['ucr']/100).astype(int).isin([9,10,11,12,13,52]))*ent_data['off']
    ent_data['violent_vic'] = (np.floor(ent_data['ucr']/100).astype(int).isin([9,10,11,12,13,52]))*ent_data['vic']
    ent_data['violent_vicoff'] = (np.floor(ent_data['ucr']/100).astype(int).isin([9,10,11,12,13,52]))*(ent_data['vic'] + ent_data['off'])
    
    out_vars = ['violent_off','violent_vic','violent_vicoff']
    
    cvars = ['Murder', 'Rape', 'AggAssault', 'SimpAssault', 'MVTheft', 'Weapon', 
             'Theft', 'Burglary', 'Robbery', 'Mischief', 'Drugs', 'Other']
    
    ent_data[cvars] = ent_data[cvars].astype(int)
    ent_data = ent_data.reset_index(drop=True)
    
    # Make sure crime categories are all mapped
    ch = (ent_data[cvars].sum(axis=1) == 1)
    ch_ct = er[~ch].value_counts()
    if ch_ct.shape[0] > 0:
        print('There are some crime categories not mapped')
        print(ch_ct)
    
    # Creating the combo of event types + inv_vars
    cv_iv_li = ['no_keep']
    for iv in inv_vars:
        for cv in cvars:
            cv_iv = cv + "_" + iv
            if ((iv == 'vic') & (cv == 'Murder')):
                # if you are a murder victim, prob not committing future crimes
                pass
            else:
                cv_iv_li.append(cv_iv)
                ent_data[cv_iv] = ent_data[cv]*ent_data[iv]
    
    last_date = ent_data['occ_date'].max()
    ld_m1 = last_date + pd.DateOffset(months=-12,days=1)
    ld_m1_st = ld_m1.strftime('%Y-%m-%d')
    base_dl = ['2022-01-01','2023-01-01','2024-01-01']
    
    # Function to get X data given historical time period, merges to a base dataset
    def get_x(begin,end,suffix,data=ent_data,vs=cv_iv_li,date='occ_date',idv='pin'):
        subd = data[(data[date] >= pd.to_datetime(begin)) & (data[date] < pd.to_datetime(end))]
        gs = subd.groupby(idv,as_index=False)[vs].sum()
        nc = [v + suffix for v in vs]
        gs.columns = [idv] + nc
        return gs
    
    # Getting past 3, past 1, and past 6 months
    def base_date(end):
        # getting X variables
        ed = pd.to_datetime(end)
        ey = int(end[:4])
        b3 = f'{ey - 3}-{end[5:]}'
        gs3 = get_x(b3,end,'p3')
        b1 = f'{ey - 1}-{end[5:]}'
        gs1 = get_x(b1,end,'p1')
        b6m = (ed - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
        gs6m = get_x(b3,end,'6m')
        # getting the Y variable, 1 year in future
        y1f = (ed + pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        year_out = y1f[:4]
        outV = get_x(end,y1f,'y',vs=out_vars)
        # merging all together
        gsc = pd.merge(gs3,gs1,on='pin',how='outer')
        gsc = pd.merge(gsc,gs6m,on='pin',how='outer')
        # For outcome, only want those known to the police prior in some capacity
        gsc = pd.merge(gsc,outV,on='pin',how='left') #indicator='merge_type'
        # setting to ints and filling with zero
        int_vars = list(set(list(gsc)) - set(['merge_type']))
        gsc[int_vars] = gsc[int_vars].fillna(0).astype(int)
        return gsc
    
    f_d = []
    for b in base_dl:
        bd = base_date(b).copy()
        bd['YEAR'] = int(b[:4])
        f_d.append(bd.copy())
    
    train_data = pd.concat(f_d,ignore_index=True)
    final_hold_out = base_date(ld_m1_st)
    
    # get rid of the no-keep variables
    nk_vars = ['no_keepp3','no_keepp1','no_keep6m']
    train_data = train_data[train_data[nk_vars].sum(axis=1) == 0]
    final_hold_out = final_hold_out[final_hold_out[nk_vars].sum(axis=1) == 0]
    train_data.drop(columns=nk_vars,inplace=True)
    final_hold_out.drop(columns=nk_vars,inplace=True)
    
    # Get rid of rows 
    return train_data, final_hold_out

y_vars = ['violent_offy','violent_vicy','violent_vicoffy']
no_vars = ['pin','YEAR']

if os.path.exists(train_data_path):
    train_data = pd.read_csv(train_data_path)
    holdout_data = pd.read_csv(holdout_data_path)
else:
    train_data, holdout_data = get_data()
    train_data.to_csv(train_data_path,index=False)
    holdout_data.to_csv(holdout_data_path,index=False)

x_vars = list(set(list(train_data)) - set(y_vars + no_vars))

# 19,20,21 ~ 22
# 20,21,22 ~ 23
# 21,22,23 ~ 24
# 25 all hold out

# Prior 3 years
# Prior year
# Prior 6 months







