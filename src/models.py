'''
Functions to build
models
'''


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
import numpy as np
import os
import pandas as pd
import pickle
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier, Dataset
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

# Setting the global seed
np.random.seed(10)

def top_k(y_true,y_pred,k=100):
    order = (-1*y_pred).argsort()
    res = y_true[order][:k]
    # returns both weighted and unweighted
    w = res.sum()
    u = res.clip(0,1).sum()
    return w, u

# Just easier function to reset indices
def split(data,test_size=1200,random_state=10):
    train, test = train_test_split(data,test_size=test_size,random_state=random_state)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test

def split_weight(data,test_size=1200,weight='pred_split',random_state=10):
    test = data.sample(test_size,weights=weight, random_state=random_state)
    train = data[~data['uid'].isin(test['uid'])].reset_index(drop=True)
    test.reset_index(drop=True,inplace=True)
    return train, test

# kfold with stratification
def kfold(data,k,strat=None):
    if strat:
        d2 = data[[strat]].copy()
        d2['k'] = -1
        sg = pd.unique(data[strat])
        for s in sg:
            d2l = d2[strat] == s
            ds = d2l.sum()
            tn = int(np.ceil(ds/k))
            tile = np.tile(range(k),tn)
            np.random.shuffle(tile)
            ka = tile[:ds]
            d2.loc[d2l,'k'] = ka
        ka = d2['k'].values
    else:
        ds = data.shape[0]
        tn = int(np.ceil(ds/k))
        tile = np.tile(range(k),tn)
        np.random.shuffle(tile)
        ka = tile[:ds]
    return ka

def kfold_split(data, k, split, strat=None):
    """
    Generate k-fold assignments ensuring groups are never split across folds.
    
    Parameters:
    -----------
    data : DataFrame
        Input data
    k : int
        Number of folds
    split : str
        Column name for grouping. Groups will never be split across folds.
    strat: str
        Column name for stratification. Will only be approximate across groups
        (not implemented yet, use sampling with weights to attempt!)
    
    Returns:
    --------
    numpy.ndarray
        Array of fold assignments (0 to k-1)
    """
    # Get unique groups and shuffle them
    unique_groups = data[split].unique()
    np.random.shuffle(unique_groups)
    
    # strat sampling with weights
    
    # Assign each group to a fold
    n_groups = len(unique_groups)
    tn = int(np.ceil(n_groups / k))
    tile = np.tile(range(k), tn)
    group_to_fold = dict(zip(unique_groups, tile[:n_groups]))
    
    # Map each row to its group's fold assignment
    ka = data[split].map(group_to_fold).values
    
    return ka


# Just a wrapper around sklearn
# so it returns a pandas dataframe with named
# columns
class DumEnc():
    def __init__(self,dtype=int):
        self.OHE = OneHotEncoder(dtype=dtype,
                                 handle_unknown='ignore')
        self.var_names = None
    def fit(self, X):
        self.OHE.fit(X)
        cats = self.OHE.categories_
        var = list(X)
        vn = []
        for v,ca in zip(var,cats):
            for c in ca:
                vn.append(f'{v}_{c}')
        self.var_names = vn
    def transform(self,X):
        res = pd.DataFrame(self.OHE.transform(X),
                           columns=self.var_names,
                           index=X.index)
        return res

def dummy_stats(values,begin_date):
    vdate = pd.to_datetime(values,errors='ignore')
    year = vdate.dt.year
    month = vdate.dt.month
    week_day = vdate.dt.dayofweek
    diff_days = (vdate - begin_date).dt.days
    # if binary, turn week/month into dummy variables
    return diff_days, week_day, month, year

def circle_stats(values,begin_date):
    vdate = pd.to_datetime(values,errors='ignore')
    within_year = vdate.dt.dayofyear
    week_day = vdate.dt.dayofweek
    # calculate sine/cosine for within year
    year_cos = np.cos(within_year*(2*np.pi/365))
    year_sin = np.sin(within_year*(2*np.pi/365))
    # calculate sine/cosine for within week
    week_cos = np.cos(week_day*(2*np.pi/7))
    week_sin = np.sin(week_day*(2*np.pi/7))
    diff_days = (vdate - begin_date).dt.days
    return diff_days, year_cos, year_sin, week_cos, week_sin


class DateEnc():
    def __init__(self,
                 begin = '1/1/2015',
                 dummy = True,
                 dum_types=['days','weekday','month']):
        self.begin = pd.to_datetime(begin)
        self.dummy = dummy
        # 'days','weekday','month','year'
        self.dum_types = dum_types
        self.cat_vars = []
        # Setting categorical variables
    def fit(self,X):
        # These are just fixed functions
        pass
    def transform(self,X):
        vars = list(X)
        res = []
        res_labs = []
        cat_labs = []
        if self.dummy:
            for v in vars:
                dd, week_day, month, year = dummy_stats(X[v],self.begin)
                if 'days' in self.dum_types:
                    res.append(dd) # this is not likely to be categorical
                    res_labs.append(f'days_{v}')
                if 'weekday' in self.dum_types:
                    res.append(week_day)
                    res_labs.append(f'weekday_{v}')
                    cat_labs.append(f'weekday_{v}')
                if 'month' in self.dum_types:
                    res.append(month)
                    res_labs.append(f'month_{v}')
                    cat_labs.append(f'weekday_{v}')
                if 'year' in self.dum_types:
                    res.append(year)
                    res_labs.append(f'year_{v}')
                    cat_labs.append(f'year_{v}')
            self.cat_vars = cat_labs
        else:
             for v in vars:
                dd, year_cos, year_sin, week_cos, week_sin = circle_stats(X[v],self.begin)
                res += [dd, year_cos, year_sin, week_cos, week_sin]
                res_labs += [f'days_{v}',f'yearcos_{v}',f'yearsin_{v}',f'weekcos_{v}',f'weeksin_{v}']
        res_df = pd.concat(res,axis=1)
        res_df.index = X.index
        res_df.columns = res_labs
        return res_df


# Spline encoding
# defaults to regular knots
# or specified locations
class SplEnc():
    def __init__(self):
        pass
    def fit(self,X):
        pass
    def transform(self,X):
        pass


class IdentEnc():
    def __init__(self):
        self.note = None
    def fit(self,X):
        pass
    def transform(self,X):
        return X.copy()

class SimpleOrdEnc():
    def __init__(self,
                 dtype=int,
                 unknown_value=-1,
                 lim_k=None,
                 lim_count=None):
        self.unknown_value = unknown_value
        self.dtype = dtype
        self.lim_k = lim_k
        self.lim_count = lim_count
        self.vars = None
        self.soe = None
    def fit(self, X):
        self.vars = list(X)
        # Now creating fit for each variable
        res_oe = {}
        for v in list(X):
            res_oe[v] = OrdinalEncoder(dtype=self.dtype,
                handle_unknown='use_encoded_value',
                        unknown_value=self.unknown_value)
            # Get unique values minus missing
            xc = X[v].value_counts().reset_index()
            xc.columns = [v, "Freq"]
            # If lim_k, only taking top K value
            if self.lim_k:
                top_k = self.lim_k - 1
                un_vals = xc.loc[0:top_k,:]
            # If count, using that to filter
            elif self.lim_count:
                un_vals = xc[xc["Freq"] >= self.lim_count].copy()
            # If neither
            else:
                un_vals = xc
            # Now fitting the encoder for one variable
            res_oe[v].fit(un_vals[[v]])
        # Appending back to the big class
        self.soe = res_oe
    # Defining transform/inverse_transform classes
    def transform(self, X):
        xcop = X[self.vars].copy()
        for v in self.vars:
            xcop[v] = self.soe[v].transform( X[[v]].fillna(self.unknown_value) )
        return xcop
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        xcop = X[self.vars].copy()
        for v in self.vars:
            xcop[v] = self.soe[v].inverse_transform( X[[v]].fillna(self.unknown_value) )
        return xcop


class FeatureEngine():
    def __init__(self,
                 ord_vars = None,
                 dum_vars = None,
                 spl_vars = None,
                 dat_vars = None,
                 ide_vars = None,
                 scale = None):
        self.fin_vars = None
        self.enc_dict = {}
        self.ord_vars = ord_vars
        self.dum_vars = dum_vars
        self.spl_vars = spl_vars
        self.dat_vars = dat_vars
        self.ide_vars = ide_vars
        self.cat_vars = None
        self.ord = SimpleOrdEnc()
        self.dum = DumEnc()
        self.dat = DateEnc()
        self.spl = SplEnc()
        self.ide = IdentEnc()
        self.scale = scale
        enc_vars = [ord_vars,dum_vars,spl_vars,dat_vars,ide_vars]
        enc_mods = [self.ord, self.dum, self.spl, self.dat, self.ide]
        for v,m in zip(enc_vars,enc_mods):
            if v is not None:
                self.enc_dict[tuple(v)] = m
    def fit(self, X):
        res = []
        for v,m in self.enc_dict.items():
            m.fit(X[list(v)])
            rf = m.transform(X[list(v)])
            res.append(rf)
        # doing a transform to know the variable names in the end
        res_df = pd.concat(res,axis=1)
        if self.scale is not None:
            self.scale.fit(res_df)
            res_df = pd.DataFrame(self.scale.transform(res_df),columns=list(res_df))
        self.fin_vars = list(res_df)
        # Adding categorical variables back in
        cat_vars = []
        if self.dum_vars is not None:
            cat_vars += self.dum.var_names
        if self.dat_vars is not None:
            cat_vars += self.dat.cat_vars
        if self.ord_vars is not None:
            cat_vars += self.ord_vars
        self.cat_vars = cat_vars
        return res_df
    def transform(self, X):
        res = []
        for v,m in self.enc_dict.items():
            res.append(m.transform(X[list(v)]))
        res_df = pd.concat(res,axis=1)
        if self.scale is not None:
            res_df = pd.DataFrame(self.scale.transform(res_df),columns=list(res_df))
        return res_df


# Class to insert different types of
# regression models
class Mod():
    def __init__(self,
                 ord_vars = None,
                 dum_vars = None,
                 dat_vars = None,
                 ide_vars = None,
                        y = None,
                transform = None,
                inv_trans = None,
                   weight = None,
                  scale_x = None,
                  fit_cat = False,
                  bin_y = True,
                      mod = CatBoostClassifier(n_estimators=100, max_depth=3)):
        self.fe = FeatureEngine(ord_vars=ord_vars,
                           dat_vars=dat_vars,
                           dum_vars=dum_vars,
                           ide_vars=ide_vars,
                           scale = scale_x)
        self.transform = transform
        self.inv_trans = inv_trans
        self.bin_y = bin_y
        self.mod = mod
        self.y = y
        self.resids = None
        self.cat_vars = None
        self.weight = weight
        self.metrics = None
        self.fit_cat = fit_cat
    def fit(self, X):
        if (self.weight is not None):
            sw = X[self.weight]
        else:
            sw = None
            #print('NOT using Weights in fit')
        y_dat = X[self.y].copy()
        if self.transform:
            y_dat = self.transform(y_dat)
        if self.bin_y:
            y_dat = y_dat.clip(0,1)
        X_dat = self.fe.fit(X)
        self.cat_vars = self.fe.cat_vars
        # If catboost or lightgbm, pass in categories
        if ((type(self.mod) in [CatBoostClassifier,CatBoostRegressor]) & (self.fit_cat)):
            vt = list(X_dat)
            if self.cat_vars is not None:
                ci = [vt.index(c) for c in self.cat_vars]
            else:
                ci = None
            self.mod.fit(X_dat,y_dat,sample_weight=sw,cat_features=ci)
        elif ((type(self.mod) == LGBMClassifier) & (self.fit_cat)):
            for v in self.cat_vars:
                X_dat[v] = X_dat[v].astype('category')
            self.mod.fit(X_dat, y_dat, sample_weight=sw)
        else:
            self.mod.fit(X_dat,y_dat,sample_weight=sw)
        pred = self.mod.predict(X_dat)
        self.resids = pd.Series(y_dat - pred)
    def predict(self,X,duan=True):
        X_dat = self.fe.transform(X)
        if self.fit_cat & (type(self.mod) in [LGBMRegressor,LGBMClassifier]):
            for v in self.cat_vars:
                X_dat[v] = X_dat[v].astype('category')
        if self.bin_y:
            pred = pd.Series(self.mod.predict_proba(X_dat)[:,1],X.index)
        else:
            pred = pd.Series(self.mod.predict(X_dat), X.index)
        # if transform, do Duans smearing
        if (self.transform is not None) & duan:
            resids = self.resids
            resids = resids.values.reshape(1,resids.shape[0])
            dp = self.inv_trans(pred.values.reshape(X.shape[0],1) + resids)
            pred = pd.Series(dp.mean(axis=1), X.index)
        return pred
    def feat_import(self):
        var_li = self.fe.fin_vars
        mod_fi = self.mod.feature_importances_
        res_df = pd.DataFrame(zip(var_li,mod_fi),columns = ['Var','FI'])
        res_df.sort_values('FI',ascending=False,inplace=True,ignore_index=True)
        # Normalize to sum to 1
        res_df['FI'] = res_df['FI']/res_df['FI'].sum()
        return res_df
    def met_eval(self,data,ki):
        metric_dat = []
        kvals = pd.unique(ki)
        kvals.sort()
        for k in kvals:
            train = data[ki != k].reset_index(drop=True)
            test = data[ki == k].reset_index(drop=True)
            self.fit(train)
            test['pred'] = self.predict(test)
            met_di = metrics(self.y,'pred',test)
            met_di['k'] = k
            metric_dat.append(met_di.copy())
        mpd = pd.DataFrame(metric_dat)
        self.metrics = mpd


def metrics(y_true,y_pred,data):
    # RMSE
    #rmse = mean_squared_error(data[y_true],data[y_pred])
    # weighted ROC/AUC
    y1 = data[y_true].clip(0,1)
    auc = roc_auc_score(y1,data[y_pred],sample_weight=data[y_true].clip(1))
    # non weighted ROC/AUC
    auc_nw = roc_auc_score(y1,data[y_pred])
    # Top 100 captures
    we, unwe = top_k(data[y_true],data[y_pred])
    # Top 1000 captures
    we1k, unwe1k = top_k(data[y_true],data[y_pred],k=1000)
    # R2
    #r2 = r2_score(data[y_true],data[y_pred])
    # top 100
    dat = {'WeightedAUC': auc, 
           'AUC': auc_nw, 
           'Top100_Weight': we, 
           'Top100Bin': unwe, 
           'Top1000_Weight': we1k, 
           'Top1000Bin': unwe1k}
    return dat


def save_model(mod,name):
    fname = f'./models/{name}.pkl'
    outfile = open(fname,"wb")
    pickle.dump(mod,outfile)
    outfile.close()


def load_model(name):
    fname = f'./models/{name}.pkl'
    infile = open(fname, "rb")
    mod = pickle.load(infile)
    infile.close()
    return mod