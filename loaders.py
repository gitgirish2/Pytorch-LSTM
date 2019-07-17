#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
import pandas as pd
import torch
import numpy as np
import copy
from bisect import bisect
from datetime import datetime, timedelta
from os import listdir
from os.path import join, isdir
from collections import namedtuple
from torch.utils.data import Dataset
import random
import glob, os
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas.io import gbq
from google.cloud import bigquery
from numpy import split
from numpy import array
from scipy import optimize
from scipy.stats import norm
import math


def load_data(data_dir='/home/app/replen_data/'):
    """Processes raw data and returns ISD, SD, S, I, D dataframes.

    Both the raw data and processed datasets are cached on disk separately.
    """
    data_raw_path = os.path.join(data_dir,'raw_data.pt')
    data_post_path = os.path.join(data_dir,'ISDdata.pt')

    if not os.path.exists(data_post_path):
        # Load raw data
        if not os.path.exists(data_raw_path):
            # Download raw data
            print('Downloading raw data.')
            data_raw = _load_raw_data(data_dir)
            torch.save(data_raw, data_raw_path)  # Cache raw data locally
        else:
            print('Raw data previously saved, loading from disk.')
            data_raw = torch.load(data_raw_path)
        print('Finished loading raw data.')

        # Process raw data
        print('Processing raw data.')
        ISD, SD, S, I, D = _process_data(data_raw)
        torch.save({'ISD': ISD,'SD': SD,'S': S,'I': I,'D': D}, data_post_path)   # Cache processed data locally
    else:
        print('Processed data previously saved, loading from disk.')
        data = torch.load(data_post_path)
        ISD, SD, S, I, D = data['ISD'], data['SD'], data['S'], data['I'], data['D']
    print('Finished loading processed data.')

    return ISD, SD, S, I, D


def _load_raw_data(data_dir):
    if not os.path.exists(os.path.join(data_dir, '000000000006')):
        os.system('gsutil cp gs://replenishment-data/data/* ' + data_dir)
    raw_data = pd.concat(map(pd.read_csv, glob.glob(os.path.join(data_dir, "*"))))
    return raw_data


def _process_data(raw_data):
    raw_data.columns = ["department_nbr","department_desc","category_nbr","category_desc","upc_nbr","item_desc",
                    "cid","fineline_nbr","fineline_desc","brand_id","brand_name","sell_qty","size_desc",
                    "whse_packsize","vendor_packsize","item1_desc","base_unit_retail_amt","vendor_name",
                    "modular_based_mdse_code","season_code","item_status_code","item_status_change_date",
                    "item_type_code","item_type_desc","repl_subtype_code","repl_subtype_desc",
                    "item_create_date","item_nbr","store_nbr","cal_date","wm_yr_wk_id","cal_week_day_nbr",
                    "bus_date","unit_cost","retail_price","unit_qty","on_hand_qty","total_trans","rc_0",
                    "rc_1","rc_2","rc_3","rc_4","rc_5","rc_6","rc_7","rc_8","rc_9","aur","oh_last_day",
                    "store_name","street_addr_line1","city_name","open_date","store_type_desc","latitude_dgr",
                    "longitude_dgr"]

    data = raw_data[(raw_data['category_nbr'] == 1514)]
    print('DATA PRE:', data.shape)

    data = data.sort_values(['item_nbr','store_nbr','bus_date'],ascending=True)
    data['item_store'] = data['item_nbr'].map(str) + data['store_nbr'].map(str)
    data = data.sort_values(['item_nbr','store_nbr','bus_date'])
    data['on_hand_qty_today_EOD'] = data['on_hand_qty']
    data['on_hand_qty_lag'] = data.groupby(['item_nbr','store_nbr'])['on_hand_qty'].shift(1)
    data['on_hand_qty'] = data['on_hand_qty_lag']
    data['on_hand_qty'] = data['on_hand_qty'].replace(np.nan, 0)
    data['unit_qty'] = data['unit_qty'].replace(np.nan, 0)
    data['demand_inv_gap'] = data['on_hand_qty'] - data['unit_qty']
    data = data[(data['unit_qty'] >= 0)]
    data = data[(data['demand_inv_gap'] >= 0)]
    data = data[(data['on_hand_qty'] >= 0)]

    data['trans_flag'] = np.where(data['unit_qty'] > 0 ,1,0)
    data['last3months_flag'] = np.where(data['wm_yr_wk_id'] > 11830 ,1,0)
    data['total_count'] = data.groupby(['item_nbr','store_nbr'])['bus_date'].transform(np.size)
    data['trans_count'] = data.groupby(['item_nbr','store_nbr'])['trans_flag'].transform(np.sum)
    data['last3months_count'] = data.groupby(['item_nbr','store_nbr'])['last3months_flag'].transform(np.sum)
    data['unique_gap'] = data.groupby(['item_nbr','store_nbr'])['demand_inv_gap'].transform(unique_count)

    data = data[(data['total_count'] >= 90)]
    data = data[(data['last3months_count'] >=30)]
    data = data[(data['trans_count'] >= 30)]
    data = data[(data['unique_gap'] >= 10)]

    # Day for which scan table has transactions
    trans_subset = data[(data['trans_flag'] == 1)]
    trans_subset = trans_subset.groupby(['item_nbr','store_nbr'])['bus_date'].min().reset_index()
    trans_subset.columns = ['item_nbr','store_nbr', 'min_trans_date']
    data = pd.merge(data,trans_subset, how='left', on=['item_nbr', 'store_nbr'] )
    data = data[(data['bus_date'] >= data['min_trans_date'])]

    data = data.sort_values(['item_nbr','store_nbr','bus_date'], ascending=True)
    # Row number to the dates
    data['date_num'] = data.groupby(['item_nbr','store_nbr']).cumcount() + 1
    # Number of units recieved in shipments & actual lead time
    data['shipment_recieved'] = data['on_hand_qty_today_EOD'] - ( data['on_hand_qty'] - data['unit_qty'])

    # Shipment order gap
    data['shipment_recieved'] = np.where(data['shipment_recieved'] <= 1, 0, data['shipment_recieved'] )
    data['shipment_order_lag'] = data.item_store.groupby((data.shipment_recieved != data.shipment_recieved.shift()).cumsum()).cumcount() + 1
    data['shipment_order_gap_cons'] = data.item_store.groupby((data.shipment_recieved != data.shipment_recieved.shift()).cumsum()).transform(np.size)
    data['shipment_order_gap'] = data.groupby(['item_nbr','store_nbr'])['shipment_order_gap_cons'].transform(mod_cal)
    data['cal_date'] =  pd.to_datetime(data['cal_date'])

    # Reset the indices to be consecutive
    data = data.reset_index(drop=True)
    print('DATA POST:', data.shape)

    print('there', type(data['cal_date']))

    ISD = data.loc[:, ['item_nbr','store_nbr','cal_date','demand_inv_gap','unit_qty', 'on_hand_qty' , 'aur', 'rc_7' , 'whse_packsize',
                        'oh_last_day','oh_last_day', 'rc_8', 'shipment_recieved', 'rc_9', 'unit_cost']]

    ISD.columns =  ['ITEM_NBR','STORE_NBR','DAY','GAP','QTY_SOLD','ON_HAND_QTY','PRICE','PROMOTIONS','MODULAR_CAPACITY',
        'SAFETY_STOCK','FORECAST','FEATURE_PROMOTION','SHIPMENT_RECIEVED','ASSORTMENT_FACINGS','UNIT_COST']

    ISD.fillna(0 , inplace=True)

    ISD.GAP = ISD.GAP.astype(int)
    ISD.QTY_SOLD = ISD.QTY_SOLD.astype(int)
    ISD.ON_HAND_QTY = ISD.ON_HAND_QTY.astype(int)
    ISD.PRICE = ISD.PRICE.astype(float)
    ISD.PROMOTIONS = ISD.PROMOTIONS.astype(int)
    ISD.MODULAR_CAPACITY = ISD.MODULAR_CAPACITY.astype(int)

    ISD.SAFETY_STOCK = ISD.SAFETY_STOCK.astype(float)
    ISD.FORECAST = ISD.FORECAST.astype(float)
    ISD.FEATURE_PROMOTION = ISD.FEATURE_PROMOTION.astype(int)
    ISD.SHIPMENT_RECIEVED = ISD.SHIPMENT_RECIEVED.astype(int)
    ISD.ASSORTMENT_FACINGS = ISD.ASSORTMENT_FACINGS.astype(int)
    ISD.UNIT_COST = ISD.UNIT_COST.astype(float)

    SD = data.loc[:, ['store_nbr','cal_date', 'rc_7','aur','rc_1','rc_2','rc_3']]
    SD = SD.groupby(['store_nbr','cal_date'], as_index=False).agg({'rc_7': 'max' ,'aur' :'mean','rc_1' :'max',
                                                         'rc_2': 'max' ,'rc_3' :'max'})
    SD.columns = ['STORE_NBR','DAY','SNOW_INDICATOR', 'AVG_TEMPERATURE','NASCAR','NFL','STORE_COLLEGE_FOOTBALL']

    SD.fillna(0 , inplace=True)

    SD.STORE_NBR = SD.STORE_NBR.astype(int)
    SD.SNOW_INDICATOR = SD.SNOW_INDICATOR.astype(int)
    SD.AVG_TEMPERATURE = SD.AVG_TEMPERATURE.astype(float)
    SD.NASCAR = SD.NASCAR.astype(int)
    SD.NFL = SD.NFL.astype(int)
    SD.STORE_COLLEGE_FOOTBALL = SD.STORE_COLLEGE_FOOTBALL.astype(int)

    S = data.loc[:, ['store_nbr','rc_4','rc_5',"latitude_dgr","longitude_dgr", 'rc_7']]
    S = S.groupby(['store_nbr'], as_index=False).agg({'rc_4': 'max' ,'rc_5' :'max','latitude_dgr' :'mean',
                                                         'longitude_dgr': 'mean' ,'rc_7' :'max'})
    S.columns = ['STORE_NBR','CITY_NAME_ENCODE','STORE_TYPE','LATITUDE','LONGITUDE','TIME_SINCE_STORE_OPEN']

    S.fillna(0 , inplace=True)

    S.STORE_NBR = S.STORE_NBR.astype(int)
    S.CITY_NAME_ENCODE = S.CITY_NAME_ENCODE.astype(int)
    S.STORE_TYPE = S.STORE_TYPE.astype(int)
    S.LATITUDE = S.LATITUDE.astype(float)
    S.LONGITUDE = S.LONGITUDE.astype(float)
    S.TIME_SINCE_STORE_OPEN = S.TIME_SINCE_STORE_OPEN.astype(int)


    I = data.loc[:, ['item_nbr','department_nbr','category_nbr',"fineline_nbr","cid", 'brand_id','whse_packsize',
                     'vendor_packsize','rc_7','season_code','rc_1','rc_8']]
    I = I.groupby(['item_nbr'], as_index=False).agg({'department_nbr': 'max' ,'category_nbr' :'max','fineline_nbr' :'max',
                                                     'cid': 'max' ,'brand_id' :'max','whse_packsize' :'max',
                                                     'vendor_packsize': 'max' ,'rc_7' :'max','season_code' :'max',
                                                         'rc_1': 'max' ,'rc_8' :'max'})


    I.columns = ['ITEM_NBR','DEPARTMENT_NBR','CATEGORY_NBR','FIBELINE_NBR','CID','BRAND_ID',
            'WAREHOUSE_PACKSIZE','VENDOR_PACKSIZE','VENDOR_NAME_ENCODE','SEASON_CODE','ITEM_ACTIVE_STATUS','ITEM_CREATE_DATE_SINCE']

    I.fillna(0 , inplace=True)

    I.ITEM_NBR = I.ITEM_NBR.astype(int)
    I.DEPARTMENT_NBR = I.DEPARTMENT_NBR.astype(int)
    I.CATEGORY_NBR = I.CATEGORY_NBR.astype(int)
    I.FIBELINE_NBR = I.FIBELINE_NBR.astype(int)
    I.CID = I.CID.astype(int)
    I.BRAND_ID = I.BRAND_ID.astype(int)
    I.WAREHOUSE_PACKSIZE = I.WAREHOUSE_PACKSIZE.astype(int)
    I.VENDOR_PACKSIZE = I.VENDOR_PACKSIZE.astype(int)
    I.VENDOR_NAME_ENCODE = I.VENDOR_NAME_ENCODE.astype(int)
    I.SEASON_CODE = I.SEASON_CODE.astype(int)
    I.ITEM_ACTIVE_STATUS = I.ITEM_ACTIVE_STATUS.astype(int)
    I.ITEM_CREATE_DATE_SINCE = I.ITEM_CREATE_DATE_SINCE.astype(int)

    D = data.loc[:, ['cal_date','rc_7','cal_week_day_nbr','rc_8',"wm_yr_wk_id","rc_1", 'rc_2']]
    D = D.groupby(['cal_date'], as_index=False).agg({'rc_7': 'max' ,'cal_week_day_nbr' :'max','rc_8' :'max',
                                                         'wm_yr_wk_id': 'max' ,'rc_1' :'max','rc_2' :'max'})
    D.columns = ['DAY','HOLIDAY_EVENTS','WEEKDAY_NBR','MONTH','WEEK_NBR','WALMART_WEEK_NBR','YEAR']

    D.fillna(0 , inplace=True)

    D.HOLIDAY_EVENTS = D.HOLIDAY_EVENTS.astype(int)
    D.WEEKDAY_NBR = D.WEEKDAY_NBR.astype(int)
    D.MONTH = D.MONTH.astype(int)
    D.WEEK_NBR = D.WEEK_NBR.astype(int)
    D.WALMART_WEEK_NBR = D.WALMART_WEEK_NBR.astype(int)
    D.YEAR = D.YEAR.astype(int)

    return ISD, SD, S, I, D


def unique_count(x):
    x_array = np.array(np.unique(x, return_counts=True)).T
    x_DF = pd.DataFrame(x_array)
    x_DF.columns = ['Quantity' , 'Frequency']
    k = x_DF['Quantity'].nunique()
    return k


def mod_cal(x):
    x_array = np.array(np.unique(x, return_counts=True)).T
    x_DF = pd.DataFrame(x_array)
    x_DF.columns = ['Quantity' , 'Frequency']
    x_DF = x_DF.sort_values(['Frequency'],ascending=False)
    x_DF = x_DF.iloc[0:3]
    k = x_DF['Quantity'].mean()
    return k

class ReplenishmentDataset(Dataset):

    def __init__(self, ISD, D, SD, I, S, seq_len=14, cv=None, train=True, target='GAP'):
        """TODO: Describe parameters
        """
        self.seq_len = seq_len
        self.seq_delta = timedelta(days=seq_len)
        self.cv = cv
        self.train = train
        self.target = target
        self.to_embed = ['PROMOTIONS', 'FEATURE_PROMOTION', 'SNOW_INDICATOR', 'NASCAR', 'NFL', 'STORE_COLLEGE_FOOTBALL', 'STORE_NBR', 'CITY_NAME_ENCODE','STORE_TYPE', 'ITEM_NBR', 'DEPARTMENT_NBR', 'CATEGORY_NBR', 'FIBELINE_NBR', 'CID', 'BRAND_ID', 'VENDOR_NAME_ENCODE', 'SEASON_CODE', 'ITEM_ACTIVE_STATUS', 'HOLIDAY_EVENTS', 'WEEKDAY_NBR', 'MONTH', 'WEEK_NBR', 'WALMART_WEEK_NBR', 'YEAR']
        self.to_skip = []
#         self.to_skip = ['NASCAR', 'NFL', 'STORE_COLLEGE_FOOTBALL', 'VENDOR_NAME_ENCODE', 'ITEM_CREATE_DATE_SINCE', 'MODULAR_CAPACITY', 'SAFETY_STOCK', 'FORECAST', 'FEATURE_PROMOTION', 'SHIPMENT_RECIEVED', 'ASSORTMENT_FACINGS', 'ITEM_ACTIVE_STATUS', 'ITEM_CREATE_DATE_SINCE',' VENDOR_NAME_ENCODE']
        self.dynamic=['ISD','D','SD']
        self.static=['I','S']
        self._init_emb_dict([ISD, D, SD, I, S])
        self._init_data(ISD, D, SD, I, S)

    def _count_dim(self):
        self.num_seq_real, self.num_seq_int, self.num_stat_real, self.num_stat_int  = 0, 0, 0, 0
        # Dynamic features
        for data_id in self.dynamic:
            data = getattr(self,data_id)
            for key in data.keys():
                if key not in ['ITEM_NBR', 'STORE_NBR', 'DAY'] + self.to_skip:
                    if key in self.to_embed:
                        self.num_seq_int += 1
                    else:
                        self.num_seq_real += 1
        # Static features
        for data_id in self.static:
            data = getattr(self,data_id)
            for key in data.keys():
                if key not in self.to_skip:
                    if key in self.to_embed:
                        self.num_stat_int += 1
                    else:
                        self.num_stat_real += 1

    def _init_data(self, ISD, D, SD, I, S):
        """finish me
         here we process the data structures to
        a)V select only train or test data if cv_out=None revert to using the last two seq_len of data as test set
        b)V substitute the entries of variables that need to be used by nn.embedding layers with their index
        c) preprocess (e.g. normalization for train and test using the train set moments)
        d)V get the list of sample indices from the preprocessed data structure that can be used as input for _getitem_
         """
        data_list=[]
        for data in [ISD, D, SD, I, S]:
            data_list.append(self._init_data_helper(data))

        self.ISD, self.D, self.SD, self.I, self.S = data_list[0],data_list[1],data_list[2],data_list[3],data_list[4] #this are the outputs of
        self._init_isd_samples()   #step d)
        self._count_dim()

    def _init_emb_dict(self, data_frames):
        emb_dict = {}
        for df in data_frames:  # Loops throug the data structures
            for key in df.keys():  # Loops through the key of each data struct
                if key in self.to_embed:  # Check for keys that must be embedded
                    if key in emb_dict.keys():
                        emb_dict[key] = np.unique(np.concatenate((emb_dict.get(key), df[key].values)), axis=0)
                    else:
                        emb_dict[key] = np.unique(df[key])
        self.emb_dict = emb_dict

    def _init_data_helper(self, data):
#         print(data.head())
#         data_c= copy.deepcopy(data)
        for key in self.emb_dict.keys():
            if key in data.keys():
                data[key] = data[key].replace(list(self.emb_dict.get(key)),list(range(len(self.emb_dict.get(key)))))
#         print(data_c.head())
        return data

    def _preprocess(self):
        pass



    def __len__(self):
        return len(self.isd_samples)



    def _init_isd_samples(self):
        """Construct the mapping of sampleable isd's to index in ISD df.
        Each sample loads temporal data between (day - seq_len) and  (day + seq_len), so it's important that we
        ensure no overlap in days observed between the train and test set.
        """

        #print('yo' , type(self.ISD['DAY'].min()))
        #print('yo' , type(self.seq_delta))
        og_idxs = self.ISD.index

        # Check that ISD indices are complete and consecutive
        assert sorted(og_idxs) == list(range(min(og_idxs), max(og_idxs) + 1))

        min_day = (self.ISD['DAY'].min()) + self.seq_delta
        max_day = self.ISD['DAY'].max() - self.seq_delta

        samples = copy.deepcopy(self.ISD)
        samples['idx'] = samples.index
        #print('check_sampleidx' , samples['idx'])
        samples = samples.loc[(samples['DAY'] >= min_day) & (samples['DAY'] <= max_day)]
        #print('num samples',len(samples))
        #print('347_check_sampleidx' , samples['idx'])
        print('Sampled %:', len(samples)/len(self.ISD))

        if self.cv is not None:
            print("Inside init_isd****************")
            for k, v in self.cv.items():
                print('{}: {}'.format(k, v))

            cv_start = self.cv['start'] - self.seq_delta
            cv_end = self.cv['end'] + self.seq_delta
        else:
            # Defaults to a test set 2*seq_delta long
            cv_start = max_day - 2*self.seq_delta
            cv_end = max_day

        assert cv_start <= cv_end
        assert cv_start >= min_day
        assert cv_end <= max_day
        #print('start',cv_start)
        #print('en d',cv_end)

        if self.train:
            # Each sample will be predicted seq_len out from sample day, so ensure there's no overlap between
            # train predictions and test warmup (test day - seq_len).
            #print('in train')
            samples = samples.loc[(samples['DAY'] < cv_start - self.seq_delta)
                                 | (samples['DAY'] > cv_end + self.seq_delta)]
            #print('after loc',len(samples))
        else:
            samples = samples.loc[(samples['DAY'] >= cv_start) & (samples['DAY'] <= cv_end)]
        #print('373_check_sampleidx' , samples['idx'])
        self.isd_samples = samples['idx'].tolist()

        print('Sampled %:', len(samples)/len(self.ISD))

        og_idxs = self.ISD.index
        assert len(self.isd_samples) <= len(og_idxs)
        for idx in self.isd_samples:
            if idx not in og_idxs:
                print('index {} is not in ISD'.format(idx))
                sys.exit()

    def ISD_2_I_S_D(self, isd_idx):
        """Returns the full i, s, d and sd vectors given the isd index."""
        #print( 'check ' , max(self.isd_samples) , len(self.ISD))
        isd = self.ISD.iloc[isd_idx]
        i = self.I.loc[self.I['ITEM_NBR'] == isd['ITEM_NBR']]
#         .drop(['ITEM_NBR'], axis=1)
        s = self.S.loc[self.S['STORE_NBR'] == isd['STORE_NBR']]
        # TODO: ENSURE ISD HAS 'DAY' COLUMN (datetime)
        # TODO: ENSURE D HAS 'DAY' COLUMN (datetime)
        # TODO: ENSURE SD HAS 'STORE_NBR' COLUMN

        d = self.D.loc[(self.D['DAY'] >= (isd['DAY'])-self.seq_delta)&(self.D['DAY'] < isd['DAY']+self.seq_delta)]
        sd = self.SD.loc[(self.SD['DAY'] >= isd['DAY']-self.seq_delta)&(self.SD['DAY'] < isd['DAY']+self.seq_delta)&(self.SD['STORE_NBR']==isd['STORE_NBR'])]
        #print('SD LEN: {}'.format(len(sd)))
        return i, s, d, sd

    def debug(self):
        print("Debugging dataset...")
        num_samples = len(self.isd_samples)
        print("{} samples".format(num_samples))
        print("{} isd".format(len(self.ISD)))
        count = 0
        og_idxs = self.ISD.index
        failed = []
        for idx in self.isd_samples:
            try:
                _ = self.ISD.iloc[idx]
            except IndexError:
                print('idx {} failed iloc'.format(idx))
                failed.append(idx)
            if count % 10000 == 0:
                print("{}% complete".format(round(count/num_samples * 100.0), 3))
            count += 1
        import pdb; pdb.set_trace()

    def __getitem__(self, j):
            """get and index i as input coming from len(self.isd_samples) use it to get the isd,it,s,d,sd indices
            returns 4 data structures of sizes:
            1) 2xseq_len,self.num_seq_real
            2) 2xseq_len,self.num_seq_int
            3) self.num_stat_real
            4) self.num_stat_int
            """
            isd_idx = self.isd_samples[j]  # Returns index of an isd row in self.ISD that can be constructed into a sample
            i, s, d, sd = self.ISD_2_I_S_D(isd_idx)

            isd_seq = self.ISD.iloc[isd_idx - self.seq_len:isd_idx + self.seq_len] #### Confirm isd_idx is correctly used
            seq_real = np.zeros((2 * self.seq_len, self.num_seq_real  ), float)
            seq_int = np.zeros((2 * self.seq_len, self.num_seq_int ), int)
            y_type = self.ISD.dtypes[self.target]
            targets = np.zeros(2 * self.seq_len, y_type)
            real_id, int_id = 0, 0

            # TODO: use dict of {key: seq_int/real idx} so that we don't have to assume pandas keys are always in the same order

            seq_int_dict = {}
            seq_real_dict = {}
            stat_int_dict = {}
            stat_real_dict = {}

            ### Dynamic Features ###

            # loops through isd
            #print('isd_seq.keys()' , isd_seq.keys())
            for key in isd_seq.keys():
                if key not in ['STORE_NBR','ITEM_NBR','DAY'] + self.to_skip:
                    if key == self.target:
                        targets[:] = isd_seq[key]
                    if key in self.to_embed:
                        seq_int_dict[int_id] = key
                        seq_int[:, int_id] = isd_seq[key]
                        int_id += 1
                    else:
                        seq_real_dict[real_id] = key
                        seq_real[:, real_id] = isd_seq[key]
                        real_id += 1
            #print('isd',len(isd_seq))
            # loops through sd
            sd_seq = sd
            for key in sd_seq.keys():
                if key not in ['STORE_NBR','DAY'] + self.to_skip:
                    if key in self.to_embed:
                        seq_int_dict[int_id] = key
                        seq_int[:, int_id] = sd_seq[key]
                        int_id += 1
                    else:
                        seq_real_dict[real_id] = key
                        seq_real[:, real_id] = sd_seq[key]
                        real_id += 1

            # loops through d
            d_seq = d
            for key in d_seq.keys():
                if key not in ['DAY'] + self.to_skip:
                    if key in self.to_embed:
                        seq_int_dict[int_id] = key
                        seq_int[:, int_id] = d_seq[key]
                        int_id += 1
                    else:
                        seq_real_dict[real_id] = key
                        seq_real[:, real_id] = d_seq[key]
                        real_id += 1

            seq_int = seq_int[:,: int_id]
            seq_real = seq_real[:,: real_id]


            ### Static Features ###

            # 1,num_real 1,num_embed
            stat_real = np.zeros(self.num_stat_real, float)
            stat_int = np.zeros(self.num_stat_int, int)
            real_id, int_id = 0, 0

            # loops through I
            i_seq = i
            for key in i_seq.keys():
                if key not in self.to_skip:
                    if key in self.to_embed:
                        stat_int_dict[int_id] = key
                        #print(key)
                        stat_int[int_id] = i_seq[key]
                        int_id += 1
                    else:
                        stat_real_dict[real_id] = key
                        stat_real[real_id] = i_seq[key]
                        real_id += 1
            # loops through S
            s_seq = s
            for key in s_seq.keys():
                if key not in self.to_skip:
                    if key in self.to_embed:
                        stat_int_dict[int_id] = key
                        #print(key)
                        stat_int[int_id] = s_seq[key]
                        int_id += 1
                    else:
                        stat_real_dict[real_id] = key
                        stat_real[real_id]= s_seq[key]
                        real_id += 1


            if hasattr(self , 'seq_int_dict') is False:
                self.seq_int_dict =  seq_int_dict
                self.seq_real_dict = seq_real_dict
                self.stat_int_dict = stat_int_dict
                self.stat_real_dict = stat_real_dict
            assert self.seq_int_dict == seq_int_dict
            assert self.seq_real_dict == seq_real_dict
            assert self.stat_int_dict == stat_int_dict
            assert self.stat_real_dict == stat_real_dict

            #print('dicts' , self.seq_real_dict,self.seq_int_dict,self.stat_real_dict,self.stat_int_dict)
            #print('tensor_shape' , seq_real.shape , seq_int.shape , stat_real.shape ,stat_int.shape ,  targets.shape)
            return torch.tensor(seq_real, dtype=torch.float32), torch.LongTensor(seq_int), torch.tensor(stat_real, dtype=torch.float32), torch.LongTensor(stat_int), torch.tensor(targets, dtype=torch.float32)
