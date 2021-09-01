import pandas as pd
import sys, os
from frechet_measure import frechetDist
from dynamic_time_warp_measure import dtwDist
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pickle
from gini_measure import gini
from pointwise_measure import pointwise
import csv
import itertools
import random
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Process
from multiprocessing import Pool
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")
import math
from functools import partial
from sklearn.neighbors import NearestNeighbors
import progressbar as pb
from pandas.tseries.offsets import MonthBegin

def unique(list_1):
    list_set = set(list_1)
    new_list = list(list_set)
    return(new_list)

def make_dist_matrices(df_matrix, dist_methods):
    matrix_size = df_matrix.shape[0]
    matrix_dict = {k: np.zeros((matrix_size, matrix_size)) for (k,v) in dist_methods.items()}

    for (k,v) in dist_methods.items():
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                #print(str(i),str(j))
                dist_val = dist_methods[k](df_matrix[i], df_matrix[j])
                matrix_dict[k][i, j] = dist_val
                matrix_dict[k][j, i] = dist_val
    return(matrix_dict)

def make_clus_cols(dist_matrix_dict, clus_methods,combi, num_clus):
    clus_dict = {}
    combi_str = '-'.join(combi)
    for (clus_k,clus_v) in clus_methods.items():
        for (dist_k,dist_v) in dist_matrix_dict.items():
            if dist_v.shape[0]>1:
                if clus_k == 'DBSCAN':
                    dist_v2 = dist_v.copy()
                    dist_v2[dist_v2 == 0] = np.inf
                    nn_dist = np.min(dist_v2,axis =0)
                    histo = np.histogram(nn_dist)
                    cumdens = np.cumsum(histo[0]/histo[0].sum())
                    cutoff = np.where(cumdens<=max(min(cumdens),0.95))[0][-1]
                    eps = max(histo[1][cutoff+1],0.1)
                    clus_arr = clus_v(eps = eps,metric='precomputed', min_samples=10).fit_predict(dist_v)
                elif clus_k == 'KMeans':

                    n_clus = num_clus#(dist_v.shape[0] // 20)+1
                    cluster_inst = clus_v(n_clusters = n_clus, precompute_distances=True)
                    clus_arr = cluster_inst.fit_predict(dist_v)
                    clus_SSE = cluster_inst.inertia_
                clus_dict['clus-{}-{}'.format(dist_k, clus_k)] = [str(i) + combi_str for i in clus_arr]

    return (clus_dict, clus_SSE)

def model_family_binariser(d, label_key):

    df_train = d['train']
    df_test = d['test']
    binarize = LabelBinarizer()
    binarize.fit(pd.concat([df_train['MODEL_FAMILY'],df_test['MODEL_FAMILY']]))
    bin_model_family_train = binarize.transform(df_train['MODEL_FAMILY'])
    bin_model_family_test = binarize.transform(df_test['MODEL_FAMILY'])
    bin_df_train = pd.DataFrame(bin_model_family_train,
                                columns=[label_key + str(i) for i in range(bin_model_family_train.shape[1])],
                                index = df_train.index)
    bin_df_test = pd.DataFrame(bin_model_family_test,
                                columns=[label_key + str(i) for i in range(bin_model_family_test.shape[1])],
                                index=df_test.index)

    df_train = df_train.join(bin_df_train, how = 'inner')
    df_test = df_test.join(bin_df_test, how='inner')
    return(df_train, df_test)

def month_binariser(df_train, df_test, Month):
    month_binariser = LabelBinarizer()
    month_binariser.fit(np.arange(1,13))
    month_train = month_binariser.transform(df_train[Month])
    month_test = month_binariser.transform(df_test[Month])

    bin_df_train = pd.DataFrame(month_train,
                                columns=[Month +str(i) for i in range(1,13)],
                                index = df_train.index)
    bin_df_test = pd.DataFrame(month_test,
                                columns=[Month +str(i) for i in range(1,13)],
                                index=df_test.index)

    df_train = df_train.join(bin_df_train, how='inner')
    df_test = df_test.join(bin_df_test, how='inner')


    return(df_train, df_test)


def col_binariser(df_train_df, label_key = 'CHD_Year'):
    
    binarize_class = df_train_df[label_key].unique()
    for i in binarize_class:
        df_train_df[label_key+'_'+str(i)] = (df_train_df[label_key]==i)*1
    
    bin_df_train = pd.DataFrame(columns=[label_key + '_' +str(i) for i in binarize_class],
                                index = df_train_df.index)
    return df_train_df
    

def find_confidence_band_of_cluster(df_train, clusters_type, DaysColsOR0DayLevel):

    if DaysColsOR0DayLevel == 'DaysCols':
        dayscol = [i for i in df_train.columns.values.tolist() if isinstance(i, int)]
        col = dayscol
    elif DaysColsOR0DayLevel == '0DayLevel':
        col = '0day_sold_order_level%'
        
    df_mean_list = []
    df_lower_list = []
    df_upper_list = []

    for cl in clusters_type:
        #print(cl)
        df_mean = df_train.groupby(cl)[col].mean().reset_index()
        df_mean['is_confidence'] = 'mean'

        group = df_train.groupby(cl)[col]
   
        df_lower_coll = pd.DataFrame()        
        if DaysColsOR0DayLevel == 'DaysCols':
            for key, item in group:            
                df_lower = pd.DataFrame(group.get_group(key).quantile(0.1)).transpose()
                df_lower['{}'.format(cl)] = key
                df_lower_coll = pd.concat([df_lower_coll,df_lower], ignore_index = True)
        elif DaysColsOR0DayLevel == '0DayLevel':          
            df_lower_coll = group.quantile(0.1).reset_index()
        df_lower_coll['is_confidence'] = 'lower'
        df_lower = df_lower_coll

        df_upper_coll = pd.DataFrame()
        if DaysColsOR0DayLevel == 'DaysCols':
            for key, item in group:
                df_upper = pd.DataFrame(group.get_group(key).quantile(0.9)).transpose()
                df_upper['{}'.format(cl)] = key
                df_upper_coll = pd.concat([df_upper_coll,df_upper],ignore_index = True)
        elif DaysColsOR0DayLevel == '0DayLevel':          
            df_upper_coll = group.quantile(0.9).reset_index()            
        df_upper_coll['is_confidence'] = 'upper'
        df_upper = df_upper_coll

        df_mean_list.append(df_mean)
        df_lower_list.append(df_lower)
        df_upper_list.append(df_upper)

    d_confidence_band ={'mean': pd.concat(df_mean_list),
                 'lower': pd.concat(df_lower_list),
                 'upper': pd.concat(df_upper_list)}

    return(d_confidence_band)

def band_list_extend(band_l, k, ct, band_name, row,band_attr ):
    band_l.extend([k, ct, band_name, row.name])
    band_l.extend(row[band_attr].tolist())
    return(band_l)

def make_confidence_band_weighted_bands(band_lists, d_confidence_band, clus_dist, ct, row, band_attr, DaysColsOR0DayLevel):
    for k, v in d_confidence_band.items():
        band = v[~v[ct].isnull()]
        band = band[band[ct].isin(clus_dist.index)].set_index(ct)
        if DaysColsOR0DayLevel == 'DaysCols':
            value_vars = [i for i in v.columns.values if isinstance(i, int)]
            band = band[value_vars]
            band = band.mul(clus_dist['probability'], axis='index')
            band_l = band.sum(axis=0).sort_index().tolist()

        elif DaysColsOR0DayLevel == '0DayLevel':
            band = band['0day_sold_order_level%']
            band = band.mul(clus_dist['probability'], axis='index')
            band_l = [band.sum(axis=0)]

        band_l = band_list_extend(band_l, k, ct,'weighted_bands', row, band_attr)
        band_lists.append(band_l)
    return(band_lists)

def make_confidence_band_classified(band_lists,d_confidence_band, clus_dist, ct, row,band_attr):
    shortest = clus_dist['mean'].min()
    chosen_cluster = clus_dist[clus_dist['mean'] == shortest].index[0]

    for k, v in d_confidence_band.items():

        value_vars = [i for i in v.columns.values if isinstance(i, int)]
        band = v[~v[ct].isnull()]
        band = band[band[ct].isin(clus_dist.index)].set_index(ct)
        band = band[value_vars]
        band_l = band.loc[chosen_cluster,:].sort_index().tolist()
        band_l = band_list_extend(band_l, k, ct, 'classified', row, band_attr)

        band_lists.append(band_l)
    return (band_lists)
'''
def make_confidence_band_weighted_values(band_lists,df_train_sub, clus_dist, ct, row,band_attr):
    df_train_sub = df_train_sub.merge(clus_dist['row_probability'].reset_index(), on =ct )
    weights = df_train_sub['row_probability']
    value_vars = [i for i in df_train_sub.columns.values if isinstance(i, int)]
    id_vars = [j for j in df_train_sub.columns.values if j not in value_vars]

    band_m = df_train_sub[value_vars].apply(find_percentile,args = (weights,0.5,), axis =0).tolist()
    band_u = df_train_sub[value_vars].apply(find_percentile,args = (weights,0.9,), axis =0).tolist()
    band_l = df_train_sub[value_vars].apply(find_percentile,args = (weights,0.1,), axis =0).tolist()

    band_m = band_list_extend(band_m, 'mean', ct, 'weighted_values', row,band_attr)
    band_u = band_list_extend(band_u, 'upper', ct, 'weighted_values', row,band_attr)
    band_l = band_list_extend(band_l, 'lower', ct, 'weighted_values', row,band_attr)

    band_lists.extend([band_m,band_l,band_u])

    return (band_lists)
'''
def find_percentile(col,*args):
    weights = args[0]
    perc = args[1]
    weights = weights.tolist()
    col = col.tolist()
    sorted_weights = [x for _,x in sorted(zip(col,weights))]
    sorted_col = [y for y, _ in sorted(zip(col, weights))]
    sorted_cumweights = [sum(sorted_weights[:i+1])/sum(sorted_weights) for i,_ in enumerate(sorted_weights)]
    for ind, val in enumerate(sorted_cumweights):
        if val<=perc: continue
        perc_pos = [max(ind-1,0), ind]
        perc_val = [val for ind,val in enumerate(sorted_cumweights) if ind in perc_pos]
        if len(perc_val)>1:
            perc_weight = [(perc-perc_val[0])/(perc_val[1]-perc_val[0]), (perc_val[1]-perc)/(perc_val[1]-perc_val[0])]
            perc_col = [val for ind,val in enumerate(sorted_col) if ind in perc_pos]
            perc_col = [ val*perc_weight[ind] for ind,val in enumerate(perc_col) ]
            return (sum(perc_col))
        else:
            return(perc_val[0])
            
def make_confidence_band_simple(band_lists,df, clus_dist, ct, row, attrs,band_attr):
    sttr_str = ''.join(attrs)
    df_chosen = df.copy(deep = True)
    for attr in attrs:
        df_chosen =df_chosen[df_chosen[attr]==row[attr]]

    value_vars = [i for i in df_chosen.columns.values if isinstance(i, int)]
    id_vars = [j for j in df_chosen.columns.values if j not in value_vars]

    weights = df_chosen['RETAIL_ACTUALS_AND_FORECAST'] / df_chosen['RETAIL_ACTUALS_AND_FORECAST'].sum()

    if df_chosen.shape[0] <=20:
        band_m = df_chosen[value_vars].multiply(weights, axis=0).sum(axis =0)
        df_std = np.square((df_chosen[value_vars] - band_m))
        df_std = np.sqrt(df_std[value_vars].multiply(weights, axis=0).sum(axis =0))
        df_bw = 1.28 * df_std
        band_u = band_m + df_bw
        band_l = band_m - df_bw

        band_m =band_m.tolist()
        band_u =band_u.tolist()
        band_l =band_l.tolist()
    else:
        band_m = df_chosen[value_vars].apply(find_percentile,args = (weights,0.5,), axis =0).tolist()
        band_u = df_chosen[value_vars].apply(find_percentile,args = (weights,0.9,), axis =0).tolist()
        band_l = df_chosen[value_vars].apply(find_percentile,args = (weights,0.1,), axis =0).tolist()


    band_m = band_list_extend(band_m, 'mean', ct, sttr_str, row, band_attr)
    band_u = band_list_extend(band_u, 'upper', ct, sttr_str, row, band_attr)
    band_l = band_list_extend(band_l, 'lower', ct, sttr_str, row, band_attr)

    band_lists.extend([band_m,band_l,band_u])
    return (band_lists)

def summarise_clus_dist(df_train_sub, ct):
    clus_dist = df_train_sub.groupby(ct)['distance2'].agg(['mean', 'count'])
    #Smallest distance should have greater weight.
    #clus_dist['inv_mean'] = 1/clus_dist['mean']
    #clus_dist['max_mean-mean'] = clus_dist['mean'].max() - clus_dist['mean']
    #clus_dist['normed_max_mean-mean'] = clus_dist['max_mean-mean'] / clus_dist['max_mean-mean'].sum()
    clus_dist['normed_mean'] = clus_dist['mean'] / clus_dist['mean'].sum()
    #clus_dist['normed_inv_mean'] = clus_dist['inv_mean'] / clus_dist['inv_mean'].sum()
    clus_dist['normed_count'] = clus_dist['count'] / clus_dist['count'].sum()
    #clus_dist['weights'] = clus_dist['normed_mean'].multiply(clus_dist['normed_count'], axis='index')
    #clus_dist['weights'] = clus_dist['normed_inv_mean']
    clus_dist['weights'] = clus_dist['mean']#*clus_dist['normed_count']
    clus_dist['probability'] = clus_dist['weights'] / clus_dist['weights'].sum()
    #Amplify the probability of the high probability cluster and reduce the other way.
    #clus_dist['probability'] = clus_dist['probability']*clus_dist['probability']
    #clus_dist['probability'] = clus_dist['probability']/clus_dist['probability'].sum()
    clus_dist['row_probability'] = clus_dist['probability'] / clus_dist['count']
    return(clus_dist)

def melt_for_classify_cluster(df):
    value_vars = [i for i in df.columns.values if isinstance(i, int)]
    #value_vars2 = [i for i in df.columns.values if str(i).strip('-').isnumeric()]
    #value_vars.extend(value_vars2)
    id_vars = [j for j in df.columns.values if j not in value_vars]
    if value_vars:
        df_melt = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                          var_name='pre_order_days', value_name='sold_order_level%')
        return(df_melt)
    else:
        return(df)

def FD_Dist(x, centre, steepness):
    return (math.exp(-steepness*centre) + 1)/(math.exp((x - centre)*steepness) + 1)
'''
df_df = pd.DataFrame()
df_df['x'] = pd.Series(list(np.arange(0.0, 6.2, 0.2)))
FD_Dist_x = partial(FD_Dist, centre = 3, steepness = 1.3)
df_df['y'] = list(map(FD_Dist_x,df_df['x']))

from bokeh.plotting import figure, output_file, show
output_file("line.html")
p = figure(plot_width=400, plot_height=400)

p.line(df_df['x'], df_df['y'], line_width =2)
show(p)
'''
def err_quant(df_train_sub, day_col, lp = 0.2, mp = 0.5, hp = 0.8):
    df = df_train_sub[['row_probability', day_col]]
    df = df.dropna()
    df = df.sort_values(by = [day_col])        
    df['prob_cs'] = df['row_probability'].cumsum()
    return np.interp(x=[lp,mp,hp], xp=df.prob_cs, fp=df[day_col])   
                        
def make_cluster_to_csv(pfile,granularity,folder, cal_measure = True, validation = False, add_renorm = True):
    df_master = pd.read_pickle(pfile)

    #Use 50 units as the cut off for training
    df_master = df_master[df_master['VISTA_RETAIL_ACTUALS_AND_FORECAST'] >= 40]
    df_master = df_master.dropna()
    df_master['Target_Met'] = df_master['Target_Met'].astype('bool')

    i_cols = [i for i in df_master.columns.values.tolist() if isinstance(i, int)]
    if add_renorm == True:    
        for i in i_cols:
            df_master[i] = df_master[i] / df_master['0day_sold_order_level%']
    granular_items_list = []
    
    #Deal with zero via the final sold order level % determination.
    df_master = df_master.dropna()
    
    Level3 = pfile.split('/')[-1].split('.')[0].split('-')[1]
    if Level3 in ['USA', 'China NSC']: 
        df_master = df_master[df_master['CPDD_reporting_date'] >= '2014-05-01']
        
    for g in granularity:
        granular_items_list.append(df_master[g].unique().tolist())
        
    ###########################################################################
    clus_num_set = []
    df_list =[]
    for combi in itertools.product(*granular_items_list):
        print(combi)
        #import time
        #time.sleep(1) 
        combi = list(combi)
        df = df_master
        
        #Get test cases, using final 10% of handover month/year regardless of CPDD
        if validation:
            for i, g in enumerate(granularity):
                if g == 'SEGMENT':
                    #print(g, combi[i])
                    df_unsplit = df[df[g] == combi[i]]        
            test_index = []
            #train_index = []
            for model in df_unsplit.MODEL_FAMILY.unique():
                #print(model)
                '''
                df_unsplit_onemodel = df_unsplit[df_unsplit['MODEL_FAMILY'] == model]
                CHD_arr = np.sort(df_unsplit_onemodel["CHD_reporting_date"].map(lambda t: t.date()).unique())
                CHD_arr_test = CHD_arr[len(CHD_arr)-round(len(CHD_arr)*0.1):]
                CHD_arr_train = CHD_arr[:len(CHD_arr)-round(len(CHD_arr)*0.1)]
                test_index.extend(list(df_unsplit_onemodel[df_unsplit_onemodel['CHD_reporting_date'].isin(CHD_arr_test)].index.values))
                '''
                max_num_months_ahead = 3
                df_unsplit_onemodel = df_unsplit[df_unsplit['MODEL_FAMILY'] == model]
                CPDD_arr = np.sort(df_unsplit_onemodel["CPDD_reporting_date"].map(lambda t: t.date()).unique())
                #Get 10% of latest VALID CPDD months as test cases 
                CPDD_arr_test = CPDD_arr[len(CPDD_arr)-round((len(CPDD_arr)-max_num_months_ahead)*0.1)-max_num_months_ahead:]
                test_index.extend(list(df_unsplit_onemodel[df_unsplit_onemodel['CPDD_reporting_date'].isin(CPDD_arr_test)].index.values))
                #train_index.extend(list(df_unsplit_onemodel[df_unsplit_onemodel['CHD_reporting_date'].isin(CHD_arr_train)].index.values))
            
        for i, g in enumerate(granularity):
            #print(i,g)
            df = df[df[g] == combi[i]]   
            
        #Generate train, test dataset
        df_true = df[df['Target_Met']]
        sample_size = df_true.shape[0]

        if validation:
            train_index = [ii for ii in df_true.index.values if ii not in test_index]
            test_index = [ii for ii in df.index.values if ii in test_index]
            df_train = df_true.loc[train_index,:]
            df_test = df.loc[test_index,:]
        else:
            df_train = df_true.copy(deep = True)
            df_test = df[~df['Target_Met']]

        if df_train.empty:
            continue
        value_vars = [i for i in df.columns.values if isinstance(i, int)]
        df_train_1 = df_train[value_vars]
        df_matrix = df_train_1.as_matrix()

        #Define clustering methods
        dist_methods ={'pointwise':pointwise}#, 'dwt':dtwDist}# 'frechet':frechetDist, 'dwt':dtwDist
        clus_methods = {'KMeans': KMeans}#'DBSCAN': DBSCAN,

        #Define filenames
        gran_str = '_'.join(granularity)
        combi2 = [str(ii) for ii in combi]
        combi_str = '_'.join(combi2)
        pickle_path = '../output_vault/dwt_cluster/matrices/{}/{}/{}-{}.pkl'.format(folder,
            gran_str,combi_str,pfile.split('/')[-1].split('.')[0])

        #Create dist matrices, if don't exist, and save them to or else load them from 'pickle_path'.
        #file_exists = os.path.isfile(pickle_path)
        #if file_exists:
        #    print(pickle_path + ' already exists')
        #if cal_measure and not file_exists:
        print('calculating dist matrics for {}-{}'.format(pfile,'-'.join(combi2)))
        dist_matrix_dict = make_dist_matrices(df_matrix, dist_methods)
        pickle.dump(dist_matrix_dict, open(pickle_path,'wb'))
        #else:
        #    dist_matrix_dict =pickle.load(open(pickle_path, 'rb'))

        #Create clusters for the granularity.
        print('making cluster for {}-{}'.format(pfile, '-'.join(combi2)))
        print('\n')
        #if dist_matrix_dict['pointwise'].shape[0] ==0:
        #    return()
        
        #num_clus = 5
        elb = []
        for num_clus in list(range(1,8)):
            if df_train_1.shape[0] >= num_clus and df_train_1.shape[0] != 1:
                clus_cols, clus_SSE = make_clus_cols(dist_matrix_dict, clus_methods, combi2, num_clus)
                elb.append([num_clus, clus_SSE])
            else:
                break
        elb = pd.DataFrame(elb, columns = ['num_clus', 'SSE'])
        #elb.plot.scatter(x = 'num_clus', y = 'SSE')
        elb['SSE'] = elb['SSE']/elb['SSE'].max()
        elb['SSE_up1'] = pd.Series([np.nan]).append(elb['SSE'][:-1]).reset_index(drop = True)
        elb['d_SSE'] = - elb['SSE'] + elb['SSE_up1'] 
        #elb['SSE_up2'] = pd.Series([np.nan]).append(elb['d_SSE'][:-1]).reset_index(drop = True)
        elb['r_SSE'] = elb['d_SSE']/elb['SSE_up1'] 
        
        elb = elb[elb['d_SSE'] > 0.05]
        #elb = elb[elb['r_SSE'] < 0.20]        
        #num_clus = elb['num_clus'].min() - 1
        num_clus = elb['num_clus'].max()
        clus_num_set.append([combi2, df_train_1.shape[0], num_clus])
        ##Need to check what granularity element is missing!
        if df_train_1.shape[0] > num_clus:
            clus_cols, clus_SSE = make_clus_cols(dist_matrix_dict, clus_methods, combi2, num_clus)
        else:
            continue
        
        for (k,v) in clus_cols.items():
            df_train[k] = v
            df_test[k] = np.nan

        #Create new columns
        df_train['is_train'] = True
        df_test['is_train'] = False
        df = pd.concat([df_train,df_test])
        # df_matrix = df.drop(['reporting_date', 'reporting_max_date', 'BRAND', 'MODEL_FAMILY',
        #                      'RETAIL_ACTUALS_AND_FORECAST', 'RETAIL_FORECAST', 'Target_Met',
        #                      'RETAIL_FORECAST_cutoff', 'target', 'total_order_sold'
        #                      ], axis=1).as_matrix()
        # matrix_size = df_matrix.shape[0]
        #
        # gini_coef = np.zeros(matrix_size)
        # for i in range(matrix_size):
        #     gini_coef[i] = gini(df_matrix[i])
        #
        #df['gini_coef'] = gini_coef
        df['CHD_Month'] = df['CHD_reporting_date'].map(lambda x:x.month)
        df['CHD_Year'] = df['CHD_reporting_date'].map(lambda x: x.year)
        df['CPDD_Month'] = df['CPDD_reporting_date'].map(lambda x:x.month)
        df['CPDD_Year'] = df['CPDD_reporting_date'].map(lambda x: x.year)
        df_list.append(df)

    ###########################################################################
    if len(df_list)>0:
        df_all  = pd.concat(df_list)
        if not df_all.empty:
            Level5 = pfile.split('/')[-1].split('.')[0].split('-')[0].split('_')[1]
            Level3 = pfile.split('/')[-1].split('.')[0].split('-')[1]
            g_str = ''.join(granularity)
            df_all['LEVEL5'] = Level5
            df_all['LEVEL3'] = Level3
            df_all['GRANULARITY'] = g_str
            clustered_path = '../output_vault/dwt_cluster/clustered/{}/historical-{}-{}.csv'.format(folder,g_str,
                                                                                pfile.split('/')[-1].split('.')[0])
            df_all.to_csv(clustered_path, index_label = 'INDEX')
            
            df_all = melt_for_classify_cluster(df_all)
            clustered_path = '../output_vault/dwt_cluster/clustered/{}/historical-{}-{}-melted.csv'.format(folder,g_str,
                                                                                pfile.split('/')[-1].split('.')[0])
            df_all.to_csv(clustered_path, index_label = 'INDEX')

    else:
        df_all = pd.DataFrame()
    return df_all, clus_num_set

'''
seg = list(itertools.product(*granular_items_list))
seg_list = df_all['clus-dwt-KMeans'].unique().tolist()
seg_done = [ii[1:].split('-') for ii in seg_list if ii is not np.nan]
seg_done = [(ii[0], int(ii[1])) for ii in seg_done]
seg_done = list(set(seg_done))
seg_undone = [ii for ii in seg if ii not in seg_done]
'''

def classify_cluster_to_csv(pfile, granularity, folder, df_all = None, validation = False):
    print('classifying for ' + pfile)

    #Load results of clustering by market & granularity (trained data), with test data appended
    if not df_all:
        g_str = ''.join(granularity)
        fname = pfile.split('/')[-1].split('.')[0]
        pfile2 = '../output_vault/dwt_cluster/clustered/{}/historical-{}-{}.csv'.format(folder,g_str,fname)
        if not os.path.isfile(pfile2):
            return()
        # if os.path.isfile('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}.csv'.format(folder, g_str, fname)):
        #     return ()
        df_all =pd.read_csv(pfile2, index_col = 'INDEX')
        value_vars = [i for i in df_all.columns.values if i.strip('-').isnumeric()]
        value_vars_dict = {k:int(k) for k in value_vars}
        df_all = df_all.rename(columns = value_vars_dict)

    #Get trained and test datasets
    Level5 = fname.split('-')[0].split('_')[1]
    Level3 = fname.split('-')[1]
    g_str = ''.join(granularity)
    df_train = df_all[df_all['is_train']]
    if validation:
        df_test = df_all[~df_all['is_train']]
    else:
        df_test = pd.read_csv('../input_vault/csv/prediction_cases.csv')
        df_test = df_test[df_test['LEVEL_3_REPORTING'] == Level3]
    #df_dict = {'train': df_train, 'test':df_test}

    #Get confidence band for different clusters
    clusters_type = [i for i in df_train.columns.values.tolist() if not isinstance(i,int) and i.split('-')[0] == 'clus']
    if (not clusters_type):
        return()
    #d_confidence_band = find_confidence_band_of_cluster(df_train, clusters_type, DaysColsOR0DayLevel = 'DaysCols') #contain df mean, lower, upper for each cluster

    #Create columns for nominal variables
    #label_key = 'model_family_1'
    #df_train, df_test = model_family_binariser(df_dict, label_key)
    #df_train, df_test = month_binariser(df_train, df_test, Month = 'CHD_Month')
    #df_train, df_test = month_binariser(df_train, df_test, Month = 'CPDD_Month')

    #Create a list of attributes needed for classification
    #class_label = [i for i in df_train.columns.values if not isinstance(i,int) and i[:len(label_key)] == label_key]
    class_attr_1 = ['clus-pointwise-KMeans', 'MODEL_FAMILY', 'RETAIL_FORECAST' ,'CHD_Year', 'CPDD_Year', 'CPDD_Month', 'CHD_Month']
    #class_attr_1 = ['RETAIL_FORECAST' ,'CHD_Year', 'CPDD_Year']
    #CPDD_month_attr = ['CPDD_Month' + str(i) for i in range(1, 13)]
    #CHD_month_attr = ['CHD_Month' + str(i) for i in range(1, 13)]
    class_attr = list(class_attr_1)
    #class_attr.extend(class_label)
    #class_attr.extend(CPDD_month_attr)
    #class_attr.extend(CHD_month_attr)

    #Create a list of attributes concerning the band
    band_attr = ['months_before_handover', 'BRAND', 'MODEL_FAMILY','SEGMENT','RETAIL_FORECAST', 'CHD_reporting_date', 'CPDD_reporting_date',
                 'RETAIL_ACTUALS_AND_FORECAST', 'RETAIL_FORECAST_cutoff', 'Target_Met', 'VISTA_RETAIL_ACTUALS_AND_FORECAST', 'is_train', '0day_sold_order_level%']

    band_lists = []
    clus_info = []
    
    progress = pb.ProgressBar(widgets=[' [', pb.Timer(), '] ', pb.Bar(),' (', pb.ETA(), ') ',], max_value = df_test.shape[0]);
    progvar = 0
    
    #Go through each row in the test dataset and classify it
    for irow,row in df_test.iterrows():
        #if irow == 1954: break
        progress.update(progvar + 1)        
        progvar += 1
        #df_test = df_test.loc[1722,:].to_frame().T
        df_train_sub = df_train.copy(deep=True)

        #Filter the trained dataset for segment relevant to the particular test row
        for _, g in enumerate(granularity):
            df_train_sub = df_train_sub[df_train_sub[g] == row[g]]
        if df_train_sub.shape[0] ==0:
            continue

        #Filter trained and test row of data for columns only relevant to classification
        df_train_attr = df_train_sub[class_attr]
        row_attr = row[[ii for ii in class_attr if ii != 'clus-pointwise-KMeans']].to_frame().T
        
        #Rescale the columns.
        #scaler = MinMaxScaler()
        #df_train_attr['RETAIL_FORECAST'] = scaler.fit_transform(df_train_attr['RETAIL_FORECAST'].to_frame())
        #row_attr['RETAIL_FORECAST'] = scaler.transform(row_attr.where(~row_attr.isnull(),0)['RETAIL_FORECAST'].to_frame())

        df_train_df = pd.DataFrame()
        df_train_df['clus-pointwise-KMeans'] = df_train_attr['clus-pointwise-KMeans']
        for col in ['MODEL_FAMILY', 'CPDD_Month', 'CHD_Month']:
            df_train_df[col] = (df_train_attr[col] == row_attr.iloc[0][col]).astype(int)
            
        for col in ['CHD_Year', 'CPDD_Year']: 
            df_train_df[col] = abs(df_train_attr[col] - row_attr.iloc[0][col])
        df_train_df = col_binariser(df_train_df, label_key = 'CHD_Year')
        df_train_df = col_binariser(df_train_df, label_key = 'CPDD_Year')
        
        CHD_Year_col = [col for col in df_train_df if 'CHD_Year_' in col]
        FD_Dist_x = partial(FD_Dist, centre = 3, steepness = 1.3)
        for col in CHD_Year_col:
            df_train_df[col] = df_train_df[col] * FD_Dist_x(float(col[-1]))
  
        CPDD_Year_col = [col for col in df_train_df if 'CPDD_Year_' in col]
        FD_Dist_x = partial(FD_Dist, centre = 3, steepness = 1.3)
        for col in CPDD_Year_col:
            df_train_df[col] = df_train_df[col] * FD_Dist_x(float(col[-1]))  

        df_train_df_sum = df_train_df.groupby(by = 'clus-pointwise-KMeans')[[col for col in df_train_df if col not in ['CHD_Year', 'CPDD_Year']]].sum()
        df_train_df_cnt = df_train_df.groupby(by = 'clus-pointwise-KMeans')[['CHD_Year', 'CPDD_Year']].count()
        df_train_df = pd.concat([df_train_df_sum, df_train_df_cnt], axis = 1)
        
        for col in ['MODEL_FAMILY', 'CPDD_Month', 'CHD_Month']:
            df_train_df[col] = df_train_df[col]/df_train_df[col].sum()
        for col in CPDD_Year_col:
            df_train_df[col] = df_train_df[col] / df_train_df['CPDD_Year']
        df_train_df['CPDD_Year'] = df_train_df[CPDD_Year_col].sum(axis = 1)
        df_train_df['CPDD_Year'] = df_train_df['CPDD_Year']/df_train_df['CPDD_Year'].sum()
        for col in CHD_Year_col:
            df_train_df[col] = df_train_df[col] / df_train_df['CHD_Year']
        df_train_df['CHD_Year'] = df_train_df[CHD_Year_col].sum(axis = 1)
        df_train_df['CHD_Year'] = df_train_df['CHD_Year']/df_train_df['CHD_Year'].sum()
        
        #Null the clusters without CPDD/CHD Month
        df_train_df = df_train_df[['MODEL_FAMILY', 'CPDD_Month', 'CHD_Month', 'CPDD_Year', 'CHD_Year']]
        #df_train_df.loc[df_train_df['CPDD_Month'] == 0,:] = 0
        for col in df_train_df.columns:
            df_train_df[col] = df_train_df[col]/df_train_df[col].sum()
        prob_col = ['MODEL_FAMILY', 'CPDD_Month', 'CPDD_Year']
        df_train_df['CPDD_Month'] = 2*df_train_df['CPDD_Month']
        df_train_df['distance2'] = df_train_df[prob_col].sum(axis = 1)
        df_train_df['distance2'] = df_train_df['distance2']/df_train_df['distance2'].sum()
        df_train_df = df_train_df.reset_index()
        df_train_sub = pd.merge(df_train_sub, df_train_df[['clus-pointwise-KMeans', 'distance2']], how = 'left', left_on = 'clus-pointwise-KMeans', right_on = 'clus-pointwise-KMeans')
                    
            #interp_data[day] = partial(func, df_train_sub = df_train_sub, pct = 0.1)
            #test = df_train_sub[i_cols].applymap(lambda x: func2(day_col = x))

        if df_train_sub.shape[0] > 0:
            for ct in clusters_type:
                #Calculate (normed) mean distance to clusters, (normed) numb of cluster members count,
                #weights and probability
                #clus_dist = summarise_clus_dist(df_train_sub, ct)
                clus_dist = summarise_clus_dist(df_train_sub, ct)
                if clus_dist.shape[0] > 0:
                #Construct weighted confidence band
                    df_train_sub = pd.merge(df_train_sub,clus_dist[['row_probability']], how = 'left', left_on = 'clus-pointwise-KMeans', right_index = True)     
                    i_cols = [i for i in df_train_sub.columns.values.tolist() if isinstance(i, int)]        

                    interp_data = {}        
                    for day in i_cols:
                        interp_data[day] = err_quant(df_train_sub, day)
                    df = pd.DataFrame.from_dict(interp_data)
                    df['confidence_band'] = pd.Series(['lower', 'mean', 'upper'])
                    df = df.set_index('confidence_band')
                    for row_quant in df.index:
                        #print(row_quant)
                        band_l = band_list_extend(df.loc[row_quant,:].tolist(), row_quant, ct,'cumulative_distribution', row, band_attr)
                        band_lists.append(band_l)                    
                    
                    #band_lists = make_confidence_band_weighted_bands(band_lists,d_confidence_band, clus_dist, ct, row,band_attr, DaysColsOR0DayLevel = 'DaysCols')
                    #band_lists = make_confidence_band_classified(band_lists, d_confidence_band, clus_dist, ct, row,band_attr)
                    #band_lists = make_confidence_band_simple(band_lists, df_train, clus_dist, ct, row, ['MODEL_FAMILY','Month'],band_attr)
                    #band_lists = make_confidence_band_simple(band_lists, df_train, clus_dist, ct, row, ['MODEL_FAMILY'],band_attr)
                    #band_lists = make_confidence_band_weighted_values(band_lists, df_train_sub, clus_dist, ct, row,band_attr)
                    #print('')

                    clus_dist_new = clus_dist.copy(deep = True)
                    clus_dist_new.index = clus_dist_new.index.rename('CLUSTER')
                    clus_dist_new = clus_dist_new.reset_index()
                    clus_dist_new['INDEX'] = row.name
                    clus_dist_new['LEVEL5'] = Level5
                    clus_dist_new['LEVEL3'] = Level3
                    clus_dist_new['GRANULARITY'] = g_str
                    clus_dist_new['CLUSTER_TYPE'] = ct
                    clus_info.append(clus_dist_new)

    #Tidying band lists:
    band_df_cols = [i for i in df_train.columns.values if isinstance(i, int)]
    band_df_cols.extend(['confidence_band', 'clusters_type', 'weighted', 'INDEX' ])
    band_df_cols.extend(band_attr)
    df_band   = pd.DataFrame(band_lists, columns = band_df_cols)
    df_band_melt = melt_for_classify_cluster(df_band)

    clus_info = pd.concat(clus_info)

    #Tidying (original) df_test:
    #all_class_labels = [i for i in df_test.columns.values.tolist() if not isinstance(i,int) and i.split('_1')[0]=='model_family']
    #df_test = df_test.drop(all_class_labels, axis =1, errors = 'ignore')
    df_test = df_test.drop(clusters_type, axis =1, errors = 'ignore')
    #df_test = df_test.drop(CPDD_month_attr, axis =1, errors = 'ignore')
    #df_test = df_test.drop(CHD_month_attr, axis =1, errors = 'ignore')

    df_test = df_test.reset_index()
    df_test_melt = melt_for_classify_cluster(df_test)
    df_test_melt['confidence_band'] = 'actual'
    if not validation:
        df_test_melt = df_test_melt.rename(columns = {'LEVEL_3_REPORTING':'LEVEL3', 'index':'INDEX'})
        
        
    if not df_band_melt.empty:
        df_band_melt['LEVEL5'] = Level5
        df_band_melt['LEVEL3'] = Level3
        df_band_melt['GRANULARITY'] = g_str 
        df_band_melt.to_csv('../output_vault/dwt_cluster/classified/{}/confidence/{}-{}.csv'
                       .format(folder,g_str,fname), index = False )
    if not clus_info.empty:
        clus_info.to_csv('../output_vault/dwt_cluster/classified/{}/clus_info/{}-{}.csv'
                       .format(folder,g_str,fname),index = False)
    if not df_test_melt.empty:
        df_test_melt.to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}.csv'
                       .format(folder,g_str,fname),index = False)
    if not df_band_melt.empty:
        pd.concat([df_test_melt, df_band_melt], ignore_index = True).to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand.csv'
                       .format(folder,g_str,fname),index = False)

'''        
        #Calculate distance between row data and members of clusters/trained data
        #df_train_df = pd.DataFrame()
        for col in df_train_attr.columns.tolist():
            if col in ['CHD_Year','CPDD_Year', 'RETAIL_FORECAST']: 
                #Scale such that min differnce will be zero and max difference will be 3. 
                #Imagine scaling such that the middle will be 2.
                #Compared to months, min difference is 0 and max difference is 2.
                df_train_df[col] = (df_train_attr[col]-row_attr.iloc[0][col])*3./(2017-2013)
                df_train_df[col] = np.square(df_train_df[col])
            else:
                df_train_df[col] = np.square(df_train_attr[col]-row_attr.iloc[0][col])
        df_train_sub['distance2']= np.sqrt(df_train_df.sum(axis = 1)).reshape(df_train_df.shape[0],-1)
'''

def concat_all(folder):
    '../output_vault/dwt_cluster/classified/{}/clus_info/{}-{}.csv'.format(folder, g_str, fname)

'''            
def combine_file():
    output_path = '../output_vault/dwt_cluster/'
    all_file = 'all.csv'
    for root, dir, files in os.walk(output_path, topdown=False):
        for r in root:
            initiate = True
            for f in files:
                if f.endswith('.csv'):
                    if not  f.startswith('all'):
                        pfile = os.path.join(r,f)
                        df = pd.read_csv(pfile)
                        if initiate:
                            with open(all_file, 'w'):
                                df.to_csv(all_file, index = False, header = True)
                            initiate = False
                        else:
                            with open(all_file, 'a'):
                                df.to_csv(all_file, index = False, header = True)

                        os.remove(pfile)
'''
def combine_data(comb_path, pfolder, end_str, start, reinitialise):
    file_list = os.listdir(comb_path.format(pfolder))
    file_list = [file for file in file_list if file.startswith(start) and \
                file.endswith(end_str)]
    all_file = 'all{}'.format(end_str)
    out_file = os.path.join(comb_path.format(pfolder), all_file)

    if reinitialise:
        try:
            os.remove(out_file)
        except OSError:
            pass
    #df_test = pd.DataFrame()
    for file in file_list:
        pfile = os.path.join(comb_path.format(pfolder),file)
        df = pd.read_csv(pfile)
        #df_test = pd.concat([df_test,df], ignore_index = True)
        if all_file not in [ii for ii in os.listdir(comb_path.format(pfolder))]:
            with open(out_file, 'w') as a_file:
                print('w:', df.shape[0],file)
                df.to_csv(a_file, index = False, header = True)
        else:
            with open(out_file, 'a') as a_file:
                print('a:', df.shape[0], file)
                df.to_csv(a_file, index = False, header = False)
                
def allfunctions(f, validation, granularities):
    #path = '../pickle_vault/pre_orders_curves/'
    #granularities = [['SEGMENT'],['MODEL_FAMILY']] # BRAND,SEGMENT,MODEL,FAMILY,MONTH
    #granularities = [['SEGMENT', 'months_before_handover']]
    #validation = [True]
    
    for v in validation:
        if v:
            folder ='validation'
        else:
            folder ='actual'

        for granularity in granularities:
            pfile = path + f
            df_all, clus_num_set = make_cluster_to_csv(pfile, granularity, folder=folder,cal_measure=True, validation=v, add_renorm = True);
            df_all, clus_num_set = make_cluster_to_csv_0daylevel(pfile, granularity, folder=folder, validation=v, clus_method = DBSCAN);
            classify_cluster_to_csv(pfile, granularity,folder=folder, validation=v);
            classify_cluster_to_csv_0daylevel(pfile, granularity, folder=folder, validation=v);
            ###Combine data:
            create_final_output_table(pfile, granularity, folder=folder, validation=v);
          
def main():
    path = '../pickle_vault/pre_orders_curves/'
    multithread = False
    level3_ignore = ['Guava', 'Direct Sales']
    granularities = [['SEGMENT', 'months_before_handover']]
    validation = [True]
    
    for root,dir,files in os.walk(path):

        if multithread:
            files = [f for f in files if f.endswith('.pkl')]
            p = Pool(3)
            p.map(allfunctions, files)

        else:
            file_list = []
            for f in files:
                if f.endswith('.pkl') and f.startswith('CPDD') and f.split('.')[0].split('-')[1] not in level3_ignore:
                    print(f)
                    file_list.append(f)
                    #f = 'CPDD_United Kingdom-UK NSC.pkl'
                    #f = 'CPDD_LowVol-EU_LowVol.pkl'
                    #f = 'CPDD_North America-USA.pkl'
            for f in file_list[int(len(file_list)/2):]:#[:int(len(file_list)/2)]: #or [int(len(file_list)/2):]
                allfunctions(f, validation, granularities)
                              
    ##Combine Files
    reinitialise = True
    comb_path = '../output_vault/dwt_cluster/{}'

    for granularity in granularities:
        g_str = ''.join(granularity)        
        for v in validation:
            if v:
                folder ='validation'
                combine_data(comb_path, pfolder = 'clustered/' + folder +'/', end_str = '-0DayLevel.csv', start = 'historical' + '-' + g_str, reinitialise = reinitialise)
                combine_data(comb_path, pfolder = 'clustered/' + folder +'/', end_str = '-melted.csv', start = 'historical' + '-' + g_str, reinitialise = reinitialise)
                combine_data(comb_path, pfolder = 'classified/' + folder +'/' + 'clus_info/', end_str = '-combined.csv', start = g_str, reinitialise = reinitialise)
                combine_data(comb_path, pfolder = 'classified/' + folder +'/' + 'test_cases/', end_str = '_WithBand-combined.csv', start = g_str, reinitialise = reinitialise)
            else:
                folder ='actual'
                combine_data(comb_path, pfolder = 'clustered/' + folder +'/', end_str = '-0DayLevel.csv', start = 'historical' + '-' + g_str, reinitialise = reinitialise)
                combine_data(comb_path, pfolder = 'clustered/' + folder +'/', end_str = '-melted.csv', start = 'historical' + '-' + g_str, reinitialise = reinitialise)
                combine_data(comb_path, pfolder = 'classified/' + folder +'/' + 'test_cases/', end_str = '_WithBand-combined-DB.csv', start = g_str, reinitialise = reinitialise)
       
'''        
    clus_info.to_csv('../output_vault/dwt_cluster/classified/{}/clus_info/{}-{}-combined.csv'
    df_test.to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand-combined.csv'
    df_test.to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand-combined-DB.csv'
'''

if __name__ == '__main__':
    main()
    
##################################

def create_final_output_table(pfile, granularity, folder, validation): 
    g_str = ''.join(granularity)
    fname = pfile.split('/')[-1].split('.')[0]
    temp_Level3 = fname.split('-')[1]

    clus_info_shape = pd.read_csv('../output_vault/dwt_cluster/classified/{}/clus_info/{}-{}.csv'
                           .format(folder,g_str,fname))
    clus_info_0day = pd.read_csv('../output_vault/dwt_cluster/classified/{}/clus_info/{}-{}-0DayLevel.csv'
                           .format(folder,g_str,fname))
    clus_info = pd.concat([clus_info_shape,clus_info_0day], ignore_index = True)
    clus_info.to_csv('../output_vault/dwt_cluster/classified/{}/clus_info/{}-{}-combined.csv'
                           .format(folder,g_str,fname),index = False)
                     
    ###############################################################################                   
    df_test_0day = pd.read_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand-0DayLevel.csv'
                           .format(folder,g_str,fname))
    #Fix the four columns, which are NULLs, belonging to actuals or else can't pivot the table later.                        
    df = df_test_0day[df_test_0day['confidence_band'] == 'mean'][['INDEX', 'clusters_type', 'weighted', 'LEVEL5', 'GRANULARITY']].rename(columns = {'clusters_type':'clusters_type_mean', 'weighted':'weighted_mean', 'LEVEL5':'LEVEL5_mean', 'GRANULARITY':'GRANULARITY_mean'})
    df_test_0day = pd.merge(df_test_0day, df, how = 'left', on = 'INDEX')
    df_test_0day['clusters_type'] = df_test_0day['clusters_type'].fillna(df_test_0day.clusters_type_mean)
    df_test_0day['weighted'] = df_test_0day['weighted'].fillna(df_test_0day.weighted_mean)
    df_test_0day['LEVEL5'] = df_test_0day['LEVEL5'].fillna(df_test_0day.LEVEL5_mean)
    df_test_0day['GRANULARITY'] = df_test_0day['GRANULARITY'].fillna(df_test_0day.GRANULARITY_mean)
    
    if not validation:
        df_test_0day = df_test_0day.drop(['RETAIL_ACTUALS_AND_FORECAST','RETAIL_FORECAST_cutoff', 'Target_Met', 'VISTA_RETAIL_ACTUALS_AND_FORECAST', 'is_train' ] ,axis = 1)
        #No actual values for prediction. So, just discard these rows.        
        df_test_0day = df_test_0day[df_test_0day['confidence_band'] != 'actual']
        #df_test.isnull().values.any()

    #Pivot table to get 'lower', 'mean' & 'upper' columns.
    ls = [ii for ii in df_test_0day.columns.values if ii not in ['confidence_band','0day_sold_order_level%_band', 'clusters_type_mean', 'weighted_mean', 'LEVEL5_mean', 'GRANULARITY_mean']]
    df_test_0day = pd.pivot_table(df_test_0day, index = ls, columns = 'confidence_band' , values = '0day_sold_order_level%_band').reset_index()
    
    #Rename some columns so they don't conflict with shape part.    
    cols =  ['clusters_type', 'weighted', 'actual', 'lower', 'mean', 'upper']
    if not validation:
        cols.remove('actual')
    col_name_change = {}
    right_table_cols = ['INDEX']
    for col_name in cols:
        col_name_change[col_name] = '0daylevel_' + col_name
        right_table_cols.extend(['0daylevel_' + col_name])
    df_test_0day = df_test_0day.rename(columns = col_name_change)
    
    ###############################################################################
    df_test_shape = pd.read_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand.csv'
                           .format(folder,g_str,fname))
    if not validation:
        df_test_shape = df_test_shape.drop(['RETAIL_ACTUALS_AND_FORECAST','RETAIL_FORECAST_cutoff', 'Target_Met', 
                                            'VISTA_RETAIL_ACTUALS_AND_FORECAST', 'is_train'] ,axis = 1)
        df_test_shape = df_test_shape[df_test_shape['confidence_band'] != 'actual']
        
    #Fix the four columns, which are NULLs, belonging to actuals or else can't pivot the table later.                        
    df_test_shape.loc[df_test_shape['confidence_band'] == 'actual','clusters_type'] = 'clus-pointwise-KMeans'
    df_test_shape.loc[df_test_shape['confidence_band'] == 'actual','weighted'] = 'cumulative_distribution'                       
    df_test_shape = df_test_shape[df_test_shape['weighted'] == 'cumulative_distribution']
    
    #Pivot table to get 'lower', 'mean' & 'upper' columns.
    ls = [ii for ii in df_test_shape.columns.values if ii not in ['confidence_band','sold_order_level%', '0day_sold_order_level%', 'CHD_Month', 'CHD_Year', 'CPDD_Month', 'CPDD_Year']]                      
    df_test_shape = pd.pivot_table(df_test_shape, index = ls, columns = 'confidence_band' , values = 'sold_order_level%').reset_index()
    
    #Rename some columns so they don't conflict with shape part.    
    cols =  ['clusters_type', 'weighted', 'actual', 'lower', 'mean', 'upper']
    col_name_change = {}
    for col_name in cols:
        col_name_change[col_name] = 'shape_' + col_name
    df_test_shape = df_test_shape.rename(columns = col_name_change)
    
    df_test = pd.merge(df_test_shape, df_test_0day[right_table_cols], on = 'INDEX', how = 'left')
    
    ##############################################################################
    #Combine shape and 0 day level results
    df_test['average'] = df_test['0daylevel_mean']*df_test['shape_mean']
    def upper_lower(val1, mean1, val2, mean2):
        return np.sqrt( (val1-mean1)**2/mean1**2 + (val2-mean2)**2/mean2**2 )
    df_test['upper_error'] = upper_lower(df_test['shape_upper'], df_test['shape_mean'], df_test['0daylevel_upper'], df_test['0daylevel_mean'])*df_test['average']
    df_test['lower_error'] = upper_lower(df_test['shape_lower'], df_test['shape_mean'], df_test['0daylevel_lower'], df_test['0daylevel_mean'])*df_test['average']
    #test = df_test[(df_test['pre_order_days'] == -1) & (df_test['MODEL_FAMILY'] == 'RANGE ROVER') & (df_test['CPDD_reporting_date'] == '2017-06-30')]
    ###############################################################################

    #Prepare output table    
    if not validation:
        df_test['CHD_reporting_date'] = pd.to_datetime(df_test['CHD_reporting_date'])
        df_test['CPDD_reporting_date'] = pd.to_datetime(df_test['CPDD_reporting_date'])
        df_test['Year'] = df_test['CPDD_reporting_date'].dt.year
        df_test['Month'] = df_test['CPDD_reporting_date'].dt.month
        df_test = df_test.rename(columns={'LEVEL5':'Region', 'LEVEL3':'Market', 'BRAND':'Brand', 'average':'Target', 'pre_order_days':'Day'})
        df_test = df_test[['Region','Market', 'Brand', 'MODEL_FAMILY', 'Year', 'Month', 'Day', 'Target', 'lower_error', 'upper_error', 'CHD_reporting_date', 'CPDD_reporting_date']]

        mkt_mapping = pd.read_csv('../input_vault/csv/LEVEL3_LEVEL5_TEMPLEVEL3_mapping.csv')
        level3_mapping = mkt_mapping.groupby('TEMP_LEVEL_3_REPORTING')['LEVEL_3_REPORTING'].apply(list)
        region_mapping = mkt_mapping.groupby('LEVEL_3_REPORTING')['LEVEL_5_REPORTING'].apply(list)

        prediction_cases_mkt_unagg = pd.read_csv('../input_vault/csv/prediction_cases_mkt_unagg.csv')
        
        for Level3 in level3_mapping[temp_Level3]:
            #print(Level3)
            prediction_cases_mkt_unagg_single = prediction_cases_mkt_unagg[prediction_cases_mkt_unagg['LEVEL_3_REPORTING'] == Level3]
            prediction_cases_mkt_unagg_single['CHD_reporting_date'] = pd.to_datetime(prediction_cases_mkt_unagg_single['CHD_reporting_date'])
            prediction_cases_mkt_unagg_single['CPDD_reporting_date'] = pd.to_datetime(prediction_cases_mkt_unagg_single['CPDD_reporting_date'])
            prediction_cases_mkt_unagg_single = pd.merge(prediction_cases_mkt_unagg_single, df_test, on = ['CHD_reporting_date', 'CPDD_reporting_date', 'MODEL_FAMILY' ], how = 'left')
            #test = prediction_cases_mkt_unagg_single[(prediction_cases_mkt_unagg_single['Target'].isnull())]
            prediction_cases_mkt_unagg_single['CHD_reporting_date'] = prediction_cases_mkt_unagg_single['CHD_reporting_date'] - MonthBegin(1)
            prediction_cases_mkt_unagg_single = prediction_cases_mkt_unagg_single[['Region', 'LEVEL_3_REPORTING', 'Brand', 'MODEL_FAMILY', 'Year', 'Month', 'Day', 'Target', 'lower_error', 'upper_error', 'CHD_reporting_date', 'RETAIL_FORECAST']] 
            prediction_cases_mkt_unagg_single = prediction_cases_mkt_unagg_single.rename(columns = {'CHD_reporting_date':'Handover Month-Year', 'MODEL_FAMILY':'Model_Family', 'LEVEL_3_REPORTING':'Market'})
            prediction_cases_mkt_unagg_single['Region'] = region_mapping[Level3][0]  
            prediction_cases_mkt_unagg_single = prediction_cases_mkt_unagg_single[~prediction_cases_mkt_unagg_single['Target'].isnull()]
            prediction_cases_mkt_unagg_single = prediction_cases_mkt_unagg_single.fillna(0)
            fname_new = 'CPDD_{}_{}'.format(region_mapping[Level3][0], Level3)
            prediction_cases_mkt_unagg_single.to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand-combined-DB.csv'
                           .format(folder,g_str,fname_new),index = False)  
                           
    else:                   
        df_test['average_abs'] = df_test['average']*df_test['RETAIL_FORECAST']
        df_mean = df_test.groupby(['MODEL_FAMILY', 'CPDD_reporting_date', 'pre_order_days'])['average_abs'].sum().reset_index()
        
        df_test['upper_error_abs'] = (df_test['upper_error']*df_test['RETAIL_FORECAST'])**2
        df_upper = df_test.groupby(['MODEL_FAMILY', 'CPDD_reporting_date', 'pre_order_days'])['upper_error_abs'].sum().reset_index()
        df_upper['upper_error_abs'] = np.sqrt(df_upper['upper_error_abs'])
        
        df_test['lower_error_abs'] = (df_test['lower_error']*df_test['RETAIL_FORECAST'])**2
        df_lower = df_test.groupby(['MODEL_FAMILY', 'CPDD_reporting_date', 'pre_order_days'])['lower_error_abs'].sum().reset_index()
        df_lower['lower_error_abs'] = np.sqrt(df_lower['lower_error_abs'])
        ###############################################################################
              
        df = pd.merge(df_mean, df_upper, on = ['MODEL_FAMILY', 'CPDD_reporting_date', 'pre_order_days'])
        df = pd.merge(df, df_lower, on = ['MODEL_FAMILY', 'CPDD_reporting_date', 'pre_order_days'])
        df_test = df_test.drop(['average_abs', 'upper_error_abs', 'lower_error_abs'], axis = 1)
        df_test = pd.merge(df_test, df, on = ['MODEL_FAMILY', 'CPDD_reporting_date', 'pre_order_days'], how = 'left')
        
        
        df_test.to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand-combined.csv'
                               .format(folder,g_str,fname),index = False)
                               
                               
'''
        df_test = pd.read_csv('../output_vault/dwt_cluster/classified/actual/test_cases/all_WithBand-combined-DB.csv')
        df_test[['Month', 'Year']] = df_test[['Month', 'Year']].astype(int).astype(str)
        df_test['CPDD_reporting_date'] = '01'+ '/' + df_test['Month'] + '/' + df_test['Year']
        df_test['CPDD_reporting_date'] = pd.to_datetime(df_test['CPDD_reporting_date'], format="%d/%m/%Y")

        df_test['average_abs'] = df_test['Target']*df_test['RETAIL_FORECAST']
        df_mean = df_test.groupby(['Market', 'Model_Family', 'Month', 'Year', 'Day'])['average_abs'].sum().reset_index()
        
        df_test['upper_error_abs'] = (df_test['upper_error']*df_test['RETAIL_FORECAST'])**2
        df_upper = df_test.groupby(['Market', 'Model_Family', 'Month', 'Year', 'Day'])['upper_error_abs'].sum().reset_index()
        df_upper['upper_error_abs'] = np.sqrt(df_upper['upper_error_abs'])
        
        df_test['lower_error_abs'] = (df_test['lower_error']*df_test['RETAIL_FORECAST'])**2
        df_lower = df_test.groupby(['Market', 'Model_Family', 'Month', 'Year', 'Day'])['lower_error_abs'].sum().reset_index()
        df_lower['lower_error_abs'] = np.sqrt(df_lower['lower_error_abs'])
        ###############################################################################
              
        df = pd.merge(df_mean, df_upper, on = ['Market', 'Model_Family', 'Month', 'Year', 'Day'])
        df = pd.merge(df, df_lower, on = ['Market', 'Model_Family', 'Month', 'Year', 'Day'])
        df_test = df_test.drop(['average_abs', 'upper_error_abs', 'lower_error_abs'], axis = 1)
        df_test = pd.merge(df_test, df, on = ['Market', 'Model_Family', 'Month', 'Year', 'Day'], how = 'left')
        
        df_test.to_csv('../output_vault/dwt_cluster/classified/actual/test_cases/all_WithBand-combined-DB_calc.csv',index = False)
        
        
        import pandas as pd
        df_test = pd.read_csv('../output_vault/dwt_cluster/classified/actual/test_cases/all_WithBand-combined-DB.csv')
        df_test = df_test.drop('RETAIL_FORECAST', axis = 1)
        df_test.columns
        df_test[['Year', 'Month', 'Day']] = df_test[['Year', 'Month', 'Day']].astype(int)        
        df_test.to_csv('../output_vault/dwt_cluster/classified/actual/test_cases/OIDB_DAY_TARGETS_2018_03.csv', index = False)
               
        
'''





