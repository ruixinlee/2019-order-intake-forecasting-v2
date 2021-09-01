# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:56:26 2018

@author: HCHANG
"""

#df_all, clus_num_set = make_cluster_to_csv_0daylevel(pfile, granularity, folder=folder, validation=v, clus_method = DBSCAN);
from sklearn.cluster import DBSCAN


def make_cluster_to_csv_0daylevel(pfile,granularity, folder, df_all = None, validation = False, clus_method = DBSCAN):
    print('clustering for ' + pfile)
    
    ############################################################################################
    #Load results of clustering by market & granularity (trained data), with test data appended#
    ############################################################################################
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
    #value_vars = [i for i in df_all.columns.values.tolist() if isinstance(i, int)]
    df_all = df_all.drop(value_vars, axis = 1)
    df_all['0day_sold_order_level_Cluster'] = np.NAN
    df_all['0day_sold_order_level%_scaled'] = np.NAN

    ########################################
    #Split data into trained and test sets.#
    ########################################
    #Level5 = fname.split('-')[0]
    #Level3 = fname.split('-')[1]
    g_str = ''.join(granularity)
    df_train = df_all[df_all['is_train']]
    df_test = df_all[~df_all['is_train']]

    ###############################
    #Cluster 0day_sold_order_level#
    ###############################
    clus_num_set = []
    df_list =pd.DataFrame()
    granular_items_list = []

    #Unique list of values for each granularity
    for g in granularity:
        granular_items_list.append(df_train[g].unique().tolist())
        
    for combi in itertools.product(*granular_items_list):
        print(combi)
        #import time
        #time.sleep(1) 
        combi = list(combi)
        df_train_itergran = df_train
        #df_test_itergran = df_test
        #Filter the df_train to only a specific granularity combination 
        for i, g in enumerate(granularity):
            df_train_itergran = df_train_itergran[df_train_itergran[g] == combi[i]]
            #df_test_itergran = df_test_itergran[df_test_itergran[g] == combi[i]]
         
        if df_train_itergran.shape[0] > 0: 
            df_train_itergran['0day_sold_order_level%_scaled'] = pd.DataFrame(MinMaxScaler().fit_transform(df_train_itergran['0day_sold_order_level%'].reshape(-1,1)), index = df_train_itergran.index)
        else:
            continue           
            
        #Reorganize data for clustering
        df_train_itergran_index = df_train_itergran.reset_index()['INDEX']
        df_train_itergran_clus_uscl = df_train_itergran['0day_sold_order_level%'].reshape(-1,1)
        df_train_itergran_clus_scl = df_train_itergran['0day_sold_order_level%_scaled'].reshape(-1,1)
        #df_test_itergran_index = df_test_itergran.reset_index()['INDEX']
        #df_test_itergran_clus = df_test_itergran['0day_sold_order_level%'].reshape(-1,1)
        
        #################################
        #Optimize for number of clusters#
        #################################
        if clus_method == KMeans:       
            elb = []
            for num_clus in list(range(1,8)):
                if df_train_itergran.shape[0] >= num_clus and df_train_itergran.shape[0] != 1:
                    kmeans_inst = KMeans(n_clusters = num_clus, precompute_distances=True)         
                    clus_cols = kmeans_inst.fit_predict(df_train_itergran_clus_scl)
                    clus_SSE = kmeans_inst.inertia_
                    elb.append([num_clus, clus_SSE])
                else:
                    break
                  
            elb = pd.DataFrame(elb, columns = ['num_clus', 'SSE'])
            elb['SSE'] = elb['SSE']/elb['SSE'].max()
            #elb.plot.scatter(x = 'num_clus', y = 'SSE')
            elb['SSE_up1'] = pd.Series([np.nan]).append(elb['SSE'][:-1]).reset_index(drop = True)
            elb['d_SSE'] = - elb['SSE'] + elb['SSE_up1'] 
            #elb['SSE_up2'] = pd.Series([np.nan]).append(elb['d_SSE'][:-1]).reset_index(drop = True)
            elb['r_SSE'] = elb['d_SSE']/elb['SSE_up1']
            
            elb = elb[elb['d_SSE'] > 0.05]
            #elb = elb[elb['r_SSE'] < 0.20]        
            #num_clus = elb['num_clus'].min() - 1
            num_clus = elb['num_clus'].max()
            clus_num_set.append([combi, df_train_itergran.shape[0], num_clus])
            ##Need to check what granularity element is missing!
        
        ####################
        #Determine clusters#
        ####################
        if clus_method == KMeans:
            del kmeans_inst
            if df_train_itergran.shape[0] > num_clus:
                kmeans_inst = KMeans(n_clusters = num_clus, precompute_distances=True)         
                clus_cols_train = kmeans_inst.fit_predict(df_train_itergran_clus_scl)
            else:
                continue
        
        elif clus_method == DBSCAN:       
            if df_train_itergran.shape[0] > 2:
                '''
                nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(df_train_itergran_clus_scl)
                nn_dist, indices = nbrs.kneighbors(df_train_itergran_clus_scl)
                nn_dist = nn_dist[:,1]
                histo = np.histogram(nn_dist, bins = 20)
                cumdens = np.cumsum(histo[0]/histo[0].sum())
                cutoff = np.where(cumdens<=max(min(cumdens),0.90))[0][-1]
                eps = max(histo[1][cutoff+1],0.1)
                '''
                df_train_itergran_clus_scl_flat = df_train_itergran_clus_scl.reshape(1,len(df_train_itergran_clus_scl))
                df_train_itergran_clus_scl_flat = np.sort(df_train_itergran_clus_scl_flat)
                df_train_itergran_clus_scl_flat = np.diff(df_train_itergran_clus_scl_flat)
                histo = np.histogram(df_train_itergran_clus_scl_flat, bins = 50)
                cumdens = np.cumsum(histo[0]/histo[0].sum())
                #5 clusters or 4 max next neigbour distances
                cutoff = np.where(cumdens<=max(min(cumdens),(1-4/histo[0].sum())))[0][-1]
                eps = histo[1][cutoff+1]#max(histo[1][cutoff+1],0.05)
                db_inst = DBSCAN(eps, min_samples=1)
                clus_cols_train = db_inst.fit_predict(df_train_itergran_clus_scl)
            else:
                continue
        df_train_itergran_clus = pd.DataFrame(clus_cols_train, columns = ['0day_sold_order_level%_cluster'], index = df_train_itergran_index)
        #df_train_itergran.loc[df_train_itergran_clus.index.tolist(), ['0day_sold_order_level_Cluster']] = df_train_itergran_clus['0day_sold_order_level%_cluster']
        df_train_itergran['0day_sold_order_level_Cluster'] = df_train_itergran_clus['0day_sold_order_level%_cluster']
        df_train_itergran['0day_sold_order_level_Cluster'] = df_train_itergran['0day_sold_order_level_Cluster'].astype(str) + combi[0] + '-'+ str(combi[1])
        df_list = pd.concat([df_list, df_train_itergran])
    
    df_all = pd.concat([df_list, df_test])

    if not df_all.empty:
        clustered_path = '../output_vault/dwt_cluster/clustered/{}/historical-{}-{}-0DayLevel.csv'.format(folder,g_str,
                                                                                pfile.split('/')[-1].split('.')[0])
        df_all.to_csv(clustered_path, index_label = 'INDEX')
    else:
        df_all = pd.DataFrame()
    return df_all, clus_num_set

#classify_cluster_to_csv_0daylevel(pfile, granularity, folder=folder, validation=v)

def make_confidence_band_weighted_bands_0daylevel(df_train_sub):
    d = df_train_sub['0day_sold_order_level%']
    w = df_train_sub['distance2']
    wavg = (d * w).sum() / w.sum() if w.sum() != 0 else np.nan
    wstd = np.sqrt(((d-wavg)*(d-wavg)*w).sum()/w.sum()) if w.sum() != 0 else np.nan
    return {'mean': [wavg], 'lower': [wavg - wstd*1.282], 'upper': [wavg + wstd*1.282]}
    
def make_confidence_band_relevant_month_0daylevel(df_train_sub_relmonth, row):
    d = df_train_sub_relmonth['0day_sold_order_level%']
    FD_Dist_x = partial(FD_Dist, centre = 3, steepness = 1.3)
    df_train_sub_relmonth['weight'] = pd.Series([FD_Dist_x(ii) for ii in abs(df_train_sub_relmonth['CPDD_Year']-row['CPDD_Year'])], index = [ii for ii in df_train_sub_relmonth.index])
    #df_train_sub_relmonth['weight'] = df_train_sub_relmonth['weight']/df_train_sub_relmonth['weight'].sum()    
    w = df_train_sub_relmonth['weight']
    wavg = (d * w).sum() / w.sum() if w.sum() != 0 else np.nan
    wstd = np.sqrt(((d-wavg)*(d-wavg)*w).sum()/w.sum()) if w.sum() != 0 else np.nan
    return {'mean': [wavg], 'lower': [wavg - wstd*1.282], 'upper': [wavg + wstd*1.282]}
    
def classify_cluster_to_csv_0daylevel(pfile, granularity, folder, df_all = None, validation = False):
    print('classifying for ' + pfile)

    #Load results of clustering by market & granularity (trained data), with test data appended
    if not df_all:
        g_str = ''.join(granularity)
        fname = pfile.split('/')[-1].split('.')[0]
        pfile2 = '../output_vault/dwt_cluster/clustered/{}/historical-{}-{}-0DayLevel.csv'.format(folder,g_str,fname)
        if not os.path.isfile(pfile2):
            return()
        # if os.path.isfile('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}.csv'.format(folder, g_str, fname)):
        #     return ()
        df_all =pd.read_csv(pfile2, index_col = 'INDEX')
        
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

    #d_confidence_band = find_confidence_band_of_cluster(df_train, clusters_type = ['0day_sold_order_level_Cluster'], DaysColsOR0DayLevel = '0DayLevel')

    #Create a list of attributes concerning the band
    band_attr = ['months_before_handover', 'BRAND', 'MODEL_FAMILY','SEGMENT','RETAIL_FORECAST', 'CHD_reporting_date', 'CPDD_reporting_date',
                 'RETAIL_ACTUALS_AND_FORECAST', 'RETAIL_FORECAST_cutoff', 'Target_Met', 'VISTA_RETAIL_ACTUALS_AND_FORECAST', 'is_train', '0day_sold_order_level%']

    #Create a list of attributes needed for classification
    class_attr_1 = ['0day_sold_order_level_Cluster', 'MODEL_FAMILY', 'RETAIL_FORECAST' ,'CHD_Year', 'CPDD_Year', 'CPDD_Month', 'CHD_Month']
    class_attr = list(class_attr_1)
    clus_info = []
    band_lists = []

    progress = pb.ProgressBar(widgets=[' [', pb.Timer(), '] ', pb.Bar(),' (', pb.ETA(), ') ',], max_value = df_test.shape[0]);
    progvar = 0
    
    for irow,row in df_test.iterrows():

        #if irow == 2262: break
        #df_test = df_test.loc[1722,:].to_frame().T
        df_train_sub = df_train.copy(deep=True)

        #Filter the trained dataset for segment relevant to the particular test row
        for _, g in enumerate(granularity):
            df_train_sub = df_train_sub[df_train_sub[g] == row[g]]
        if df_train_sub.shape[0] ==0:
            continue

        #Filter trained and test row of data for columns only relevant to classification
        df_train_attr = df_train_sub[class_attr]
        row_attr = row[[ii for ii in class_attr if ii != '0day_sold_order_level_Cluster']].to_frame().T
        
        #Rescale the columns.
        #scaler = MinMaxScaler()
        #df_train_attr['RETAIL_FORECAST'] = scaler.fit_transform(df_train_attr['RETAIL_FORECAST'].to_frame())
        #row_attr['RETAIL_FORECAST'] = scaler.transform(row_attr.where(~row_attr.isnull(),0)['RETAIL_FORECAST'].to_frame())

        df_train_df = pd.DataFrame()
        df_train_df['0day_sold_order_level_Cluster'] = df_train_attr['0day_sold_order_level_Cluster']
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

        df_train_df_sum = df_train_df.groupby(by = '0day_sold_order_level_Cluster')[[col for col in df_train_df if col not in ['CHD_Year', 'CPDD_Year']]].sum()
        df_train_df_cnt = df_train_df.groupby(by = '0day_sold_order_level_Cluster')[['CHD_Year', 'CPDD_Year']].count()
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
        df_train_sub = pd.merge(df_train_sub, df_train_df[['0day_sold_order_level_Cluster', 'distance2']], how = 'left', left_on = '0day_sold_order_level_Cluster', right_on = '0day_sold_order_level_Cluster')        
        df_train_sub_relmonth = df_train_sub[(df_train_sub['months_before_handover']==row['months_before_handover']) & (df_train_sub['CPDD_Month'] == row['CPDD_Month'])]

        if df_train_sub.shape[0] > 0:
            clus_dist = summarise_clus_dist(df_train_sub, '0day_sold_order_level_Cluster')
            
            if clus_dist.shape[0] > 0:
                #Construct weighted confidence band
                if df_train_sub_relmonth.shape[0] >= 5 and Level3 != 'USA':
                    ct = '0day_sold_order_level_SimilarMonths'
                    band_name = 'similar_months' 
                    #FD_Dist_x = partial(FD_Dist, centre = 3, steepness = 1.3)
                    #df_train_sub_relmonth['row_probability'] = pd.Series([FD_Dist_x(ii) for ii in abs(df_train_sub_relmonth['CPDD_Year']-row['CPDD_Year'])], index = [ii for ii in df_train_sub_relmonth.index])
                    #df_train_sub_relmonth['row_probability'] = df_train_sub_relmonth['row_probability']/df_train_sub_relmonth['row_probability'].sum()
                    interp_data = {}
                    interp_data['0day_sold_order_level%'] = df_train_sub_relmonth['0day_sold_order_level%'].quantile([.2, .5, 0.8]).tolist()
                    #interp_data['0day_sold_order_level%'] = err_quant(df_train_sub_relmonth, '0day_sold_order_level%')
                    #d_confidence_band = make_confidence_band_relevant_month_0daylevel(df_train_sub_relmonth, row)
                    #for k, v in d_confidence_band.items():
                    #print(k,v)
                        #band_l = band_list_extend(band_l = v, k = k, ct = '0day_sold_order_level_SimilarMonths', band_name = 'similar_months', row = row, band_attr = band_attr)                    
                        #band_lists.append(band_l)
                else:
                    #d_confidence_band = find_confidence_band_of_cluster(df_train_sub, clusters_type = ['0day_sold_order_level_Cluster'], DaysColsOR0DayLevel = '0DayLevel')
                    #band_lists = make_confidence_band_weighted_bands(band_lists,d_confidence_band, clus_dist, ct = '0day_sold_order_level_Cluster', row = row, band_attr = band_attr, DaysColsOR0DayLevel = '0DayLevel')           
                    ct = '0day_sold_order_level_Cluster'
                    band_name = 'cumulative_distribution'
                    df_train_sub = pd.merge(df_train_sub,clus_dist[['row_probability']], how = 'left', left_on = '0day_sold_order_level_Cluster', right_index = True)     
                    interp_data = {}        
                    interp_data['0day_sold_order_level%'] = err_quant(df_train_sub, '0day_sold_order_level%')
                    
                df = pd.DataFrame.from_dict(interp_data)
                df['confidence_band'] = pd.Series(['lower', 'mean', 'upper'])
                df = df.set_index('confidence_band')
                for row_quant in df.index:
                    #print(row_quant)
                    band_l = band_list_extend(df.loc[row_quant,:].tolist(), row_quant, ct, band_name, row, band_attr)
                    band_lists.append(band_l)

                clus_dist_new = clus_dist.copy(deep = True)
                clus_dist_new.index = clus_dist_new.index.rename('CLUSTER')
                clus_dist_new = clus_dist_new.reset_index()
                clus_dist_new['INDEX'] = row.name
                clus_dist_new['LEVEL5'] = Level5
                clus_dist_new['LEVEL3'] = Level3
                clus_dist_new['GRANULARITY'] = g_str
                clus_dist_new['CLUSTER_TYPE'] = '0day_sold_order_level_Cluster'
                clus_info.append(clus_dist_new)
        progress.update(progvar + 1)        
        progvar += 1
        
    #Tidying band lists:
    band_df_cols = ['0day_sold_order_level%_band']
    band_df_cols.extend(['confidence_band', 'clusters_type', 'weighted', 'INDEX' ])
    band_df_cols.extend(band_attr)
    df_band_melt = pd.DataFrame(band_lists, columns = band_df_cols)
    df_band_melt = df_band_melt.drop('0day_sold_order_level%', axis =1, errors = 'ignore') 
    
    clus_info = pd.concat(clus_info)
    
    df_test['confidence_band'] = 'actual'
    if validation:
        df_test['0day_sold_order_level%_band'] = df_test['0day_sold_order_level%']
    #df_test['clusters_type'] = '0day_sold_order_level_Cluster'
    #df_test['weighted'] = 'weighted_bands'
    clusters_type = [i for i in df_train.columns.values.tolist() if not isinstance(i,int) and i.split('-')[0] == 'clus']
    clusters_type.extend(['0day_sold_order_level%', '0day_sold_order_level_Cluster','CHD_Month', 'CHD_Year', 'CPDD_Month', 'CPDD_Year', '0day_sold_order_level%_scaled'])
    df_test = df_test.drop(clusters_type, axis =1, errors = 'ignore')
    df_test_melt = df_test.reset_index()
    if not validation:
        df_test_melt = df_test_melt.rename(columns = {'LEVEL_3_REPORTING':'LEVEL3', 'index':'INDEX'})
    
    if not df_band_melt.empty:
        df_band_melt['LEVEL5'] = Level5
        df_band_melt['LEVEL3'] = Level3
        df_band_melt['GRANULARITY'] = g_str 
        df_band_melt.to_csv('../output_vault/dwt_cluster/classified/{}/confidence/{}-{}-0DayLevel.csv'
                       .format(folder,g_str,fname), index = False )
    if not clus_info.empty:
        clus_info.to_csv('../output_vault/dwt_cluster/classified/{}/clus_info/{}-{}-0DayLevel.csv'
                       .format(folder,g_str,fname),index = False)
    if not df_test_melt.empty:
        df_test_melt.to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}-0DayLevel.csv'
                       .format(folder,g_str,fname),index = False)
    if not df_band_melt.empty:
        pd.concat([df_test_melt, df_band_melt], ignore_index = True).to_csv('../output_vault/dwt_cluster/classified/{}/test_cases/{}-{}_WithBand-0DayLevel.csv'
                       .format(folder,g_str,fname),index = False)
                       
#[ii for ii in df_band_melt.columns.values if ii not in df_test_melt.columns.values]