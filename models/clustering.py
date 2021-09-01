"""
.. module:: clustering
.. moduleauthor:: Hui Chang <hchang@jaguarlandrover.com>, Rui Xin Lee <rlee7@jaguarlandrover.com>
.. refactoredby:: Emmanuel Aroge <earoge@jaguarlandrover.com>

"""

import itertools
import math
import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler

from config.conf import env, gcp, logger
from config.model_config import (band_attr, class_attr_1, date_columns,
                                 level_3_date_truncate, main_config)
from config.model_config import pre_order_curves_columns_mapping as pre_col_map
from config.model_config import pre_order_day_col, tables
from utils.bq import bq_table_to_dataframe, load_df_to_bigquery, bq_query_to_dataframe
from utils.pointwise_measure import pointwise

warnings.filterwarnings("ignore")

if env == 'test':
    LIMIT = True
else:
    LIMIT = False


class ModelParameters:
    # define columns
    is_train_col = pre_col_map["is_train"]
    level_3_reporting_col = pre_col_map["level_3_reporting"]
    level_5_reporting_col = pre_col_map["level_5_reporting"]
    zero_day_sold_order_level_perc_col = pre_col_map["0day_sold_order_level_prec"]
    zero_day_sold_order_level_cluster_col = pre_col_map["0day_sold_order_level_cluster"]
    zero_day_sold_order_perc_scaled_col = pre_col_map["0day_sold_order_level%_scaled"]
    distance_col = pre_col_map["distance"]
    months_before_handover_col = pre_col_map["months_before_handover"]
    CPDD_month_col = pre_col_map["CPDD_month"]
    CPDD_year_col = pre_col_map["CPDD_year"]
    CHD_year_col = pre_col_map["CHD_year"]
    CHD_month_col = pre_col_map["CHD_month"]
    index_col = pre_col_map["index"]
    level_5_col = pre_col_map["Level_5"]
    level_3_col = pre_col_map["Level_3"]
    granularity_col = pre_col_map["Granularity"]
    cluster_type_col = pre_col_map["cluster_type"]
    zero_day_sold_order_level_band_col = pre_col_map["0day_sold_order_level%_band"]
    zero_day_sold_order_level_col = pre_col_map["0day_sold_order_level_prec"]
    zero_day_sold_order_level_perc_scaled_col = pre_col_map[
        "0day_sold_order_level%_scaled"
    ]
    zero_day_sold_order_level_perc_cluster_col = pre_col_map[
        "0day_sold_order_level%_cluster"
    ]
    zero_day_sold_order_level_perc_col = pre_col_map["0day_sold_order_level_prec"]
    confidence_band_col = pre_col_map["confidence_band"]
    weighted_col = pre_col_map["weighted"]
    cluster_type_mean_col = pre_col_map["clusters_type_mean"]
    weighted_mean_col = pre_col_map["weighted_mean"]
    level_5_mean_col = pre_col_map["level_5_mean"]
    granularity_mean_col = pre_col_map["granularity_mean"]
    vista_actuals_and_forecasts = pre_col_map["vista_retail_actuals"]
    sold_order_level_perc = pre_col_map["sold_order_level_perc"]
    retail_actuals_and_forecast_col = pre_col_map["retail_sales_and_forecast"]
    retail_forecast_cutoff_col = pre_col_map["retail_forecast_cutoff"]
    target_met_col = pre_col_map["target_met"]
    vista_retail_actuals_and_forecast_col = pre_col_map["vista_retail_actuals"]
    is_train_col = pre_col_map["is_train"]
    CHD_reporting_date_col = pre_col_map["CHD_reporting_date"]
    CPDD_reporting_date_col = pre_col_map["CPDD_reporting_date"]
    year_col = pre_col_map["year"]
    month_col = pre_col_map["month"]
    model_family_col = pre_col_map["model_family"]
    classify_cluster_var = pre_col_map["classify_cluster_var"]
    classify_cluster_val = pre_col_map["classify_cluster_val"]
    cluster_type = pre_col_map["cluster_type"]


def df_extend(df, row):
    df[ModelParameters.confidence_band_col] = ["lower", "mean", "upper"]
    df['MONTHS_BEFORE_HANDOVER'] = row['MONTHS_BEFORE_HANDOVER']
    df['BRAND'] = row['BRAND']
    df['MODEL_FAMILY'] = row['MODEL_FAMILY']
    df['SEGMENT'] = row['SEGMENT']
    df['RETAIL_FORECAST'] = row['RETAIL_FORECAST']
    df['CHD_REPORTING_DATE'] = row['CHD_REPORTING_DATE']
    # df['CPDD_REPORTING_DATE'] = row['CPDD_REPORTING_DATE']
    df['RETAIL_ACTUALS'] = np.nan
    df['RETAIL_FORECAST_cutoff'] = np.nan
    df['TARGET_MET'] = np.nan
    df['VISTA_RETAIL_ACTUALS'] = np.nan
    df[ModelParameters.is_train_col] = np.nan

    return df

def make_dist_matrices(df_matrix, dist_methods):
    """
    Used in :func:`make_cluster_to_csv`.
    **For shape clustering: calculates the distance between glide paths.**
    """

    matrix_size = df_matrix.shape[0]
    matrix_dict = {
        k: np.zeros((matrix_size, matrix_size)) for (k, v) in dist_methods.items()
    }

    for (k, v) in dist_methods.items():
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                dist_val = dist_methods[k](df_matrix[i], df_matrix[j])
                matrix_dict[k][i, j] = dist_val
                matrix_dict[k][j, i] = dist_val
    return matrix_dict


def make_clus_cols(dist_matrix_dict, clus_methods, combi, num_clus):
    """
    Used in :func:`make_cluster_to_csv`.\n
    **For shape clustering: executes the clustering.**
    """
    clus_dict = {}
    combi_str = "-".join(combi)
    for (clus_k, clus_v) in clus_methods.items():
        for (dist_k, dist_v) in dist_matrix_dict.items():
            if dist_v.shape[0] > 1:
                if clus_k == "DBSCAN":
                    dist_v2 = dist_v.copy()
                    dist_v2[dist_v2 == 0] = np.inf
                    nn_dist = np.min(dist_v2, axis=0)
                    histo = np.histogram(nn_dist)
                    cumdens = np.cumsum(histo[0] / histo[0].sum())
                    cutoff = np.where(cumdens <= max(
                        min(cumdens), 0.95))[0][-1]
                    eps = max(histo[1][cutoff + 1], 0.1)
                    clus_arr = clus_v(
                        eps=eps, metric="precomputed", min_samples=10
                    ).fit_predict(dist_v)
                elif clus_k == "KMeans":

                    n_clus = num_clus  # (dist_v.shape[0] // 20)+1
                    cluster_inst = clus_v(
                        n_clusters=n_clus, precompute_distances=True, random_state=5
                    )
                    clus_arr = cluster_inst.fit_predict(dist_v)
                    clus_SSE = cluster_inst.inertia_
                clus_dict["clus_{}_{}".format(dist_k, clus_k)] = [
                    str(i) + combi_str for i in clus_arr
                ]

    return (clus_dict, clus_SSE)


def err_quant2(x, row_probability, lp=0.2, mp=0.5, hp=0.8):
    df = pd.DataFrame({x.name: x, row_probability.name: row_probability})
    df = df.sort_values(by=x.name)
    df["prob_cs"] = df["row_probability"].cumsum()
    # if not df.prob_cs.empty:
    return np.interp(x=[lp, mp, hp], xp=df.prob_cs, fp=df[x.name])


def col_binariser(df_train_df, label_key=ModelParameters.CHD_year_col):
    """Used in :func:`find_similarity_of_test_to_clus`.
    **Encodes a column into columns with name linked to the (unique) values in the original column.**
    """

    binarize_class = df_train_df[label_key].unique()
    for i in binarize_class:
        df_train_df[label_key + "_" +
                    str(i)] = (df_train_df[label_key] == i) * 1

    return df_train_df


def band_list_extend(band_l, k, ct, band_name, row, band_attr):
    """
    Used in :func:`classify_cluster_to_csv`.
    """
    band_l.extend([k, ct, band_name, row.name])
    band_l.extend(row[band_attr].tolist())
    return band_l


def summarise_clus_dist(df_train_sub, ct):
    """
    Used in :func:`classify_cluster_to_csv`.
    Summarizes the relevance of a test/prediction case to each
    of the cluster belonging to the relevant granularity values.

    :param df_train_sub: Dataframe containing a column 'distance2',
    quantifying the relevance of the test/prediction case to each of the cluster.
    :type df_train_sub: DataFrame
    :param ct: Name of the column containing the cluster names.
    :type ct: str.
    :returns: DataFrame
    """

    clus_dist = df_train_sub.groupby([ct])[ModelParameters.distance_col].agg(
        ["mean", "count"])

    clus_dist["normed_mean"] = clus_dist["mean"] / clus_dist["mean"].sum()
    clus_dist["normed_count"] = clus_dist["count"] / clus_dist["count"].sum()
    clus_dist["weights"] = clus_dist["mean"]
    clus_dist["probability"] = clus_dist["weights"] / \
        clus_dist["weights"].sum()
    clus_dist["row_probability"] = clus_dist["probability"] / \
        clus_dist["count"]
    return clus_dist


def melt_for_classify_cluster(df):
    """
    Used in :func:`make_cluster_to_csv` and :func:`classify_cluster_to_csv`.

    **Changes columns of pre_order_days into a column with the pre_order_days.**

    :param df: DataFrame with pre_order_days columns and 'sold_order_level%' column.
    :type df: DataFrame
    :returns: DataFrame
    """
    value_vars = [i for i in df.columns.values if i in pre_order_day_col]
    id_vars = [j for j in df.columns.values if j not in value_vars]
    if value_vars:
        df_melt = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=ModelParameters.classify_cluster_var,
            value_name=ModelParameters.classify_cluster_val,
        )
        return df_melt
    else:
        return df


def FD_Dist(x, centre, steepness):
    """
    Used in :func:`find_similarity_of_test_to_clus`.
    Step-like function: used to generate weights for years. Symmetric about
    x = 0; x = 0 gives 1; x -> inf gives 0.

    :param x: Independent variable.
    :param centre: The x-point where output is approx. 0.5.
    :param steepness: Steepness of step.
    :type x: Float
    :type centre: Float
    :type steepness: Float
    :returns: Float
    """
    return (math.exp(-steepness * centre) + 1) / (
        math.exp((x - centre) * steepness) + 1
    )


def err_quant(df_train_sub, day_col, lp=0.2, mp=0.5, hp=0.8):
    """
    Used in :func:`classify_cluster_to_csv`.\n

    **Calculates the median and the band for a partiular test/prediction
    case using the clusters and the corresponding relevance metrics.**

    :param df_train_sub: Dataframe containing all members of clusters
        relevant to the test/prediction case\
    and the relevance (metric) of each member to the test/prediction case.
    :type df_train_sub: DataFrame
    :param day_col: Column(s) to aggregate.
    :type day_col: List
    :param lp: Lower percentile.
    :type lp: Float
    :param mp: Median.
    :type mp: Float
    :param lp: Upper percentile.
    :type lp: Float
    :returns: DataFrame
    """
    df = df_train_sub[["row_probability", day_col]]
    # df = df.dropna()
    df = df.sort_values(by=[day_col])
    df["prob_cs"] = df["row_probability"].cumsum()
    # if not df.prob_cs.empty:
    return np.interp(x=[lp, mp, hp], xp=df.prob_cs, fp=df[day_col])


def find_similarity_of_test_to_clus(
    cluster_col, df_train_attr, row_attr, centre, steepness
):
    """
    Used in :func:`classify_cluster_to_csv`

    Calculates the median and the band (weighted percentiles)
    for a partiular test/prediction case using the clusters
    and the relevance metrics.

    :param df_train_sub: Dataframe containing all members of
        clusters relevant to the test/prediction case
        and the relevance (metric) of each of the member to
        the test/prediction case.
    :type df_train_sub: DataFrame
    :param day_col: Column(s) to aggregate.
    :type day_col: List
    :param lp: Lower percentile.
    :type lp: Float
    :param lp: Median.
    :type lp: Float
    :param lp: Upper percentile.
    :type lp: Float
    :returns: DataFrame
    """
    df_train_df = pd.DataFrame()
    df_train_df[cluster_col] = df_train_attr[cluster_col].values.tolist()

    columns = [ModelParameters.model_family_col,ModelParameters.CHD_month_col]

    for col in columns:
        df_train_df[col] = (df_train_attr[col] ==
                            row_attr.iloc[0][col]).astype(int).tolist()

        #TODO not sure what this is for
        # df_train_df[col] = (df_train_attr[col] ==
        #                     row_attr.iloc[0][col]).astype(int)

    for col in [ModelParameters.CHD_year_col]:
        df_train_df[col] = abs(df_train_attr[col] - row_attr.iloc[0][col]).tolist()
    df_train_df = col_binariser(
        df_train_df, label_key=ModelParameters.CHD_year_col)
    #
    cpdd_year_col_nm = ModelParameters.CHD_year_col + "_"
    CPDD_Year_col = [
        col for col in df_train_df if cpdd_year_col_nm in col and "nan" not in col
    ]
    FD_Dist_x = partial(FD_Dist, centre=centre, steepness=steepness)

    for col in CPDD_Year_col:
        df_train_df[col] = df_train_df[col] * FD_Dist_x(float(col[-1]))

    cols_dict = {col:'sum' for col in df_train_df.columns.values if col not in [ModelParameters.CHD_year_col]}
    df_train_df['MODEL_FAMILY'] = df_train_df['MODEL_FAMILY'] *1.0
    df_train_df['CHD_MONTH'] = df_train_df['CHD_MONTH'] *1.0
    df_train_df['CHD_YEAR'] = df_train_df['CHD_YEAR'] *1.0
    df_train_df_sum = df_train_df.groupby([cluster_col]).agg(cols_dict)

    df_train_df_cnt = df_train_df.groupby([cluster_col]).agg({ModelParameters.CHD_year_col: 'count'})


    df_train_df = pd.concat([df_train_df_sum, df_train_df_cnt], axis=1)

    for col in columns:
            df_train_df[col] = df_train_df[col] / df_train_df[col].sum()

    for col in CPDD_Year_col:
        df_train_df[col] = df_train_df[col] / \
            df_train_df[ModelParameters.CHD_year_col]

    df_train_df[ModelParameters.CHD_year_col] = df_train_df[CPDD_Year_col].sum(
        axis=1)

    df_train_df[ModelParameters.CHD_year_col] = (
        df_train_df[ModelParameters.CHD_year_col]
        / df_train_df[ModelParameters.CHD_year_col].sum()
    )

    df_train_df = df_train_df[
        [
            ModelParameters.model_family_col,
            ModelParameters.CHD_month_col,
            ModelParameters.CHD_year_col
        ]
    ]

    for col in df_train_df.columns:
        df_train_df[col] = df_train_df[col] / df_train_df[col].sum()

    prob_col = [
        ModelParameters.model_family_col,
        ModelParameters.CHD_month_col,
        ModelParameters.CHD_year_col,
    ]

    # df_train_df[ModelParameters.CPDD_month_col] = (
    #     2 * df_train_df[ModelParameters.CPDD_month_col]
    # )
    df_train_df[ModelParameters.distance_col] = df_train_df[prob_col].sum(
        axis=1)
    df_train_df[ModelParameters.distance_col] = (
        df_train_df[ModelParameters.distance_col]
        / df_train_df[ModelParameters.distance_col].sum()
    )
    df_train_df = df_train_df.reset_index()

    return df_train_df


def make_cluster_to_csv(granularity, validation, market, if_bq_exists):
    """
    **Cluster the glide paths based on shape (only) for a particular
    granularity and country, as indicated in *pfile*.**

    .. note::
        Remove CHD month-year and model combination with low sales figure
        VISTA_RETAIL_ACTUALS_AND_FORECAST < 40).
        Scales the glide paths by the 0day_sold_order_level% for the
        purpose of shape clustering.
        Remove earlier data in USA & China NSC (CHD < 2014-05-01).
        Prepares the train and test set of data if validation is true or
        else use all target met true as training data.

    :param granularity: (e.g. ['SEGMENT', 'months_before_handover'])
    :param validation: Choose whether to validate model (i.e. testing) (when True)
        or to do predictions (for future month-years) (for False).
    :param glide_path_dataframe:

    :type granularity: List
    :type validation: Bool.
    :type glide_path_dataframe: pandas.Dataframe
    :returns: Similar to the input data, but with cluster assigned (DataFrame)
    """

    df_master = get_data_by_market(tables["glide_path_table"], market)

    # defining columns
    # Remove CHD month-year and model with low volume sales
    df_master = df_master[
        df_master[ModelParameters.vista_retail_actuals_and_forecast_col] >= 40
    ]

    df_master = df_master.dropna()

    i_cols = pre_order_day_col.copy()

    # Scale so that the amount at 0 day is 1 -- for shape clustering.
    df_master[i_cols] = df_master[i_cols]\
        .divide(df_master[ModelParameters.zero_day_sold_order_level_perc_col],axis='rows')

    # Remove glide paths where the cumulative sold order level at 0 day is
    # zero, i.e. a zero/NULL glide path.
    df_master = df_master.dropna()

    # Remove earlier data in USA & China NSC, as they are not
    # consistent with the more recent glide paths.
    if df_master.shape[0] == 0:
        print(market)
        return
    # if df_master['LEVEL_3_REPORTING'].tolist()[0] in level_3_date_truncate:
    #     df_master = df_master[
    #         df_master[ModelParameters.CPDD_reporting_date_col]
    #         >= main_config["cpdd_reporting_start"].date()
    #     ]

    # Create a list of unique values for each granularity item
    granular_items_list = []

    for g in granularity:
        granular_items_list.append(df_master[g].unique().tolist())

    clus_num_set = []
    df_list = []

    # Iterate through every combination of values corresponding to each
    # granularity item.
    for combi in itertools.product(*granular_items_list):
        combi = list(combi)
        df = df_master.copy(deep = True)

        # Get test cases: latest 10% of CPDD month/year that could have had/have
        # n, n+1, n+2 and n+3 handover (max_num_months_ahead) at the model family
        # level for the segment concerned in the current loop.
        if validation:
            # Filter for the segment concerned within the current loop.
            for i, g in enumerate(granularity):
                if g == "SEGMENT":
                    df_unsplit = df[df[g] == combi[i]]
            test_index = []
            for model in df_unsplit.MODEL_FAMILY.unique():
                max_num_months_ahead = 3
                df_unsplit_onemodel = df_unsplit[
                    df_unsplit[ModelParameters.model_family_col] == model
                ]
                CPDD_arr = np.sort(
                    df_unsplit_onemodel["c"].map(lambda t: t.date()).unique()
                )
                CPDD_arr_test = CPDD_arr[
                    len(CPDD_arr)
                    - round((len(CPDD_arr) - max_num_months_ahead) * 0.1)
                    - max_num_months_ahead:
                ]
                test_index.extend(
                    list(
                        df_unsplit_onemodel[
                            df_unsplit_onemodel[ModelParameters.CPDD_reporting_date_col].isin(
                                CPDD_arr_test
                            )
                        ].index.values
                    )
                )

        # Filter for a particular combination of granularity values.
        for i, g in enumerate(granularity):
            df = df[df[g] == combi[i]]


        df_true = df[df[ModelParameters.target_met_col]]

        # Create train & test dataframe
        if validation:
            train_index = [
                ii for ii in df_true.index.values if ii not in test_index]
            test_index = [ii for ii in df.index.values if ii in test_index]
            df_train = df_true.loc[train_index, :]
            df_test = df.loc[test_index, :]
        else:
            df_train = df_true.copy(deep=True)
            df_test = df[~df[ModelParameters.target_met_col]]

        if df_train.empty:
            continue

        # Get a list containing the name of the pre_order_days columns
        value_vars = [i for i in df.columns.values if i in pre_order_day_col]
        df_train_1 = df_train[value_vars]
        df_matrix = df_train_1.as_matrix()

        # Define the clustering method
        # options: 'dwt':dtwDist, 'frechet':frechetDist, 'pointwise':pointwise
        dist_methods = {"pointwise": pointwise}

        # 'DBSCAN': DBSCAN, 'KMeans': KMeans
        clus_methods = {"KMeans": KMeans}

        # Calculate distance matrix
        combi2 = [str(ii) for ii in combi]
        dist_matrix_dict = make_dist_matrices(df_matrix, dist_methods)

        # Optimize number of clusters
        elb = []

        for num_clus in list(range(1, 8)):

            # Make sure that there is enough data points for clustering.
            if df_train_1.shape[0] >= num_clus and df_train_1.shape[0] != 1:
                clus_cols, clus_SSE = make_clus_cols(
                    dist_matrix_dict, clus_methods, combi2, num_clus
                )
                elb.append([num_clus, clus_SSE])
            else:
                break

        elb = pd.DataFrame(elb, columns=["num_clus", "SSE"])
        elb["SSE"] = elb["SSE"] / elb["SSE"].max()
        elb["SSE_up1"] = (
            pd.Series([np.nan]).append(elb["SSE"][:-1]).reset_index(drop=True)
        )
        elb["d_SSE"] = -elb["SSE"] + elb["SSE_up1"]

        # Get the max number of clusters after which improvement in SSE is
        # below a threshold.
        elb = elb[elb["d_SSE"] > 0.05]
        num_clus = elb["num_clus"].max()

        # For checking whether the number of clusters for all combination of
        # granularity values makes sense.
        clus_num_set.append([combi2, df_train_1.shape[0], num_clus])

        # Cluster the glide paths according to shapes.
        if df_train_1.shape[0] > num_clus:
            clus_cols, clus_SSE = make_clus_cols(
                dist_matrix_dict, clus_methods, combi2, num_clus
            )
        else:
            continue

        for (k, v) in clus_cols.items():
            df_train[k] = v
            df_test[k] = np.nan

        # Create a column indicating whether the data is a training sample or a
        # testing sample.
        df_train[ModelParameters.is_train_col] = True
        df_test[ModelParameters.is_train_col] = False
        df = pd.concat([df_train, df_test])

        # Create columns for CHD and CPDD month and year.
        df[ModelParameters.CHD_month_col] = df[
            ModelParameters.CHD_reporting_date_col
        ].map(lambda x: x.month)

        df[ModelParameters.CHD_year_col] = df[
            ModelParameters.CHD_reporting_date_col
        ].map(lambda x: x.year)

        # df[ModelParameters.CPDD_month_col] = df[
        #     ModelParameters.CPDD_reporting_date_col
        # ].map(lambda x: x.month)
        #
        # df[ModelParameters.CPDD_year_col] = df[
        #     ModelParameters.CPDD_reporting_date_col
        # ].map(lambda x: x.year)

        # Appends the result coresponding to a particular combination of
        # granularity values to the master list.
        df_list.append(df)

    if len(df_list) > 0:
        df_all = pd.concat(df_list)
        load_df_to_bigquery(
            df_all, tables["glide_path_shape_cluster"], if_exists = if_bq_exists
        )

        df_all_melted = melt_for_classify_cluster(df_all)
        load_df_to_bigquery(
            df_all_melted, tables["glide_path_shape_melted_cluster"], if_exists = if_bq_exists
        )

    logger.info("shape level clustering completed.")


def make_cluster_to_csv_0daylevel(granularity,if_bq_exists, market, **kwargs):
    """
    **Cluster the 0day_sold_order_level% (that is, relative to the retail forecast)**
    **for a particular granularity and country, as indicated in** *the data*

    .. note::
        Load result of shape clustering, which contains the train/test split assignment.
        Cluster the 0day_sold_order_level% using DBSCAN with the number of next neighbour
        distance above the threshold set to 4.

    :param granularity: (e.g. ['SEGMENT', 'months_before_handover'])
    :type granularity: List
    :returns: Similar to the input data, but with the 0day_sold_order_level%
        cluster assigned and pre_order_days info removed. (DataFrame)
    """

    df_all = get_data_by_market(tables["glide_path_shape_cluster"], market)

    if df_all.shape[0] ==0:
        return
        # Get a list containing the name of the pre_order_days columns.
    value_vars = [i for i in df_all.columns.values if i in pre_order_day_col]

    # Remove the pre_order_days columns as they are irrelevant.
    df_all = df_all.drop(value_vars, axis=1)

    df_all[ModelParameters.zero_day_sold_order_level_cluster_col] = np.NAN
    df_all[ModelParameters.zero_day_sold_order_level_perc_scaled_col] = np.NAN

    # Split the dataset into train and test sets.
    df_train = df_all[df_all[ModelParameters.is_train_col]]
    df_test = df_all[~df_all[ModelParameters.is_train_col]]


    if df_train.shape[0] ==0:
        return

    # Create a list of unique values for each granularity item.
    granular_items_list = []

    for g in granularity:
        granular_items_list.append(df_train[g].unique().tolist())

    df_list = pd.DataFrame()

    # Iterate through every combination of values corresponding to each
    # granularity item.
    for combi in itertools.product(*granular_items_list):
        combi = list(combi)
        df_train_itergran = df_train

        # Filter for a particular combination of granularity values.
        for i, g in enumerate(granularity):
            df_train_itergran = df_train_itergran[df_train_itergran[g] == combi[i]]

        # Scale the 0 day order level % for a particular combination of
        # granularity values to between 0 and 1.
        if df_train_itergran.shape[0] > 0:
            df_train_itergran[
                ModelParameters.zero_day_sold_order_level_perc_scaled_col
            ] = pd.DataFrame(
                MinMaxScaler().fit_transform(
                    np.array(
                        df_train_itergran[
                            ModelParameters.zero_day_sold_order_level_perc_col
                        ]
                    ).reshape(-1, 1)
                ),
                index=df_train_itergran.index,
            )
        else:
            continue

        df_train_itergran_index = df_train_itergran.reset_index()["index"]

        # Flatten the 0day_sold_order_level% columns for the clustering
        # algorithm.
        df_train_itergran_clus_uscl = np.array(
            df_train_itergran[ModelParameters.zero_day_sold_order_level_perc_col]
        ).reshape(-1, 1)

        df_train_itergran_clus_scl = np.array(
            df_train_itergran[ModelParameters.zero_day_sold_order_level_perc_scaled_col]
        ).reshape(-1, 1)

        # Cluster the 0day_sold_order_level% using DBSCAN
        if df_train_itergran.shape[0] > 2:
            df_train_itergran_clus_scl_flat = df_train_itergran_clus_scl.reshape(
                1, len(df_train_itergran_clus_scl)
            )

            df_train_itergran_clus_scl_flat = np.sort(
                df_train_itergran_clus_scl_flat)

            df_train_itergran_clus_scl_flat = np.diff(
                df_train_itergran_clus_scl_flat)

            histo = np.histogram(df_train_itergran_clus_scl_flat, bins=50)
            cumdens = np.cumsum(histo[0] / histo[0].sum())

            # Create >= 5 clusters by setting the number of next neigbour
            # distance above threshold to be 4.
            cutoff = np.where(cumdens <= max(min(cumdens), (1 - 4 / histo[0].sum())))[
                0
            ][-1]
            eps = histo[1][cutoff + 1]
            db_inst = DBSCAN(eps, min_samples=1)
            clus_cols_train = db_inst.fit_predict(df_train_itergran_clus_scl)
        else:
            continue

        # Add the result of the clustering to the initial dataframe
        df_train_itergran_clus = pd.DataFrame(
            clus_cols_train,
            columns=[ModelParameters.zero_day_sold_order_level_perc_cluster_col],
            index=df_train_itergran_index,
        )

        df_train_itergran[
            ModelParameters.zero_day_sold_order_level_cluster_col
        ] = df_train_itergran_clus[
            ModelParameters.zero_day_sold_order_level_perc_cluster_col
        ]

        # Rename clusters
        df_train_itergran[ModelParameters.zero_day_sold_order_level_cluster_col] = (
            df_train_itergran[
                ModelParameters.zero_day_sold_order_level_cluster_col
            ].astype(str)
            + combi[0]
            + "-"
            + str(combi[1])
        )

        # Appends the result coresponding to a particular combination of
        # granularity values to the master list.
        df_list = pd.concat([df_list, df_train_itergran])

    df_all = pd.concat([df_list, df_test])

    # Save the data
    if not df_all.empty:
        df_all = df_all.reset_index()
        load_df_to_bigquery(df_all, tables["glide_path_zero_day_cluster"], if_exists = if_bq_exists)

    logger.info("zero day level clustering completed.")

    return df_all

def classify_cluster_to_csv_shape(granularity, step, validation, if_bq_exists, market, **kwargs):
    """
    **Determine the shape component or the 0day_sold_order_level% [step], i.e. the band and the centre, for each test/prediction case\
    for a particular granularity and country, as indicated in** *pfile*.

    .. note::
        For 0day_sold_order_level%, if Level3 != 'USA' and there are >=5 data points having month similar to that of the test/prediction case,\
        use those points, instead of clusters, to predict.
        Saves the original test/prediction data together with the results of classification as \
        *classified folders & folder*/test_cases/*granularity*-*pfile*\_WithBand\-*step*.csv'.\
        Saves the info on the relevance of clusters to the test (identified by index)/prediction cases as\
        *classified folders & folder*/clus_info/*granularity*-*pfile*\-*step*.csv'.\

    :param pfile: Path to the file containing the glide paths for a particular Level3 (e.g. '../pickle_vault/pre_orders_curves/CPDD_United Kingdom-UK NSC.pkl')
    :param granularity: (e.g. ['SEGMENT', 'months_before_handover'])
    :param folder: Use either the 'validation' folder (when *validation* = True) or the 'actual' folder (when *validation* = False) in the 'clustered_folder'.
    :param validation: Choose whether to validate model (i.e. testing) (when True) or to do predictions (for future month-years) (when False).
    :param clustered_folders: Location of the folder where the clustered results are saved (e.g. '../output_vault/dwt_cluster/clustered/').
    :param classified_folders: Location of the folder where the classification results will be saved (e.g. '../output_vault/dwt_cluster/classified/').
    :param step: Solving for 'shape' or '0DayLevel'.
    :param prediction_file_path: Path to the file containing the prediction cases, the one whose Level3 is altered to the new Level3 for country aggregation (e.g. '../input_vault/csv/prediction_cases.csv').

    :type pfile: str.
    :type granularity: List
    :type folder: str.
    :type validation: Bool.
    :type clustered_folders: str.
    :type classified_folders: str.
    :type step: str.
    :type prediction_file_path: str.

    :returns: None
    """

    ### Comment: print('{} classification for '.format(step) + pfile)

    # Load results of clustering.
    g_str = ''.join(granularity)
    table_name = tables["glide_path_shape_cluster"]
    df_all = get_data_by_market(table_name, market)
    if df_all.shape[0]==0:
        return
    Level3 = market
    Level5 = df_all[ModelParameters.level_5_reporting_col][0]


    df_train = df_all[df_all[ModelParameters.is_train_col]]
    # Get the test set (validation = True) or the prediction set (validation = False)
    if validation:
        df_test = df_all[~df_all[ModelParameters.is_train_col]]
    else:
        df_test = get_data_by_market(tables["prediction_case"], market)

    # Each clustering techniques has its associated column showing the result of the clustering. The list here
    # log the name of these columns.

    i_cols = pre_order_day_col.copy()
    clusters_type = ['clus_pointwise_KMeans']

    # Loop through the different rows in the test/prediction set and try to determine the corresponding median and confidence band.
    band_lists = []
    clus_info = []

    if df_train.shape[0]==0:
        return

    for irow, row in df_test.iterrows():

        print(f'working on shape {market} irow {irow}')
        df_train_sub = df_train.copy(deep=True)

        # Filter the train set to the granularity values of the test/prediction row in question.
        for _, g in enumerate(granularity):
            df_train_sub = df_train_sub[df_train_sub[g] == row[g]]

        if df_train_sub.shape[0] > 0:
            # Loop through different clustering approach.
            for ct in clusters_type:
                # Filter the train set and the test/prediction row of data for columns only relevant to the classification step.

                class_attr = list(class_attr_1)
                class_attr.append(ct)
                df_train_attr = df_train_sub[class_attr]
                row_attr = row[[ii for ii in class_attr if ii != ct]].to_frame().T

                # Determine the relevance of the test/prediction case to each cluster.
                df_train_df = find_similarity_of_test_to_clus(cluster_col=ct, row_attr=row_attr,
                                                              df_train_attr=df_train_attr, centre=3, steepness=1.3)
                # Legacy. Merge to create df_train_sub that will be consistent with the needed input for the summarise_clus_dist function to be used later.
                df_train_sub = pd.merge(df_train_sub, df_train_df[[ct, ModelParameters.distance_col]], how='left', left_on=ct,
                                        right_on=ct)



                # having months_before_handover & CPDD month similar to those of the test/prediction case.


                # Legacy. Gives the 'probability' for each train row/glide path.
                clus_dist = summarise_clus_dist(df_train_sub, ct)
                if clus_dist.shape[0] > 0:
                    band_name = 'cumulative_distribution'
                    df_train_sub = pd.merge(df_train_sub, clus_dist[['row_probability']], how='left',
                                            left_on=ct, right_index=True)
                    interp_data = {}
                    # Get the band and the centre of band.
                    for day in i_cols:
                        interp_data[day] = err_quant(df_train_sub, day)

                    # Switch the the band and the centre of band info from dict type to dataframe type.
                    df = pd.DataFrame.from_dict(interp_data)
                    df[ ModelParameters.confidence_band_col] = pd.Series(['lower', 'mean', 'upper'])
                    df = df.set_index( ModelParameters.confidence_band_col)
                    # Include more info. Iterate through lower, mean and then upper.
                    for row_quant in df.index:
                        band_l = band_list_extend(df.loc[row_quant, :].tolist(), row_quant, ct, band_name,
                                                  row, band_attr)
                        # Export to master list for band.
                        band_lists.append(band_l)

                    # Prepare for output the info on the relevance of each cluster to the test (identified by index)/prediction case.
                    clus_dist_new = clus_dist.copy(deep=True)
                    clus_dist_new.index = clus_dist_new.index.rename('CLUSTER')
                    clus_dist_new = clus_dist_new.reset_index()
                    clus_dist_new['INDEX'] = row.name
                    clus_dist_new['LEVEL5'] = Level5
                    clus_dist_new['LEVEL3'] = Level3
                    clus_dist_new['GRANULARITY'] = g_str
                    clus_dist_new['CLUSTER_TYPE'] = ct
                    # Export to master list.
                    clus_info.append(clus_dist_new)


    band_df_cols = i_cols
    band_df_cols.extend([ ModelParameters.confidence_band_col, ModelParameters.cluster_type_col, ModelParameters.weighted_col, ModelParameters.index_col])
    band_df_cols.extend(band_attr)
    df_band = pd.DataFrame(band_lists, columns=band_df_cols)

    # For shape step, pivot the classification data and test data (or prediction data, if validation False, even though unnecessary).

    df_band_melt = melt_for_classify_cluster(df_band)

    df_test = df_test.drop(clusters_type, axis=1, errors='ignore')
    df_test = df_test.reset_index()
    df_test_melt = melt_for_classify_cluster(df_test)


    df_test_melt[ ModelParameters.confidence_band_col] = 'actual'
    # Renaming columns to be consistent with validation runs.
    if not validation:
        df_test_melt = df_test_melt.rename(columns={'LEVEL_3_REPORTING': 'LEVEL3', 'index': 'INDEX'})

    clus_info = pd.concat(clus_info)

    # Save the info on the relevance of clusters to the test (identified by index)/prediction cases

    info_tab = tables["classified_shape_cluster_info"]
    test_case_tab = tables["classified_shape_cluster_info_test_cases"]

    if not clus_info.empty:
        load_df_to_bigquery(clus_info, info_tab, if_exists=if_bq_exists)

    # Save the original test/prediction data together with the results of classification.
    if not df_band_melt.empty:
        df_band_melt['CHD_MONTH'] = pd.to_datetime(df_band_melt.CHD_REPORTING_DATE).dt.month
        df_band_melt['CHD_YEAR']= pd.to_datetime(df_band_melt.CHD_REPORTING_DATE).dt.year
        # df_band_melt['CPDD_MONTH'] = pd.to_datetime(df_band_melt.CPDD_REPORTING_DATE).dt.month
        # df_band_melt['CPDD_YEAR'] = pd.to_datetime(df_band_melt.CPDD_REPORTING_DATE).dt.year
        df_band_melt['LEVEL_3_REPORTING'] = market
        df_band_melt['LEVEL_5_REPORTING'] = Level5
        load_df_to_bigquery(df_band_melt, test_case_tab, if_exists = if_bq_exists)


def classify_cluster_to_csv_0day_level(granularity, step, validation, if_bq_exists, market, **kwargs):
    """
    **Determine the shape component or the 0day_sold_order_level% [step], i.e. the band and the centre, for each test/prediction case\
    for a particular granularity and country, as indicated in** *pfile*.

    .. note::
        For 0day_sold_order_level%, if Level3 != 'USA' and there are >=5 data points having month similar to that of the test/prediction case,\
        use those points, instead of clusters, to predict.
        Saves the original test/prediction data together with the results of classification as \
        *classified folders & folder*/test_cases/*granularity*-*pfile*\_WithBand\-*step*.csv'.\
        Saves the info on the relevance of clusters to the test (identified by index)/prediction cases as\
        *classified folders & folder*/clus_info/*granularity*-*pfile*\-*step*.csv'.\

    :param pfile: Path to the file containing the glide paths for a particular Level3 (e.g. '../pickle_vault/pre_orders_curves/CPDD_United Kingdom-UK NSC.pkl')
    :param granularity: (e.g. ['SEGMENT', 'months_before_handover'])
    :param folder: Use either the 'validation' folder (when *validation* = True) or the 'actual' folder (when *validation* = False) in the 'clustered_folder'.
    :param validation: Choose whether to validate model (i.e. testing) (when True) or to do predictions (for future month-years) (when False).
    :param clustered_folders: Location of the folder where the clustered results are saved (e.g. '../output_vault/dwt_cluster/clustered/').
    :param classified_folders: Location of the folder where the classification results will be saved (e.g. '../output_vault/dwt_cluster/classified/').
    :param step: Solving for 'shape' or '0DayLevel'.
    :param prediction_file_path: Path to the file containing the prediction cases, the one whose Level3 is altered to the new Level3 for country aggregation (e.g. '../input_vault/csv/prediction_cases.csv').

    :type pfile: str.
    :type granularity: List
    :type folder: str.
    :type validation: Bool.
    :type clustered_folders: str.
    :type classified_folders: str.
    :type step: str.
    :type prediction_file_path: str.

    :returns: None
    """

    ### Comment: print('{} classification for '.format(step) + pfile)

    # Load results of clustering.
    g_str = ''.join(granularity)
    table_name = tables["glide_path_zero_day_cluster"]
    df_all = get_data_by_market(table_name, market)

    if df_all.shape[0]==0:
        return

    Level5 = df_all[ModelParameters.level_3_reporting_col][0]
    Level3 = df_all[ModelParameters.level_5_reporting_col][0]

    # For classifying shape, change column name type to integer for the pre_order_days columns.


    df_train = df_all[df_all[ModelParameters.is_train_col]]
    # Get the test set (validation = True) or the prediction set (validation = False)
    if validation:
        df_test = df_all[~df_all[ModelParameters.is_train_col]]
    else:
        df_test = get_data_by_market(tables["prediction_case"], market)


    # Each clustering techniques has its associated column showing the result of the clustering. The list here
    # log the name of these columns.

    i_cols = ['_0_DAY_SOLD_ORDER_LEVEL']
    clusters_type = ['_0_DAY_SOLD_ORDER_LEVEL_CLUSTER']

    # Loop through the different rows in the test/prediction set and try to determine the corresponding median and confidence band.
    band_lists = []
    clus_info = []

    if df_train.shape[0]==0:
        return
    for irow, row in df_test.iterrows():
        print(f'working on 0-day-level-{market} irow {irow}')
        df_train_sub = df_train.copy(deep=True)

        # Filter the train set to the granularity values of the test/prediction row in question.
        for _, g in enumerate(granularity):
            df_train_sub = df_train_sub[df_train_sub[g] == row[g]]
        if df_train_sub.shape[0] == 0:
            continue

        if df_train_sub.shape[0] > 0:
            # Loop through different clustering approach.
            for ct in clusters_type:
                # Filter the train set and the test/prediction row of data for columns only relevant to the classification step.

                class_attr = list(class_attr_1)
                class_attr.append(ct)
                df_train_attr = df_train_sub[class_attr]
                row_attr = row[[ii for ii in class_attr if ii != ct]].to_frame().T

                # Determine the relevance of the test/prediction case to each cluster.
                df_train_df = find_similarity_of_test_to_clus(cluster_col=ct, row_attr=row_attr,
                                                              df_train_attr=df_train_attr, centre=3, steepness=1.3)
                # Legacy. Merge to create df_train_sub that will be consistent with the needed input for the summarise_clus_dist function to be used later.
                df_train_sub = pd.merge(df_train_sub, df_train_df[[ct, ModelParameters.distance_col]], how='left', left_on=ct,
                                        right_on=ct)

                # For 0DayLevel step, create a dataframe keeping only the 0day_sold_order_level% data points (or glide paths)
                # having months_before_handover & CPDD month similar to those of the test/prediction case.

                df_train_sub_relmonth = df_train_sub[
                    (df_train_sub['MONTHS_BEFORE_HANDOVER'] == row['MONTHS_BEFORE_HANDOVER']) & (
                            df_train_sub['CPDD_MONTH'] == row['CPDD_MONTH'])]


                # Legacy. Gives the 'probability' for each train row/glide path.
                clus_dist = summarise_clus_dist(df_train_sub, ct)
                if clus_dist.shape[0] > 0:
                    # For 0DayLevel step, allow the possibility of aggregating data points of similar months.
                    if df_train_sub_relmonth.shape[0] >= 5 and Level3 != 'USA':
                        ct = ct.replace('CLUSTER', 'SIMIlARMONTHS')
                        band_name = 'SIMILAR_MONTHS'
                        interp_data = {}
                        # Get the band and the centre of band.
                        interp_data['_0_DAY_SOLD_ORDER_LEVEL'] = df_train_sub_relmonth[
                            '_0_DAY_SOLD_ORDER_LEVEL'].quantile(
                            [.2, .5, 0.8]).tolist()
                    # Append 'probability' for each train row/glide path to the corresponding row.
                    else:
                        band_name = 'cumulative_distribution'
                        df_train_sub = pd.merge(df_train_sub, clus_dist[['row_probability']], how='left',
                                                left_on=ct, right_index=True)
                        interp_data = {}
                        # Get the band and the centre of band.
                        for day in i_cols:
                            interp_data[day] = err_quant(df_train_sub, day)

                    # Switch the the band and the centre of band info from dict type to dataframe type.
                    df = pd.DataFrame.from_dict(interp_data)
                    df['CONFIDENCE_BAND'] = pd.Series(['lower', 'mean', 'upper'])
                    df = df.set_index('CONFIDENCE_BAND')
                    # Include more info. Iterate through lower, mean and then upper.
                    for row_quant in df.index:
                        band_l = band_list_extend(df.loc[row_quant, :].tolist(), row_quant, ct, band_name,
                                                  row, band_attr)
                        # Export to master list for band.
                        band_lists.append(band_l)

                    # Prepare for output the info on the relevance of each cluster to the test (identified by index)/prediction case.
                    clus_dist_new = clus_dist.copy(deep=True)
                    clus_dist_new.index = clus_dist_new.index.rename('CLUSTER')
                    clus_dist_new = clus_dist_new.reset_index()
                    clus_dist_new['INDEX'] = row.name
                    clus_dist_new['LEVEL5'] = Level5
                    clus_dist_new['LEVEL3'] = Level3
                    clus_dist_new['GRANULARITY'] = g_str
                    clus_dist_new['CLUSTER_TYPE'] = ct
                    # Export to master list.
                    clus_info.append(clus_dist_new)

    # Convert master list for band to dataframe.

    i_cols = ['_0_DAY_SOLD_ORDER_LEVEL_PERC_BAND']
    band_df_cols = i_cols
    band_df_cols.extend([ ModelParameters.confidence_band_col, ModelParameters.cluster_type_col, ModelParameters.weighted_col, ModelParameters.index_col])
    band_df_cols.extend(band_attr)
    df_band = pd.DataFrame(band_lists, columns=band_df_cols)

    # For shape step, pivot the classification data and test data (or prediction data, if validation False, even though unnecessary).
    # For 0DayLevel step, just do some tidying.

    df_band_melt = df_band.drop('_0_DAY_SOLD_ORDER_LEVEL', axis=1, errors='ignore')

    clusters_type = [i for i in df_train.columns.values.tolist() if
                     not isinstance(i, int) and i.split('-')[0] == 'clus']
    clusters_type.extend(
        ['_0_DAY_SOLD_ORDER_LEVEL', '_0_DAY_SOLD_ORDER_LEVEL_CLUSTER', 'CHD_MONTH', 'CHD_YEAR', 'CPDD_MONTH',
         'CPDD_YEAR',
         '_0_DAY_SOLD_ORDER_LEVEL_PERC_SCALED'])
    if validation:
        df_test['_0_DAY_SOLD_ORDER_LEVEL_PERC_BAND'] = df_test['_0_DAY_SOLD_ORDER_LEVEL']
    df_test = df_test.drop(clusters_type, axis=1, errors='ignore')
    df_test = df_test.reset_index()
    df_test_melt = df_test

    df_test_melt['CONFIDENCE_BAND'] = 'actual'
    # Renaming columns to be consistent with validation runs.
    if not validation:
        df_test_melt = df_test_melt.rename(columns={'LEVEL_3_REPORTING': 'LEVEL3', 'index': 'INDEX'})

    clus_info = pd.concat(clus_info)

    info_tab = tables["classified_zero_day_cluster_info"]
    test_case_tab = tables["classified_zero_day_cluster_info_test_cases"]

    # Save the info on the relevance of clusters to the test (identified by index)/prediction cases
    if not clus_info.empty:
        load_df_to_bigquery(clus_info, info_tab, if_exists=if_bq_exists)
    # Save the original test/prediction data together with the results of classification.
    if not df_band_melt.empty:
        df_band_melt[ModelParameters.CHD_month_col] = pd.to_datetime(df_band_melt.CHD_REPORTING_DATE).dt.month
        df_band_melt[ModelParameters.CHD_year_col]= pd.to_datetime(df_band_melt.CHD_REPORTING_DATE).dt.year
        # df_band_melt[ModelParameters.CPDD_month_col] = pd.to_datetime(df_band_melt.CPDD_REPORTING_DATE).dt.month
        # df_band_melt[ModelParameters.CPDD_year_col] = pd.to_datetime(df_band_melt.CPDD_REPORTING_DATE).dt.year
        df_band_melt[ModelParameters.level_3_col] = market
        df_band_melt[ModelParameters.level_5_col] = Level5
        load_df_to_bigquery(df_band_melt, test_case_tab, if_exists = if_bq_exists)


def loop_validation_granularity(validation, granularities, markets, if_bq_exists):
    """
    Calls -
    :func:`make_cluster_to_csv`,
    :func:`make_cluster_to_csv_0daylevel`,
    :func:`classify_cluster_to_csv` **(shape & 0DayLevel)** and
    **for each option in the granularity list and each option in the
        validation list for a particular country.**

    :param validation: Choose whether to validate model (i.e. testing) (when True)
        or to do predictions (for future month-years) (for False).
    :param granularities: List of granularity options (e.g. [['SEGMENT', 'months_before_handover'], [...]]).
    :param glide_path_dataframe: This is the 400_Glide_Path Table generated from ETL Processes preceeding this
        script.

    :type validation: Bool.
    :type granularities: List of list
    :type glide_path_dataframe: pandas.Dataframe.

    :returns: None
    """

    # df_glide_path_country = get_data_by_market(tables["glide_path_table"], markets)

    for granularity in granularities:
        int_arg = {
            "granularity": granularity,
            "validation": validation,
            "if_bq_exists": if_bq_exists,
            "market": markets
        }

        # Clustering

        print(markets)

        logger.info("started shape clustering.")
        print('started shape clustering')
        make_cluster_to_csv(**int_arg)



        # Classification
        logger.info("started shape cluster classification.")
        print("started shape cluster classification.")
        classify_cluster_to_csv_shape(step="shape", **int_arg)




def get_markets():
    sql = """SELECT distinct LEVEL_3_REPORTING FROM `{}.{}.{}`""".format(gcp["project"],gcp["dataset"], tables["glide_path_table"])
    return bq_query_to_dataframe(sql)


def get_data_by_market(table,market):
    sql = """SELECT * FROM `{}.{}.{}` where LEVEL_3_REPORTING = '{}'""".format(gcp["project"],gcp["dataset"], table, market)
    return bq_query_to_dataframe(sql)


def main():
    """
        Run clustering and classification functions.
    """
    print("Running...")
    if_bq_exists = 'replace'

    markets = get_markets().iloc[:,0].tolist()
    markets = [m for m in markets if m not in ['Taiwan', 'CJLR']]
    # for m in markets:
    for m in ['LACRO', 'Turkey', 'Singapore', 'Sub Sahara Africa']:
        print('running market {}'.format(m))
        loop_validation_granularity(
            validation=False,
            granularities=main_config["granularities"],
            markets=m,
            if_bq_exists = if_bq_exists
        )
        if_bq_exists = 'append'

if __name__ == "__main__":
    import time

    start = time.time()

    main()
    finish = time.time()

    print(f"time taken:{finish - start}")
