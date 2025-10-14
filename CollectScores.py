# from CausalLearn.causallearn.utils.cit import *
from causallearn.utils.cit import *
# from causallearn.search.ConstraintBased.PC import pc
# from causallearn.utils.GraphUtils import GraphUtils

from causallearn.score.LocalScoreFunction import (
    local_score_BDeu,
    local_score_BIC,
    local_score_BIC_from_cov,
    local_score_cv_general,
    local_score_cv_multi,
    local_score_marginal_general,
    local_score_marginal_multi,
)
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics.pairwise import paired_distances


import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset, linear_lm

from npeet import entropy_estimators as ee

from DegenerateGaussianScore import DegenerateGaussianScore
import Utils as Utils
from causallearn.search.FCMBased.ANM.ANM import ANM
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.PNL.PNL import PNL

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import multiprocessing
import time
from datetime import timedelta
import argparse


# ### As for all scores in Tetrad, higher scores mean more dependence, and negative scores indicate independence.


def get_local_score_BIC(covariance_matrix,num_of_instances, X_idx, Y_idx, parameters):
    
    # Calculate the *negative* local score with BIC for the linear Gaussian continue data case

    # Parameters
    # ----------
    # Data: ndarray, (sample, features)
    # i: current index
    # PAi: parent indexes
    # parameters: lambda_value, the penalty discount of bic

    # Returns
    # -------
    # score: local BIC score
    # local_score_BIC(Data: ndarray, i: int, PAi: List[int], parameters=None)
    # parameters = {
    #     "lambda_value": 10,  # 10 fold cross validation
    #     "lambda": 0.01,
    # }  # regularization parameter
    #result = local_score_BIC(data, X_idx, [Y_idx], parameters=parameters);
    result = local_score_BIC_from_cov((covariance_matrix,num_of_instances), X_idx, [Y_idx], parameters=parameters);
    return result.item((0, 0)) ;

def get_local_score_cv_general(data, X_idx, Y_idx, parameters):
    
    """
    Calculate the local score
    using negative k-fold cross-validated log likelihood as the score
    based on a regression model in RKHS

    Parameters
    ----------
    Data: (sample, features)
    Xi: current index
    PAi: parent indexes
    parameters:
                   kfold: k-fold cross validation
                   lambda: regularization parameter

    Returns
    -------
    score: local score
    """
    # local_score_cv_general(Data: ndarray, Xi: int, PAi: List[int], parameters: Dict[str, Any])
    result = local_score_cv_general(data, X_idx, [Y_idx], parameters=parameters);
    return result;

def get_local_score_BDeu(data, X_idx, Y_idx, parameters):
    """
    Calculate the *negative* local score with BDeu for the discrete case

    Parameters
    ----------
    Data: (sample, features)
    i: current index
    PAi: parent indexes
    parameters:
             sample_prior: sample prior
             structure_prior: structure prior
             r_i_map: number of states of the finite random variable X_{i}

    Returns
    -------
    score: local BDeu score
    """
    # local_score_BDeu(Data: ndarray, i: int, PAi: List[int], parameters=None)
    
    result = local_score_BDeu(data, X_idx, [Y_idx], parameters=parameters);
    return result;


def collect_test_scores(data, var1, var2, varList, fisherz_indep_test=None, mv_fisherz_indep_test=None
                       ,chisq_indep_test=None,kci_indep_test=None,covariance_matrix=None,num_of_instances=0
                       ,degenerate_gaussian=None, temp_index_data_type_dict=None, anm=None,VARLiNGAM=None
                       ,pvalue_threshold = 0.05, pvalue_filter_threshold = 0.07, nonlinearity_pvalue_threshold = 0.05):
    #var1 is X and var2 is Y
    var1_var2_score_test_dict = {};
    
    #var2 is X and var1 is Y
    var2_var1_score_test_dict = {};
    
    condition_set = list(set(varList) - {var1, var2});
    
    #--------------------------------------------------------------------
    
    indep_test = "fisherz";
    
    # X, Y and condition_set : column indices of data
    fisherz_pvalue = fisherz_indep_test(var1,var2,condition_set);
    
    var1_var2_score_test_dict[indep_test] = fisherz_pvalue;
    var2_var1_score_test_dict[indep_test] = fisherz_pvalue;
    
    
    #Add another information, the chosen direction. 1- is not determine, 1 is cause, and 0 is effect
    fisherz_score_name = indep_test+"_binary";
    var1_var2_score_test_dict[fisherz_score_name] = -1;
    var2_var1_score_test_dict[fisherz_score_name] = -1;
    
    if fisherz_pvalue>pvalue_threshold:
        var1_var2_score_test_dict[fisherz_score_name] = 0;
        var2_var1_score_test_dict[fisherz_score_name] = 0;
    else:
        var1_var2_score_test_dict[fisherz_score_name] = 1;
        var2_var1_score_test_dict[fisherz_score_name] = 1;
    
    # fisherz_pvalue = get_indep_test(data, var1, var2, condition_set, indep_test);
    #TODO is this invoke neccessary??? Does switching orders for var1 and var2 generate a different score? TOTOTOTOTOTOTOTOTOTOTOTOTTOTOTOTOTOTTOTOTO
    # fisherz_pvalue2 = fisherz_indep_test(var2, var1, condition_set);
    # if fisherz_pvalue2>pvalue_threshold:
    #     var2_var1_score_test_dict[fisherz_score_name] = 0;
    # else:
    #     var2_var1_score_test_dict[fisherz_score_name] = 1;
    
    #--------------------------------------------------------------------
    
    #print(f'test_stat_old: {test_stat_old}, p_old: {p_old}')

    indep_test = "hsic_test_gamma";
    #stat,hisc_pvalue = get_hsic_test_gamma(data, var1, var2);
    stat,hisc_pvalue = hsic_test_gamma(data[:,var1], data[:,var2]);
    var1_var2_score_test_dict[indep_test] = hisc_pvalue;
    var1_var2_score_test_dict[indep_test+"_stat"] = stat;
    
    
    stat,hisc_pvalue2 = hsic_test_gamma(data[:,var2], data[:,var1]);
    var2_var1_score_test_dict[indep_test] = hisc_pvalue2;
    var2_var1_score_test_dict[indep_test+"_stat"] = stat;
    
    #Add another information, the chosen direction. 1- is not determine, 1 is cause, and 0 is effect
    hsic_score_name = indep_test+"_binary";
    var1_var2_score_test_dict[hsic_score_name] = -1;
    var2_var1_score_test_dict[hsic_score_name] = -1;
    
    if hisc_pvalue>pvalue_threshold:
        var1_var2_score_test_dict[hsic_score_name] = 0;
        #var2_var1_score_test_dict[hsic_score_name] = 0;
    else:
        var1_var2_score_test_dict[hsic_score_name] = 1;
        #var2_var1_score_test_dict[hsic_score_name] = 1;
        
    if hisc_pvalue2>pvalue_threshold:
        var2_var1_score_test_dict[hsic_score_name] = 0;
    else:
        var2_var1_score_test_dict[hsic_score_name] = 1;
    
    #--------------------------------------------------------------------
    #KCI always returns 0 and mv_fisherz always returns the same pvalue with fisherz, so use fisherz and hisc.
    #Use the interaction if one of the fisherz and hisc returns a smaller pvalue than the defined threshold.
    
    #and kci_pvalue> pvalue_filter_threshold and mv_fisherz_pvalue>pvalue_filter_threshold
    if fisherz_pvalue > pvalue_filter_threshold and hisc_pvalue>pvalue_filter_threshold and hisc_pvalue2>pvalue_filter_threshold:
        print(str(fisherz_pvalue)+", "+str(hisc_pvalue)+", "+str(hisc_pvalue2)+". Fail to pass the independence tests.");
        return None;
    #--------------------------------------------------------------------
    #     indep_test = "chisq";
#     print(condition_set);
#     chisq_pvalue = chisq_indep_test(var1, var2, condition_set);
#     var1_var2_score_test_dict[indep_test] = chisq_pvalue;
    
#     chisq_pvalue = chisq_indep_test(var2, var1, condition_set);
#     var2_var1_score_test_dict[indep_test] = chisq_pvalue;
    
    #condition_set = list(set(varList) - {var1, var2});
    #condition_set = list(set(varList) - {var1, var2});
    #--------------------------------------------------------------------
    
    
    #KCI is also super slow and always returns 0..............................................
    
    indep_test = "kci";
    
    kci_pvalue1 = kci_indep_test(var1, var2, condition_set);
    var1_var2_score_test_dict[indep_test] = kci_pvalue1;
    
    var2_var1_score_test_dict[indep_test] = kci_pvalue1;
    
    #Add another information, the chosen direction. 1- is not determine, 1 is cause, and 0 is effect
    kci_score_name = indep_test+"_binary";
    var1_var2_score_test_dict[kci_score_name] = -1;
    var2_var1_score_test_dict[kci_score_name] = -1;
    
    if kci_pvalue1>pvalue_threshold:
        var1_var2_score_test_dict[kci_score_name] = 0;
        var2_var1_score_test_dict[kci_score_name] = 0;
    else:
        var1_var2_score_test_dict[kci_score_name] = 1;
        var2_var1_score_test_dict[kci_score_name] = 1;
      
    # if kci_pvalue2>pvalue_threshold:
    #     var2_var1_score_test_dict[kci_score_name] = 0;
    # else:
    #     var2_var1_score_test_dict[kci_score_name] = 1;
    # kci_pvalue2 = kci_indep_test(data, var2, var1, condition_set, indep_test);
    # condition_set = list(set(varList) - {var1, var2});
    #condition_set = list(set(varList) - {var1, var2});
    #--------------------------------------------------------------------
    
    
    indep_test = "mv_fisherz";
    
    mv_fisherz_pvalue = mv_fisherz_indep_test(var1,var2,condition_set);
    
    var1_var2_score_test_dict[indep_test] = mv_fisherz_pvalue;
    var2_var1_score_test_dict[indep_test] = mv_fisherz_pvalue;
    
    #Add another information, the chosen direction. 1- is not determine, 1 is cause, and 0 is effect
    mv_fisherz_score_name = indep_test+"_binary";
    var1_var2_score_test_dict[mv_fisherz_score_name] = -1;
    var2_var1_score_test_dict[mv_fisherz_score_name] = -1;
    
    if mv_fisherz_pvalue>pvalue_threshold:
        var1_var2_score_test_dict[mv_fisherz_score_name] = 0;
        var2_var1_score_test_dict[mv_fisherz_score_name] = 0;
    else:
        var1_var2_score_test_dict[mv_fisherz_score_name] = 1;
        var2_var1_score_test_dict[mv_fisherz_score_name] = 1;
    
    #condition_set = list(set(varList) - {var1, var2});
#     mv_fisherz_pvalue = get_indep_test(data, var1, var2, condition_set, indep_test);
#     var1_var2_score_test_dict[indep_test] = mv_fisherz_pvalue;
    
#     condition_set = list(set(varList) - {var1, var2});
#     pvalue = get_indep_test(data, var2, var1, condition_set, indep_test);
#     var2_var1_score_test_dict[indep_test] = pvalue;
    #--------------------------------------------------------------------
    
    
    score_name = "local_score_cv_general";
    parameters = {}
    parameters["kfold"] = 10# 10 fold cross validation
    parameters["lambda"] = 0.01# regularization parameter 
    
    key_score_name = score_name+"_"+str(parameters["kfold"])+"_"+str(parameters["lambda"]);
    
    s1 = get_local_score_cv_general(data, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = s1;
    
    s2 = get_local_score_cv_general(data, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = s2;
    
    parameters["lambda"] = 0.05# regularization parameter 
    
    key_score_name = score_name+"_"+str(parameters["kfold"])+"_"+str(parameters["lambda"]);
    
    result = get_local_score_cv_general(data, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_cv_general(data, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    
    parameters["lambda"] = 0.1# regularization parameter 
    
    key_score_name = score_name+"_"+str(parameters["kfold"])+"_"+str(parameters["lambda"]);
    
    result = get_local_score_cv_general(data, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_cv_general(data, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    
    
    #Add another information, the chosen direction. 1- is not determine, 1 is cause, and 0 is effect
    cv_score_name = score_name+"_binary";
    var1_var2_score_test_dict[cv_score_name] = -1;
    var2_var1_score_test_dict[cv_score_name] = -1;
    
    if s1<s2:
        #print("Cause: "+str(n2_idx)+", Effect: "+str(n1_idx));
        var1_var2_score_test_dict[cv_score_name] = 0;
        var2_var1_score_test_dict[cv_score_name] = 1;
    elif s1>s2:
        #print("Cause: "+str(n1_idx)+", Effect: "+str(n2_idx));
        var1_var2_score_test_dict[cv_score_name] = 1;
        var2_var1_score_test_dict[cv_score_name] = 0;
    
    
    #--------------------------------------------------------------------
    
    
    score_name = "if_cause_is_dicrete";
    #Save if the cause node and effect node are discrete or not
    var1_var2_score_test_dict[score_name] = temp_index_data_type_dict[var1];
    var2_var1_score_test_dict[score_name] = temp_index_data_type_dict[var2];
    
    score_name = "if_effect_is_dicrete";
    var1_var2_score_test_dict[score_name] = temp_index_data_type_dict[var2];
    var2_var1_score_test_dict[score_name] = temp_index_data_type_dict[var1];
    
    #--------------------------------------------------------------------
    score_name = "mutual_info";
    if temp_index_data_type_dict[var2]==1 or temp_index_data_type_dict[var2]==2:
        #If x is discrete
        if temp_index_data_type_dict[var1]==1 or temp_index_data_type_dict[var1]==2:
            # Estimate mutual information for a discrete target variable.
            mir = mutual_info_classif(np.reshape(data[:,var1], (-1, 1)), np.reshape(data[:,var2], (-1, 1)),discrete_features=[0]);
            #print("mutual_info_classif with x being discrete: {}".format(mir));
        else:
            #print("mutual_info_classif: {}".format(mir));
            mir = mutual_info_classif(np.reshape(data[:,var1], (-1, 1)), np.reshape(data[:,var2], (-1, 1)));
    else:
        #If x is discrete
        if temp_index_data_type_dict[var1]==1 or temp_index_data_type_dict[var1]==2:
            #print("mutual_info_regression with x being discrete: {}".format(mir));
            mir = mutual_info_regression(np.reshape(data[:,var1], (-1, 1)), np.reshape(data[:,var2], (-1, 1)),discrete_features=[0]);
        else:
            #print("mutual_info_regression: {}".format(mir));
            mir = mutual_info_regression(np.reshape(data[:,var1], (-1, 1)), np.reshape(data[:,var2], (-1, 1)));
            
    var1_var2_score_test_dict[score_name] = mir[0];
    
    if temp_index_data_type_dict[var1]==1 or temp_index_data_type_dict[var1]==2:
        #If x is discrete
        if temp_index_data_type_dict[var2]==1 or temp_index_data_type_dict[var2]==2:
            # Estimate mutual information for a discrete target variable.
            mir = mutual_info_classif(np.reshape(data[:,var2], (-1, 1)), np.reshape(data[:,var1], (-1, 1)),discrete_features=[0]);
            #print("mutual_info_classif with x being discrete: {}".format(mir));
        else:
            #print("mutual_info_classif: {}".format(mir));
            mir = mutual_info_classif(np.reshape(data[:,var2], (-1, 1)), np.reshape(data[:,var1], (-1, 1)));
    else:
        #If x is discrete
        if temp_index_data_type_dict[var2]==1 or temp_index_data_type_dict[var2]==2:
            #print("mutual_info_regression with x being discrete: {}".format(mir));
            mir = mutual_info_regression(np.reshape(data[:,var2], (-1, 1)), np.reshape(data[:,var1], (-1, 1)),discrete_features=[0]);
        else:
            #print("mutual_info_regression: {}".format(mir));
            mir = mutual_info_regression(np.reshape(data[:,var2], (-1, 1)), np.reshape(data[:,var1], (-1, 1)));
            
    var2_var1_score_test_dict[score_name] = mir[0];
    
    
    
    #--------------------------------------------------------------------
    score_name = "cond_mut_info";
    x = data[:,[var1]].tolist();
    y = data[:,[var2]].tolist();
    z = data[:,condition_set].tolist();
    cond_mi = ee.cmi(x, y, z);
    var1_var2_score_test_dict[score_name] = cond_mi;
    var2_var1_score_test_dict[score_name] = cond_mi;
    
    #--------------------------------------------------------------------
    #TODO should add one binary score???
    score_name = "additive_noise_model";
    p_value_forward, p_value_backward = anm.cause_or_effect(data[:, var1].reshape(-1, 1), data[:, var2].reshape(-1, 1));
    var1_var2_score_test_dict[score_name] = p_value_forward;
    var2_var1_score_test_dict[score_name] = p_value_backward;
    
    
    
    #--------------------------------------------------------------------
    isNonlinear1 = False;
    isNonlinear2 = False;
    
    score_name = "RESET";
    
    
    # regression = sm.OLS(data[:,var2], data[:,var1]);
    # result = regression.fit();
    
    x = data[:,var1];
    y = data[:,var2];

    # Add a constant to the independent variable
    x = sm.add_constant(x);

    # Fit the model
    model_result = sm.OLS(y, x).fit();
    
    linear_lm_test = linear_lm(model_result.resid,exog=model_result.model.exog);
    linear_lm_pvale = linear_lm_test[1];
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=True, test_type='fitted');
    linear_reset_fitted_pvalue = Utils.parse_reset_sum(test_res.summary());
    
    linear_reset_exog_pvalue = 0;
    try:
        test_res = linear_reset(model_result, power = [2,3,4], use_f=True, test_type='exog');
        linear_reset_exog_pvalue = Utils.parse_reset_sum(test_res.summary());
    except ValueError:
        print("Oops!  Model contains only constant or binary data.");
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=True, test_type='princomp');
    linear_reset_pricomp_pvalue = Utils.parse_reset_sum(test_res.summary());
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=False, test_type='fitted');
    linear_reset_chisq_fitted_pvalue = Utils.parse_reset_sum(test_res.summary(),left_str = ", p-value=");
    
    linear_reset_chisq_exog_pvalue = 0;
    try:
        test_res = linear_reset(model_result, power = [2,3,4], use_f=False, test_type='exog');
        linear_reset_chisq_exog_pvalue = Utils.parse_reset_sum(test_res.summary(),left_str = ", p-value=");
    except ValueError:
        print("Oops!  Model contains only constant or binary data.");
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=False, test_type='princomp');
    linear_reset_chisq_precomp_pvalue = Utils.parse_reset_sum(test_res.summary(),left_str = ", p-value=");
    
    if (linear_lm_pvale<nonlinearity_pvalue_threshold and linear_reset_fitted_pvalue<nonlinearity_pvalue_threshold 
        and linear_reset_exog_pvalue<nonlinearity_pvalue_threshold and linear_reset_pricomp_pvalue<nonlinearity_pvalue_threshold 
        and linear_reset_chisq_fitted_pvalue<nonlinearity_pvalue_threshold and linear_reset_chisq_exog_pvalue<nonlinearity_pvalue_threshold 
        and linear_reset_chisq_precomp_pvalue<nonlinearity_pvalue_threshold):
        var1_var2_score_test_dict[score_name+"_is_nonlinear"] = 1;
        
        isNonlinear1 = True;
    else:
        var1_var2_score_test_dict[score_name+"_is_nonlinear"] = 0;
        
    var1_var2_score_test_dict[score_name+"_linear_lm_pval"] = linear_lm_pvale;
    var1_var2_score_test_dict[score_name+"_linear_reset_fitted_pvalue"] = linear_reset_fitted_pvalue;
    var1_var2_score_test_dict[score_name+"_linear_reset_exog_pvalue"] = linear_reset_exog_pvalue;
    var1_var2_score_test_dict[score_name+"_linear_reset_pricomp_pvalue"] = linear_reset_pricomp_pvalue;
    var1_var2_score_test_dict[score_name+"_linear_reset_chisq_fitted_pvalue"] = linear_reset_chisq_fitted_pvalue;
    var1_var2_score_test_dict[score_name+"_linear_reset_chisq_exog_pvalue"] = linear_reset_chisq_exog_pvalue;
    var1_var2_score_test_dict[score_name+"_linear_reset_chisq_precomp_pvalue"] = linear_reset_chisq_precomp_pvalue;
        
    
    x = data[:,var2];
    y = data[:,var1];

    # Add a constant to the independent variable
    x = sm.add_constant(x);

    # Fit the model
    model_result = sm.OLS(y, x).fit();
    
    linear_lm_test = linear_lm(model_result.resid,exog=model_result.model.exog);
    linear_lm_pvale = linear_lm_test[1];
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=True, test_type='fitted');
    linear_reset_fitted_pvalue = Utils.parse_reset_sum(test_res.summary());
    
    linear_reset_exog_pvalue = 0;
    try:
        test_res = linear_reset(model_result, power = [2,3,4], use_f=True, test_type='exog');
        linear_reset_exog_pvalue = Utils.parse_reset_sum(test_res.summary());
    except ValueError:
        print("Oops!  Model contains only constant or binary data.");
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=True, test_type='princomp');
    linear_reset_pricomp_pvalue = Utils.parse_reset_sum(test_res.summary());
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=False, test_type='fitted');
    linear_reset_chisq_fitted_pvalue = Utils.parse_reset_sum(test_res.summary(),left_str = ", p-value=");
    
    linear_reset_chisq_exog_pvalue = 0;
    try:
        test_res = linear_reset(model_result, power = [2,3,4], use_f=False, test_type='exog');
        linear_reset_chisq_exog_pvalue = Utils.parse_reset_sum(test_res.summary(),left_str = ", p-value=");
    except ValueError:
        print("Oops!  Model contains only constant or binary data.");
    
    test_res = linear_reset(model_result, power = [2,3,4], use_f=False, test_type='princomp');
    linear_reset_chisq_precomp_pvalue = Utils.parse_reset_sum(test_res.summary(),left_str = ", p-value=");
    
    if (linear_lm_pvale<nonlinearity_pvalue_threshold and linear_reset_fitted_pvalue<nonlinearity_pvalue_threshold 
        and linear_reset_exog_pvalue<nonlinearity_pvalue_threshold and linear_reset_pricomp_pvalue<nonlinearity_pvalue_threshold 
        and linear_reset_chisq_fitted_pvalue<nonlinearity_pvalue_threshold and linear_reset_chisq_exog_pvalue<nonlinearity_pvalue_threshold 
        and linear_reset_chisq_precomp_pvalue<nonlinearity_pvalue_threshold):
        var2_var1_score_test_dict[score_name+"_is_nonlinear"] = 1;
        isNonlinear2 = True;
    else:
        var2_var1_score_test_dict[score_name+"_is_nonlinear"] = 0;
        
    var2_var1_score_test_dict[score_name+"_linear_lm_pval"] = linear_lm_pvale;
    var2_var1_score_test_dict[score_name+"_linear_reset_fitted_pvalue"] = linear_reset_fitted_pvalue;
    var2_var1_score_test_dict[score_name+"_linear_reset_exog_pvalue"] = linear_reset_exog_pvalue;
    var2_var1_score_test_dict[score_name+"_linear_reset_pricomp_pvalue"] = linear_reset_pricomp_pvalue;
    var2_var1_score_test_dict[score_name+"_linear_reset_chisq_fitted_pvalue"] = linear_reset_chisq_fitted_pvalue;
    var2_var1_score_test_dict[score_name+"_linear_reset_chisq_exog_pvalue"] = linear_reset_chisq_exog_pvalue;
    var2_var1_score_test_dict[score_name+"_linear_reset_chisq_precomp_pvalue"] = linear_reset_chisq_precomp_pvalue;
            
    
    #--------------------------------------------------------------------
    
    score_name = "local_score_BIC";
    parameters = {};
    parameters["lambda_value"] = 10;
    parameters["lambda"] = 0.1;
    key_score_name = score_name+"_"+str(parameters["lambda"]);
    
    result = get_local_score_BIC(covariance_matrix,num_of_instances, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_BIC(covariance_matrix,num_of_instances, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    
    parameters["lambda"] = 0.5
    key_score_name = score_name+"_"+str(parameters["lambda"]);
    result = get_local_score_BIC(covariance_matrix,num_of_instances, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_BIC(covariance_matrix,num_of_instances, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    #TODO delete this duplicate code.................................................................
    parameters["lambda"] = 1
    key_score_name = score_name+"_"+str(parameters["lambda"]);
    s1 = get_local_score_BIC(covariance_matrix,num_of_instances, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = s1;
    
    s2 = get_local_score_BIC(covariance_matrix,num_of_instances, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = s2;
    
    parameters["lambda"] = 1.5
    key_score_name = score_name+"_"+str(parameters["lambda"]);
    result = get_local_score_BIC(covariance_matrix,num_of_instances, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_BIC(covariance_matrix,num_of_instances, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    
    
    #Add another information, the chosen direction. 1- is not determine, 1 is cause, and 0 is effect
    bic_score_name = score_name+"_binary";
    var1_var2_score_test_dict[bic_score_name] = -1;
    var2_var1_score_test_dict[bic_score_name] = -1;
    
    if s1<s2:
        #print("Cause: "+str(n2_idx)+", Effect: "+str(n1_idx));
        #TODO if a relationship is nonlinear the DG and BIC score will be reverse...TODO..............................................................
        if isNonlinear1:
            var1_var2_score_test_dict[bic_score_name] = 1;
        else:
            var1_var2_score_test_dict[bic_score_name] = 0;

        if isNonlinear2:
            var2_var1_score_test_dict[bic_score_name] = 0;
        else:
            var2_var1_score_test_dict[bic_score_name] = 1;
        
    elif s1>s2:
        #print("Cause: "+str(n1_idx)+", Effect: "+str(n2_idx));
        if isNonlinear1:
            var1_var2_score_test_dict[bic_score_name] = 0;
        else:
            var1_var2_score_test_dict[bic_score_name] = 1;

        if isNonlinear2:
            var2_var1_score_test_dict[bic_score_name] = 1;
        else:
            var2_var1_score_test_dict[bic_score_name] = 0;
    
    #--------------------------------------------------------------------

    #both discrete->opposite
    #one discrete and one continuous->normal
    #one continuous and one discrete->opposite
    #both continuous->normal
    
    score_name = "degenerate_score";
    
    s1 = degenerate_gaussian.localScore(var1,{var2});
    s2 = degenerate_gaussian.localScore(var2,{var1});
    
    var1_var2_score_test_dict[score_name] = s1;
    var2_var1_score_test_dict[score_name] = s2;
    
    #Add another information, the chosen direction. 1- is not determine, 1 is cause, and 0 is effect
    dg_score_name = score_name+"_binary";
    var1_var2_score_test_dict[dg_score_name] = -1;
    var2_var1_score_test_dict[dg_score_name] = -1;

    if s1<s2:
        #print("Cause: "+str(n2_idx)+", Effect: "+str(n1_idx));
        if isNonlinear1:
            var1_var2_score_test_dict[dg_score_name] = 1;
        else:
            var1_var2_score_test_dict[dg_score_name] = 0;

        if isNonlinear2:
            var2_var1_score_test_dict[dg_score_name] = 0;
        else:
            var2_var1_score_test_dict[dg_score_name] = 1;
    elif s1>s2:
        #print("Cause: "+str(n1_idx)+", Effect: "+str(n2_idx));
        if isNonlinear1:
            var1_var2_score_test_dict[dg_score_name] = 0;
        else:
            var1_var2_score_test_dict[dg_score_name] = 1;

        if isNonlinear2:
            var2_var1_score_test_dict[dg_score_name] = 1;
        else:
            var2_var1_score_test_dict[dg_score_name] = 0;
        
    #--------------------------------------------------------------------

    # BDeu is slow............................Delete???
    '''
    score_name = "local_score_BDeu";
    parameters = {};
    parameters["sample_prior"] = 0.5;
    parameters["structure_prior"] = 0.5;
    r_i_map = {
        i: len(np.unique(np.asarray(data[:, i]))) for i in range(data.shape[1])
    };
    parameters["r_i_map"] = r_i_map;

    key_score_name = score_name+"_"+str(parameters["sample_prior"])+"_"+str(parameters["structure_prior"]);
    
    result = get_local_score_BDeu(data, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_BDeu(data, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    
    
    parameters["sample_prior"] = 1;
    parameters["structure_prior"] = 1;
    r_i_map = {
        i: len(np.unique(np.asarray(data[:, i]))) for i in range(data.shape[1])
    };
    parameters["r_i_map"] = r_i_map;

    key_score_name = score_name+"_"+str(parameters["sample_prior"])+"_"+str(parameters["structure_prior"]);
    
    result = get_local_score_BDeu(data, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_BDeu(data, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    
    parameters["sample_prior"] = 1.5;
    parameters["structure_prior"] = 1.5;
    r_i_map = {
        i: len(np.unique(np.asarray(data[:, i]))) for i in range(data.shape[1])
    };
    parameters["r_i_map"] = r_i_map;

    key_score_name = score_name+"_"+str(parameters["sample_prior"])+"_"+str(parameters["structure_prior"]);
    
    result = get_local_score_BDeu(data, var1, var2, parameters);
    var1_var2_score_test_dict[key_score_name] = result;
    
    result = get_local_score_BDeu(data, var2, var1, parameters);
    var2_var1_score_test_dict[key_score_name] = result;
    '''
    
    #--------------------------------------------------------------------
    #--------------------------------------------------------------------
    #Super slow...
    '''
    score_name = "PNL";
    p_value_forward_1, p_value_backward_1 = pnl.cause_or_effect(data[:, var1].reshape(-1, 1), data[:, var2].reshape(-1, 1));
    var1_var2_score_test_dict[score_name] = p_value_forward_1;
    var2_var1_score_test_dict[score_name] = p_value_backward_1;
    print("pnl x->y: "+str(p_value_forward_1)+", y->x: "+str(p_value_backward_1));
    '''
    #--------------------------------------------------------------------

    #This score does not have any impact on the final prediction outcome.
    '''
    score_name = "VAR_LiNGAM";
    est_val = VARLiNGAM.estimate_total_effect(data, var1, var2);
    var1_var2_score_test_dict[score_name] = est_val;
    
    est_val = VARLiNGAM.estimate_total_effect(data, var2, var1);
    var2_var1_score_test_dict[score_name] = est_val;
    '''
    #print("VARLiNGAM effect: "+str(est_val));
    
    '''
    #--------------------------------------------------------------------
    score_name = "euclidean";
    dist_val = paired_distances([list(data[var1])], [list(data[var2])], metric=score_name);
    var1_var2_score_test_dict[score_name] = dist_val[0];
    var2_var1_score_test_dict[score_name] = dist_val[0];
    
    #--------------------------------------------------------------------
    score_name = "manhattan";
    # “euclidean”, “manhattan”, or “cosine”.
    dist_val = paired_distances([list(data[var1])], [list(data[var2])], metric=score_name);
    var1_var2_score_test_dict[score_name] = dist_val[0];
    var2_var1_score_test_dict[score_name] = dist_val[0];
    
    #--------------------------------------------------------------------
    score_name = "cosine";
    dist_val = paired_distances([list(data[var1])], [list(data[var2])], metric=score_name);
    var1_var2_score_test_dict[score_name] = dist_val[0];
    var2_var1_score_test_dict[score_name] = dist_val[0];
    
    #--------------------------------------------------------------------
    score_name = "cityblock";
    dist_val = paired_distances([list(data[var1])], [list(data[var2])], metric=score_name);
    var1_var2_score_test_dict[score_name] = dist_val[0];
    var2_var1_score_test_dict[score_name] = dist_val[0];
    
    #--------------------------------------------------------------------
    score_name = "l1";
    dist_val = paired_distances([list(data[var1])], [list(data[var2])], metric=score_name);
    var1_var2_score_test_dict[score_name] = dist_val[0];
    var2_var1_score_test_dict[score_name] = dist_val[0];
    
    #--------------------------------------------------------------------
    score_name = "l2";
    dist_val = paired_distances([list(data[var1])], [list(data[var2])], metric=score_name);
    var1_var2_score_test_dict[score_name] = dist_val[0];
    var2_var1_score_test_dict[score_name] = dist_val[0];
    '''
        
    
    
    return var1_var2_score_test_dict, var2_var1_score_test_dict;


def multi_proc_wrapper(args):
    createdProc = multiprocessing.Process();
    currentProc = multiprocessing.current_process();
    print(createdProc.name+" --- "+currentProc.name+" starts."+" --- The number of relations to process: "+str(len(args)));
    
    #Get graph
    directed_graph = args.pop(len(args)-1);
    
    #Get and remove the data matrix from the list
    dataset = args.pop(len(args)-1);
    data_mat = dataset.to_numpy();
    
    temp_index_data_type_dict = args.pop(len(args)-1);#index_data_type_dict.copy();
    
    #print(len(args));
    #dataset = pd.read_csv("SimulationData/Continuous/testingdata.csv", index_col=0);
    #data_mat = dataset.to_numpy();
    
    var_idx_list = list(range(data_mat.shape[1]));
    
    indep_test = "fisherz";
    fisherz_indep_test = CIT(data_mat, indep_test);
    
    indep_test = "mv_fisherz";
    mv_fisherz_indep_test = CIT(data_mat, indep_test);
    
    indep_test = "chisq";
    chisq_indep_test = CIT(data_mat, indep_test);
    
    indep_test = "kci";
    kci_indep_test = CIT(data_mat, indep_test);
    
    covariance_matrix = np.cov(data_mat.T);
    num_of_instances = data_mat.shape[0];
    
    
    
    # #Initialize DG object
    # #Please set the ordinal discrete variables or the variables that should be handled as continuous variables as continuous variables.
    #both discrete->opposite
    #one discrete and one continuous->normal
    #one continuous and one discrete->opposite
    #both continuous->normal
    continuous_list = [];#"Ferritin_cat","MAS","Death"
    dg = DegenerateGaussianScore(dataset,continuous_list=continuous_list,discrete_threshold=0.05);
    
    anm = ANM();
    
    #pnl = PNL()
    
    VARLiNGAM_model = None;#lingam.VARLiNGAM();
    # VARLiNGAM_model.fit(data_mat);
    
    pvalue_threshold = 0.05;
    pvalue_filter_threshold = 0.07;
    nonlinearity_pvalue_threshold = 0.05;#300, 250
    
    temp_list = [];
    temp_key_list = [];
    
    #print(args);
    for ele in args:
        idx1 = ele[0];
        idx2 = ele[1];
        
        #print(ele[0]);
        score_test_dict = collect_test_scores(data_mat, idx1, idx2, var_idx_list, fisherz_indep_test=fisherz_indep_test
                                              , mv_fisherz_indep_test = mv_fisherz_indep_test, chisq_indep_test=chisq_indep_test
                                              , kci_indep_test=kci_indep_test,covariance_matrix=covariance_matrix,num_of_instances=num_of_instances
                                              , degenerate_gaussian=dg,temp_index_data_type_dict=temp_index_data_type_dict,anm=anm,VARLiNGAM=VARLiNGAM_model
                                              ,pvalue_threshold=pvalue_threshold,pvalue_filter_threshold=pvalue_filter_threshold
                                              ,nonlinearity_pvalue_threshold = 0.05);
        if score_test_dict is not None:
            if directed_graph.number_of_edges()>0:
                if directed_graph.has_edge(idx1, idx2):
                    score_test_dict[0]["Label"] = 1;
                else:
                    score_test_dict[0]["Label"] = 0;

            key = str(idx1)+"_causes_"+str(idx2);
            temp_key_list.append(key);
            temp_list.append(score_test_dict[0]);

            if directed_graph.number_of_edges()>0:
                if directed_graph.has_edge(idx2, idx1):
                    score_test_dict[1]["Label"] = 1;
                else:
                    score_test_dict[1]["Label"] = 0;
                    
            key = str(idx2)+"_causes_"+str(idx1);
            temp_key_list.append(key);
            temp_list.append(score_test_dict[1]);
        else:
            if directed_graph.has_edge(idx1, idx2) or directed_graph.has_edge(idx2, idx1):
                print(str(idx1)+"---"+str(idx2)+"___Incorrect filtering using independence tests..........................................");
                    
    result_df = pd.DataFrame(temp_list,index=temp_key_list);
    # display(result_df);
    #result_df.to_csv("SimulationData/Continuous/testing_data_result_df.csv");
    print("=====================");
    return result_df;



if __name__ == "__main__":
    
    start_time = time.monotonic();
    
    #----------------------------Parameters----------------------------------------------------
    
    #======Simulation training and testing data=======
    # file_path = "SimulationData/Mixed/Test/";
    # input_file = "mixed_sim_data.csv";
    # graph_edge_file = None;#"mixed_sim_graph_edges.csv";
    
    # file_path = "SimulationData/Nonlinear/Training1/";
    # input_file = "nonlinear_sim_data.csv";
    # graph_edge_file = "nonlinear_sim_graph_edges.csv";
    
    # file_path = "SimulationData/Continuous/Training1/";
    # input_file = "continuous_sim_data_testing.csv";
    # graph_edge_file = "continuous_sim_graph_edges_testing.csv";

    
    #=======Real-world training data=======
    # file_path = "RWData/Sepsis/";
    # file_path = "RWData/ChronicKidneyDisease/";
    # file_path = "RWData/SmokerStatusPredictionBio-Signals/";
    # file_path = "RWData/OvarianCancer/";
    # file_path = "RWData/Diabetes/";
    # file_path = "RWData/national-health-and-nutrition-examination-survey/";
    # file_path = "RWData/RNA-Seq-Long-Read Sequencing of an Advanced Cancer Cohort Resolves Rearrangements, Unravels Haplotypes, and Reveals Methylation Landscapes/";
    # file_path = "RWData/scRNA-seq-A Molecular Switch from Tumor Suppressor to Oncogene in ER-Positive Breast Cancer - Role of Androgen Receptor, JAK-STAT, and Lineage Plasticity/";
    # file_path = "RWData/scRNA-seq-Gene expression profile at single cell level of a rare seen and young tracheal adenoid cystic carcinoma/";
    # file_path = "RWData/scRNA-Seq-Spatial Multiomics Reveals Metabolic Reprogramming and Calcification Characteristics of Diabetic Macroangiopathy/";

    
    # input_file = "processed_data.csv";
    # graph_edge_file = "ges_fci_pc_noncausal_output_file.csv";
    
    #========Real-world estimation data========
    # file_path = "RWDataEstimation/PAM50/";
    # input_file = "BRCADataPM50.csv";
    # graph_edge_file = None;
    
    # file_path = "RWDataEstimation/HCV_UCI/";
    # input_file = "processed_data.csv";
    # graph_edge_file = None;
    
    # file_path = "RWDataEstimation/MyocardialUCI/";
    # input_file = "processed_data.csv";
    # graph_edge_file = None;

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="The score collection process.");

    # Argument for the input file path
    parser.add_argument("--file_path","-fp",type=str,default="",help="The path to the input file.");

    # Argument for the input file name
    parser.add_argument("--input_file","-if",type=str,default="",help="The name of the input file.");

    # Argument for the graph file
    parser.add_argument("--graph_file","-gf",type=str,default="",help="The name of the graph file.");

    # The number of processes to start for the task.
    parser.add_argument("--number_of_processes","-nop",type=int,help="The number of processes to start for the task.");

    # Parse the arguments
    args = parser.parse_args();

    file_path = args.file_path;
    input_file = args.input_file;
    
    graph_edge_file = None;
    if args.graph_file:
        graph_edge_file = args.graph_file;
        print(f"Graph file: {graph_edge_file}");
        
    
    full_info = f"{file_path} - {input_file} - {graph_edge_file}";
    print(full_info);
    
    num_of_processes_defined = 6;
    #If the user sets the number of processes to start for the task.
    if args.number_of_processes:
        num_of_processes_defined = args.number_of_processes;
        print(f"Number of processes: {num_of_processes_defined}");
    
    output_file = "score_test_result.csv";
    
    

    #Target feature list
    target_var_list = list();
    # target_var_list = ['PAM50Call_RNAseq'];

    #----------------------------Load and prepare data------------------------------------------
    #dataset, directed_graph 
    dataset, directed_graph, data_type_dict, discrete_list, name_index_mapping, index_name_mapping, index_data_type_dict,relation_list = Utils.read_input_data(file_path=file_path,input_file=input_file,graph_edge_file=graph_edge_file);
    

    #Relations that are sent to subprocesses to collect their scores
    temp_relations_list = [];
    
    if len(target_var_list) == 0 and len(relation_list)>0:
        temp_relations_list = relation_list;
        print("Start with a predefined causal relation list.");
        
    #When there is a target feature list defined, the iteration strategy is different when there is not target features defined.
    elif len(target_var_list)>0:
        print("Start with a list of target variables.");
        var_list = list(dataset.columns);
        for target in target_var_list:
            for var in var_list:
                if var!=target:
                    idx1 = name_index_mapping[target];
                    idx2 = name_index_mapping[var];
                    temp_relations_list.append((idx1,idx2));
                    #print(str(target)+"-"+str(var));
                    
    #There is no target variables. Perform causal discovery among all the variables.                 
    else:
        print("Start with all potential causal relations.");
        var_list_length = dataset.shape[1];#len(var_list)#6;#len(var_list);
        var_idx_list = list(range(var_list_length));#list(range(dataset.shape[1]));
        for idx1 in range(var_list_length):
            for idx2 in range(idx1+1,var_list_length):
                #print("idx1={}, idx2={}".format(idx1,idx2));
                temp_relations_list.append((idx1,idx2));
                
    
    #--------------------------Send data to subprocesses----------------------------------------------
    #print(temp_relations_list);
    list_of_groups = [];
    temp_list_of_groups = list(np.array_split(temp_relations_list, num_of_processes_defined));
    print("The total number of relations: {}".format(len(temp_relations_list)));
    for ele in temp_list_of_groups:
        #print(len(ele));
        temp_ele_list = ele.tolist();
        
        #Send the data type info to the process
        temp_index_data_type_dict = index_data_type_dict.copy();
        temp_ele_list.append(temp_index_data_type_dict);
        
        #Send the dataset to the process
        #data_mat = dataset.to_numpy();
        data_frame = dataset.copy();
        temp_ele_list.append(data_frame);
        
        #Send the graph
        temp_ele_list.append(directed_graph.copy());
        
        list_of_groups.append(temp_ele_list);
        #print(len(temp_ele_list));
    
    
    #https://www.reddit.com/r/Python/comments/6shzvu/when_is_it_ever_a_good_idea_to_use/
    pool = multiprocessing.Pool(num_of_processes_defined);
    result_dfs = pool.map(multi_proc_wrapper, list_of_groups);
    pool.close();
    pool.join();
    merged_df = pd.concat(result_dfs);
    print(merged_df);
    # result_df = pd.DataFrame(temp_list,index=temp_key_list);
    # display(result_df);
    merged_df.to_csv(file_path+output_file);
        
    end_time = time.monotonic();
    print('Time used:')
    print(timedelta(seconds=end_time - start_time));
    