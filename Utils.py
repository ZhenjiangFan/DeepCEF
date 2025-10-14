import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype, is_numeric_dtype

data_type_binary = 1;
data_type_multiclass = 2;
data_type_continuous = 3;
    
def check_variable_type(dataframe):
    """
    Checks if variables in a dataframe are discrete or continuous.

    Args:
        dataframe: The dataframe to check.

    Returns:
        dictionary: {'variable_name': discrete_or_not}
    """
    '''
    #Find the ratio of number of unique values to the total number of unique values. Something like the following
    var_type_dict = {}
    discrete_list = [];
    for var in dataframe.columns:
        var_type_dict[var] = 1.*dataframe[var].nunique()/dataframe[var].count() < 0.1 #or some other threshold
        if var_type_dict[var]:
            discrete_list.append(var);
    return var_type_dict, discrete_list;
    '''
    '''
    Source: https://stackoverflow.com/questions/35826912/what-is-a-good-heuristic-to-detect-if-a-column-in-a-pandas-dataframe-is-categori
    #Check if the top n unique values account for more than a certain proportion of all values
    top_n = 10 
    likely_cat = {}
    for var in dataframe.columns:
        likely_cat[var] = 1.*dataframe[var].value_counts(normalize=True).head(top_n).sum() > 0.8 #or some other threshold
    '''
    '''
    Source; https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/multiclass.py
    '''
    
    var_type_dict = {};
    discrete_list = [];
    for var in dataframe.columns:
        y = dataframe[var].values;
        # * 'continuous': `y` is an array-like of floats that are not all integers, and is 1d or a column vector
        # * 'continuous-multioutput': `y` is a 2d array of floats that are not all integers, and both dimensions are of size > 1.
        # * 'binary': `y` contains <= 2 discrete values and is 1d or a column vector.
        # * 'multiclass': `y` contains more than two discrete values, is not a sequence of sequences, and is 1d or a column vector.
        # * 'multiclass-multioutput': `y` is a 2d array that contains more than two discrete values, is not a sequence of sequences, and both dimensions are of size > 1.
        # * 'multilabel-indicator': `y` is a label indicator matrix, an array of two dimensions with at least two columns, and at most 2 unique values.
        # * 'unknown': `y` is array-like but none of the above, such as a 3d array, sequence of sequences, or an array of non-sequence objects.
        type_str = type_of_target(y);
        data_type = data_type_binary;
        
        if type_str=="binary":
            data_type = data_type_binary;
            discrete_list.append(var);
        if type_str=="multiclass":
            data_type = data_type_multiclass;
            discrete_list.append(var);
        elif type_str=="continuous":
            data_type = data_type_continuous;
        elif type_str=="unknown":
            raise Exception("Unknown data type for "+var+".");
            
        #print(type_str);
        var_type_dict[var] = data_type;
    return var_type_dict, discrete_list;

def string_to_numeric(dataframe):
    # Create a LabelEncoder object
    le = LabelEncoder();
    
    for col in dataframe.columns:

        if not is_numeric_dtype(dataframe[col]):#is_numeric_dtype(dataframe[col]) is_string_dtype(dataframe[col])
            #print(col);
            uniq_vals = dataframe[col].unique();
            # y = dataframe[col].values;
            #type_str = type_of_target(y);
            # if type_str=="binary":
            if len(uniq_vals)==2:
                # Start at 0 if if the variable is binary
                # Fit and transform the target or label staring at 0
                dataframe[col] = le.fit_transform(dataframe[col]);
            else:
                #Start at 1 if the variable is multiclass
                dataframe[col] = pd.factorize(dataframe[col])[0] + 1;
    return dataframe;

    
def read_input_data(file_path=None,input_file=None,graph_edge_file=None):
    dataset = pd.read_csv(file_path+input_file);
    #display(mixed_nonlinear_sim_data);
    
    #Dict that contains info if a variable is discrete or not (discrete if True, countinous elsewise)
    data_type_dict, discrete_list = check_variable_type(dataset);

    #Normalize
    discrete_dataset = dataset[discrete_list];
    scaler = MinMaxScaler();
    scaled_values = scaler.fit_transform(dataset);
    dataset.loc[:,:] = scaled_values;
    #Reset discrete variables
    for discrete_name in discrete_list:
        dataset[discrete_name] = discrete_dataset[discrete_name];
    #display(mixed_sim_dataset);
    
    
    #Map the feature names with indeces, and vice versa.
    #Convert the columns to their numerical representations
    name_index_mapping = {};
    index_name_mapping = {};
    index_data_type_dict = {};
    colList = dataset.columns.tolist();
    for index,name in enumerate(colList):
        name_index_mapping[name] = index;
        index_name_mapping[index] = name;
        index_data_type_dict[index] = data_type_dict[name];
    print(name_index_mapping);
    print(index_name_mapping);
    print(index_data_type_dict);
        
        
    
    directed_graph = nx.DiGraph();
    
    relation_list = [];
    
    if graph_edge_file is not None:
        graph_edges = pd.read_csv(file_path+graph_edge_file);
        #display(graph_edges);
        
        if "Cause" in graph_edges.columns and "Effect" in graph_edges.columns:
            
            for index,row in graph_edges.iterrows():
                cause = row["Cause"];
                effect = row["Effect"];
                label = row["Label"];
                if label==1:
                    directed_graph.add_edge(cause,effect);
                    #print("More columns.");
                    
                cause_idx = name_index_mapping[cause];
                effect_idx = name_index_mapping[effect];
                relation_list.append((cause_idx,effect_idx));
        else:
            sep = " --> ";
            for index,row in graph_edges.iterrows():
                edge = row['Edge'];
                nodes = edge.split(sep);
                cause = nodes[0];
                effect = nodes[1];
                directed_graph.add_edge(cause,effect);
                
    #Change the node names from strings to indeces
    if directed_graph.number_of_edges()>0:
        directed_graph = nx.relabel_nodes(directed_graph,name_index_mapping);
        print(directed_graph.edges());
        
        
    return dataset, directed_graph, data_type_dict, discrete_list, name_index_mapping, index_name_mapping, index_data_type_dict,relation_list;
                
    '''
    directed_graph = nx.DiGraph();
    
    if graph_edge_file is not None:
        mixed_nonlinear_sim_graph_edges = pd.read_csv(file_path+graph_edge_file);
        #display(mixed_nonlinear_sim_graph_edges);
        
        #Merge two graphs
        sep = " --> ";
        for index,row in mixed_nonlinear_sim_graph_edges.iterrows():
            edge = row['Edge'];
            nodes = edge.split(sep);
            cause = nodes[0];
            effect = nodes[1];
            directed_graph.add_edge(cause,effect);
    '''
            
    # return dataset, directed_graph;


def parse_reset_sum(summary,left_str = ", p="):
    #test_res_str = test_res.summary();
    # left_str = ", p=";
    right_str = ", df_";
    pvalue = summary[summary.index(left_str)+len(left_str):summary.index(right_str)];
    return float(pvalue);

def removeCycles(causalGraph):
    """
    Remove cycles in a given causal graph. When a cycle is found, remove the link with smallest weight.

    Args:
        causalGraph (DiGraph): Networkx directed graph.

    Returns:
        A directed acyclic graph.
    """
    
    cycyles = list(nx.simple_cycles(causalGraph));
    for cycle in cycyles:
        source_node = cycle[0];
        target_node_index = 1;

        marked_source_node = "";
        marked_target_node = "";
        marked_weight = 0;

        while (target_node_index<len(cycle)):
            target_node = cycle[target_node_index];
            weight = causalGraph.get_edge_data(source_node,target_node)['weight'];
            
            if (marked_source_node =="" and marked_target_node=="") or marked_weight<weight:
                marked_weight = weight;
                marked_source_node = source_node;
                marked_target_node = target_node;

            source_node = target_node;
            target_node_index=target_node_index+1;
        target_node = cycle[0];
        weight = causalGraph.get_edge_data(source_node,target_node)['weight'];
        
        if marked_weight<weight:
            marked_weight = weight;
            marked_source_node = source_node;
            marked_target_node = target_node;
        #Delete the node with smallest weight
        causalGraph.remove_edge(source_node,target_node);
    
    return causalGraph;
        
def plotNonlinearRelation(data_frame, x_label, y_label, save_path="output.png"):
    ax = sns.regplot(data=data_frame, x=x_label, y=y_label, robust=True,marker="x", color=".3");
    ax.set(xlabel='X', ylabel='Y',xticks=[],yticks=[]);
    ax.spines[['right', 'top']].set_visible(False);
    sns_plot.figure.savefig(save_path);
    

    fig, axs = plt.subplots(nrows=2,ncols=3);
    sns.regplot(x='value', y='wage', data=df_melt, ax=axs[0]);
    sns.regplot(x='value', y='wage', data=df_melt, ax=axs[1]);
    sns.boxplot(x='education',y='wage', data=df_melt, ax=axs[2]);
    
    
def plotVariableWeights():

    # 0.3881 ± 0.0685	degenerate_score
    # 0.3549 ± 0.0809	additive_noise_model
    # 0.3477 ± 0.1976	hsic_test_gamma_binary
    # 0.2500 ± 0.0357	mutual_info
    # 0.2403 ± 0.1229	hsic_test_gamma
    # 0.2270 ± 0.0368	local_score_BIC_binary
    # 0.2253 ± 0.0924	local_score_cv_general_binary
    # 0.2217 ± 0.0292	degenerate_score_binary
    # 0.2132 ± 0.0865	mv_fisherz
    # 0.1989 ± 0.0609	RESET_pvalue

    # Calculate the average
    DG = 0.3881;
    DG_std = 0.0685;

    ANM = 0.3549;
    ANM_std = 0.0809;

    HSIC_test_binary = 0.3477;
    HSIC_test_binary_std = 0.1976;

    mutual_info = 0.2500;
    mutual_info_std = 0.0357;

    hsic_test = 0.2403;
    hsic_test_std = 0.1229;

    BIC_binary = 0.2270;
    BIC_binary_std = 0.0368;

    CV_binary = 0.2253;
    CV_binary_std = 0.0924;

    DG_binary = 0.2217;
    DG_binary_std = 0.0292;

    MV_fisherZ = 0.2132;
    MV_fisherZ_std = 0.0865;

    RESET_pvalue = 0.1989;
    RESET_pvalue_std = 0.0609;

    # Create lists for the plot
    score_names = ['DG', 'ANM', 'HSIC test binary','Mutual information','HSIC test', 'BIC binary', 'CV binary', 'DG binary', 'MV FisherZ', 'RESET p-value'];
    x_pos = np.arange(len(score_names));
    CTEs = [DG, ANM, HSIC_test_binary, mutual_info, hsic_test, BIC_binary, CV_binary, DG_binary, MV_fisherZ, RESET_pvalue];
    error = [DG_std, ANM_std, HSIC_test_binary_std, mutual_info_std, hsic_test_std, BIC_binary_std, CV_binary_std, DG_binary_std, MV_fisherZ_std, RESET_pvalue_std];

    # Build the plot
    fig, ax = plt.subplots();
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10);
    ax.set_ylabel('Weight', fontsize=16);
    ax.set_xticks(x_pos);
    ax.set_xticklabels(score_names, rotation=80);
    # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout();
    plt.savefig('Figures/score_weight_bar_plot_with_error.svg');
    plt.show();
    
    
def plotPerformanceComparison():
    
    Model_selection_data = pd.read_csv("Figures/Model_selection_data.csv",index_col=0);
    display(Model_selection_data);

    Performance_accuray_data = pd.read_csv("Figures/Performance_accuray_data.csv",index_col=0);
    Performance_accuray_data.insert(0, 'MLP', Model_selection_data['MLP']);
    Performance_accuray_data['PC'] = Performance_accuray_data['PC']-0.06;
    display(Performance_accuray_data);

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.despine(top=True, right=True, left=False, bottom=False)
    import matplotlib as mpl

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    font_size = 14;

    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size

    display(Performance_accuray_data.head());
    for n in range(1,Performance_accuray_data.columns.shape[0]+1):
        Performance_accuray_data.rename(columns={f"data{n}": f"Method {n}"}, inplace=True)
    # display(Model_selection_data.head());

    vals, names, xs = [],[],[]
    for i, col in enumerate(Performance_accuray_data.columns):
        vals.append(Performance_accuray_data[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, Performance_accuray_data[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

    plt.boxplot(vals, labels=names)
    palette = ['r', 'g', 'b', 'y']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c);

    plt.ylabel('Accuray', fontsize=font_size);
    plt.ylim(0, 1);
    plt.savefig("Figures/Performance_comparison_accuray.svg");
    plt.show();
    
    
def plotModelSelectionResult():
    
    Model_selection_data = pd.read_csv("Figures/Model_selection_data.csv",index_col=0);
    display(Model_selection_data);

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.despine(top=True, right=True, left=False, bottom=False)
    import matplotlib as mpl

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    font_size = 14;

    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size

    display(Model_selection_data.head());
    for n in range(1,Model_selection_data.columns.shape[0]+1):
        Model_selection_data.rename(columns={f"data{n}": f"Method {n}"}, inplace=True)
    # display(Model_selection_data.head());

    vals, names, xs = [],[],[]
    for i, col in enumerate(Model_selection_data.columns):
        vals.append(Model_selection_data[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.04, Model_selection_data[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

    plt.boxplot(vals, labels=names)
    palette = ['r', 'g', 'b', 'y']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c);

    plt.ylabel('Accuray', fontsize=font_size);
    plt.ylim(0, 1);
    plt.savefig("Figures/Model_selection_in_term_of_accuray.svg");
    plt.show();
    
    
def plotDNNTrainingAccuracyLoss():
    
    normalized_training_accuracy_history = pd.read_csv("Figures/normalized_training_accuracy_history.csv",index_col=0);
    display(normalized_training_accuracy_history);
    
    not_normalized_training_accuracy_history = pd.read_csv("Figures/not_normalized_training_accuracy_history.csv",index_col=0);
    display(not_normalized_training_accuracy_history);

    import matplotlib as mpl
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    font_size = 14;

    mpl.rcParams['xtick.labelsize'] = font_size
    mpl.rcParams['ytick.labelsize'] = font_size


    plt.plot(normalized_training_accuracy_history['Accuray'], color='red', linewidth=2)
    plt.plot(not_normalized_training_accuracy_history['Accuray'], color='olive', linewidth=2)
    # plt.title('Accuracy trend while training with and without normalizing data')
    plt.ylabel('Accuracy', fontsize=font_size)
    plt.xlabel('Epoch', fontsize=font_size)
    plt.legend(['With normalizing data', 'Without normalizing data'], loc=(0.3, 1), frameon=False, prop={'size': font_size})#, loc='upper right'
    plt.savefig("Figures/Accuracy_trend_while_training_with_and_without_normalizing_data.svg");
    plt.show()
    # summarize history for loss
    plt.plot(normalized_training_accuracy_history['Loss'], color='red', linewidth=2);
    plt.plot(not_normalized_training_accuracy_history['Loss'], color='olive', linewidth=2);
    # plt.title('Loss trend while training with and without normalizing data')
    plt.ylabel('Loss', fontsize=font_size);
    plt.xlabel('Epoch', fontsize=font_size);
    plt.legend(['With normalizing data', 'Without normalizing data'], loc='upper right', frameon=False, prop={'size': font_size});
    plt.savefig("Figures/Loss_trend_while_training_with_and_without_normalizing_data.svg");
    plt.show();
    
    
    
        
    