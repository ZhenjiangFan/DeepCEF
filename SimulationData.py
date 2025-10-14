import pandas as pd

from nonlinearsim.NonlinearSimulation import NonlinearSimulation
from pytetrad import SimulateMixedData as SimMixed


def generate_mixed_sim_data(num_of_feature,num_of_samples,save_path):
    
    mixed_data_DF, graph_mat, mixed_edge_DF = SimMixed.simulate_data(num_of_features=num_of_feature, num_of_samples=num_of_samples);
    mixed_edge_DF['Edge'] = mixed_edge_DF['Edge'].astype(str);

    #Add a prefix to all the variables
    prefix_str = 'Mixed_';
    name_mapping = {};
    for col in mixed_data_DF.columns:
        name_mapping[col]=prefix_str+col;
    
    mixed_data_DF.rename(columns=name_mapping, inplace=True);

    #Change names of the cause and effect
    name_mapping = {};
    sep = " --> ";
    for index,row in mixed_edge_DF.iterrows():
        edge = row['Edge'];
        nodes = edge.split(sep);
        cause = nodes[0];
        effect = nodes[1];
        new_str = prefix_str+cause+sep+prefix_str+effect;
        name_mapping[edge]=new_str;
    
    mixed_edge_DF['Edge'] = mixed_edge_DF['Edge'].map(name_mapping);

    #Save the simulation data matrix and graph edge data
    file_name = 'mixed_sim_data.csv';
    mixed_data_DF.to_csv(save_path+file_name, index=False);
    file_name = 'mixed_sim_data.txt';
    mixed_data_DF.to_csv(save_path+file_name, index=False, sep='\t');

    file_name = 'mixed_sim_graph_edges.csv';
    mixed_edge_DF.to_csv(save_path+file_name, index=False);

    print("Mixed type simulation data have been saved to "+save_path+".");

def generate_nonlinear_sim_data(num_of_feature,num_of_samples,save_path):
    NLSim = NonlinearSimulation(sample_size=num_of_samples, num_of_features=num_of_feature,starting_index=1);
    nl_data_DF,directed_graph = NLSim.simulate_nl_rel();

    #Add a prefix to variable names
    prefix_str = 'Nonlinear_';
    name_mapping = {};
    for col in nl_data_DF.columns:
        name_mapping[col]=prefix_str+col;
    
    nl_data_DF.rename(columns=name_mapping, inplace=True);

    sep = " --> ";
    edge_list = [];
    for edge in directed_graph.edges():
        cause = edge[0];
        effect = edge[1];
        edge_str = name_mapping[cause]+sep+name_mapping[effect];
        edge_list.append(edge_str);
    nl_edge_DF = pd.DataFrame(edge_list, columns=["Edge"]);

    #Save
    file_name = 'nonlinear_sim_data.csv';
    nl_data_DF.to_csv(save_path+file_name, index=False);
    file_name = 'nonlinear_sim_data.txt';
    nl_data_DF.to_csv(save_path+file_name, index=False, sep='\t');
    
    file_name = 'nonlinear_sim_graph_edges.csv';
    nl_edge_DF.to_csv(save_path+file_name, index=False);

    print("Nonlinear simulation data have been saved to "+save_path+".");


def generate_continuous_sim_data(num_of_feature, num_of_samples, save_path):
    
    mixed_data_DF, graph_mat, mixed_edge_DF = SimMixed.simulateContinuous(num_of_features=num_of_feature, num_of_samples=num_of_samples);
    mixed_edge_DF['Edge'] = mixed_edge_DF['Edge'].astype(str);

    #Add a prefix to variable names
    prefix_str = 'Continuous_';
    name_mapping = {};
    for col in mixed_data_DF.columns:
        name_mapping[col]=prefix_str+col;
        
    mixed_data_DF.rename(columns=name_mapping, inplace=True);

    name_mapping = {};
    #Merge two graphs
    sep = " --> ";
    for index,row in mixed_edge_DF.iterrows():
        edge = row['Edge'];
        nodes = edge.split(sep);
        cause = nodes[0];
        effect = nodes[1];
        new_str = prefix_str+cause+sep+prefix_str+effect;
        name_mapping[edge]=new_str;
        
    mixed_edge_DF['Edge'] = mixed_edge_DF['Edge'].map(name_mapping);
    #display(mixed_edge_DF);

    #Save
    file_name = '/continuous_sim_data.csv';
    mixed_data_DF.to_csv(save_path+file_name, index=False);
    file_name = '/continuous_sim_data.txt';
    mixed_data_DF.to_csv(save_path+file_name, index=False, sep='\t');
    
    file_name = '/continuous_sim_graph_edges.csv';
    mixed_edge_DF.to_csv(save_path+file_name, index=False);

    print("Continuous simulation data have been saved to "+save_path+".");
    
def generate(num_of_feature_mixed=50,num_of_samples_mixed=1500,save_path_mixed=None
            ,num_of_feature_nonlinear=50, num_of_samples_nonlinear=1500, save_path_nonlinear=None
            ,num_of_feature_continuous=50, num_of_samples_continuous=1500, save_path_continuous=None):

    '''
    For each type of simulation data (mixed-type, nonlinear, and continuous), three files will be generated:
        1. [data type]_sim_data.csv - simulation data matrix in CSV format, including variable names as headers.
        2. [data type]_sim_data.txt - simulation data matrix in TXT format separated via '\t', including variable names as headers.
        3. [data type]__sim_graph_edges.csv - A list of edges of the simulated causal graph.
    '''
    
    generate_mixed_sim_data(num_of_feature_mixed, num_of_samples_mixed, save_path_mixed)
    generate_nonlinear_sim_data(num_of_feature_nonlinear,num_of_samples_nonlinear,save_path_nonlinear);
    generate_continuous_sim_data(num_of_feature_continuous, num_of_samples_continuous, save_path_continuous);

    