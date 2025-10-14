import jpype.imports


import os
import sys

dir_path = os.getcwd();
#jar path
jar_path = os.path.join(dir_path, 'pytetrad'+os.path.sep+'resources'+os.path.sep+'tetrad-current.jar');
print(jar_path);

try:
    jpype.startJVM();
except OSError:
    pass

#Add the jar to Java class path
jpype.addClassPath(jar_path);


import numpy as np
import pandas as pd
from pandas import DataFrame

import java.util as util

from edu.cmu.tetrad import data as td
#import edu.cmu.tetrad.data as td

import edu.cmu.tetrad.graph as tg

from edu.cmu.tetrad.util import Params, Parameters
import edu.cmu.tetrad.algcomparison.simulation as sim
import edu.cmu.tetrad.algcomparison.graph as graph



def tetrad_data_to_pandas(data: td.DataSet):
    names = data.getVariableNames()
    columns_ = []

    for name in names:
        columns_.append(str(name))

    df: DataFrame = pd.DataFrame(columns=columns_, index=range(data.getNumRows()))

    for row in range(data.getNumRows()):
        for col in range(data.getNumColumns()):
            df.at[row, columns_[col]] = data.getObject(row, col)

    return df

## The defaults here are for the PCALG style of general graph endpoint matrices, but
## the user can use whichever endpoint encoding they like.
def graph_to_matrix(g, nullEpt = 0, circleEpt = 1, arrowEpt = 2, tailEpt = 3):
    endpoint_map = {"NULL": nullEpt,
                    "CIRCLE": circleEpt,
                    "ARROW": arrowEpt,
                    "TAIL": tailEpt}

    nodes = g.getNodes()
    p = g.getNumNodes()
    A = np.zeros((p, p), dtype=int)

    for edge in g.getEdges():
        i = nodes.indexOf(edge.getNode1())
        j = nodes.indexOf(edge.getNode2())
        A[j][i] = endpoint_map[edge.getEndpoint1().name()]
        A[i][j] = endpoint_map[edge.getEndpoint2().name()]

    columns_ = []

    for name in nodes:
        columns_.append(str(name))

    return pd.DataFrame(A, columns=columns_)

def tetrad_matrix_to_numpy(array):
    np_array = np.zeros((array.getNumRows(), array.getNumColumns()), dtype=float)

    for i in range(array.getNumRows()):
        for j in range(array.getNumColumns()):
            np_array[i][j] = array.get(i, j)

    return np_array

def tetrad_matrix_to_pandas(array, variables):
    np_array = tetrad_matrix_to_numpy(array)
    columns = [str(variables.get(i)) for i in range(array.getNumColumns())]
    return pd.DataFrame(np_array, columns=columns)


    
# Simuolates a mixed continuous/discrete dataset using the Lee-Hastic method with the given arguments
# and returns the dataset as a pandas dataframe.
def simulateLeeHastie(num_meas = 20, num_lat = 0, avg_deg = 3, min_cat=2, max_cat=10, perc_disc=50, samp_size=1000):

    # Set the parameters for the simulation
    params = Parameters()

    params.set(Params.NUM_MEASURES, num_meas)
    params.set(Params.NUM_LATENTS, num_lat)
    params.set(Params.AVG_DEGREE, avg_deg)

    params.set(Params.MIN_CATEGORIES, min_cat)
    params.set(Params.MAX_CATEGORIES, max_cat)
    params.set(Params.PERCENT_DISCRETE, perc_disc)
    params.set(Params.DIFFERENT_GRAPHS, False)

    params.set(Params.RANDOMIZE_COLUMNS, True) # Preents some algorithsm from taking advantage of causal order
    params.set(Params.SAMPLE_SIZE, samp_size)
    params.set(Params.SAVE_LATENT_VARS, False)
    # params.set(Params.SEED, 29493L)

    params.set(Params.NUM_RUNS, 10)

    # Do the simulation and grab the dataset and generative graph
    sim_ = sim.LeeHastieSimulation(graph.RandomForward())
    sim_.createData(params, True)
    D = sim_.getDataModel(0)
    G = sim_.getTrueGraph(0)

    return D, G

    # D_ = tr.tetrad_to_pandas(D)
    # G_ = tr.tetrad_graph_to_causal_learn(G)
    #
    # return D_, G_

def simulate_data(num_of_features=30, num_of_samples=1000, save_path=None):

    # num_of_meas = 30;
    # samp_size = 2000;

    ## Simulates data with both continuous and discrete columns.
    D, G = simulateLeeHastie(num_meas=num_of_features, samp_size=num_of_samples);

    #print("");
    #print(G.getEdges());
    edge_DF = pd.DataFrame(G.getEdges(), columns=["Edge"]);#.to_csv(save_path+'/mixed_sim_graph_edges.csv', index=False);

    D = tetrad_data_to_pandas(D);
    G = graph_to_matrix(G);

    # # Save data to a file
    # D.to_csv(save_path+'/mixed_sim_data.csv', index=False);
    # G.to_csv(save_path+'/mixed_sim_graph_matrix.csv', index=False);
    
    return D, G, edge_DF;


# Simuolates a continuous dataset with the given arguments and returns the dataset as a pandas datafram
def simulateContinuous(num_of_features = 20, num_of_samples = 200, save_path=None):
    num_lat = 0; 
    avg_deg = 4;
    coef_low = 0;
    coef_high = 1;
    var_low = 1;
    var_high = 3;
    rand_cols=False;
    
    # Set the parameters for the simulation
    params = Parameters()

    params.set(Params.SAMPLE_SIZE, num_of_samples)
    params.set(Params.NUM_MEASURES, num_of_features)
    params.set(Params.AVG_DEGREE, avg_deg)
    params.set(Params.NUM_LATENTS, num_lat)
    params.set(Params.RANDOMIZE_COLUMNS, rand_cols) # Prevents some algorithsm from taking advantage of true causal order
    params.set(Params.COEF_LOW, coef_low)
    params.set(Params.COEF_HIGH, coef_high)
    params.set(Params.VAR_LOW, var_low)
    params.set(Params.VAR_HIGH, var_high)
    params.set(Params.INTERVAL_BETWEEN_SHOCKS, 30)
    params.set(Params.INTERVAL_BETWEEN_RECORDINGS, 30)
    params.set(Params.VERBOSE, False)
    params.set(Params.NUM_RUNS, 1)
    # params.set(Params.SEED, 29483)

    # Do the simulation and grab the dataset and generative graph
    sim_ = sim.LinearFisherModel(graph.RandomForward())
    sim_.createData(params, True)

    D = sim_.getDataModel(0)
    G = sim_.getTrueGraph(0)

    # return D, G
    #print(G.getEdges());
    edge_DF = pd.DataFrame(G.getEdges(), columns=["Edge"]);#.to_csv(save_path+'/mixed_sim_graph_edges.csv', index=False);

    D = tetrad_data_to_pandas(D);
    G = graph_to_matrix(G);

    # # Save data to a file
    # D.to_csv(save_path+'/mixed_sim_data.csv', index=False);
    # G.to_csv(save_path+'/mixed_sim_graph_matrix.csv', index=False);
    
    return D, G, edge_DF;


#     num_of_meas = 20;
#     ## Simulates data with both continuous and discrete columns.
#     D, G = sim.simulateNLSem(num_meas=num_of_meas, samp_size=samp_size);

#     print(G.getEdges());
#     pd.DataFrame(G.getEdges(), columns=["Edge"]).to_csv('../nonlinear_sim_true_graph.csv', index=False);

#     D = tr.tetrad_data_to_pandas(D);
#     G = tr.graph_to_matrix(G);

#     # Save data to a file
#     D.to_csv('../nonlinear_sim_data.csv', index=False);
#     G.to_csv('../nonlinear_sim_graph.csv', index=False);


'''
num_of_meas = 30;
samp_size = 2000;

## Simulates data with both continuous and discrete columns.
D, G = simulateLeeHastie(num_meas=num_of_meas, samp_size=samp_size);
#print("");
print(G.getEdges());
pd.DataFrame(G.getEdges(), columns=["Edge"]).to_csv(save_path+'/mixed_sim_graph_edges.csv', index=False);

D = tetrad_data_to_pandas(D);
G = graph_to_matrix(G);

# Save data to a file
D.to_csv(save_path+'/mixed_sim_data.csv', index=False);
G.to_csv(save_path+'/mixed_sim_graph_matrix.csv', index=False);
'''

'''
try:
    jpype.startJVM(classpath=[f"resources/tetrad-current.jar"])
except OSError:
    print("JVM already started")

# import tools.translate as tr
# import tools.simulate as sim

# from tools import translate as tr
# from tools import simulate as sim

# from .translate import translate as tr
# from .simulate import simulate as sim

#from .translate import translate as tr
#from .simulate import simulate as sim
'''

'''
def pandas_data_to_tetrad(df: DataFrame, int_as_cont=False):
    dtypes = ["float16", "float32", "float64"]
    if int_as_cont:
        for i in range(3, 7):
            dtypes.append(f"int{2 ** i}")
            dtypes.append(f"uint{2 ** i}")
    cols = df.columns
    discrete_cols = [col for col in cols if df[col].dtypes not in dtypes]
    category_map = {col: {val: i for i, val in enumerate(df[col].unique())} for col in discrete_cols}
    df = df.replace(category_map)
    values = df.values
    n, p = df.shape

    variables = util.ArrayList()
    for col in cols:
        if col in discrete_cols:
            categories = util.ArrayList()
            for category in category_map[col]:
                categories.add(str(category))
            variables.add(td.DiscreteVariable(str(col), categories))
        else:
            variables.add(td.ContinuousVariable(str(col)))

    print(discrete_cols)

    if len(discrete_cols) == len(cols):
        databox = td.IntDataBox(n, p)
    elif len(discrete_cols) == 0:
        databox = td.DoubleDataBox(n, p)
    else:
        databox = td.MixedDataBox(variables, n)

    for col, var in enumerate(values.T):
        for row, val in enumerate(var):
            databox.set(row, col, val)

    return td.BoxDataSet(databox, variables)
'''
'''
# Input a square int[][] array with only 0's and 1's, where a[i][j] = 1 just in case
# j->i. Returns a Java graph object for this.
def adj_matrix_to_graph(adjMatrix):
    rows, cols = adjMatrix.shape

    if rows != cols:
        raise ValueError("The matrix is not square. Rows and columns must be equal.")

    variable_names = ["X" + str(i) for i in range(1, rows + 1)]
    variables = util.ArrayList()

    for i in range(0, rows):
        variables.append(tg.GraphNode(variable_names[i]))

    graph = tg.EdgeListGraph(variables)

    for i, row in enumerate(adjMatrix):
        for j, value in enumerate(row):
            if (adjMatrix[i][j]):
                graph.addDirectedEdge(variables.get(i), variables.get(j))

    return graph
'''

'''
# PASS ME A GraphViz Graph object and call it gdot!
def write_gdot(g, gdot):
    endpoint_map = {"TAIL": "none",
                    "ARROW": "empty",
                    "CIRCLE": "odot"}

    for node in g.getNodes():
        gdot.node(str(node.getName()),
                  shape='circle',
                  fixedsize='true',
                  style='filled',
                  color='lightgray')

    for edge in g.getEdges():
        node1 = str(edge.getNode1().getName())
        node2 = str(edge.getNode2().getName())
        endpoint1 = str(endpoint_map[edge.getEndpoint1().name()])
        endpoint2 = str(endpoint_map[edge.getEndpoint2().name()])
        color = "blue"
        if (endpoint1 == "empty") and (endpoint2 == "empty"): color = "red"
        gdot.edge(node1, node2,
                  arrowtail=endpoint1,
                  arrowhead=endpoint2,
                  dir='both', color=color)

    return gdot
'''
    
'''
# Prints a Java graph object as a Tetrad-style string. Needed from R.
def print_java(java_graph: tg.Graph):
    print(java_graph)
'''