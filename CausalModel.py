import pandas as pd
import numpy as np
import networkx as nx

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

import keras

import tensorflow as tf

import Utils as Utils


def build_model_silu(feature_vector_length, input_shape, act_function, kernel_init, num_classes, loss_function, optimizer_name):


    # Create the model
    model = Sequential()
    model.add(Dense(feature_vector_length, input_shape=input_shape, activation=act_function, kernel_initializer=kernel_init))
    model.add(Dense(64, activation=act_function
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(BatchNormalization());#???Does it work???
    model.add(Dense(128, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(256, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    # model.add(Dense(512, activation=act_function2, kernel_initializer=kernel_init
    #                 # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #             # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #                 # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #                ))
    # model.add(Dense(512, activation=act_function3, kernel_initializer=kernel_init
    #                 # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #             # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #                 # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #                ))
    # model.add(Dropout(0.1));
    model.add(Dense(512, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dropout(0.1));
    model.add(Dense(256, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(128, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(64, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(32, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(16, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(8, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(4, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(num_classes, activation='sigmoid'))


    # Configure the model and start training
    model.compile(loss=loss_function, optimizer=optimizer_name
                            , metrics=[keras.metrics.Precision()
                            #, keras.metrics.Recall()
                            #, keras.metrics.SpecificityAtSensitivity(0.5)
                            #, keras.metrics.SensitivityAtSpecificity(0.5)
                            , 'accuracy']);
    return model;

def build_model_relu(feature_vector_length, input_shape, act_function, kernel_init, num_classes, loss_function, optimizer_name):


    # Create the model
    model = Sequential()
    model.add(Dense(feature_vector_length, input_shape=input_shape, activation=act_function, kernel_initializer=kernel_init))
    model.add(Dense(64, activation=act_function
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(BatchNormalization());#???Does it work???
    model.add(Dense(128, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(256, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(512, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    # model.add(Dense(512, activation=act_function3, kernel_initializer=kernel_init
    #                 # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #             # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #                 # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    #                ))
    # model.add(Dropout(0.1));
    model.add(Dense(512, activation=act_function2, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dropout(0.1));
    model.add(Dense(256, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(128, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(64, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(32, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(16, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(8, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(4, activation=act_function, kernel_initializer=kernel_init
                    # ,kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                # ,bias_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                    # ,activity_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
                   ))
    model.add(Dense(num_classes, activation='sigmoid'))


    # Configure the model and start training
    model.compile(loss=loss_function, optimizer=optimizer_name
                            , metrics=[keras.metrics.Precision()
                            #, keras.metrics.Recall()
                            #, keras.metrics.SpecificityAtSensitivity(0.5)
                            #, keras.metrics.SensitivityAtSpecificity(0.5)
                            , 'accuracy']);
    return model;

    

def train():
    data_path = "MergedScoreData/";
    score_data_file_name = "score_data.csv";

    training_data_result_df = pd.read_csv(data_path+score_data_file_name,index_col=0);
    display(training_data_result_df);

    # Normalize the dataset if neccesary. Not normalize discrete variables
    var_type_dict, discrete_list = Utils.check_variable_type(training_data_result_df);
    print(discrete_list);
    
    discrete_dataset = training_data_result_df[discrete_list];
    
    scaler = MinMaxScaler();
    scaled_values = scaler.fit_transform(training_data_result_df);
    training_data_result_df.loc[:,:] = scaled_values;
    #Reset discrete variables
    for discrete_name in discrete_list:
        training_data_result_df[discrete_name] = discrete_dataset[discrete_name];
    
    display(training_data_result_df.tail(10));


    Y_train = training_data_result_df["Label"].values;
    X_train = training_data_result_df.drop(columns=["Label"]).values;
    
    print(Y_train);
    print(X_train);
    print(X_train.shape);
    
    num_classes = 2;
    # Convert target classes to categorical ones
    Y_train = to_categorical(Y_train, num_classes)
    print(Y_train);
    print(Y_train.shape);

    # Configuration options
    feature_vector_length = X_train.shape[1];
    # Set the input shape
    input_shape = (feature_vector_length,);
    
    eposh_num = 10;#1000
    batch_size = 128;
    validation_persentage=0.10;
        
    
    optimizer_name = "adamw";
    loss_function = "binary_crossentropy";
    act_function = "silu";#silu,relu,selu,leaky_relu
    # relu,sigmoid,softmax,softplus,softsign,tanh,selu,elu,exponential,leaky_relu,relu6,silu,hard_silu,gelu,hard_sigmoid,linear,mish,log_softmax
    act_function2 = 'silu';
    act_function3 = 'tanh';
    
    kernel_init='he_normal';
    # 'glorot_uniform' (or 'xavier_uniform'): This is the default initializer for most Keras layers. It draws samples from a uniform distribution within a range calculated based on the number of input and output units.
    # 'glorot_normal' (or 'xavier_normal'): Similar to glorot_uniform, but draws samples from a normal distribution.
    # 'he_uniform': This initializer is recommended for layers with ReLU activation. It draws samples from a uniform distribution within a range calculated based on the number of input units.
    # 'he_normal': Similar to he_uniform, but draws samples from a normal distribution.

    # Create a basic model instance
    if act_function=="silu":
        model = build_model_silu(feature_vector_length, input_shape, act_function, kernel_init, num_classes, loss_function, optimizer_name);
    else:
        model = build_model_relu(feature_vector_length, input_shape, act_function, kernel_init, num_classes, loss_function, optimizer_name);
    
    #Save the best model: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    checkpoint_filepath = 'SavedModel/MLP/'+act_function+'/'+str(eposh_num)+'epoches_'+'causal_model.keras';
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',#val_precision,val_accuracy
        mode='max',
        save_best_only=True);
    
    training_info = model.fit(X_train, Y_train, epochs=eposh_num, batch_size=batch_size, validation_split=validation_persentage, verbose=1, callbacks=[model_checkpoint_callback]);

    #Save the best model: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    checkpoint_filepath = 'SavedModel/MLP/'+act_function+'/'+str(eposh_num)+'epoches_'+'causal_model_last.keras';
    model.save(checkpoint_filepath)




def mlp_estimate(X_test):
    
    eposh_num = 1000;

    # Reload Keras models from the .keras zip archive:
    act_function = "silu";#selu,relu,silu,leaky_relu

    checkpoint_filepath = 'SavedModel/MLP/'+act_function+'/'+str(eposh_num)+'epoches_'+'causal_model.keras';
    silu_model = tf.keras.models.load_model(checkpoint_filepath);
    checkpoint_filepath = 'SavedModel/MLP/'+act_function+'/'+str(eposh_num)+'epoches_'+'causal_model_last.keras';
    silu_model_last = tf.keras.models.load_model(checkpoint_filepath);


    act_function = "relu";#selu,relu,silu,leaky_relu

    checkpoint_filepath = 'SavedModel/MLP/'+act_function+'/'+str(eposh_num)+'epoches_'+'causal_model.keras';
    relu_model = tf.keras.models.load_model(checkpoint_filepath);

    checkpoint_filepath = 'SavedModel/MLP/'+act_function+'/'+str(eposh_num)+'epoches_'+'causal_model_last.keras';
    relu_model_last = tf.keras.models.load_model(checkpoint_filepath);

    # keras.utils.plot_model(model, to_file="Figures/MLP_Model.png", show_shapes=True)
    

    y_pred = silu_model.predict(X_test);
    class_predictions_silu = np.argmax (y_pred, axis = 1);
    print(class_predictions_silu);


    y_pred = silu_model_last.predict(X_test);
    class_predictions_silu_last = np.argmax (y_pred, axis = 1);
    print(class_predictions_silu_last);
    
    y_pred = relu_model.predict(X_test);
    class_predictions_relu = np.argmax (y_pred, axis = 1);
    print(class_predictions_relu);
    
    y_pred = relu_model_last.predict(X_test);
    class_predictions_relu_last = np.argmax (y_pred, axis = 1);
    print(class_predictions_relu_last);
    
    merged_pred = [];
    for p_silu,p_silu_last, p_relu,p_relu_last in zip(class_predictions_silu,class_predictions_silu_last, class_predictions_relu, class_predictions_relu_last):
        if p_silu == 1 or p_silu_last ==1 or p_relu==1 or p_relu_last==1:
            merged_pred.append(1);
        else: 
            merged_pred.append(0);

    print(merged_pred);
    return merged_pred;
    

def estimate(data_path, data_file_name, score_data_file_name):
    
    
    estimation_result_file_name = "estimation_result_mlp.csv";
    
    score_data_df = pd.read_csv(data_path+score_data_file_name,index_col=0);
    # display(score_data_df);
    dataset = pd.read_csv(data_path+data_file_name);
    # display(dataset);
    
    # Drop rows with any missing values
    score_data_df.fillna(0,inplace=True);
    # display(score_data_df);

    # #Normalize the dataset if neccesary. Not normalize discrete variables
    var_type_dict, discrete_list = Utils.check_variable_type(score_data_df);
    # print(discrete_list);
    
    discrete_dataset = score_data_df[discrete_list];
    
    scaler = MinMaxScaler();
    scaled_values = scaler.fit_transform(score_data_df);
    score_data_df.loc[:,:] = scaled_values;
    #Reset discrete variables
    for discrete_name in discrete_list:
        score_data_df[discrete_name] = discrete_dataset[discrete_name];
    
    # display(score_data_df);

    # Remove the label column if it is included.
    num_classes = 2;
    # score_data_df.columns
    if 'Label' in score_data_df.columns:
        Y_test = score_data_df["Label"].values;
        
        # Convert target classes to categorical ones
        Y_test_cate = to_categorical(Y_test, num_classes)
        # print(Y_test_cate);
        # print(Y_test_cate.shape);
        
        score_data_df.drop(['Label'], axis=1, inplace=True);
    else:
        Y_test = None;
        Y_test_cate = None;
        
    X_test = score_data_df.values;
    # print(X_test);
    # print(X_test.shape);


    merged_pred = mlp_estimate(X_test);

    #Save estimation result
    result_DF = pd.DataFrame(merged_pred,columns=['Estimation'],index=score_data_df.index);
    # display(result_DF);

    return dataset, result_DF, score_data_df;


def dag(dataset, result_DF, score_data_df):
    '''
    A directed acyclic graph in the 'dot' format will be saved in the same directory as the input data.
    '''
    
    # #Normalize the dataset if neccesary. Not normalize discrete variables
    dataset_var_type_dict, dataset_discrete_list = Utils.check_variable_type(dataset);
    # print(discrete_list);
    
    directed_graph = nx.DiGraph();
    
    sep = "_causes_";
    idx_list = result_DF.index.tolist();
    while idx_list:
        rel1 = idx_list.pop();
        # print(rel1);
        rel1_strs = rel1.split(sep);
        cause = rel1_strs[0];
        effect = rel1_strs[1];
        
        rel1_bic = score_data_df.loc[rel1]['local_score_BIC_1'];
        
        rel2 = effect+sep+cause;
        if rel2 in idx_list:
            idx_list.remove(rel2);
            # print('==============================================');
            
            rel2_bic = score_data_df.loc[rel2]['local_score_BIC_1'];
            
            rel1_dg = score_data_df.loc[rel1]['degenerate_score'];
            rel2_dg = score_data_df.loc[rel2]['degenerate_score'];
    
            
            #If both variables are continuous or discrete, use DG; If one of the two variables is discrete and the other is continuous, use BIC.
            if (cause in dataset_discrete_list and effect not in dataset_discrete_list) or (cause not in dataset_discrete_list and effect in dataset_discrete_list):
                print("One variable is discrete and the other is continuous.");
                if rel1_bic>=rel2_bic:
                    print("Estimated relation: "+rel2);
                    directed_graph.add_edge(effect,cause, weight=rel1_bic);
                else:
                    print("Estimated relation: "+rel1);
                    #Add it to graph
                    directed_graph.add_edge(cause,effect, weight=rel2_bic);
            else:
                print("Both variables are the same data type.");
                if rel1_dg>=rel2_dg:
                    print("Estimated relation: "+rel2);
                    directed_graph.add_edge(effect,cause, weight=rel2_bic);
                else:
                    print("Estimated relation: "+rel1);
                    #Add it to graph
                    directed_graph.add_edge(cause,effect, weight=rel1_bic);
                
        else:
            #Add it to graph
            directed_graph.add_edge(cause,effect, weight=rel1_bic);
    
    #Remove cycles in graph
    causal_graph = Utils.removeCycles(directed_graph);
    graph_save_path = data_path+"estimated_causal_graph.dot";
    # graph_save_path = "estimated_causal_graph.dot";
    nx.drawing.nx_pydot.write_dot(causal_graph,graph_save_path);
    # result_DF.to_csv(data_path+estimation_result_file_name);
