import numpy as np
import pandas as pd
import networkx as nx
import random

class NonlinearSimulation:
    
    def __init__(self,sample_size=2000, num_of_features=20, starting_index=1):
        self.sample_size = sample_size;
        # num_of_feat = 20;#Must be equal to or bigger than 5.
        self.num_of_feat=num_of_features;
        self.start_val_linspace = 5;
        self.stop_val_linspace = 0;
        self.starting_index = starting_index;

    def generate_two_to_one_nl_rel(self):
        flag = bool(random.randrange(2));
        if flag:
            
            x1 = np.random.uniform(size=self.sample_size)*6;
            x2 = np.random.uniform(size=self.sample_size)*2;
            noise = 0.1*np.random.normal(0, 2, self.sample_size);
            y = np.arctan(x1**4)+np.cos(x2**2.5);
            ydata = y + noise;

        else:
            
            x1 = np.random.uniform(size=self.sample_size)*4;
            x2 = np.random.uniform(size=self.sample_size)*1;
            noise = 0.1*np.random.normal(0, 2, self.sample_size);
            y = -np.tanh(x1**5)+np.arcsin(x2**4);
            ydata = y + noise;

        return x1,x2,ydata;

    def generate_one_to_one_nl_rel(self,trend_type=1):

        reverse = bool(random.randrange(2));

        if trend_type==1:
            x = np.random.uniform(size=self.sample_size);
            if reverse:
                y = -np.power(x,7);
            else:
                y = np.power(x,7);
            noise = 0.2*np.random.normal(size=self.sample_size);
            ydata = y+ noise;
            
            #print('trend_type==1');
            
        elif trend_type==2:
            x = np.random.uniform(size=self.sample_size)*2;
            noise = 0.06*np.random.normal(3, 6, self.sample_size);
            if reverse:
                y = -np.sin(x*1.5);
            else:
                y = np.sin(x*1.5);
            ydata = y + noise;
            
            #print('trend_type==2');

        elif trend_type==3:
            x = np.random.uniform(size=self.sample_size)+2;
            noise = 0.5*np.random.normal(0, 2, self.sample_size);
            if reverse:
                y = -np.cos(x**2.5);
            else:
                y = np.cos(x**2.5);
            ydata = y + noise;
            #print('trend_type==3');
            
        elif trend_type==4:
            x = np.random.uniform(size=self.sample_size);
            noise = 0.2*np.random.normal(0,2, self.sample_size);
            if reverse:
                y = -np.arccos(x**8);
            else:
                y = np.arccos(x**8);
            ydata = y + noise;
            #print('trend_type==4');
            
        elif trend_type==5:
            x = np.random.uniform(size=self.sample_size)*3;
            noise = 0.2*np.random.normal(0, 2, self.sample_size);
            if reverse:
                y = -np.arctan(x**19);
            else:
                y = np.arctan(x**19);
            ydata = y + noise;
            #print('trend_type==5');
            
        elif trend_type==6:
            x = np.random.uniform(size=self.sample_size);
            noise = 0.1*np.random.normal(0, 2, self.sample_size);
            if reverse:
                y = -np.arcsin(x**4);
            else:
                y = np.arcsin(x**4);
            ydata = y + noise;
            #print('trend_type==6');
            
        elif trend_type==7:
            x = np.random.uniform(size=self.sample_size)*5;
            noise = 0.1*np.random.normal(0, 2, self.sample_size);
            if reverse:
                y = -np.tanh(x**4);
            else:
                y = np.tanh(x**4);
            ydata = y + noise;
            #print('trend_type==7');
            
        return x, ydata;

    def simulate_nl_rel(self):

        data_df = pd.DataFrame();
        directed_graph = nx.DiGraph();

        #Simulate one-to-one relations
        trend_type = 1;
        num_of_relations = int((self.num_of_feat-3)/2);
        
        end_idx = num_of_relations+self.starting_index;
        for first_idx in range(self.starting_index,end_idx):
            
            second_idx = first_idx+num_of_relations;
            
            first_idx_str = 'X'+str(first_idx);
            second_idx_str = 'X'+str(second_idx);
            
            x, ydata = self.generate_one_to_one_nl_rel(trend_type=trend_type);
            # print(x);
            # print(ydata);

            data_df[first_idx_str] = x;
            data_df[second_idx_str] = ydata;
            directed_graph.add_edge(first_idx_str,second_idx_str);
            
            trend_type=1 if trend_type==7 else trend_type+1;

        # Simulate two-to-one relations
        parent_idx1 = second_idx+1;
        parent_idx_str1 = "X"+str(parent_idx1);
        
        parent_idx2 = second_idx+2;
        parent_idx_str2 = "X"+str(parent_idx2);
        
        child_idx = second_idx+3;
        child_idx_str = "X"+str(child_idx);

        x1,x2,ydata = self.generate_two_to_one_nl_rel();
        data_df[parent_idx_str1] = x1;
        data_df[parent_idx_str2] = x2;
        data_df[child_idx_str] = ydata;
        
        directed_graph.add_edge(parent_idx_str1,child_idx_str);
        directed_graph.add_edge(parent_idx_str2,child_idx_str);
        
        last_idx = self.starting_index+self.num_of_feat;
        if child_idx<last_idx:
            for idx in range(child_idx+1,last_idx):
                temp_idx_str = "X"+str(idx);
                x = np.random.uniform(size=self.sample_size);
                data_df[temp_idx_str] = x;

        #display(data_df);
        #print(directed_graph.edges());
        return data_df,directed_graph;