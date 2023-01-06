
import os
import json
import sys
if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    import pickle
else:
    import pickle5 as pickle

import pandas as pd

import argparse

from argparse import ArgumentParser

import torch
import torch_geometric
import torchvision
from torch_geometric.loader import DataLoader


from tqdm import tqdm
import datetime
from collections import Counter



from scipy.sparse import csr_matrix, triu
from scipy.spatial import distance_matrix
from scipy.sparse import coo_matrix
import re
import matplotlib
from sklearn.preprocessing import minmax_scale
import math
import matplotlib.pyplot as plt
import numpy as np
from model import *

from sklearn import preprocessing
import datetime
os.umask(0)
from sklearn.preprocessing import MinMaxScaler




def load_pdb_data(args,bs):
    

    
    l_amino_acid = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',]


    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l_amino_acid)
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False,categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


    target_distance = 9
    pdb_atom_format_field = [(1, 6, '"ATOM"'),
                                 (7, 11, 'serial'),
                                 (13, 16, 'name'),
                                 (17, 17, 'altLoc'),
                                 (18, 20, 'resName'),
                                 (22, 22, 'chainID'),
                                 (23, 26, 'resSeq'),
                                 (27, 27, 'iCode'),
                                 (31, 38, 'x'),
                                 (39, 46, 'y'),
                                 (47, 54, 'z'),
                                 (55, 60, 'occupanc'),
                                 (61, 66, 'tempFact'),
                                 (77, 78, 'element'),
                                 (79, 80, 'charge')]


    path = args.target_source

    data_list = []

    for pdb_files in os.listdir(args.target_source):

        pdb_name_s = pdb_files.split('.')[0]




        if args.type == 'af2':
            df = pd.read_fwf(path+'/'+pdb_name_s.lower()+'.pdb',colspecs=[(h[0]-1,h[1]) for h in pdb_atom_format_field],header=None, dtype=str)
            df.columns = [h[2] for h in pdb_atom_format_field]
            if len(df[df['"ATOM"']=='MODEL']) > 1:
                df_model = df[df['"ATOM"']=='MODEL']
                df = df[:df_model.iloc[1].name]
        else:
            df = pd.read_fwf(path+'/'+pdb_name_s.lower()+'.pdb',colspecs=[(h[0]-1,h[1]) for h in pdb_atom_format_field],header=None, dtype=str)
            df.columns = [h[2] for h in pdb_atom_format_field]
            if len(df[df['"ATOM"']=='MODEL']) > 0:
                df_model = df[df['"ATOM"']=='MODEL']
                df = df[:df_model.iloc[1].name]




        df_o = df[(df['"ATOM"']=='ATOM')&(df['resName'].isin(l_amino_acid))].reset_index(drop=True).copy()
        

        
        for pdb_chain in df_o['chainID'].unique():
            pdb_name_chain = pdb_name_s + '_' + pdb_chain
            
            df = df_o[(df_o['chainID'] == pdb_chain)].reset_index(drop=True)


            df['chainID_resSeq'] = df['chainID'] + df['resSeq'].str.zfill(5)
            # df['resSeq'] = df['resSeq'].apply(int)

            df['x'] = df['x'].apply(float)
            df['y'] = df['y'].apply(float)
            df['z'] = df['z'].apply(float)
            df_1 = df.groupby('chainID_resSeq').agg({'resName':'first','name':'count','x':'sum','y':'sum','z':'sum'})
            df_1['name'] = df_1['name'].apply(float)
            df_1['x'] = df_1['x']/df_1['name']
            df_1['y'] = df_1['y']/df_1['name']
            df_1['z'] = df_1['z']/df_1['name']
            df_1 = df_1.drop('name',axis=1)
            df_node = df_1.reset_index()
            # df_distance = pd.DataFrame(distance_matrix(df_1[['x','y','z']].values, df_1[['x','y','z']].values))
            d1 = distance_matrix(df_1[['x','y','z']].values, df_1[['x','y','z']].values)



            d1[d1>target_distance] = 0
            d2 = 1-d1/target_distance
            d2[d2==1] = 0
                
                
            coo_edges = coo_matrix(d2)
            # df_distance.values
            
            
            node_labels = df_1['resName']
            node_labels_1 = onehot_encoder.transform(label_encoder.transform(node_labels).reshape(-1,1))
            
            
            
            
            x = torch.from_numpy(node_labels_1).type(torch.FloatTensor)

            y = torch.tensor([[0]], dtype=torch.float64)

            edge_index = torch.tensor(np.array([coo_edges.row,coo_edges.col]), dtype=torch.int64)
            
            edge_weight = torch.tensor(np.array(coo_edges.data), dtype=torch.float32)
            data = torch_geometric.data.Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight,name=pdb_name_chain)



            data_list += [data]

    loader = DataLoader(data_list, batch_size=bs)

    return loader


def grad_cam(final_conv_acts, final_conv_grads):
    # print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map

def test(loader):
    model.eval()
    l_pred, l_true = [],[]
    correct = 0
    for local_batch in loader:
        pred = (torch.sigmoid(model(local_batch.to(device))).squeeze(dim=1)>0.5)*1
        correct += pred.eq(local_batch.y).sum().item()

        l_pred += pred.tolist()
        l_true += local_batch.y.tolist()    
    return l_pred


if 1:

    datetime_st = datetime.datetime.now()


    parser = ArgumentParser()
    parser.add_argument("-s", "--task_name", dest="task_name",default='example')
    parser.add_argument("-t", "--target_type", dest="type",default='pdb',help='pdb or af2')
    

    if '-f' in sys.argv or '--ip=127.0.0.1' in sys.argv:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    args.target_source = './data/'+args.task_name


    test_loader = load_pdb_data(args,32)
    
    df_threshold = pd.read_excel('./excel/model_threshold.xlsx')
    df_threshold['f1_score'].mean()
    df_threshold = df_threshold.drop('Unnamed: 0',axis=1)
    l_ec_score_available = df_threshold['ec_class'].to_list()


    d_threshold = {}
    for index,row in df_threshold.iterrows():
        d_threshold[row['ec_class']] = row['thresholds']


    # l_path_target_2 = [h for h in l_path_target_1 if h in l_ec_score_available]
    l_path_target_2 = l_ec_score_available
    l_path_target_2.sort()
    print(len(l_path_target_2))


    datetime_s1 = datetime.datetime.now()

    d_result = {}
    # d_score = {}
    # l_path_target_1 = [l_path_target_1[1]]


    d_model_datetime = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_ec = l_path_target_2[0]
    for target_ec in l_path_target_2:
        

        model = torch.load('./models/ec_'+target_ec+'.pt', map_location=device)
        # model1 = torch.nn.DataParallel(model, device_ids = [0,1])
        model.eval()


        l_pred, l_true, l_name = [],[],[]
        correct = 0
        for local_batch in test_loader:
            # break

            name = local_batch.name
            pred = torch.sigmoid(model(local_batch.to(device)))

            l_pred += pred.tolist()
            l_name += local_batch.name

        d_result['EC '+target_ec+' >= thresholds'] = {}
        d_result['EC '+target_ec+' predicted prob.'] = {}


        # with open(path_load_evaluation_1+'score.txt') as f1:
        #     d_score = json.load(f1)
        model_score_info = df_threshold[df_threshold['ec_class']==target_ec].reset_index(drop=True).T
        thresholds = model_score_info.loc['thresholds'][0]
        d_score = model_score_info.to_dict()[0]
        for k,v in d_score.items():
            d_result['EC '+target_ec+' >= thresholds']['_'+k] = v
        for k,v in d_score.items():
            d_result['EC '+target_ec+' predicted prob.']['_'+k] = v

        # model1.__dict__

        for h in zip(l_name,l_pred):
            d_result['EC '+target_ec+' >= thresholds'][h[0]] = (h[1][0]>=thresholds)*1
            d_result['EC '+target_ec+' predicted prob.'][h[0]] = h[1][0]
            # d_result[dataset_name][h[0][0]] = h[1][0]
            
        torch.cuda.empty_cache()

    datetime_s2 = datetime.datetime.now()


    df_results = pd.DataFrame(d_result)
    os.makedirs('./results/'+args.task_name,exist_ok=True)
    df_results.to_excel('./results/'+args.task_name+'/prediction.xlsx')
