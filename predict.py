
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import math
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




        if args.target_type == 'af2':
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


def draw_prediction(d_config,weights,df_uniprot_sites, df_mcsa_data):
    DRAW_UNIPROT_SITES = 1


    def f_amino_acid(x):
        return d_amino_acid[x]



    def f_1(x):
        if x == 0:
            return '  0  '
        elif x%10==0:
            return str(x)
        else:
            return ''

    def f_valid_chain(x):
        if x['valid_chain'] == True and x['valid_chain_uniprot']==True:
            return 'B'
        elif x['valid_chain'] == True:
            return 'M'
        elif x['valid_chain_uniprot']==True:
            return 'U'
        else:
            return ''

    d_amino_acid = {'ALA':'A',
        'ARG':'R',
        'ASN':'N',
        'ASP':'D',
        'CYS':'C',
        'GLN':'Q',
        'GLU':'E',
        'GLY':'G',
        'HIS':'H',
        'ILE':'I',
        'LEU':'L',
        'LYS':'K',
        'MET':'M',
        'PHE':'F',
        'PRO':'P',
        'SER':'S',
        'THR':'T',
        'TRP':'W',
        'TYR':'Y',
        'VAL':'V',
        'SEC':'U',
        'PYL':'O',}



    
    l_colors = ['#ced4da','#fd6104','#CC0000']
    l_colors_1 = [l_colors[0]]*18+[l_colors[1]]+[l_colors[2]]

    
    cmap = (matplotlib.colors.ListedColormap(l_colors_1))
    cmap.set_under('#FFFFFF')



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

    pdb_name = d_config['pdb_name']
    pdb_name_s = pdb_name.split('_')[0]
    pdb_chain = pdb_name.split('_')[1:]


    path_pdb = d_config['path_pdb']
    path_output_img = d_config['path_output_s1']+d_config['pdb_name']+' '+d_config['ec']+'/'
    os.makedirs(path_output_img,exist_ok=True)

    if d_config['type'] == 'af2':
        df = pd.read_fwf(path_pdb+'/'+pdb_name_s.lower()+'.pdb',colspecs=[(h[0]-1,h[1]) for h in pdb_atom_format_field],header=None, dtype=str)
        df.columns = [h[2] for h in pdb_atom_format_field]
        if len(df[df['"ATOM"']=='MODEL']) > 1:
            df_model = df[df['"ATOM"']=='MODEL']
            df = df[:df_model.iloc[1].name]
    elif d_config['type'] == 'pdb':
        df = pd.read_fwf(path_pdb+'/'+pdb_name_s.lower()+'.pdb',colspecs=[(h[0]-1,h[1]) for h in pdb_atom_format_field],header=None, dtype=str)
        df.columns = [h[2] for h in pdb_atom_format_field]
        if len(df[df['"ATOM"']=='MODEL']) > 0:
            df_model = df[df['"ATOM"']=='MODEL']
            df = df[:df_model.iloc[1].name]



    pdb_dbref_format_field = [(1, 6, 0),
                                    (7,999,1)]

    df_full = pd.read_fwf(path_pdb+'/'+pdb_name_s.lower()+'.pdb',colspecs=[(h[0]-1,h[1]) for h in pdb_dbref_format_field],header=None, dtype=str)

    l_dbref = df_full[df_full[0] == 'DBREF'][1].tolist()
    d_dbref = {}
    for l_dbref_q in l_dbref:
        l_dbref_s = l_dbref_q.split()
        
        if l_dbref_s[4] == 'UNP':
            chain = l_dbref_s[1]
            d_dbref[chain] = {}
            d_dbref[chain]['pdb_st'] = int(l_dbref_s[2])
            d_dbref[chain]['pdb_ed'] = int(l_dbref_s[3])

            
            d_dbref[chain]['species'] = l_dbref_s[-3]
            d_dbref[chain]['aa_st'] = int(l_dbref_s[-2])
            d_dbref[chain]['aa_ed'] = int(l_dbref_s[-1])
            d_dbref[chain]['pos_diff'] = int(l_dbref_s[2]) - int(l_dbref_s[-2])

    for l_dbref_q in l_dbref:

        l_dbref_s = l_dbref_q.split()
        
    
        chain = l_dbref_s[1]
        if chain not in d_dbref:
            d_dbref[chain] = {}
            
            d_dbref[chain]['pdb_st'] = int(l_dbref_s[2])
            d_dbref[chain]['pdb_ed'] = int(l_dbref_s[3])

            
            d_dbref[chain]['species'] = l_dbref_s[-3]
            d_dbref[chain]['aa_st'] = int(l_dbref_s[-2])
            d_dbref[chain]['aa_ed'] = int(l_dbref_s[-1])
            d_dbref[chain]['pos_diff'] = int(l_dbref_s[2]) - int(l_dbref_s[-2])

    if len(l_dbref) == 0:
        l_dbref_1 = df_full[df_full[0] == 'DBREF1'][1].tolist()
        l_dbref_2 = df_full[df_full[0] == 'DBREF2'][1].tolist()
        for q1,q2 in zip(l_dbref_1,l_dbref_2):
            
            l_dbref_s_q1 = q1.split()
            l_dbref_s_q2 = q2.split()
            l_dbref_s = l_dbref_s_q1 + l_dbref_s_q2[2:]

            
            chain = l_dbref_s[1]
            d_dbref[chain] = {}
            
            d_dbref[chain]['pdb_st'] = int(l_dbref_s[2])
            d_dbref[chain]['pdb_ed'] = int(l_dbref_s[3])

            
            d_dbref[chain]['species'] = l_dbref_s[-3]
            d_dbref[chain]['aa_st'] = int(l_dbref_s[-2])
            d_dbref[chain]['aa_ed'] = int(l_dbref_s[-1])
            d_dbref[chain]['pos_diff'] = int(l_dbref_s[2]) - int(l_dbref_s[-2])

    df_dbref = pd.DataFrame(d_dbref).T

    if d_config['task_name'][:7]=='deepfri':
        l_amino_acid = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','SEC','PYL','XAA','ASX','GLX','XLE']
    else:
        l_amino_acid = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',]
    
    df = df[(df['"ATOM"']=='ATOM')&(df['resName'].isin(l_amino_acid))&(df['chainID'].isin(pdb_chain))].reset_index().drop('index',axis=1)
    df['chainID_resSeq'] = df['chainID'] + df['resSeq'].str.zfill(5)
    
    df['x'] = df['x'].apply(float)
    df['y'] = df['y'].apply(float)
    df['z'] = df['z'].apply(float)
    df_1 = df.groupby('chainID_resSeq').agg({'resName':'first','name':'count','chainID':'first','resSeq':'first','x':'sum','y':'sum','z':'sum','tempFact':'first'})
    


    weights = minmax_scale(weights, feature_range=(0, 1), axis=0)
    
    df_1['weights'] = weights

    top_n = math.ceil(len(weights)*0.05)
    # weights[np.argpartition(weights,-top_n)][-top_n:]
    top_x = weights[np.argsort(weights)[-top_n:]][0]
    top_x1 = max(top_x,1e-99)


    top_n = math.ceil(len(weights)*0.1)
    # weights[np.argpartition(weights,-top_n)][-top_n:]
    top_x = weights[np.argsort(weights)[-top_n:]][0]
    top_x2 = max(top_x,1e-99)



    def f_weights(x,top_x1,top_x2):
        if x >= top_x1:
            return 0.95
        elif x >= top_x2:
            return 0.9
        else:
            return 0


    df_1['weights'] = df_1['weights'].apply(lambda x: f_weights(x,top_x1,top_x2))
    
    
    df_1['resName_s'] = df_1['resName'].apply(lambda x:f_amino_acid(x))

    df_1['weights_+/-_1'] = df_1['weights'].rolling(2*1+1,center=True,min_periods=0).max()
    df_1['weights_+/-_2'] = df_1['weights'].rolling(2*2+1,center=True,min_periods=0).max()
    df_1['weights_+/-_3'] = df_1['weights'].rolling(2*3+1,center=True,min_periods=0).max()
    df_1['weights_+/-_4'] = df_1['weights'].rolling(2*4+1,center=True,min_periods=0).max()
    df_1['weights_+/-_5'] = df_1['weights'].rolling(2*5+1,center=True,min_periods=0).max()
    df_1['weights_+/-_6'] = df_1['weights'].rolling(2*6+1,center=True,min_periods=0).max()
    df_1['weights_+/-_7'] = df_1['weights'].rolling(2*7+1,center=True,min_periods=0).max()
    df_1['weights_+/-_8'] = df_1['weights'].rolling(2*8+1,center=True,min_periods=0).max()
    df_1['weights_+/-_9'] = df_1['weights'].rolling(2*9+1,center=True,min_periods=0).max()
    

    df_1['resSeq'] = df_1['resSeq'].apply(int)

    df_1 = df_1.merge(df_dbref,left_on='chainID',right_index=True)
    df_1['resSeq_uniprot'] = df_1['resSeq'].apply(int) - df_1['pos_diff']
    
    df_uniprot_sites_1 = df_uniprot_sites[df_uniprot_sites['pdb_id']==pdb_name_s.split('.')[0]].reset_index().drop('index',axis=1)
    df_uniprot_sites_1['pdb_chain_s'] = df_uniprot_sites_1['pdb_chain'].apply(lambda x:x.split('/'))
    df_uniprot_sites_1 = df_uniprot_sites_1.explode(['pdb_chain_s'])
    df_uniprot_sites_1 = df_uniprot_sites_1.explode(['sites_pos'])
    df_uniprot_sites_1['valid_chain_uniprot'] = True
    df_uniprot_sites_1 = df_uniprot_sites_1.drop_duplicates(subset=['pdb_id','pdb_chain_s','sites_pos'],keep='first')


    df_2 = df_1.merge(df_uniprot_sites_1,how='left',left_on=['chainID','resSeq_uniprot'],right_on=['pdb_chain_s','sites_pos'])
    

    df_uniprot_sites_data_1 = df_1.merge(df_uniprot_sites_1,left_on=['chainID','resSeq_uniprot'],right_on=['pdb_chain_s','sites_pos'])
    df_uniprot_sites_data_1['valid_chain'] = True


    if d_config['type'] == 'af2':
        df_mcsa_data_1 = df_mcsa_data[df_mcsa_data['uniprot_id']==pdb_name_s.split('-')[0].upper()].reset_index().drop('index',axis=1)
        df_mcsa_data_1 = df_mcsa_data_1.rename({'auth_resid/chebi id':'resid/chebi id auth'},axis=1)
    else:
        df_mcsa_data_1 = df_mcsa_data[df_mcsa_data['PDB']==pdb_name_s].reset_index().drop('index',axis=1)
    

    df_mcsa_data_2 = df_mcsa_data_1.merge(df_1,how='left',left_on=['chain/kegg compound','resid/chebi id auth'],right_on=['chainID','resSeq'])
    df_2 = df_2.merge(df_mcsa_data_1,how='left',left_on=['chainID','resSeq'],right_on=['chain/kegg compound','resid/chebi id auth'])
    df_2['sites_combined'] = df_2.apply(f_valid_chain,axis=1)

    df_2.to_excel(path_output_img+'/'+pdb_name+'_weights.xlsx')




    # Draw
    for chain_id_s in set(df_1['chainID'].to_list()):
        df_1_s = df_1[df_1['chainID'] == chain_id_s].copy()


        if d_config['type'] in ['pdb']:
            text = 'open '+pdb_name_s+'\n'
        if d_config['type']=='af2':
            text = 'alphafold fetch '+pdb_name_s.split('-')[0]+'\n'

        text += 'set bgColor white\n'
        text += 'hide atoms\n'
        
        text += "2dlab text 'Grad-CAM Weights (Top n%)' size 20 xpos .53 ypos .15\n"
        text += "key  "+l_colors[2]+":  "+l_colors[1]+": "+l_colors[0]+": pos .51,.12 size .32,.02 colorTreatment distinct  numericLabelSpacing  proportional \n"
        text += "2dlab text '0%' size 18 x 0.50 y .08; 2dlab text '5%' size 18 x .61 y .08; 2dlab text '10%' size 18 x .71 y .08; ; 2dlab text '100%' size 18 x .81 y .08; 2dlab text '...' size 18 x .77 y .09;\n"

        for chain_id_s1  in set(df_1['chainID'].to_list()):
            if chain_id_s1 != chain_id_s:
                text += 'delete /'+chain_id_s1+'\n'

        
        for weights_s in set(set(df_1_s['weights'])):
            text_aa = ''
            for index,row in df_1_s[df_1_s['weights'] == weights_s].iterrows():
                text_aa += '/'+row['chainID']+':'+str(row['resSeq'])+' '
            text += 'color '+text_aa+' '+matplotlib.colors.to_hex(cmap(weights_s))+'\n'
            
        for index,row in df_mcsa_data_2.iterrows():
            
            text += 'show /'+str(row['chainID'])+':'+str(row['resSeq'])+' atoms\n'
            if row['weights'] >= 0.5:
                text += 'shape sphere radius 2 center /'+str(row['chainID'])+':'+str(row['resSeq'])+' color #F8CBAD60\n'
            else:
                text += 'shape sphere radius 2 center /'+str(row['chainID'])+':'+str(row['resSeq'])+' color #d7e5f860\n'


        with open(path_output_img+pdb_name_s+'_'+chain_id_s+'_mcsa.cxc','w') as f1:
            f1.writelines(text)







    for chain_id_s in set(df_1['chainID'].to_list()):
        df_uniprot_sites_data_1_s = df_uniprot_sites_data_1[df_uniprot_sites_data_1['chainID']==chain_id_s].copy()
            
        df_1_s = df_1[df_1['chainID'] == chain_id_s].copy()


        if d_config['type'] in ['pdb']:
            text = 'open '+pdb_name_s+'\n'
        if d_config['type']=='af2':
            text = 'alphafold fetch '+pdb_name_s.split('-')[0]+'\n'


        text += 'set bgColor white\n'
        text += 'hide atoms\n'
        
        text += "2dlab text 'Grad-CAM Weights (Top n%)' size 20 xpos .53 ypos .15\n"
        text += "key  "+l_colors[2]+":  "+l_colors[1]+": "+l_colors[0]+": pos .51,.12 size .32,.02 colorTreatment distinct  numericLabelSpacing  proportional \n"
        text += "2dlab text '0%' size 18 x 0.50 y .08; 2dlab text '5%' size 18 x .61 y .08; 2dlab text '10%' size 18 x .71 y .08; ; 2dlab text '100%' size 18 x .81 y .08; 2dlab text '...' size 18 x .77 y .09;\n"



        for chain_id_s1  in set(df_1['chainID'].to_list()):
            if chain_id_s1 != chain_id_s:
                text += 'delete /'+chain_id_s1+'\n'

        
        for weights_s in set(set(df_1_s['weights'])):
            text_aa = ''
            for index,row in df_1_s[df_1_s['weights'] == weights_s].iterrows():
                text_aa += '/'+row['chainID']+':'+str(row['resSeq'])+' '
            text += 'color '+text_aa+' '+matplotlib.colors.to_hex(cmap(weights_s))+'\n'

        for index,row in df_uniprot_sites_data_1_s.iterrows():
            text += 'show /'+str(row['chainID'])+':'+str(row['resSeq'])+' atoms\n'
            if row['weights'] >= 0.5:
                text += 'shape sphere radius 2 center /'+str(row['chainID'])+':'+str(row['resSeq'])+' color #F8CBAD60\n'
            else:
                text += 'shape sphere radius 2 center /'+str(row['chainID'])+':'+str(row['resSeq'])+' color #d7e5f860\n'



        with open(path_output_img+pdb_name_s+'_'+chain_id_s+'_uniprot_sites.cxc','w') as f1:
            f1.writelines(text)
    
    
    for chain_id_s in set(df_2['chainID'].to_list()):
        df_1_s = df_2[df_2['chainID']==chain_id_s].copy()
        df_1_s['n'] = df_1_s['resSeq_uniprot'].apply(f_1)
        if df_1_s.iloc[0]['resSeq_uniprot']%10 < 9:
            df_1_s.at[0,'n'] = str(df_1_s.iloc[0]['resSeq_uniprot'])
        df_1_s_1 = df_1_s
        amino_acid_len_s_st_min = df_1_s_1.iloc[0]['resSeq_uniprot']
        amino_acid_len_s_st_max = df_1_s_1.iloc[-1]['resSeq_uniprot']
        amino_acid_len_s_0 = int(np.floor(amino_acid_len_s_st_min/100)) 
    
        amino_acid_len =  int(np.ceil((amino_acid_len_s_st_max)/100))-int(np.floor(amino_acid_len_s_st_min/100))
        
        fig, ax1 = plt.subplots(nrows=amino_acid_len, figsize=(12, amino_acid_len))

        for amino_acid_len_s in range(int(np.floor(amino_acid_len_s_st_min/100)),int(np.ceil((amino_acid_len_s_st_max)/100))):
        
            amino_acid_len_s_st = amino_acid_len_s*100
            amino_acid_len_s_ed = (amino_acid_len_s+1)*100

            amino_acid_len_ax = amino_acid_len_s - amino_acid_len_s_0
            if amino_acid_len == 1:
                ax = ax1
            else:
                ax = ax1[amino_acid_len_ax]
            
            df_1_s_2 = df_1_s_1[(df_1_s_1['resSeq_uniprot']>amino_acid_len_s_st) & (df_1_s_1['resSeq_uniprot']<=amino_acid_len_s_ed)]
            
            df_ref = pd.DataFrame([range(amino_acid_len_s_st+1,amino_acid_len_s_ed+1)]).T
            
            df_ref = df_ref.rename({0:'resSeq_ref'},axis=1)
            df_1_s_3 = df_ref.merge(df_1_s_2,how='left',left_on='resSeq_ref',right_on='resSeq_uniprot')
            df_1_s_3['weights'] = df_1_s_3['weights'].fillna(-1)
            weights_s = df_1_s_3['weights'].to_numpy()+0.0001
            
            extent = [0,100,0,0.6]

            
            
            ax.imshow(weights_s[np.newaxis,:],cmap=cmap,norm=matplotlib.colors.Normalize(vmin=0, vmax=1), aspect="auto", extent=extent)


            ax.set_yticks([])
            
            x_mcsa = df_1_s_3['sites_combined'].to_list()

            for h1 in range(len(x_mcsa)):
                if x_mcsa[h1] in ['M','B']:
                    ax.add_patch(matplotlib.patches.Rectangle((h1,0), 1, extent[3], hatch='///', fill=False, snap=False))
                if DRAW_UNIPROT_SITES == 1:
                    if x_mcsa[h1] in ['M','U']:
                        ax.add_patch(matplotlib.patches.Rectangle((h1,0), 1, extent[3], hatch='\\\\\\', fill=False, snap=False))

            x_label_major = df_1_s_2[df_1_s_2['n']!='']['n'].to_list()
            x_label_minor = df_1_s_2['resSeq_uniprot'].tolist()

            x_label_major = df_1_s_2[df_1_s_2['n']!='']['n'].to_list()
            x_label_minor = df_1_s_2['resSeq_uniprot'].tolist()

            ax.set_xticks([(int(a)-amino_acid_len_s_st)/10*10-0.5 for a in x_label_major])
            ax.set_xticks([a-amino_acid_len_s_st-0.5 for a in x_label_minor], minor = True)
            ax.set_xticklabels(x_label_major,fontsize=14)

            ax.tick_params(which='both', width=1.5)
            ax.tick_params(which='major', length=6)
            ax.tick_params(which='minor', length=4)
            
            ax.set_xlim(0,100)
            

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        fig.tight_layout()
        plt.savefig(path_output_img+pdb_name_s+'_'+chain_id_s+'.svg')
        plt.close()


if 1:

    datetime_st = datetime.datetime.now()


    parser = ArgumentParser()
    parser.add_argument("-s", "--task_name", dest="task_name",default='Task 1')
    parser.add_argument("-t", "--target_type", dest="target_type",default='pdb',help='pdb or af2')
    

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
    for target_ec in tqdm(l_path_target_2):
        

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


if 1:
    with open('./excel/uniprot_sites.pkl','rb') as f1:
        df_uniprot_sites = pickle.load(f1)


    df_uniprot_sites['sites'] = df_uniprot_sites['sites_metal']+df_uniprot_sites['sites_active'] +df_uniprot_sites['sites_binding']

    df_uniprot_sites['sites_pos'] = df_uniprot_sites['sites'].apply(lambda x:[re.sub("[^0-9]", "", h) for h in x.split('ExactPosition')])
    df_uniprot_sites['sites_pos'] = df_uniprot_sites['sites_pos'].apply(lambda x:[h for h in x if h != ''])
    df_uniprot_sites['sites_pos'] = df_uniprot_sites['sites_pos'].apply(lambda x:x[1::2])
    df_uniprot_sites['sites_pos'] = df_uniprot_sites['sites_pos'].apply(lambda x:[int(h) for h in x])
    df_uniprot_sites['sites_pos'] = df_uniprot_sites['sites_pos'].apply(lambda x:list(set(x)))


    df_uniprot_sites['sites_metal_pos'] = df_uniprot_sites['sites_metal'].apply(lambda x:[re.sub("[^0-9]", "", h) for h in x.split('ExactPosition')])
    df_uniprot_sites['sites_metal_pos'] = df_uniprot_sites['sites_metal_pos'].apply(lambda x:[h for h in x if h != ''])
    df_uniprot_sites['sites_metal_pos'] = df_uniprot_sites['sites_metal_pos'].apply(lambda x:x[1::2])

    df_uniprot_sites['sites_active_pos'] = df_uniprot_sites['sites_active'].apply(lambda x:[re.sub("[^0-9]", "", h) for h in x.split('ExactPosition')])
    df_uniprot_sites['sites_active_pos'] = df_uniprot_sites['sites_active_pos'].apply(lambda x:[h for h in x if h != ''])
    df_uniprot_sites['sites_active_pos'] = df_uniprot_sites['sites_active_pos'].apply(lambda x:x[1::2])

    df_uniprot_sites['sites_binding_pos'] = df_uniprot_sites['sites_binding'].apply(lambda x:[re.sub("[^0-9]", "", h) for h in x.split('ExactPosition')])
    df_uniprot_sites['sites_binding_pos'] = df_uniprot_sites['sites_binding_pos'].apply(lambda x:[h for h in x if h != ''])
    df_uniprot_sites['sites_binding_pos'] = df_uniprot_sites['sites_binding_pos'].apply(lambda x:x[1::2])


    if args.target_type == 'af2':
        df_mcsa_data = pd.read_excel('./excel/mcsa_data_u.xlsx')
        df_mcsa_data['resid/chebi id'] = df_mcsa_data['uniprot_resid'].apply(int)
        df_mcsa_data['uniprot_id'] = df_mcsa_data['uniprot_id'].apply(lambda x:x.lower())
        l_mcsa_data = set(df_mcsa_data['uniprot_id'].tolist())
        
    
    else:
        df_mcsa_data = pd.read_excel('./excel/mcsa_data_t.xlsx')
        df_mcsa_data['resid/chebi id'] = df_mcsa_data['resid/chebi id'].apply(int)
        df_mcsa_data['PDB'] = df_mcsa_data['PDB'].apply(lambda x:x.lower())
        
        l_mcsa_data = set([tuple(x) for x in df_mcsa_data[['PDB','chain/kegg compound']].to_numpy()])
    






test_loader = load_pdb_data(args,1)

datetime_s3 = datetime.datetime.now()

target_ec = l_path_target_2[0]
for target_ec in l_path_target_2:


    
    df_results_1 = df_results[['EC '+target_ec+' >= thresholds']]
    df_results_2 = df_results_1.loc[[h for h in df_results_1.index if h[0] != '_']]
    df_results_2 = df_results_2[df_results_2['EC '+target_ec+' >= thresholds']>0]
    if len(df_results_2)==0:
        continue
    
    target_pdb = df_results_2.index.tolist()

    d_pred_weights = {}
    
    print(target_ec)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


    model_score_info = df_threshold[df_threshold['ec_class']==target_ec].reset_index(drop=True).T
    thresholds = model_score_info.loc['thresholds'][0]
    

    model = torch.load('./models/ec_'+target_ec+'.pt', map_location=device)
    model.eval()
    
    
    if model.final_conv_grads.mean() == 0:
        print('Zero gradient mean')
        continue
    
    counter = 0
    

    for target_loader in [test_loader]:
        # break
        for local_batch in tqdm(target_loader):
            # break

            
            if local_batch.name[0] in target_pdb:
                
                

                model.train()
                local_batch = local_batch.to(device)
                

                out = model(local_batch.to(device))





                mol_num = local_batch.name
                

                final_conv_acts = model.final_conv_acts
                final_conv_grads = model.final_conv_grads
                grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)[:local_batch.x.detach().shape[0]]
                scaled_grad_cam_weights = MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )
                
                weights = scaled_grad_cam_weights


                pdb_name = local_batch.name[0]




                d_config = {'type':args.target_type}
                d_config['task_name'] = args.task_name
                d_config['path_pdb'] = './data/'+args.task_name+'/'
                d_config['path_output_s1'] = './results/'+args.task_name+'/'
                d_config['ec'] = target_ec
                d_config['pdb_name'] = pdb_name



                draw_prediction(d_config,weights,df_uniprot_sites, df_mcsa_data)






