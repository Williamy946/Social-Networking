import sklearn
import pandas as pd
import numpy as np
import random
import torch
import collections
import logging
import  pickle
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.utils import subgraph
from torch import nn
import torch.nn.functional as F
from GCN import GNNModel
import os
import sys
#from se_data_process import load_data_valid, load_testdata

datasets_path = "../process_data/"

user_group_df = pd.read_csv(datasets_path + "user_group.csv", sep=',')
user_event_df = pd.read_csv(datasets_path + "user_event.csv", sep=',')
event_info_df = pd.read_csv(datasets_path + "event_info.csv", sep=',') # 55393
user_feature_df = pd.read_csv("../fixed_data/user_features.csv", sep='\t') # 73685
event_feature_df = pd.read_csv("../fixed_data/event_features.csv", sep='\t')
group_graph_df = pd.read_csv("../fixed_data/group_net.csv")
event_info_df["hold_member"] = event_info_df["hold_member"].fillna(method="ffill")
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(len(user_event_df))
user_user_df = pd.read_csv(datasets_path + "user_user_net.csv")
#user_user_df = user_user_df.drop(["Unnamed: 0"],axis=1)
com_network_list = user_user_df.values.tolist()

event_info_df["time"] = (event_info_df["time"]- event_info_df["time"].mean())/event_info_df["time"].std()
event_info_df["limit"].loc[event_info_df["limit"]>=100] = 0
event_info_df["limit"] = (event_info_df["limit"]- event_info_df["limit"].mean())/event_info_df["limit"].std()


def group_based_network(user_group_df):
    '''
    :return: user-user matrix in list form
    ex: undirected user network
    '''

    def complete_network_generator(user_df,network_list):
        proportion = 0.08
        for u in user_df:
            for v in user_df:
                if u != v:
                    if np.random.rand() < proportion:
                        network_list.append([u,v,1])
        return

    network_list = []
    for i in sorted(user_group_df["group"].unique()):
        group_user = user_group_df.loc[user_group_df['group'] == i]['uid']
        #group_user = np.random.choice(group_user,int(0.2*len(group_user)))
        complete_network_generator(group_user,network_list)

    return network_list

def event_based_network(user_event_df):
    
    def complete_network_generator(user_df,network_dict):
        proportion = 0.5
        for u in user_df:
            for v in user_df:
                if u != v:
                    if u not in network_dict.keys():
                        network_dict[u] = {v:1}
                    else:
                        if v not in network_dict[u].keys():
                            if np.random.rand() < proportion:
                                network_dict[u][v] = 1
                        else:
                            network_dict[u][v] += 1
        return

    network_dict = {}
    network_list = []
    for i in sorted(user_event_df["event"].unique()):
        event_user = user_event_df.loc[user_event_df["event"] == i]['uid']
        #event_user = np.random.choice(event_user,int(0.1*len(event_user)))
        complete_network_generator(event_user,network_dict)
    max_num = 0
    for key in network_dict.keys():
        for v in network_dict[key].keys():
            if network_dict[key][v] > max_num:
                max_num = network_dict[key][v]
            
    for key in network_dict.keys():
        for v in network_dict[key].keys():
            network_list.append([key,v,float(network_dict[key][v])/float(max_num)])

    return network_list

'''
tag = 0
total = 0
for i in sorted(user_event_df["event"].unique()):
    event_user = user_event_df.loc[user_event_df["event"] == i]['uid']
    total += len(event_user)
    event_group = event_info_df.loc[event_info_df["event"] == i]["group"]
    group_user = user_group_df.loc[user_group_df['group'] == event_group.values[0]]['uid']
    for u in event_user.values:
        if u not in group_user.values:
            tag += 1
'''
print("generate graph")
#if not os.path.exists(datasets_path + "user_user.csv"):
group_network_list = group_based_network(user_group_df)

network_list = []

com_user_user_dict = {}
for n in com_network_list:
    if n[0] not in com_user_user_dict.keys():
        com_user_user_dict[n[0]] = {n[1]:n[2]}
    else:
        com_user_user_dict[n[0]][n[1]] = n[2]

group_user_user_dict = {}

for n in group_network_list:
    if n[0] not in group_user_user_dict.keys():
        group_user_user_dict[n[0]] = {n[1]:n[2]}
    else:
        group_user_user_dict[n[0]][n[1]] = n[2]

user_user_dict = {}
for u in group_user_user_dict.keys():
    for v in group_user_user_dict[u].keys():
        if u in com_user_user_dict.keys():
            if v in com_user_user_dict[u].keys():
                network_list.append([u,v,com_user_user_dict[u][v]])
            else:
                network_list.append([u, v, 0.5])

for u in com_user_user_dict.keys():
    if u not in group_user_user_dict.keys():
        for v in com_user_user_dict[u].keys():
            network_list.append([u,v,com_user_user_dict[u][v]])
    else:
        for v in com_user_user_dict[u].keys():
            if v not in group_user_user_dict[u].keys():
                network_list.append([u, v, com_user_user_dict[u][v]])

for n in network_list:
    if n[0] not in user_user_dict.keys():
        user_user_dict[n[0]] = {n[1]:n[2]}
    else:
        user_user_dict[n[0]][n[1]] = n[2]

user_user_matrix = pd.DataFrame(network_list,columns=["user1","user2","score"])
user_user_matrix.to_csv(datasets_path + "user_user.csv", sep='\t')

print("generate train/test data")
## generate train/test data
'''
user_event_list = []
willingness_type = ["yes","no","may"]
for user in user_feature_df['uid']:
    for i in range(len(willingness_type)):
        if len(user_feature_df.loc[user][willingness_type[i]+"_event"]) > 2:
            event_list = list(map(int, user_feature_df.loc[user][willingness_type[i]+'_event'][1:-1].split(',')))
            cur_user_event_list = [[user,j,i] for j in event_list]
            user_event_list += (cur_user_event_list)

user_event_list = sorted(user_event_list, key=lambda x:x[1])
'''

events = user_event_df["event"].unique()
n_event = max(events)
print(n_event)
event_train = np.random.choice(user_event_df["event"].unique(),int(0.8*len(events)),replace=False)
event_test = np.setdiff1d(events,event_train)

event_tag = np.array([0 for i in range(n_event+1)])
event_tag[event_train] = 1

train_event_user = []
test_event_user = []

event_index = np.array(range(len(user_event_df)))
np.random.shuffle(event_index)

datanum = 2000000
yes_num = 0
no_num = 0

for n in event_index:
    event_user_pair = user_event_df.loc[n]
    event_info = event_info_df[event_info_df["event"] == event_user_pair["event"]]
    #event_user_pair = user_event_df[n:n+1]
    #event_info = event_info_df[event_info_df["event"] == event_user_pair["event"].values[0]]
    #if (yes_num >= datanum/2 and no_num >= datanum/2):
    #    break
    if event_tag[int(event_user_pair["event"])] == 1: # train data
        if (event_user_pair["answer"] == 2 and yes_num < datanum/2) or (event_user_pair["answer"] == 0 and no_num < datanum/2) or event_user_pair["answer"] == 1:
            train_event_user.append([event_user_pair["uid"],
                             event_user_pair["event"], event_user_pair["answer"], event_info["hold_group"].values,
                             event_info["hold_member"].values, event_info["time"].values, event_info["limit"].values])
            if event_user_pair["answer"] == 2:
                yes_num += 1
            if event_user_pair["answer"] == 0:
                no_num += 1
    if event_tag[int(event_user_pair["event"])] == 0: # test data
        test_event_user.append([event_user_pair["uid"],
            event_user_pair["event"], event_user_pair["answer"], event_info["hold_group"].values,
            event_info["hold_member"].values, event_info["time"].values, event_info["limit"].values])

## batch generation
batch_size = 1000

np.random.shuffle(np.array(train_event_user))
batch_train = [train_event_user[i*batch_size:(i+1)*batch_size] for i in range(int(len(train_event_user)/batch_size))]



## Network construction



user_edge_index = torch.tensor([[u[0] for u in network_list],
                          [u[1] for u in network_list]], dtype=torch.long)

user_edge_weight = torch.tensor([u[2] for u in network_list], dtype=torch.float)

group_graph = group_graph_df.values.tolist()

group_edge_index = torch.tensor([[u[0] for u in group_graph],
                          [u[1] for u in group_graph]], dtype=torch.long)

group_edge_weight = torch.tensor([u[2] for u in group_graph], dtype=torch.float)


x = sorted(list(set([u[0] for u in network_list] + [u[1] for u in network_list])))

x = torch.tensor(x,dtype=torch.long)

data = Data(x=x, user_edge_index = user_edge_index, user_edge_weight = user_edge_weight,
            group_edge_index = group_edge_index, group_edge_weight=group_edge_weight)

#data_loader = DataLoader(data,batch_size=100,shuffle=True)

lr = 2e-3
l2 = 5e-3
lr_dc_step = 5
lr_dc_step2 = 20
lr_dc = 0.5
hidden_size = 100
model = GNNModel(n_events=55400, n_user= 73700, n_group=50000 , hidden_size=hidden_size, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_dc_step, lr_dc_step2], gamma=lr_dc)
epoch_num = 50
ans_trans = [0.0,0.5,1.0]
max_precision = 0.0
cor_loss = 0.0
loss = 0.0


test_event_host_list = [pair[4] for pair in test_event_user]
test_user_list = [pair[0] for pair in test_event_user]
test_users_host_weight_list = [0 for i in range(len(test_user_list))]

for ind in range(len(test_event_host_list)):
    for h in test_event_host_list[ind]:
        if h in user_user_dict.keys():
            if test_user_list[ind] in user_user_dict[h].keys():
                test_users_host_weight_list[ind] += user_user_dict[h][test_user_list[ind]]

test_users = torch.tensor([pair[0] for pair in test_event_user],dtype=torch.long).to(device)
test_events = torch.tensor([pair[1] for pair in test_event_user],dtype=torch.long).to(device)
test_ans = torch.tensor([ans_trans[int(pair[2])] for pair in test_event_user],dtype=torch.float).view(-1,1).to(device)
test_events_group = torch.tensor([np.random.choice(pair[3],1) for pair in test_event_user],dtype=torch.long).to(device)
test_events_users = torch.tensor([np.random.choice(pair[4],1) for pair in test_event_user],dtype=torch.long).to(device)
test_events_assist = torch.tensor([[np.array(pair[5])[0],np.array(pair[6])[0]] for pair in test_event_user], dtype=torch.float).to(device)
test_user_host_weight = torch.tensor(test_users_host_weight_list,dtype=torch.float).reshape(len(test_users),1).to(device)

# local network
assist_embedding = nn.Embedding(2,hidden_size).to(device)
layer1 = nn.Linear(3*hidden_size, hidden_size).to(device)
layer2 = nn.Linear(2*hidden_size, 1).to(device)
tl_layer1 = nn.Linear(2, hidden_size).to(device)
tl_layer2 = nn.Linear(hidden_size, hidden_size).to(device)
for train_epoch in range(epoch_num):
    #for i, batch in enumerate(data_loader):
    model.train()
    mean_loss = 0.0
    print("Epoch: " + str(train_epoch))
    batchnum = 0
    for batch in batch_train:
        batchnum += 1
        optimizer.zero_grad()
        user_embedding, group_embedding = model(data.to(device))

        users = torch.tensor([pair[0] for pair in batch],dtype=torch.long).to(device)
        events = torch.tensor([pair[1] for pair in batch],dtype=torch.long).to(device)
        events_group = torch.tensor([np.random.choice(pair[3],1) for pair in batch],dtype=torch.long).to(device)
        events_users = torch.tensor([np.random.choice(pair[4],1) for pair in batch],dtype=torch.long).to(device)
        events_assist = torch.tensor([[np.array(pair[5])[0],np.array(pair[6])[0]] for pair in batch], dtype=torch.float).to(device)

        # social cascade
        event_host_list = [pair[4] for pair in batch]
        user_list = [pair[0] for pair in batch]
        users_host_weight_list = [0 for i in range(batch_size)]
        for ind in range(batch_size):
            for h in event_host_list[ind]:
                if h in user_user_dict.keys():
                    if user_list[ind] in user_user_dict[h].keys():
                        users_host_weight_list[ind] += user_user_dict[h][user_list[ind]]

        user_host_weight = torch.tensor(users_host_weight_list, dtype=torch.float).reshape(batch_size,1).to(device)
        #cascade

        ans = torch.tensor([ans_trans[int(pair[2])]for pair in batch],dtype=torch.float).view(-1,1).to(device)

        assist_event_emb = (F.relu(tl_layer1(events_assist)))#torch.matmul(events_assist, assist_embedding.weight)

        batch_user_emb = user_embedding[users]#F.tanh(layer2(torch.cat((user_embedding[users],group_embedding[events_group].reshape(batch_size,hidden_size)),dim=1)))
        batch_event_emb = F.tanh(layer1(torch.cat((group_embedding[events_group].reshape(batch_size,hidden_size),
                          user_embedding[events_users].reshape(batch_size,hidden_size),assist_event_emb), dim=1)))
        #batch_event_emb = user_embedding[events_users]#layer2(batch_event_emb)
        score = (torch.bmm(batch_user_emb.reshape(batch_size,1,hidden_size),
                          batch_event_emb.reshape(batch_size,hidden_size,1))).reshape(batch_size,1).to(device)
        #score = layer2(torch.cat((batch_user_emb, batch_event_emb),dim=1))
        #score = F.cosine_similarity(batch_user_emb,batch_event_emb,dim=1).reshape(batch_size,1).to(device)
        # cascade
        #score = score + 0.3*user_host_weight
 
        loss = model.loss_function(score,ans)

        loss.backward()
        optimizer.step()

        mean_loss += loss/len(batch_train)

        dis_ans = torch.tensor([pair[2] for pair in batch],dtype=torch.float)
        tmp_list = [0 if s < 0.5 else s for s in score.reshape(batch_size).tolist()]
        #tmp_list = [2 if s > 0.505 else s for s in tmp_list]
        dis_target = torch.tensor([2 if s >= 0.5 else s for s in tmp_list])#torch.tensor([1 if 495 <= s <= 0.505 else s for s in tmp_list])
        
        corr = 0
        for i in range(batch_size):
            if dis_ans[i] == dis_target[i]:
                corr += 1
        if batchnum%100 == 0:
            print("Precision: " + str(corr/batch_size))
            print("Loss: " + str(loss))

    scheduler.step()
    print("Mean loss: " + str(mean_loss))
    print()

    model.eval()
    test_size = len(test_users)
    test_user_emb = user_embedding[test_users]#F.tanh(layer2(torch.cat((user_embedding[test_users],group_embedding[test_events_group].reshape(test_size,hidden_size)),dim=1)))#user_embedding[test_users]
    #test_size = len(test_users)
    assist_event_emb = (F.relu(tl_layer1(test_events_assist)))#torch.matmul(test_events_assist, assist_embedding.weight)
      
    test_event_emb = F.tanh(layer1(torch.cat((group_embedding[test_events_group].reshape(test_size,hidden_size),
                     user_embedding[test_events_users].reshape(test_size,hidden_size),assist_event_emb),dim=1)))
    #test_event_emb = user_embedding[test_events_users]#layer2(test_event_emb)
    score = (torch.bmm(test_user_emb.reshape(test_size,1,hidden_size),
                      test_event_emb.reshape(test_size,hidden_size,1))).reshape(test_size,1).to(device)
    #score = layer2(torch.cat((test_user_emb, test_event_emb),dim=1))
    # Cascade
    #score = score + 0.3*test_user_host_weight
    #score = (F.cosine_similarity(test_user_emb,test_event_emb,dim=1).reshape(test_size,1).to(device))
    dis_ans = torch.tensor([pair[2] for pair in test_event_user], dtype=torch.float)
    tmp_list = [0 if s < 0.5 else s for s in score.reshape(test_size).tolist()]
    #tmp_list = [2 if s > 0.505 else s for s in tmp_list]
    dis_target = torch.tensor([2 if s >= 0.5 else s for s in tmp_list])#torch.tensor([1 if 495 <= s <= 0.505 else s for s in tmp_list])
    corr = 0
    yes_num = 0
    no_num = 0
    may_num = 0
    yes_TP, yes_FP, yes_TN, yes_FN = 0, 0, 0, 0
    no_TP, no_FP, no_TN, no_FN = 0, 0, 0, 0
    may_TP, may_FP, may_TN, may_FN = 0, 0, 0, 0
    yes_pre, yes_recall, yes_f1 = 0.0, 0.0, 0.0
    no_pre, no_recall, no_f1 = 0.0, 0.0, 0.0
    may_pre, may_recall, may_f1 = 0.0, 0.0, 0.0 
 
    for i in range(test_size):
        if dis_ans[i] == dis_target[i]:
            corr += 1
        if (dis_ans[i] == 2) and (dis_target[i] == 2):
            yes_TP += 1
            yes_num += 1
        if (dis_ans[i] == 2) and (dis_target[i] != 2):
            yes_FN += 1
        if (dis_ans[i] != 2) and (dis_target[i] == 2):
            yes_FP += 1
            yes_num += 1
        if (dis_ans[i] != 2) and (dis_target[i] != 2):
            yes_TN += 1

        if (dis_ans[i] == 0) and (dis_target[i] == 0):
            no_TP += 1
            no_num += 1
        if (dis_ans[i] == 0) and (dis_target[i] != 0):
            no_FN += 1
        if (dis_ans[i] != 0) and (dis_target[i] == 0):
            no_FP += 1
            no_num += 1
        if (dis_ans[i] != 0) and (dis_target[i] != 0):
            no_TN += 1

        if (dis_ans[i] == 1) and (dis_target[i] == 1):
            may_TP += 1
            may_num += 1
        if (dis_ans[i] == 1) and (dis_target[i] != 1):
            may_FN += 1
        if (dis_ans[i] != 1) and (dis_target[i] == 1):
            may_FP += 1
            may_num += 1
        if (dis_ans[i] != 1) and (dis_target[i] != 1):
            may_TN += 1
    
    yes_pre = yes_TP/(yes_TP + yes_FP+1)
    yes_recall = yes_TP/(yes_TP + yes_FN+1)
    yes_f1 = 2*yes_pre*yes_recall/(yes_pre + yes_recall+0.0001)
    
    no_pre = no_TP/(no_TP + no_FP+1)
    no_recall = no_TP/(no_TP + no_FN+1)
    no_f1 = 2*no_pre*no_recall/(no_pre + no_recall+0.0001)

    may_pre = may_TP/(may_TP + may_FP+1)
    may_recall = may_TP/(may_TP + may_FN+1)
    may_f1 = 2*may_pre*may_recall/(may_pre + may_recall+0.0001)

    total_pre = (yes_pre*yes_num+no_pre*no_num+may_pre*may_num)/test_size
    total_recall = (yes_recall*yes_num+no_recall*no_num+may_recall*may_num)/test_size
    total_f1 = (yes_f1*yes_num+no_f1*no_num+may_f1*may_num)/test_size

    print("\t Precision\t recall \t f1 \t Num")
    print("yes/2\t "+str(yes_pre)+"\t"+str(yes_recall)+"\t"+str(yes_f1)+"\t"+str(yes_num))
    print("may/1\t "+str(may_pre)+"\t"+str(may_recall)+"\t"+str(may_f1)+"\t"+str(may_num))
    print("no/0\t "+str(no_pre)+"\t"+str(no_recall)+"\t"+str(no_f1)+"\t"+str(no_num))
    print("total\t "+str(total_pre)+"\t"+str(total_recall)+"\t"+str(total_f1)+"\t"+str(test_size))

    print("Precision: " + str(corr / test_size))
    
    loss = model.loss_function(score, test_ans)
    print("Test Loss: " + str(loss))
    if (corr / test_size) > max_precision:
        max_precision = corr / test_size
        cor_loss = loss
print("Max Precision: " + str(max_precision))
print("Cor Loss: " + str(loss))


