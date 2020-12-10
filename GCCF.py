'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCCF(nn.Module):
    def __init__(self, n_user, n_item, args, uv_adj, vu_adj, d_i_train, d_j_train):
        super(GCCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.reg

        self.d_i_train = torch.FloatTensor(d_i_train).view(-1, 1).cuda()
        self.d_j_train = torch.FloatTensor(d_j_train).view(-1, 1).cuda()

        self.d_i_train = self.d_i_train.expand(-1, self.emb_size)
        self.d_j_train = self.d_j_train.expand(-1, self.emb_size)

        self.uv_adj = uv_adj
        self.vu_adj = vu_adj

        self.embed_user = nn.Embedding(n_user, self.emb_size)
        self.embed_item = nn.Embedding(n_item, self.emb_size) 
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01) 

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    
    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss#, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def getScores(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        if neg_items is None:
            return pos_scores
        else:
            neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
            return pos_scores, neg_scores
    

    # def forward(self, sparse_norm_adj, users, pos_items, neg_items, drop_flag=True):
    def forward(self, user, item_i, item_j):
        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight  

        gcn1_users_embedding = (torch.sparse.mm(self.uv_adj, items_embedding) + users_embedding.mul(self.d_i_train))
        gcn1_items_embedding = (torch.sparse.mm(self.vu_adj, users_embedding) + items_embedding.mul(self.d_j_train))
        
        # gcn2_users_embedding = (torch.sparse.mm(self.uv_adj, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train))
        # gcn2_items_embedding = (torch.sparse.mm(self.vu_adj, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train))
          
        # gcn3_users_embedding = (torch.sparse.mm(self.uv_adj, gcn2_items_embedding) + gcn2_users_embedding.mul(self.d_i_train))
        # gcn3_items_embedding = (torch.sparse.mm(self.vu_adj, gcn2_users_embedding) + gcn2_items_embedding.mul(self.d_j_train))
        
        gcn_users_embedding= torch.cat((users_embedding, gcn1_users_embedding), -1)
        gcn_items_embedding= torch.cat((items_embedding, gcn1_items_embedding), -1)

        # gcn_users_embedding= torch.cat((users_embedding, gcn1_users_embedding, gcn2_users_embedding, gcn3_users_embedding), -1)
        # gcn_items_embedding= torch.cat((items_embedding, gcn1_items_embedding, gcn2_items_embedding, gcn3_items_embedding), -1)
        

        user = F.embedding(user, gcn_users_embedding)
        item_i = F.embedding(item_i, gcn_items_embedding)
        item_j = F.embedding(item_j, gcn_items_embedding)  

        return user, item_i, item_j


    def getEmbeds(self):
        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight  

        gcn1_users_embedding = (torch.sparse.mm(self.uv_adj, items_embedding) + users_embedding.mul(self.d_i_train))
        gcn1_items_embedding = (torch.sparse.mm(self.vu_adj, users_embedding) + items_embedding.mul(self.d_j_train))
        
        
        gcn_users_embedding= torch.cat((users_embedding, gcn1_users_embedding), -1)
        gcn_items_embedding= torch.cat((items_embedding, gcn1_items_embedding), -1)

        return gcn_users_embedding, gcn_items_embedding
        
