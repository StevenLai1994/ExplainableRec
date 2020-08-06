import torch
from torch import nn
from modules.SemanticModule import SemanticModule
import torch.nn.functional as F
import numpy as np

class GATGraph(nn.Module):
    def __init__(self, opt, semantic_encoder):
        super(GATGraph, self).__init__()
        self.user_ne_items, self.item_ne_users, \
                self.user_ne_users, self.item_ne_items, self.user_item_review = opt["relations"]
        self.toolkits = opt["toolkits"]
        self.device = opt["device"]
        self.max_text_len = opt["max_text_len"]
        self.max_sent_len = opt["max_sent_len"]
        self.max_neighbors = opt["max_neighbors"]
        self.num_users = opt["num_users"]
        self.num_items = opt["num_items"]
        self.hidden_size = opt["hidden_size"]
        self.dropout = opt["dropout"]
        self.semantic_encoder = semantic_encoder
        self.node_encoder = NodeEncoder(self.num_users, self.num_items, self.hidden_size, self.dropout)
        self.att = Attention(self.hidden_size, self.dropout)
        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def get_user_text(self, uind):
        ne_items = self.user_ne_items[uind]
        ret = ""
        for iind in ne_items:
            edge = (uind, iind)
            ret += self.user_item_review[edge].rstrip('\n')
        return ret

    def get_item_text(self, iind):
        ne_users = self.item_ne_users[iind]
        ret = ""
        for uind in ne_users:
            edge = (uind, iind)
            ret += self.user_item_review[edge].rstrip('\n')
        return ret
    
    def get_review_text(self, ind1, ind2, ntype="user"):
        edge = (ind1, ind2) if ntype == "user" else (ind2, ind1)
        return self.user_item_review[edge].rstrip('\n')

    def texts2vectors(self, texts):
        text_vecs = []
        sents_vecs = []
        conceptnet_text_vecs = []
        every_sent_pad_lens = []
        for text in texts:
            text_vec = self.toolkits.text2vec(text)
            sents_vec, maxlen_of_every_sent = self.toolkits.text2sentences_vec(text)
            conceptnet_text_vec = self.toolkits.indvvec2indcvec(text_vec)
            text_vecs.append(text_vec)
            sents_vecs.append(sents_vec)
            conceptnet_text_vecs.append(conceptnet_text_vec)
            every_sent_pad_lens.append(maxlen_of_every_sent)

        text_lens = np.array([len(text) for text in text_vecs])
        sents_lens = np.array([len(sent) for sent in sents_vecs])

        text_pad_len = min(max(text_lens), self.max_text_len)
        sents_pad_len = min(max(sents_lens), self.max_sent_len)
        every_sent_pad_len = min(max(every_sent_pad_lens), self.max_text_len)
        text_pad_vecs = self.toolkits.pad2d(text_vecs, text_pad_len)
        sents_pad_vecs = self.toolkits.pad3d(sents_vecs, sents_pad_len, every_sent_pad_len)
        conceptnet_text_pad_vecs = self.toolkits.pad2d(conceptnet_text_vecs, text_pad_len)

        text_sorted_idx = np.argsort(text_lens)[::-1]
        sents_sorted_idx = np.argsort(sents_lens)[::-1]
        text_lens = text_lens[text_sorted_idx]
        text_lens[text_lens > self.max_text_len] = self.max_text_len
        sents_lens = sents_lens[sents_sorted_idx]
        sents_lens[sents_lens > self.max_sent_len] = self.max_sent_len
        text_pad_vecs = text_pad_vecs[text_sorted_idx]
        sents_pad_vecs = sents_pad_vecs[sents_sorted_idx]
        conceptnet_text_pad_vecs = conceptnet_text_pad_vecs[text_sorted_idx]

        return (torch.from_numpy(text_pad_vecs).long().to(self.device), torch.from_numpy(text_lens).long(), \
                torch.from_numpy(sents_pad_vecs).long().to(self.device), torch.from_numpy(sents_lens).long(), \
                    torch.from_numpy(conceptnet_text_pad_vecs).long().to(self.device))

    def node_preference(self, ind, ntype="user"):
        diff_type = "item" if ntype == "user" else "user"
        node_ne_same_nodes = getattr(self, "{}_ne_{}s".format(ntype, ntype))
        node_ne_diff_nodes = getattr(self, "{}_ne_{}s".format(ntype, diff_type))
        ind2text = getattr(self, "get_{}_text".format(ntype))
        
        ne_diff_nodes = node_ne_diff_nodes[ind][:self.max_neighbors]
        ne_same_nodes = node_ne_same_nodes[ind][:self.max_neighbors]
        len_diff = len(ne_diff_nodes)
        len_same = len(ne_same_nodes)

        ne_nodes = ne_diff_nodes + ne_same_nodes
        all_nodes = ne_nodes + [ind]
        batch_texts = [ind2text(i) for i in all_nodes]
        review_texts = [self.get_review_text(ind, ind2, ntype) for ind2 in ne_diff_nodes]
        batch_texts.extend(review_texts)

        batch_vectors = self.texts2vectors(batch_texts)
        all_tensors = self.semantic_encoder(*batch_vectors)
        all_nodes_tensors = self.node_encoder(all_tensors[:len(all_nodes)], torch.LongTensor(all_nodes).to(self.device), ntype, len_diff)

        ne_nodes_tensors = all_nodes_tensors[:-1, :]
        this_node_tensor = all_nodes_tensors[-1, :]
        same_relations_tensors = ne_nodes_tensors[len_diff:, :] * this_node_tensor
        diff_relations_tensors = all_tensors[(len_diff+len_same)+1:, :]
        relations_tensors = torch.cat((diff_relations_tensors, same_relations_tensors), dim=0)
        node_preference = self.att(this_node_tensor, relations_tensors, ne_nodes_tensors)
        return torch.cat((this_node_tensor, node_preference))

    def transform(self, inputs):
        x = F.relu(self.linear1(inputs))
        x = self.dropout_layer(x)
        x = F.relu(self.linear2(x))
        return x

    def get_relations_pref(self, users_ind, items_ind):
        #TODO: deal none review text user-item pairs
        review_texts = [self.get_review_text(ind1, ind2, ntype="user") for ind1, ind2 in zip(users_ind, items_ind)]
        review_vectors = self.texts2vectors(review_texts)
        return self.semantic_encoder(*review_vectors)

    def forward(self, users_ind, items_ind):
        batch_size = len(users_ind)
        users_pref = torch.empty(batch_size, 2 * self.hidden_size).to(self.device)
        items_pref = torch.empty(batch_size, 2 * self.hidden_size).to(self.device)
        relations_pref = self.get_relations_pref(users_ind, items_ind)

        for i, (uind, iind) in enumerate(zip(users_ind, items_ind)):
            u_pref = self.node_preference(uind, "user")
            i_pref = self.node_preference(iind, "item")
            users_pref[i, :] = u_pref
            items_pref[i, :] = i_pref
        users_pref = self.transform(users_pref)
        items_pref = self.transform(items_pref)

        return users_pref, items_pref, relations_pref

class NodeEncoder(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, dropout=0.1):
        super(NodeEncoder, self).__init__()
        self.user_emb = nn.Embedding(num_users, hidden_size)
        self.item_emb = nn.Embedding(num_items, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
    
    def forward(self, text_tensors, inds, ntype, len_diff):
        diff_emb_layer = self.item_emb if ntype == "user" else self.user_emb
        same_emb_layer = self.user_emb if ntype == "user" else self.item_emb
        diff_inds = inds[:len_diff]
        same_inds = inds[len_diff:]
        diff_emb = diff_emb_layer(diff_inds)
        same_emb = same_emb_layer(same_inds)
        emb = torch.cat((diff_emb, same_emb), dim=0)
        out = torch.cat((text_tensors, emb), dim=1)
        out = F.relu(self.linear1(out))
        out = self.dropout(out)
        out = F.relu(self.linear2(out))
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(Attention, self).__init__()
        self.att1 = nn.Linear(hidden_size * 3, hidden_size)
        self.att2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, this_node, relations, ne_nodes):
        this_nodes = this_node.expand(ne_nodes.shape)
        x = torch.cat((this_nodes, relations, ne_nodes), dim=1)
        x = F.relu(self.att1(x))
        x = self.dropout(x)
        x = F.relu(self.att2(x))
        att = F.softmax(x, dim=1)
        out = torch.mm(att.t(), ne_nodes).squeeze(0)
        return out
