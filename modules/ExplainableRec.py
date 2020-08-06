import torch
from torch import nn
from modules.SemanticModule import SemanticModule
from modules.GATGraph import GATGraph
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np


class ExplainableRec(nn.Module):
    def __init__(self, opt):
        super(ExplainableRec, self).__init__()
        self.dict = opt["dict"]
        self.device = opt["device"]
        self.toolkits = opt["toolkits"]
        self.hidden_size = opt["hidden_size"]
        self.dropout = opt["dropout"]
        self.max_tip_len = opt["max_tip_len"]
        self.max_copy_len = opt["max_copy_len"]
        self.gru_num_layers = opt["gru_num_layers"]
        self.end_ind = self.toolkits.tok2ind(self.dict.end_tok)
        self.semantic_encoder = SemanticModule(opt)
        self.embedding_layer = self.semantic_encoder.word2vec_encoder.w2v_emb
        self.gat = GATGraph(opt, self.semantic_encoder)
        self.recommender = Recommender(self.hidden_size)
        self.gru_explainer = GruExplainer(self.embedding_layer, self.end_ind, self.dict, \
                self.hidden_size, self.max_tip_len, self.max_copy_len, self.gru_num_layers, self.device)
        self.h0_layer = nn.Linear(3 * self.hidden_size, self.hidden_size)
        

    def forward(self, users_ind, items_ind, ys=None):
        users_pref, items_pref, relations_pref = self.gat(users_ind, items_ind)
        h0 = F.tanh(self.h0_layer(torch.cat((users_pref, items_pref, relations_pref), dim=1)))
        r_hat = self.recommender(users_pref, items_pref, relations_pref)
        masks = self.get_texts_copy_mask(users_ind, items_ind)
        _, predicts_tips = self.gru_explainer(masks, h0, ys)
        return r_hat, predicts_tips
    

    def get_texts_copy_mask(self, items_ind):
        bs = users_ind.shape[0]
        texts = [self.gat.get_user_text(uind) + self.gat.get_item_text(iind) for uind, iind in zip(users_ind, items_ind)]
        texts_vecs = self.gat.texts2vectors(texts)
        texts_vecs = texts_vecs[0][:, :self.max_copy_len]
        mask = torch.ones(bs, len(self.dict))
        for i in range(bs):
            indices = text_vecs[i]
            mask[i, indices] = 0
        return masks == 1

class Recommender(nn.Module):
    def __init__(self, hidden_size):
        super(Recommender, self).__init__()
        self.hidden_size = hidden_size
        # self.dropout = dropout

        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # self.dropout1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # self.dropout2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(2 * self.hidden_size, 1)
    
    def forward(self, users_pref, items_pref, relations_pref):
        i_uv = F.relu(self.linear1(torch.cat((users_pref, items_pref), dim=1)))
        z_h = i_uv * relations_pref
        z_m = F.relu(self.linear2(torch.cat((i_uv, relations_pref), dim=1)))
        r_uv = F.relu(self.linear3(torch.cat((z_h, z_m), dim=1)))
        return r_uv    

class GruExplainer(nn.Module):
    def __init__(self, embedding_layer, end_ind, word_dict, hidden_size, max_tip_len, max_copy_len, num_layers, device, dropout=0):
        super(GruExplainer, self).__init__()
        self.end_ind = end_ind
        self.dict = word_dict
        self.hidden_size = hidden_size
        self.emb_size = embedding_layer.embedding_dim
        self.embedding_layer = embedding_layer
        self.gru_decoder = nn.GRU(self.emb_size, hidden_size, batch_first=True)
        self.out_linear = nn.Linear(hidden_size, len(self.dict))
        self.copy_trans = nn.Linear(2 * hidden_size, len(word_dict))

    def neginf(self, dtype):
            """Returns a representable finite number near -inf for a dtype."""
            if dtype is torch.float16:
                return -NEAR_INF_FP16
            else:
                return -NEAR_INF

    def decode_forced(self, masks, ys, h0):
        bs = ys.size(0)
        starts = torch.LongTensor([1]).repeat((bs, 1)).to(self.device)
        seqlen = ys.size(1)
        masks = masks.expand(-1, seqlen, -1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat((starts, inputs), 1)
        x = self.embedding_layer(imputs)
        hs, hn = self.gru_decoder(x, h0.view(1, bs, -1))
        x1 = F.softmax(self.out_linear(hs), dim=-1)
        h0 = h0.repeat.unsqueeze(1).repeat(-1, seqlen, -1)
        x2 = self.copy_trans(torch.cat((hs, h0), dim=1))
        x2 = F.softmax(x2.masked_fill_(masks, self.neginf(x2.dtype)))
        scores = x1 + x2
        predicts = scores.max(dim=-1)
        return scores, predicts

    def decode_greedy(self, h0):
        bs = h0.size(0)
        x = torch.LongTensor([1]).repeat((bs, 1)).to(self.device)
        logits = []
        for i in range(self.max_tip_len):
            x = self.embedding_layer(x)
            hs, hn = self.gru_decoder(x, h0.view(1, bs, -1))
            hs = hs[:, -1, :]
            x1 = F.softmax(self.out_linear(hs), dim=-1)
            x2 = self.copy_trans(torch.cat((hs, h0), dim=1))
            x2 = F.softmax(x2.masked_fill_(masks, self.neginf(x2.dtype)), dim=-1)
            scores = x1 + x2
            logits.append(scores.view(bs, 1, -1))
            predicts = scores.max(dim=-1)
            x = torch.cat((x, predicts), dim=1)
            all_finished = ((predicts == self.end_ind).sum(dim=1) > 0).sum().item() == bs
            if all_finished:
                break
        scores = torch.cat(logits, 1)

        return scores, x

    def forward(self, masks, h0, ys=None):
        if ys is None:
            scores, predicts = self.decode_forced(ys, h0)
        else:
            scores, predicts = self.decode_greedy(h0)

        return scores, predicts