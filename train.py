import argparse
from dataset.build_dict import Dict
from dataset import utils
from dataset.data_loaders import RecDatasetManager
from modules.GATGraph import GATGraph
from modules.SemanticModule import SemanticModule
from modules.ExplainableRec import ExplainableRec
import torch
from tqdm import tqdm
import numpy as np
import random

torch.cuda.set_device(0)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def train(opt):
    word_dict = Dict(opt)
    word_dict.build(sort=True)
    opt["dict"] = word_dict
    utils.prepare_datas(opt)
    toolkits = utils.Toolkits(opt)
    opt["toolkits"] = toolkits
    datamanager = RecDatasetManager(opt)
    # model = ExplainableRec(opt).cuda()
    semanticModule = SemanticModule(opt)
    model = GATGraph(opt, semanticModule).cuda()
    for uinds, iinds, ratings, tips in datamanager.dataloader:
        model(uinds, iinds)
        pass
    # pass
    # from test_code import function
    # function(opt)
    ########################################
    # import torch
    # batch_size = opt["batch_size"]
    # model = SemanticModule(opt).cuda()
    # text_vec = torch.randint_like(torch.zeros((batch_size, 60)), len(word_dict)-1).long().cuda()
    # sent_vec = torch.randint_like(torch.zeros((batch_size, 10, 30)), len(word_dict)-1).long().cuda()
    # text_lens = torch.Tensor(sorted([random.randint(16, 60) for x in range(batch_size)], reverse=True)).long()
    # sent_lens = torch.Tensor(sorted([random.randint(2, 10) for x in range(batch_size)], reverse=True)).long()
    # conceptnet_text_vec = torch.randint_like(torch.zeros((batch_size, 60)), len(word_dict)-1).long().cuda()
    # model(text_vec, text_lens, sent_vec, sent_lens, conceptnet_text_vec)
    ########################################
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(
        data_source="Amazon",
        data_path="dataset/tini_Electronics_5.json",
        splits="8:1:1",
        save_dir="saved",
        conceptnet_dir="saved/conceptnet",
        conceptnet_emb_type="float16",
        tokenizer="nltk",
        dict_language="english",
        fp16=True,
        batch_size=8,
        dropout=0.1,
        num_heads=2,
        words_topk=10,
        user_topk=30,
        min_support = 0,
        min_conf = 0,
        min_tip_len=5,
        rec_topk=10,
        w2v_emb_size=64,
        bilstm_hidden_size=32,
        hidden_size=8,
        bilstm_num_layers=2,
        gru_num_layers=2,
        max_text_len=256,
        max_sent_len=16,
        max_tip_len=64,
        max_neighbors = 3,
        max_copy_len=256,
        tensorboard_tag="task,model,batchsize,dim,learningrate,model_file",
        tensorboard_metrics="loss,base_loss,kge_loss,l2_loss,acc,auc,recall@1,recall@10,recall@50",
    )
    opt = vars(parser.parse_args())
    opt["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    train(opt)
    pass
