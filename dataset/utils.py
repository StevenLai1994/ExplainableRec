import json
import numpy as np
import os
from collections import defaultdict
from gensim.models import word2vec, Word2Vec
from tqdm import tqdm
import string
import numpy as np
import pickle

def build_dict_from_txt(file, key_type=str, val_type=int, seg='\t'):
    ret_dict = {}
    with open(file, 'r', encoding="utf-8") as fin:
        for line in fin:
            key, val = line.strip("\n").split(seg)
            key, val = key_type(key), val_type(val)
            ret_dict[key] = val
    return ret_dict

def prepare_w2v(opt):
    word_dict = opt.get("dict")
    w2v_emb_size = opt["w2v_emb_size"]
    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "w2v_{}_{}.txt".format(len(word_dict), w2v_emb_size))
    save_npy_path = os.path.join(save_dir, "w2v_{}_{}.npy".format(len(word_dict), w2v_emb_size))
    opt["w2v_weight_path"] = save_npy_path

    if os.path.exists(save_npy_path):
        print("pretrained_word2vec {} is already exists".format(save_npy_path))
        return

    print("training word2vec weight from dataset {}".format(data_path))
    sentences = []
    if opt["data_source"] == "Amazon":
        text_fields = ["reviewText", "summary"]
    elif opt["data_source"] == "Yelp":
        text_fields = []
        #TODO:  add yelp
    with open(opt["data_path"]) as f_data:
        for line in tqdm(f_data):
            instance = json.loads(line)
            for field in text_fields:
                text = instance[field]
                sentences.append(list(word_dict.tokenize(text)))
    
    model = word2vec.Word2Vec(sentences, size=w2v_emb_size, window=5, min_count=0, workers=4)
    model.wv.save_word2vec_format(save_path, binary=False)
    to_save = np.random.randn(len(word_dict), w2v_emb_size)
    for vec, tok in zip(model.wv.vectors, model.wv.index2word):
        idx = word_dict.tok2ind.get(tok, None)
        if idx is not None:
            to_save[idx, :] = vec
    np.save(save_npy_path, to_save)

def build_conceptnet(opt):
    '''
    save:
        tok2uri.txt
        indv2indc.txt: vocab index to conceptnet index
        indv2idnc.txt: exchange vocab index to conceptnet index 
        request_cache.txt: cache of request
        conceptnet_emb.npy: numpy weight   N * 300,  N is concepnet vocab len
    '''
    word_dict = opt.get("dict")
    save_dir = opt["conceptnet_dir"]
    ind2tok = word_dict.ind2tok
    tok2ind = word_dict.tok2ind

    tok2uri = None
    uri2indc = None
    conceptnet_emb = None
    ind2topk_emb = None

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    this_save_dir = os.path.join(save_dir, data_name)
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)

    numberbatch_path = os.path.join(save_dir, "numberbatch-en.txt")
    if not os.path.exists(numberbatch_path):
        raise("please Download concept numberbatch at https://github.com/commonsense/conceptnet-numberbatch")
    tok2uri_path = os.path.join(this_save_dir, "tok2uri.txt")
    uri2indc_path = os.path.join(this_save_dir, "uri2indc.txt")
    indv2idnc_path = os.path.join(this_save_dir, "indv2indc.txt")
    request_cache_path = os.path.join(save_dir, "request_cache.txt")
    conceptnet_emb_path = os.path.join(save_dir, "conceptnet_emb_{}.npy".format(opt["conceptnet_emb_type"]))
    opt["conceptnet_emb_path"] = conceptnet_emb_path
    opt["ind2topk_conceptnet_ind_path"] = indv2idnc_path

    with open(numberbatch_path) as f_numberbatch:
        for line in f_numberbatch:
            conceptnet_len, conceptnet_emb_size = line.split()
            conceptnet_len, conceptnet_emb_size = int(conceptnet_len), int(conceptnet_emb_size)
            opt["conceptnet_len"] = conceptnet_len
            opt["conceptnet_emb_size"] = conceptnet_emb_size
            break
    
    ###############Create tok2uri################
    if os.path.exists(tok2uri_path):
        print("tok2uri file {} is already exists".format(tok2uri_path))
    else:
        print("Creating tok2uri.txt, saving at {}".format(tok2uri_path))
        
        def tok2uri_function(tok, lanuage='en'):
            from .text2url import standardized_uri
            return standardized_uri(lanuage, tok)

        tok2uri = {}
        with open(tok2uri_path, 'w', encoding="utf-8") as f_tok2uri:
            for tok in tqdm(ind2tok.values()):
                uri = tok2uri_function(tok)
                uri = uri.split('/')[-1]
                f_tok2uri.write("{}\t{}\n".format(tok, uri))
                tok2uri[tok] = uri
    
    
    ###############Create uri2indc################
    if os.path.exists(uri2indc_path):
        print("uri2indc file {} is already exists".format(uri2indc_path))
    else:
        print("Creating uri2indc.txt, saving at {}".format(uri2indc_path))
        uri2indc = {}
        with open(numberbatch_path, 'r', encoding="utf-8") as f_numberbatch, open(uri2indc_path, 'w', encoding="utf-8") as f_uri2indc:
            for i, line in enumerate(tqdm(f_numberbatch, total=conceptnet_len+1)):
                if i != 0:
                    uri = line.split()[0]
                    f_uri2indc.write("{}\t{}\n".format(uri, i-1))
                    uri2indc[uri] = i-1
        

    ###############Create indv2indc################
    if os.path.exists(indv2idnc_path):
        print("indv2indc file {} is already exists".format(indv2idnc_path))
        indv2indc = build_dict_from_txt(indv2idnc_path, key_type=int, val_type=int)
    else:
        print("Creating indv2indc, saving at {}".format(indv2idnc_path))
        indv2indc = {}
        if os.path.exists(request_cache_path):
            request_cache = build_dict_from_txt(request_cache_path, key_type=str, val_type=int)
        else:
            request_cache = {}
        len_cache = len(request_cache)
        def find_uri_indc(uri):
            if uri in uri2indc:
                return uri2indc[uri]
            elif uri in request_cache:
                return request_cache[uri]
            else:
                import requests
                related_url = "http://api.conceptnet.io/related/c/en/{}?filter=/c/en"
                while True:
                    try:
                        obj = requests.get(related_url.format(uri)).json()
                        break
                    except Exception as e:
                        print(e)
                        continue
                for related in obj["related"]:
                    related_uri = related["@id"].split('/')[-1]
                    if related_uri in uri2indc:
                        indc =  uri2indc[related_uri]
                indc = uri2indc[tok2uri[word_dict.null_tok]]
                request_cache[uri] = indc
            return indc

        with open(indv2idnc_path, 'w', encoding="utf-8") as f_indv2indc:
            if tok2uri is None:
                tok2uri = build_dict_from_txt(tok2uri_path, key_type=str, val_type=str)
            if uri2indc is None:
                uri2indc = build_dict_from_txt(uri2indc_path, key_type=str, val_type=int)
            none_uris = []

            for indv, tok in tqdm(ind2tok.items()):
                indc = find_uri_indc(tok2uri[tok])
                f_indv2indc.write("{}\t{}\n".format(indv, indc))
                indv2indc[indv] = indc
                if indc == uri2indc[tok2uri[word_dict.null_tok]]:
                    none_uris.append(tok)
        #保存请求缓存
        if len(request_cache) > len_cache:
            with open(request_cache_path, 'w', encoding="utf-8") as f_cache:
                for uri, indc in request_cache.items():
                    f_cache.write("{}\t{}\n".format(uri, indc))
    opt["indv2indc"] = indv2indc

    ###############Create conceptnet_emb################
    if os.path.exists(conceptnet_emb_path):
        print("Conceptnet embedding {} is already exists".format(conceptnet_emb_path))
    else:
        print("Creating conceptnet_emb, saving at {}".format(conceptnet_emb_path))
        import pandas as pd
        with open(numberbatch_path, 'r', encoding="utf-8") as f_numberbatch:
            for line in f_numberbatch:
                break
            emb_df = pd.read_table(f_numberbatch, sep=' ', header=None, index_col=0).astype(getattr(np, opt["conceptnet_emb_type"]))
            np.save(conceptnet_emb_path, emb_df.values)

def build_user_item_datas(opt):
    '''
    save:
        user2ind.txt
        item2ind.txt
        user_item_matrix.json
        user_ne_items.txt
        item_ne_users.txt
        user_item_review.txt
        datasaet.pkl
    '''
    word_dict = opt.get("dict")
    data_path = opt["data_path"]
    total_instances = opt["total_instances"]
    min_tip_len = opt["min_tip_len"]
    save_dir = opt["save_dir"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    user2ind_path = os.path.join(save_dir, "user2ind.txt")
    item2ind_path = os.path.join(save_dir, "item2ind.txt")
    user_item_matrix_path = os.path.join(save_dir, "user_item_matrix.json")
    user_ne_items_path = os.path.join(save_dir, "user_ne_items.txt")
    item_ne_users_path = os.path.join(save_dir, "item_ne_users.txt")
    user_item_review_path = os.path.join(save_dir, "user_item_review.txt")
    dataset_path = os.path.join(save_dir, "dataset_tiplen{}.pkl".format(min_tip_len))
    opt["user_item_matrix_path"] = user_item_matrix_path
    opt["user_ne_items_path"] = user_ne_items_path
    opt["item_ne_users_path"] = item_ne_users_path
    opt["user_item_review_path"] = user_item_review_path
    opt["dataset_path"] = dataset_path

    if opt["data_source"] == "Amazon":
        text_field = "reviewText"
        user_field = "reviewerID"
        item_field = "asin"
        rating_field = "overall"
        tip_field = "summary"
    elif opt["data_source"] == "Yelp":
        text_field = ""
        user_field = ""
        item_field = ""
        rating_field = ""
        #TODO:  add yelp

    user_item_matrix = {}

    #build user2ind and item2ind
    user2ind = {}
    item2ind = {}
    if os.path.exists(user2ind_path) and os.path.exists(item2ind_path):
        print("user2ind {} and item2ind file {} is already exist".format(user2ind_path, item2ind_path))
        user2ind = build_dict_from_txt(user2ind_path, key_type=str, val_type=int)
        item2ind = build_dict_from_txt(item2ind_path, key_type=str, val_type=int)
    else:
        print("Building user2ind {} and item2ind file {} ".format(user2ind_path, item2ind_path))
        with open(data_path, 'r', encoding="utf-8") as f_data:
            for line in tqdm(f_data, total=total_instances):
                instance = json.loads(line)
                user = instance[user_field]
                item = instance[item_field]
                if user not in user2ind:
                    user2ind[user] = len(user2ind)
                if item not in item2ind:
                    item2ind[item] = len(item2ind)
        with open(user2ind_path, 'w', encoding="utf-8") as f_user2ind, open(item2ind_path, 'w', encoding="utf-8") as f_item2ind:
            key = lambda x:x[1]
            for (user, uind) in sorted(user2ind.items(), key=key):
                f_user2ind.write("{}\t{}\n".format(user, uind))
            for (item, iind) in sorted(item2ind.items(), key=key):
                f_item2ind.write("{}\t{}\n".format(item, iind))

    opt["user2ind"] = user2ind
    opt["item2ind"] = item2ind
    opt["num_users"] = len(user2ind)
    opt["num_items"] = len(item2ind)

    #write user_item_matrix.json
    if os.path.exists(user_item_matrix_path):
        print("User and item interactive matrix file {} is already exist".format(user_item_matrix_path))
    else:
        print("Building user and item interactive matrix file {}.".format(user_item_matrix_path))
        with open(data_path, 'r', encoding="utf-8") as f_data:
            with open(user_item_matrix_path, 'w', encoding="utf-8") as f_matrix:
                for line in tqdm(f_data, total=total_instances):
                    instance = json.loads(line)
                    user = instance[user_field]
                    item = instance[item_field]
                    rating = float(instance[rating_field])
                    if user in user_item_matrix:
                        if item in user_item_matrix[user]:
                            user_item_matrix[user][item].append(rating)
                        else:
                            user_item_matrix[user][item] = [rating]
                    else:
                        user_item_matrix[user] = {item:[rating]}
                user_item_matrix = sorted(user_item_matrix.items(), key=lambda x:user2ind[x[0]])
                for user, itemdict in user_item_matrix:
                    for item, rating in itemdict.items():
                        itemdict[item] = sum(rating) / len(rating)
                    line = itemdict
                    f_matrix.write(json.dumps(line) + '\n')

    #build dataset
    user_ne_items = defaultdict(list)
    item_ne_users = defaultdict(list)
    user_item_review = defaultdict(str)
    users = []
    items = []
    ratings = []
    tips = []
    if os.path.exists(dataset_path):
        print("dataset file {} is already exist".format(dataset_path))
    else:
        print("Building dataset file {}.".format(dataset_path))
        with open(data_path, 'r', encoding="utf-8") as f_data:
            for line in tqdm(f_data, total=total_instances):
                instance = json.loads(line)
                user_ind = user2ind[instance[user_field]]
                item_ind = item2ind[instance[item_field]]
                rating = float(instance[rating_field])
                text = instance[text_field]
                tip = instance[tip_field]
                pair = (user_ind, item_ind)
                user_item_review[pair] += text
                if item_ind not in user_ne_items[user_ind]:
                    user_ne_items[user_ind].append(item_ind)
                if user_ind not in item_ne_users[item_ind]:
                    item_ne_users[item_ind].append(user_ind)

                if (len(tip.split()) < min_tip_len):
                    for tip in word_dict.sent_tok.tokenize(text):
                        if (len(tip.split()) >= min_tip_len):
                            break
                if (len(tip.split()) >= min_tip_len):
                    users.append(user_ind)
                    items.append(item_ind)
                    ratings.append(rating)
                    tips.append(tip)

        user_ne_users = get_neighbor_users(opt)
        item_ne_items = get_neighbor_items(opt)

        datas = (users, items, ratings, tips, user_ne_items, item_ne_users, user_ne_users, item_ne_items, user_item_review)
        with open(dataset_path, 'wb') as f_dataset:
            pickle.dump(datas, f_dataset)

        with open(user_item_review_path, 'w', encoding="utf-8") as f_uir:
            for (user, item), review in user_item_review.items():
                f_uir.write("{}\t{}\t{}\n".format(user, item, review))
    
        with open(user_ne_items_path, 'w', encoding="utf-8") as f_ui:
            for user, items in user_ne_items.items():
                f_ui.write("{}\t{}\n".format(user, " ".join(map(str, items))))
        with open(item_ne_users_path, 'w', encoding="utf-8") as f_iu:
            for item, users in item_ne_users.items():
                f_iu.write("{}\t{}\n".format(item, "\t".join(map(str, users))))
    # opt["user_ne_items"] = user_ne_items
    # opt["item_ne_users"] = item_ne_users
    # opt["user_item_review"] = user_item_review

    
def get_neighbor_users(opt):
    '''
    save:
        user_ne_k.txt
    '''
    topk = opt["user_topk"]
    user_item_matrix_path = opt["user_item_matrix_path"]
    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    user_ne_k_path = os.path.join(save_dir, "user_ne_{}.txt".format(topk))

    user_ne_users = defaultdict(list)
    if os.path.exists(user_ne_k_path):
        print("The file of user-user neighbors is Already exists in {}".format(user_ne_k_path))
        with open(user_ne_k_path, 'r', encoding="utf-8") as f_user_ne:
            for line in f_user_ne:
                user, ne_users = line.split('\t')
                user = int(user)
                ne_users = list(map(int, ne_users.split()))
                user_ne_users[user] = ne_users
    else:
        print("Building the file of user-user neighbors {}".format(user_ne_k_path))
        from datasketch import MinHashLSHForest, MinHash
        minHashs = []
        forset = MinHashLSHForest(num_perm=128)
        with open(user_item_matrix_path, 'r', encoding="utf-8") as f_user_item:
            for i, line in tqdm(enumerate(f_user_item), total=opt["num_users"]):
                line = json.loads(line)
                m = MinHash(num_perm=128)
                for d in line.keys():
                    m.update(d.encode("utf8"))
                forset.add(i, m)
                minHashs.append(m)
        forset.index()

        with open(user_ne_k_path, 'w', encoding="utf-8") as f_user_ne:
            for h, minHash in tqdm(enumerate(minHashs), total=opt["num_users"]):
                topk_relate = forset.query(minHash, 10)
                topk_relate.remove(h)
                if len(topk_relate) > 0:
                    line_ne = "{}\t{}\n".format(h, ' '.join(map(str, topk_relate)))
                    user_ne_users[h] = topk_relate
                    f_user_ne.write(line_ne)
    return user_ne_users

def get_neighbor_items(opt):
    '''
    save:
        item_ne.txt
    '''
    min_support = opt["min_support"]
    min_conf = opt["min_conf"]
    item2ind = opt["item2ind"]
    user_item_matrix_path = opt["user_item_matrix_path"]
    save_dir = opt["save_dir"]
    data_path = opt["data_path"]
    data_name = os.path.basename(data_path).split('.')
    data_name = ".".join(data_name[:-1])
    save_dir = os.path.join(save_dir, data_name)
    item_ne_path = os.path.join(save_dir, "item_ne.txt")

    item_ne_items = defaultdict(list)
    if os.path.exists(item_ne_path):
        print("The file of item-item neighbor is Already exists in {}".format(item_ne_path))
        with open(item_ne_path, 'r', encoding="utf-8") as f_item_ne:
            for line in f_item_ne:
                item, items_ne = line.split('\t')
                item = int(item)
                items_ne = list(map(lambda x:int(x.split(':')[0]), items_ne.split()))
                item_ne_items[item] = items_ne
    else:
        print("Building the file of item-item neighbors is Already exists in {}".format(item_ne_path))

        support1 = defaultdict(float)
        support2 = defaultdict(float)
        min_sum = opt["num_users"] * min_support
        with open(user_item_matrix_path, 'r', encoding="utf-8") as f_user_item:
            for i, line in tqdm(enumerate(f_user_item), total=opt["num_users"]):
                line = json.loads(line)
                items = list(map(lambda x:item2ind[x], line.keys()))
                for i in range(len(items)):
                    support1[str(items[i])] += 1.
        for key, val in support1.items():
            if val < min_sum:
                del support1[key]

        with open(user_item_matrix_path, 'r', encoding="utf-8") as f_user_item:
            for i, line in tqdm(enumerate(f_user_item), total=opt["num_users"]):
                line = json.loads(line)
                items = list(map(lambda x:item2ind[x], line.keys()))
                for i in range(len(items)):
                    if str(items[i]) not in support1:
                        continue
                    for j in range(i+1, len(items)):
                        if str(items[j]) not in support1:
                            continue
                        h, t = items[i], items[j]
                        if (h, t) not in support2 and (t, h) not in support2:
                            support2[(h, t)] += 1.
        for key, val in support2.items():
            if val < min_sum:
                del support2[key]

        confs = defaultdict(dict)
        for (i, j), val in support2.items():
            confs_i_j = support2[(i, j)] / support1[str(i)]
            confs_j_i = support2[(i, j)] / support1[str(j)]
            if confs_i_j >= min_conf:
                confs[str(i)][str(j)] = confs_i_j
            if confs_j_i >= min_conf:
                confs[str(j)][str(i)] = confs_j_i
        del support1, support2

        with open(item_ne_path, 'w', encoding="utf-8") as f_item_ne:
            for h, val in confs.items():
                tails = []
                c_vals = []
                for t, c_val in val.items():
                    tails.append(t)
                    c_vals.append(c_val)
                item_ne_items[h] = tails
                line_ne = "{}\t{}\n".format(h, ' '.join(("{}:{}".format(i, v) for i, v in zip(tails, c_vals))))
                f_item_ne.write(line_ne)
    return item_ne_items

def split_datas(opt):
    splits = opt.get("splits", "8:1:1")
    train, valid, test = splits.split(':')
    train, valid, test = int(train), int(valid), int(test)
    #TODO: split datas

def prepare_datas(opt):
    prepare_w2v(opt)
    build_conceptnet(opt)
    build_user_item_datas(opt)

class Toolkits(object):
    def __init__(self, opt):
        self.word_dict = opt["dict"]
        self.user2ind_dict = opt["user2ind"]
        self.item2ind_dict = opt["item2ind"]
        self.indv2indc_dict = opt["indv2indc"]
    
    def tok2ind(self, tok):
        return self.word_dict.tok2ind.get(tok, self.word_dict.tok2ind[self.word_dict.null_tok])

    def user2ind(self, user):
        return self.user2ind_dict.get(user, -1)
    
    def item2ind(self, item):
        return self.item2ind_dict.get(item, -1)
    
    def indv2indc(self, indv):
        return self.indv2indc_dict.get(indv, self.indv2indc_dict[0])

    def uind2entity(self, uind):
        return uind
    
    def iind2entity(self, iind):
        return len(self.user2ind_dict) + iind

    def indvvec2indcvec(self, indvvec):
        ret = []
        for indv in indvvec:
            ret.append(self.indv2indc(indv))
        return ret

    def text2vec(self, text, add_start=False, add_end=False):
        ret = []
        if add_start:
            ret.append(self.tok2ind(self.word_dict.start_tok))
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        for tok in self.word_dict.tokenize(text):
            ret.append(self.tok2ind(tok))
        if add_end:
            ret.append(self.tok2ind(self.word_dict.end_tok))
        if len(ret) == 0:
            ret.append(self.tok2ind(self.word_dict.null_tok))
        return ret
    
    def text2sentences_vec(self, text):
        ret = []
        max_len = 0
        sentences = self.word_dict.sent_tok.tokenize(text)
        for sent in sentences:
            ret.append(self.text2vec(sent))
            max_len = max(max_len, len(ret[-1]))
        if len(ret) == 0:
            ret.append([self.tok2ind(self.word_dict.null_tok)])
            max_len = 1
        return ret, max_len
    
    def pad2d(self, datas, pad_len, pad_ind=None):
        pad_ind = pad_ind if pad_ind is not None else self.tok2ind(self.word_dict.null_tok)
        for data in datas:
            if pad_len > len(data):
                data.extend([pad_ind] * (pad_len - len(data)))
            else:
                del data[pad_len:]
        return np.array(datas)
    
    def pad3d(self, datas, len1, len2, pad_ind=None):
        pad_ind = pad_ind if pad_ind is not None else self.tok2ind(self.word_dict.null_tok)
        pad_dim1 = [pad_ind] * len2
        for i, data in enumerate(datas):
            pad_data = self.pad2d(data, len2)
            if len1 > len(data):
                pad_data = np.pad(pad_data, ((0, len1 - len(data)), (0, 0)), constant_values=(pad_ind,))
            else:
                pad_data = pad_data[:len1]
            datas[i] = pad_data
        return np.array(datas)