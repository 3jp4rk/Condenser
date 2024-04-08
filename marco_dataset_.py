import os
import struct
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import time
import ijson
import random
import json
import sys

os.chdir('/root')

def unescape_csv(bdata) :
    bcopy = bytearray()
    dquote = False

    for i, b in enumerate(bdata) :
        if b == 34 : # '"' is chr(34)
            if i == 0 or i == len(bdata) - 1 :
                continue
            elif dquote :
                dquote = False
                continue
            else :
                dquote = True
        elif dquote :
            dquote = False
        bcopy.append(b)

    return bcopy.decode('utf-8')

def get_csv_len(fname) :
    idx_name = fname + '.idx'
    fsize = os.stat(idx_name).st_size
    return int((fsize - 8) / 8) - 1

def get_csv_item(fname, idx, len) :
    try :
        idx_name = fname + '.idx'
        with open(idx_name, 'rb') as h :
            h.seek((idx + 1) * 8)
            start = struct.unpack('>Q', h.read(8))[0]
            if idx == len - 1 :
                end = -1
            else :
                h.seek((idx + 2) * 8)
                end = struct.unpack('>Q', h.read(8))[0]
        # print(f'start : { start }, end : { end }')
        with open(fname, 'rb') as h :
            h.seek(start)
            if end >= 0 :
                row_data = h.read(end - start)
            else :
                row_data = h.read(os.stat(fname).st_size - start)
        return unescape_csv(row_data[:-1])
    except FileNotFoundError as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e
    except struct.error as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e


def get_csv_len(fname) :
    idx_name = fname + '.idx'
    fsize = os.stat(idx_name).st_size 
    return int((fsize - 8) / 8) - 1 # 아~~ 0부터 count하니까 1 뺐나 보다 


# fname, offset, length (idx를)
def get_jsonl_data_item_at(fname, start, len) :
    try :
        end = start + len
        # print(f'start : { start }, end : { end }')
        with open(fname, 'rb') as h :
            h.seek(start)
            bdata = h.read(end - start)
            sdata = bdata.decode('utf8')
            json_data = json.loads(sdata)
            # json.dumps(sdata)
        return json_data # torch.tensor(token_data, dtype=torch.long), attention_mask
    except FileNotFoundError as e :
        print(f'fname : {fname}, offset: {start}, len: {len}, error : {str(e)} ')
        raise e
    except struct.error as e :
        print(f'fname : {fname}, offset: {start}, len: {len}, error : {str(e)} ')
        raise e

import nltk
# nltk.download('punkt')
def get_token_num(tokenizer, text, truncate=False):
    max_length = 512-2 # special tokens will be addeds
    ### 공통 dict 한번에 처리 
    if truncate:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=truncate,
            return_attention_mask=False,
            return_token_type_ids=False,
            max_length=max_length
        ) ["input_ids"]
        total_token_len = len(ids)
    else:
        sentences = nltk.sent_tokenize(text)
        ids = [
            tokenizer(
                sent,
                add_special_tokens=False,
                truncation=truncate,
                return_attention_mask=False,
                return_token_type_ids=False,
                max_length=max_length
            ) ["input_ids"] for sent in sentences
        ]
        total_token_len = sum([len(id) for id in ids])
        for idx, sent in enumerate(ids[:]):
            if len(sent) > max_length:
                del ids[idx]
                for i in range(0, len(sent), max_length):
                    ids.insert(idx, sent[i:i+max_length])

    # 해당 item의 총 token 길이, tokenized
    # print(ids)
    return total_token_len, ids



'''
[ // doc_list
    {
        'doc_id': doc_id,
        'passages': [
            {
                "passage_id":,
                "offset":,
                "length":,
                "line_no":,
                "doc_id":,
            }, // passage_dict
        ]
    }, // doc_dict
]
'''




def get_idx(file_path):
    with open(file_path, 'rb') as file:
        return json.load(file)
    


print('-------------------------------------------')
print('loading json . . .')
#ds_key = ['en', 'ko', 'ko-en-same', 'ko-en-diff']
ds_key = ['en', 'ko', 'ko-en-diff']
lang_list = ["en", "ko"]
data_idx = {}
start = time.time()
marco_path = f"/home/ubuntu/ejpark/new_json_diet.json"
marco_path = f"/home/ubuntu/ejpark/debugging_12000.json"
# print(f"{lang} data length: {count_jsonl_objects(marco_path)}")
    # item 2개가 안 되면 그 document는 제외해야 함
    # print(f"{lang} number of doc: {len(get_doc_id(marco_path))}")
    # data_idx[lang] = get_doc_idx(marco_path, lang)
data_idx = get_idx(marco_path)

print(f"{marco_path} load complete! {time.time()-start} seconds took.")
        
data_len = {}
data_start = {}

# 솎아내기 전 (문서 개수 기준)
DATA_LENGTH = 2000 # 2k
data_len = {f'{key}': DATA_LENGTH for key in ds_key}
# data_len = {f'{key}': 11527 for key in ds_key}

# data_len = {f'{ds_key[idx]}': idx * 11527002 for idx in range(len(ds_key))}
    # data_len['en'] = 11527002
    # data_len['ko'] = 11527002


    # data_len['en'] = len(data_idx['en'])
    # data_len['ko'] = len(data_idx['ko'])

    # print(data_len)
    # print(f"{time.time() - start} seconds.")
    
data_start = {f'{ds_key[idx]}': idx * data_len[ds_key[idx]] for idx in range(len(ds_key))}


# doc_id_set = get_doc_ids(data_idx['en'])

# prune dataset (passage 개수가 최소 2개는 되어야 함)
# for lang in lang_list:
#     docs = data_idx[lang]
#     for doc in data_idx[lang][:]:
#         doc_id = doc['doc_id']
#         if len(doc['passages']) < 2:
#             data_idx[lang].remove(doc)





# enumerate로 바꿀 수도 있으려나? 

# data_start = {f'{ds_key[idx]}': idx * 11527002 for idx in range(len(ds_key))}

# data_start['en'] = 0
# data_start['ko'] = data_len['en']






def find_in_list(ds_list, idx) :
    for i, v in enumerate(ds_list) :
        idx -= v
        if idx < 0 :
            return i, idx + v

# en_ratio = { 0:0.7, 1:0.6, 2:0.5 , 3:0.6, 4:0.5 }
# en_ratio = { 0 : 0.65, 1: 0.5 }
# en_ratio = {i:0.5 for i in range(8)}
# en_ratio = {i:1 for i in range(8)}


def get_doc_passages(data_idx, ds, idx, passage_id=None):
    
    # iterating dict
    passages = []
    for k, v in data_idx[ds].items():
        if k == passage_id: # 제외하고 선택
            continue
        if v[3] == idx: # doc id를 알 수 있는 방법이 없음 ㅋㅋ 
            passages.append({
                'passage_id': k,
                'items': v # offset, length, line_no, doc_id
            })
    return passages

def get_passages(lang_index, doc_id, lang, idx=None):
    passages = []
    if idx:
        return lang_index[lang][idx]
    for key, value in lang_index[lang].items():
        if value[3] == doc_id:
            passages.append((key, value))
    return passages


def get_doc_passage(data_idx, ds, idx):
    
    lang = 'ko' if ds == 'ko-en-same' or ds == 'ko-en-diff' else ds
    
    
    
    doc_id, doc_passages = list(data_idx[lang].items())[idx] # passage_idx
    
    passages = []
    ko_passage = None
    en_passage = None
    if ds == 'en' or ds == 'ko':
        passages = random.sample(doc_passages, 2)
    else:
        passage_idx = random.randint(0, len(doc_passages)-1)
        ko_passage = doc_passages[passage_idx]
        if ds == 'ko-en-same':
            en_passage = data_idx['en'][doc_id][passage_idx]
        elif ds == 'ko-en-diff':
            idx_list = [i for i in range(len(doc_passages)) if i != passage_idx]
            try:
                en_idx = random.sample(idx_list, 1)[0]
            except:
                print(f"{idx} is wrong in {ds}... ")
                en_idx = passage_idx # 에러 나면 그냥 ko랑 같은 index로 가자... 
            en_passage = data_idx['en'][doc_id][en_idx]
        
        passages = [ko_passage, en_passage]

    return passages

class ComplicatedDataset(Dataset):
    def __init__(self, tokenizer, max_length, namuwiki=False, epoch=-1, train_ratio=0.8):
        
        ### 여기서 dataset 초기화를 해야 할 듯
        self.data_idx = data_idx
        self.data_len = data_len
        self.data_start = data_start
        self.ds_key = ds_key

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token
        self.train_ratio = train_ratio
        # self.data_order = data_order
        self.data_order = self.ds_key
 
        total_len = self.data_len['en'] * 
        eval_offset = {}
        eval_len = {}
        # train_data_start
        train_data_start = {}
        train_data_start['en'] = 0
        eval_data_start = {}
        eval_data_start['en'] = 0
                        
        for i, key in enumerate(self.ds_key):
            eval_len[key] = int((1 - train_ratio) * self.data_len[key])
            eval_offset[key] = self.data_len[key] - eval_len[key]
            if i == 0:
                train_data_start[key] = 0
                eval_data_start[key] = 0
            else:
                prev = self.ds_key[i-1]
                train_data_start[key] = train_data_start[prev] + eval_offset[prev]
                eval_data_start[key] = eval_len[prev]
 
        # train_data_start['ko'] = train_data_start['en'] + eval_offset['en']
        self.train_data_start = train_data_start

        # eval_data_start
        
        # eval_data_start['ko'] = eval_len['en']
        self.eval_data_start = eval_data_start
        
        self.eval_offset = eval_offset
        self.eval_len = eval_len
        
        total_eval_len = 0
        for l in eval_len.values() :
            total_eval_len += l
        self.total_train_len = total_len - total_eval_len
        self.total_eval_len = total_eval_len

        class EvalDataset(Dataset):
            def __init__(self, outer):
                self.outer = outer
                
            def __len__(self) :
                return self.outer.total_eval_len

            def __getitem__(self, idx) :
                return self.outer.eval_data(idx)
                
        self.eval_dataset = EvalDataset(self)
        # self.eval_mode = False

    # def switch_to_eval(self):
    #     self.eval_mode = True
    
    # def switch_to_train(self):
    #     self.eval_mode = False
    
    
    def find_dataset(self, idx) :
        for ds in reversed(self.data_order) :
            if self.data_start[ds] <= idx :
                return ds
    
    def __len__(self):
        # total number of samples
        # return data_start['culen'] + self.culen_size
        # if self.eval_mode :
        #     return self.total_eval_len
        return self.total_train_len

    
    def __getitem__(self, idx):
        # if self.eval_mode :
        #     return self.eval_data(idx)
        # print(self.get_data(self.train2full(idx)))
        return self.get_data(self.train2full(idx))

    def find_train_dataset(self, idx) :
        for ds in reversed(self.data_order) :
            if self.train_data_start[ds] <= idx :
                return ds
                
    def find_eval_dataset(self, idx) :
        for ds in reversed(self.data_order) :
            if self.eval_data_start[ds] <= idx :
                return ds

    def train2full(self, idx) :
        found = self.find_train_dataset(idx)
        for ds in self.data_order :
            if ds == found :
                break
            idx += self.eval_len[ds]
        return idx
        
    def eval2full(self, idx) :
        found = self.find_eval_dataset(idx)
        for ds in self.data_order :
            idx += self.eval_offset[ds]
            if ds == found :
                break
        return idx
        
    def get_data(self, idx) :
        ds = self.find_dataset(idx)
        # print(f'{idx} is in {ds}')
        idx -= self.data_start[ds]
        
        '''
        'en': {  'passage_key': [   ], ... }
        'ko': {  'passage_key': [   ], ... }
        '''
        
        selected = get_doc_passage(self.data_idx, ds, idx)

        # selected = []
        # if ds == 'en' or ds == 'ko':
        #     passages = get_doc_passages(data_idx, ds, idx)
        #     selected = random.sample(passages, 2)
        # else:
        #     passages = get_doc_passages(data_idx, 'ko', idx)
        #     ko_selected = random.sample(passages, 1)
        #     passage_id = ko_selected[0]
        #     if ds == 'ko-en-same':
        #         en_selected = data_idx['en'][passage_id]
        #     elif ds == 'ko-en-diff':
        #         en_passages = get_doc_passages(data_idx, 'en', idx, passage_id)
        #     selected = [ko_selected, en_selected]
        
        lang = ['ko', 'en'] if ds == 'ko-en-same' or ds == 'ko-en-diff' else [ds, ds]

        inputs = []
        for idx, item in enumerate(selected):
            # passage_id = item['passage_id']
            # item: ('passage_offset', [offset, length, line_no, doc_id])
            passage_id = item[0].split("_")[0]
            fname = f'msmarco/{lang[idx]}/msmarco_passage_{passage_id}' # msmarco_passage_{passage_id}
            # fname, idx, len
            # idx2 = item['offset']
            offset, length, line_no, _ = item[1]
            # idx2 = item[1][]
            # text_offset = item[1][0]
            # text_length = item[1][1]
            # file_len = get_csv_len(fname)
            # input = get_csv_item(fname, idx2, file_len)
            input = get_jsonl_data_item_at(fname, offset, length)['passage']
            inputs.append(input)

        truncate = True
        total_token_len, ids = get_token_num(self.tokenizer, inputs, truncate)
        
        return ids

    def eval_data(self, idx) :
        return self.get_data(self.eval2full(idx))

