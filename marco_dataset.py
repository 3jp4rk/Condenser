import os
import struct
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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





### 
import nltk
# nltk.download('punkt')
def get_token_num(text, truncate=False):
    tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-en-base')
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

import ijson
import json
def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        # data = json.load(f)
        # parser = ijson.parse(f)
        # for idx, (prefix, event, value) in enumerate(parser):
        #     # print(prefix, event, value)
        #     if event == 'map_key':
        #         key = value
        #     elif event == 'string' and key is not None:
        #         print(f"{key}: {value}")
        #         return
        #     # prefix: 
        objects = ijson.items(f, '38_835390905')
        for o in objects:
            print(o) # [835390905, 385, 77919332, '33_1546968948']
            # 
        
read_json("msmarco/en_index.json")

quit()
# data files
data_root = "msmarco"



data_len = {}



## read_index













data_len['sciencedaily'] = 168329 # 0 ~ 168328
# data_len['namuwiki'] = 1041222 # 0 ~ 1041221
data_len['newspepper'] = 7954 # 0 ~ 7953

data_len_ko_txt = data_len['newspepper']
data_len_en_txt = data_len['sciencedaily']
data_len_txt = data_len_ko_txt + data_len_en_txt

data_len['kowiki'] = []
data_len['culko'] = []
data_len['culen'] = []

data_len_csv = 0

data_len['kowiki'].clear()

# 'kowiki/part_00000.csv' ~ 'kowiki/part_00003.csv'
for i in range(4) :
    f = f'kowiki/part_{{num:05d}}.csv'.format(num=i)
    data_count = get_csv_len(f)
    data_len['kowiki'].append(data_count)

# print(data_len['kowiki']) # [111474, 111473, 111473, 83605] (csv 파일이 4개임)
print(f"kowiki: {sum(data_len['kowiki'])}") # [111474, 111473, 111473, 83605] (csv 파일이 4개임)

data_len['culko'].clear()

# 'CulturaX/ko/ko_part_00000.csv' ~ 'CulturaX/ko/ko_part_00031.csv'
for i in range(7) :
    f = f'CulturaX/ko/ko_part_{{num:05d}}.csv'.format(num=i)
    data_count = get_csv_len(f)
    data_len['culko'].append(data_count)

print(f"cultura ko: {sum(data_len['culko'])}")

data_len['culen'].clear()

# 'CulturaX/en/en_part_00000.csv' ~ 'CulturaX/en/en_part_00031.csv'
# for i in range(3072) :
for i in range(4) :
    f = f'CulturaX/en/en_part_{{num:05d}}.csv'.format(num=i)
    try :
        data_count = get_csv_len(f)
        data_len['culen'].append(data_count)
    except FileNotFoundError :
        data_len['culen'].append(0)

print(f"cultura en: {sum(data_len['culen'])}")

# need to handle CulturaX en shuffling
# the constants are the data count before CulturaX en (including CulturaX ko)
# CulturaX en range should be different on each epochs
# (won't randomize across epochs. Will use different sets on different epochs)

print(f'txt data : {data_len_txt}, en txt data : { data_len_en_txt }, ko txt data : { data_len_ko_txt }')
print(sum(data_len['kowiki']))
print(sum(data_len['culko']))

data_len_ko = data_len_ko_txt + sum(data_len['kowiki']) + sum(data_len['culko'])
print(data_len_ko)

data_len_en_except_culen = data_len_en_txt
data_len_except_culen = data_len_txt + sum(data_len['kowiki']) + sum(data_len['culko'])
# current en data가 culen 제외하고였잖아 ㅋㅋ 
print(f'total ko data : { data_len_ko }, current en data : { data_len_en_except_culen }, (ko - current en) data : { data_len_ko - data_len_en_except_culen }')

def get_culen_size(en_ratio) :
    return int(data_len_ko * en_ratio / (1.0 - en_ratio) - data_len_en_except_culen)

print(f'when 90% en data, cul en size should be { get_culen_size(0.9) }')
print(f'when 80% en data, cul en size should be { get_culen_size(0.8) }')
print(f'when 70% en data, cul en size should be { get_culen_size(0.7) }')
print(f'when 60% en data, cul en size should be { get_culen_size(0.6) }')
print(f'when 50% en data, cul en size should be { get_culen_size(0.5) }')


# csv with idx
#   - CulturaX/en/en_part_00000.csv ~ en_part_03071.csv
#   - CulturaX/ko/ko_part_00000.csv ~ ko_part_00031.csv
#   - kowiki/part_00000.csv ~ part_00003.csv
# txt (idx is file name)
#   - sciencedaily/0.txt ~ 168328.txt
#   - namuwiki/0.txt ~ 1041292.txt
#   - newspepper/0.txt ~ 7953.txt

data_start = {}
data_start['sciencedaily'] = 0
# data_start['namuwiki'] = data_len['sciencedaily']
data_start['newspepper'] = data_len['sciencedaily']

data_start['kowiki'] = data_start['newspepper'] + data_len['newspepper']
data_start['culko'] = data_start['kowiki'] + sum(data_len['kowiki']) # list
data_start['culen'] = data_start['culko'] + sum(data_len['culko']) # list

data_order = [ 'sciencedaily', 'newspepper', 'kowiki', 'culko', 'culen' ]


def find_dataset(idx) :
    for ds in reversed(data_order) :
        if data_start[ds] <= idx :
            return ds

def is_in_csv_data(idx) :
    return idx >= data_start['kowiki']

def get_non_csv(ds, idx) :
    fname = ds + '/' + str(idx - data_start[ds]) + '.txt'
    with open(fname, mode='r', encoding='utf-8') as f :
        return f.read()

def get_csv_fname(ds, ds_pos) :
    if ds == 'kowiki' :
        return f'kowiki/part_{{num:05d}}.csv'.format(num=ds_pos)
    elif ds == 'culko' :
        return f'CulturaX/ko/ko_part_{{num:05d}}.csv'.format(num=ds_pos)
    return f'CulturaX/en/en_part_{{num:05d}}.csv'.format(num=ds_pos)

def find_in_list(ds_list, idx) :
    for i, v in enumerate(ds_list) :
        idx -= v
        if idx < 0 :
            return i, idx + v

# en_ratio = { 0:0.7, 1:0.6, 2:0.5 , 3:0.6, 4:0.5 }
# en_ratio = { 0 : 0.65, 1: 0.5 }
en_ratio = {i:0.5 for i in range(8)}
# en_ratio = {i:1 for i in range(8)}

class ComplicatedDataset(Dataset):
    def __init__(self, tokenizer, max_length, namuwiki=False, epoch=-1, train_ratio=0.8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.pad_token
        self.train_ratio = train_ratio
        self.data_order = data_order

        # if not namuwiki :
        #     self.data_order = [ a for a in data_order]
        #     self.data_order.remove('namuwiki')

        ratio = en_ratio[epoch] if epoch >= 0 and epoch < len(en_ratio) else en_ratio[len(en_ratio) - 1]
        self.culen_size = get_culen_size(ratio)
        self.culen_offset = 0
        
        # for shuffle
        for i in range(epoch) :
            if i == 0 :
                continue
            self.culen_offset += get_culen_size(en_ratio[epoch - 1])
            print(self.culen_offset)
            
        total_len = data_start['culen'] + self.culen_size
        eval_offset = {}
        eval_len = {}
        
        eval_len['sciencedaily'] = int((1 - train_ratio) * data_len['sciencedaily'])
        # if namuwiki : 
        #     eval_len['namuwiki'] = int((1 - train_ratio) * data_len['namuwiki'])
        eval_len['newspepper'] = int((1 - train_ratio) * data_len['newspepper'])
        eval_len['kowiki'] = int((1 - train_ratio) * sum(data_len['kowiki']))
        eval_len['culko'] = int((1 - train_ratio) * sum(data_len['culko']))
        eval_len['culen'] = int((1 - train_ratio) * self.culen_size)

        # eval_offset in each datasets
        eval_offset['sciencedaily'] = data_len['sciencedaily'] - eval_len['sciencedaily']
        # if namuwiki : 
        #     eval_offset['namuwiki'] = data_len['namuwiki'] - eval_len['namuwiki']
        eval_offset['newspepper'] = data_len['newspepper'] - eval_len['newspepper']
        eval_offset['kowiki'] = sum(data_len['kowiki']) - eval_len['kowiki']
        eval_offset['culko'] = sum(data_len['culko']) - eval_len['culko']
        eval_offset['culen'] = self.culen_size - eval_len['culen']
        
        self.eval_offset = eval_offset
        self.eval_len = eval_len

        # train_data_start
        train_data_start = {}
        train_data_start['sciencedaily'] = 0
        if namuwiki : 
            train_data_start['namuwiki'] = eval_offset['sciencedaily']
            train_data_start['newspepper'] = train_data_start['namuwiki'] + eval_offset['namuwiki']
        else : 
            train_data_start['newspepper'] = eval_offset['sciencedaily']

        train_data_start['kowiki'] = train_data_start['newspepper'] + eval_offset['newspepper']
        train_data_start['culko'] = train_data_start['kowiki'] + eval_offset['kowiki']
        train_data_start['culen'] = train_data_start['culko'] + eval_offset['culko']
        self.train_data_start = train_data_start

        # eval_data_start
        eval_data_start = {}
        eval_data_start['sciencedaily'] = 0
        # if namuwiki : 
        #     eval_data_start['namuwiki'] = eval_len['sciencedaily']
        #     eval_data_start['newspepper'] = eval_data_start['namuwiki'] + eval_len['namuwiki']
        # else :
        eval_data_start['newspepper'] = eval_len['sciencedaily']

        eval_data_start['kowiki'] = eval_data_start['newspepper'] + eval_len['newspepper']
        eval_data_start['culko'] = eval_data_start['kowiki'] + eval_len['kowiki']
        eval_data_start['culen'] = eval_data_start['culko'] + eval_len['culko']
        self.eval_data_start = eval_data_start
        
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
        ds = find_dataset(idx)
        if not is_in_csv_data(idx) :
            input = get_non_csv(ds, idx)
        else :
            if ds == 'culen' :
                idx += self.culen_offset
            idx -= data_start[ds]
            ds_pos, idx2 = find_in_list(data_len[ds], idx)
            fname = get_csv_fname(ds, ds_pos)
            input = get_csv_item(fname, idx2, data_len[ds][ds_pos])

        # tokenize the text 
        # return self.tokenizer(input, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        truncate = True
        total_token_len, ids = get_token_num(input, truncate)
        # return [{'text': id} for id in ids]
        return {'text': ids if truncate else ids[0]}
        # labels = input_ids[1:] + [ self.pad_token ]  # Shift the input_ids to the right
        # return {'input_ids': input_ids, 'labels': labels}

    def eval_data(self, idx) :
        return self.get_data(self.eval2full(idx))

