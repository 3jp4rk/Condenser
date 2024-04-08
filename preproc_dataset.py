import os
# to use CPU for not distrupting GPU since it's been using for training
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import struct
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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


from enko_dataset_ import ComplicatedDataset
tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-en-base')
enko_dataset = ComplicatedDataset(
    tokenizer=tokenizer,
    max_length=512,
    namuwiki=False,
    train_ratio=0.9
    )


'''
data_len = 
'''

# 7876539 (train_ratio 0.8)
print(enko_dataset.total_train_len)

for i in range(enko_dataset.total_train_len):
    item = enko_dataset.__getitem__(i)
    print(len(item['text'])) # 512 
    quit()
    








