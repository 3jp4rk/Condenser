import os
import csv
import pandas as pd
import glob
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
import multiprocessing


import sys
sys.stdout.reconfigure(encoding='utf-8') # python a.py > log.txt 로 내보낼 때 cp949 codec can't encode ~ 에러 발생 방지 
csv.field_size_limit(200000) # _csv.Error: field larger than field limit (131072) 에러 발생 방지 

def get_files(path, ext):
    return sorted(glob.glob(os.path.abspath(path) + '**/*.'+ext, recursive=True))

def get_length(text):
    
    llama_tokenizer = AutoTokenizer.from_pretrained(
    './merged_tokenizer/merged_tokenizer/', 
    local_files_only=True) # LlamaTokenizerFast
    if not isinstance(text, str):
        text = " ".join(word for word in text)
    words = text.split()
    tokenized = llama_tokenizer(text, add_special_tokens=False)
    input_ids = tokenized['input_ids']
    
    return len(words), len(input_ids)

# 1 shard per 1 "line"
def open_csv(path):
    data = []
    total_len = 0
    out_len = 0
    with open(path, mode='r', encoding='utf-8', newline='') as file:
        lines = csv.reader(file)
        for idx, line in enumerate(lines):
            if idx == 5:
                break
            # print(f"=======================sample {idx} ==============================")
            # print(line) # 그냥 터미널에 찍는 건 괜찮은데 txt 파일로 내보내는 순간 cp959 codec can't encode character error가 남
            word_len, token_len = get_length(line)
            if token_len < 512:
                out_len += 1
                continue
            else:
                data.append({
                    "text": line,
                    "word_len": word_len,
                    "token_len": token_len
                })
    return data

# 1 shard per 1 "file"
def open_txt(path):
    total_len = 0
    out_len = 0
    data = {}
    with open(path, mode='r', encoding='utf-8', newline='') as file:
        lines = file.read()
        word_len, token_len = get_length(lines)
        print(token_len)
        if token_len < 512:
            return
        else:
            data = {
                "text": lines,
                "word_len": word_len,
                "token_len": token_len
            }

    return data


def check_data(item):
    
    '''
    for test
    
    '''
    data_root = "D:\\cocondenser_data\\mp_test"

    data_key = item["key"]
    data_path = item["path"]
    data_ext = item["ext"]
    
    print(f"working on {item['key']}")
    data_path = os.path.join(data_root, data_path)
    file_list = get_files(data_path, data_ext)
    
    print(file_list)
    
    data_desc = None
    temp_data = []
    for idx, file in enumerate(file_list):
        if data_ext == "csv":
            data = open_csv(file)
            if data:
                temp_data.extend(data)
        elif data_ext == "txt":
            data = open_txt(file)
            if data:
                temp_data.append(data)

    data_desc = {
        "name": data_key,
        "data": temp_data,
        "total_token_len": sum([i["token_len"] for i in temp_data]),
        "total_word_len": sum([i["word_len"] for i in temp_data])
    }
    
    return data_desc

def worker(param, result_queue):
    result = check_data(param)
    result_queue.put(result)

def run_processes(param_list):
    result_queue = multiprocessing.Queue()

    processes = []
    for param in param_list:
        process = multiprocessing.Process(target=worker, args=(param, result_queue))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
        

    results = []
    while not result_queue.empty():
        result = result_queue.get()
        results.append(result)
    
    return results


if __name__ == "__main__":

    data_root = "D:\\cocondenser_data"
    dir_list = [
        {
            "key": "kowiki",
            "path": "kowiki",
            "ext": "csv"
        },
        {
            "key": "namuwiki",
            "path": "Namuwiki231219\\Namuwiki231219",
            "ext": "txt"
        },
        {
            "key": "newspeppermint",
            "path": "NewsPeppermint",
            "ext": "txt"
        },
        # {
        #     "key": "Sciencedaily",
        #     "path": "Sciencedaily240102",
        #     "ext": "txt"
        # }
    ]
    
    dir_list = [
        {
            "key": "kowiki",
            "path": "kowiki",
            "ext": "csv"
        },
        {
            "key": "namu",
            "path": "namuwiki",
            "ext": "txt"
        }
        
    ]
    
    final_result = run_processes(dir_list)
    for item in final_result:
        print("=====================================")
        total_token_len = item["total_token_len"]
        data_key = item["name"]
        total_word_len = item["total_word_len"]
        
        print(f"data name: {data_key}")
        # print(f"total token: {total_token_len}, total word: {total_word_len}")
        print(item)

    
    # 확인해야 하는 것
    '''
    512로 잘랐을 때 살아남는 text의 개수?
    '''
