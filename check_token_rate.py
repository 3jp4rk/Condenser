import os
import csv
import pandas as pd
import glob
from bs4 import BeautifulSoup


import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig

if __name__ == "__main__":

    # 사용할 tokenizer load하여 문서 5개 각각 단어 수 & 토큰 수 세서 평균 비율 계산
    tokenizer_ours = AutoTokenizer.from_pretrained(
        './merged_tokenizer/merged_tokenizer/', 
        local_files_only=True) # LlamaTokenizerFast

    tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')

    path = "D:\\cocondenser_data\\Namuwiki231219\\Namuwiki231219\\%2B%28%EC%9D%8C%EB%B0%98%29.txt" # 나무위키

    # kowiki 기준 
    # path = "D:\\cocondenser_data\\kowiki\\part_00000.csv"
    check_ = []
    with open(path, mode='r', encoding='utf-8', newline='') as file:
        lines = file.read()
        # print(lines)
        print(type(lines))
        print(len(lines))
        quit()
    # with open(path, mode='r', encoding='utf-8', newline='') as file:
    #     lines = csv.reader(file)
    #     for idx, line in enumerate(lines):
    #         if idx == 0:
    #             continue
    #         if idx == 101: # 0번이 'text'라서
    #             break
    #         check_.append(line)


    count_result = []
    for item in check_:
        # tokenizer length
        text = " ".join(word for word in item)
        tokenized = tokenizer_ours(text, add_special_tokens=False)
        # tokenized_ = tokenizer_ours(text)
        input_ids = tokenized['input_ids']
        # input_ids_ = tokenized_['input_ids']
        # print(input_ids)
        # print(input_ids_)
        # print(len(input_ids))
        # print(input_ids[0])
        # print(input_ids[-1])
        
        # print(text.split())
        
        token_len = len(input_ids)
        word_len = len(text.split())
        
        count_result.append({
            "token_len": token_len,
            "word_len": word_len,
            "ratio": token_len/word_len if token_len >= word_len else word_len/token_len
        })
        
    # print(count_result)
    
    '''
    [{'token_len': 4156, 'word_len': 1319, 'ratio': 3.150871872630781}, {'token_len': 672, 'word_len': 218, 'ratio': 3.0825688073394497}, {'token_len': 1720, 'word_len': 504, 'ratio': 3.4126984126984126}, {'token_len': 146, 'word_len': 56, 'ratio': 2.607142857142857}, 
    {'token_len': 512, 'word_len': 154, 'ratio': 3.324675324675325}]
    '''
    
    '''
    대략적 비율이 2.6~3.41 
    '''
    
    # 100개 평균
    # average token length & word length
    lst = [item["ratio"] for item in count_result]
    print(f"100개 average: {sum(lst)/len(lst)}") # 3.03
    print(f"min value: {min(lst)}") # 2.28
    print(f"min value: {max(lst)}") # 4.76
    
    
    
    
    
    
    # soup = BeautifulSoup(content, 'html.parser')
    # non_specific_texts = [element.get_text(strip=True) for element in soup.find_all(lambda tag: tag.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span', 'td', 'tr', 'time', 'table', 'title'])]

    # # plain_text = soup.get_text(separator='\n', strip=True)
    # # plain_text = soup.find_all(text=True)
    # # pure_text_list = [text.strip() for text in soup.find_all(text=True)]

    # print('\n'.join(non_specific_texts))

    # # for text in pure_text_list:
    # #     print(text)
