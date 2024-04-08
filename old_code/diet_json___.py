import json
from multiprocessing import Process, Queue
import os
import time

def load_json(file_path, lang):
    
    print(f"{file_path} loading start!")
    
    start = time.time()
    with open(file_path, 'rb') as file:
        data = json.load(file)
    # print(data)
    print(f"{file_path} loaded complete! {time.time() - start} seconds took.")
    
    # return  {"lang": lang, "data": data}
    return data


# 한글 pid
# 영어 pid는 동일하구나 ㅋㅋ 


# 영어랑 pid 동일한 ko를 찾으면 되는 거아냐?


# ko:                                                                             │···································································
# 30_753639335                                                                    │···································································
# [1297143221, 1160, 61717344, '26_1999986136']                                   │···································································
# ================================                                                │···································································
# en:                                                                             │···································································
# 38_283710562                                                                    │···································································
# [283710562, 468, 76652483, '33_602842870']  


# 30_753639335                                                                    │···································································
# [1297143221, 1160, 61717344, '26_1999986136']                                   │···································································
# ================================                                                │···································································
# en:                                                                             │···································································
# 38_283710562                                                                    │···································································
# [283710562, 468, 76652483, '33_602842870'] 
# [746735259, 842, 6984530, '03_459946499']

  
def iter_json(json_dicts):
    
    en_dict = json_dicts[0]
    ko_dict = json_dicts[1]

    start = time.time()
    
    for idx, origin in enumerate(en_dict.items()):

        origin_k, (offset, length, line_no, doc_id) = origin

        pair_item = ko_dict.get(origin_k)
        pair_k, (pair_offset, pair_length, pair_line_no, pair_doc_id) = pair_k
        
        assert doc_id == pair_doc_id
        
        pair_item = {
            "en": (offset, length),
            "ko": (offset, length)
        }
        
        if doc_id in new_dict:
            new_dict[doc_id].append(new_item)
            else:
                new_dict[doc_id][lang] = [new_item]
        else:
            new_dict[doc_id] = {
                f"{lang}": [new_item]
            }
            
        # if idx % 10000 == 1:
        #     print(f"working on {idx} row... ")
        #     try:
        #         if len(new_dict[doc_id][lang]) > 0:
        #             print(f"{idx}: {new_dict[doc_id][lang][0]}")
        #         else:
        #             print("passages didn't append yet...")
        #     except:
        #         import traceback
        #         print(traceback.format_exc())
        #         pass

    print(f"{time.time() - start} seconds took.")
    return new_dict      


def write(new_dict):
    with open('./new_index_diet.json', 'w') as newfile:
        json.dump(new_dict, newfile, ensure_ascii=False)
    

if __name__ == "__main__":
    
    # a = load_json("./sample.json", "sample")
    # b = a.get("1234555")
    # for item in b:
    #     print(item)
    # # print(type(b))
    # quit()
    
    
    # File paths
    root = '/root/msmarco'
    # file_paths = ['en_index.json', 'ko_index.json']
    langs = ["en", "ko"]
    
    # Start a process for each file
    loaded_data = []
    for lang in langs:
        file_path = f"{lang}_index.json"
        json_dict = load_json(os.path.join(root, file_path), lang)
        loaded_data.append(json_dict)
        
    new_dict = iter_json(loaded_data)
        

    print("start writing . . . ")
    start = time.time()
    print(f"total doc_id {len(new_dict)}")
    write(new_dict)
    print(f"{time.time() - start} seconds took!")
    print("task compelete!")

