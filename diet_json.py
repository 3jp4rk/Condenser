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
        pair_offset, pair_length, pair_line_no, pair_doc_id = pair_item
        
        file_no = origin_k.split("_")[0]
        
        assert doc_id == pair_doc_id
        
        pair_item = {
            "en": (file_no, offset, length),
            "ko": (file_no, pair_offset, pair_length)
        }
        
        if doc_id in new_dict:
            new_dict[doc_id].append(pair_item)
        else:
            new_dict[doc_id] = [pair_item]
            
        # if idx % 1000000 == 1:
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
    with open('./new_index_diet__.json', 'w') as newfile:
        json.dump(new_dict, newfile, ensure_ascii=False)

def check_length(new_dict):
    
    delete_key = [k for k, v in new_dict.items() if len(v) < 2]
    print(f"items to delete ... {len(delete_key)}")

    for key in delete_key:
        print(f"deleting doc_id: {key}")
        del new_dict[key]
        
    return new_dict
    

if __name__ == "__main__":
    
    # a = load_json("./sample.json", "sample")
    # b = a.get("1234555")
    # for item in b:
    #     print(item)
    # # print(type(b))
    # quit()
    
    
    # File paths
    # root = '/root/msmarco'
    # # file_paths = ['en_index.json', 'ko_index.json']
    # langs = ["en", "ko"]
    
    # # Start a process for each file
    # loaded_data = []
    # for lang in langs:
    #     file_path = f"{lang}_index.json"
    #     json_dict = load_json(os.path.join(root, file_path), lang)
    #     loaded_data.append(json_dict)
        
    # # new_dict = iter_json(loaded_data, new_dict)
    # new_dict = {}
    # en_dict = loaded_data[0]
    # ko_dict = loaded_data[1]

    start = time.time()

    # new_dict = load_json("./new_index_diet__.json", "prune")
    pruned = load_json("./new_index_diet_pruned.json", "prune")
    
    # print("after pruned: ", len(pruned.items())) # 11132703
    
    # quit()
    # print("working on pruning json . . . ")
    # start = time.time()
    # new_dict = check_length(new_dict)

    # with open('./new_index_diet_pruned.json', 'w') as newfile:
    #     json.dump(new_dict, newfile, ensure_ascii=False)
    
    print(f"{time.time()-start} second took.")
        
    print("prepraring debugging json...")
    start = time.time()
    debug_dict = list(pruned.items())[:12000]
    debug_dict = dict(debug_dict)
    with open('./debugging_12000.json', 'w') as debugging:
        json.dump(debug_dict, debugging, ensure_ascii=False)
    
    # for idx, origin in enumerate(en_dict.items()):

    #     origin_k, (offset, length, line_no, doc_id) = origin

    #     pair_item = ko_dict.get(origin_k)
    #     pair_offset, pair_length, pair_line_no, pair_doc_id = pair_item
        
    #     file_no = origin_k.split("_")[0]
        
        
    #     assert doc_id == pair_doc_id
        
    #     pair_item = {
    #         "en": (file_no, offset, length),
    #         "ko": (file_no, pair_offset, pair_length)
    #     }
        
    #     if doc_id in new_dict:
    #         new_dict[doc_id].append(pair_item)
    #     else:
    #         new_dict[doc_id] = [pair_item]
            
        # if idx % 1000000 == 1:
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

    # print(f"{time.time() - start} seconds took.")
    
    # print("start writing . . . ")
    # start = time.time()
    # print(f"total doc_id len: {len(list(new_dict.keys()))}")
    # write(new_dict)
    # print(f"{time.time() - start} seconds took!")
    print("task compelete!")

