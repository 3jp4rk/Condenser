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
    
    return  {"lang": lang, "data": data}

# search
def _check(ko_items, origin_k, doc_id, line_no):
    
    file_no = origin_k.split("_")[0]
    
    found = False
    target = tuple()
    for item in ko_items:
        pair_k, (pair_offset, pair_length, pair_line_no, pair_doc_id) = item
        pair_file_no = pair_k.split("_")[0]
        if pair_doc_id == doc_id:
            if pair_file_no == file_no:
                if pair_line_no == line_no:
                    found = True
                    target = item
                    break

    return found, target
                

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
    
    # Start a process for each file
    # loaded_data = []
    # for lang in langs:
    #     file_path = f"{lang}_index.json"
    #     file_path = os.path.join(root, file_path)
    #     loaded_data.append(load_json(file_path, lang))
    
    new_dict = load_json("./new_index_diet__.json", "prune")

    print("working on pruning json . . . ")
    start = time.time()
    new_dict = check_length(new_dict["data"])

    with open('./new_index_diet_pruned.json', 'w') as newfile:
        json.dump(new_dict, newfile, ensure_ascii=False)
    
    print(f"{time.time()-start} second took.")
        
    print("prepraring debugging json...")
    start = time.time()
    debug_dict = list(new_dict.items())[:2000]
    debug_dict = dict(debug_dict)
    with open('./debugging.json', 'w') as debugging:
        json.dump(debug_dict, debugging, ensure_ascii=False)

    # Join processes
    # for p in processes:
    #     p.join()

    # Process loaded data (example)
    # print("working on dieting json . . . ")
    # start = time.time()
    # new_dict = {}
    # with open('./new_index_diet_pruned.json', 'w') as newfile:
        
    #     en_items = loaded_data[0]["data"].items()
    #     ko_items = loaded_data[1]["data"].items()
        
    #     assert len(list(en_items)) == len(list(ko_items))
        
    #     for idx, origin in enumerate(en_items):
            
    #         pair_k = None
    #         pair_offset = None
    #         pair_length = None
    #         pair_line_no = None
    #         pair_doc_id = None
            
    #         origin_k, (offset, length, line_no, doc_id) = origin
            
    #         # get other language pair
    #         found, pair = _check(ko_items, origin_k, doc_id, line_no)
            
    #         if found:
    #             pair_k, (pair_offset, pair_length, pair_line_no, pair_doc_id) = pair
    #             try:
    #                 assert doc_id == pair_doc_id
    #             except:
    #                 print(f"{idx, pair_doc_id} didn't match!")
    #                 continue
    #         else:
    #             print("NOT FOUND MATCH")
    #             print(origin)
    #             continue

    #         pair_item = {
    #                         "en": (origin_k, length, line_no),
    #                         "ko": (pair_k, pair_length, pair_line_no)
    #                     }
            
    #         if doc_id in new_dict:
    #             new_dict[doc_id].append(pair_item)
    #         else:
    #             new_dict[doc_id] = [pair_item]
                
    #         if idx % 100 == 1:
    #             print(f"working on {idx} row... ")
    #             try:
    #                 print(f"{idx}: {len(new_dict[doc_id])}")
    #             except:
    #                 print("not generated...")
    #                 pass
    #     new_dict = check_length(new_dict)
    #     json.dump(new_dict, newfile, ensure_ascii=False)

    print(f"{time.time() - start} seconds took.")
    print("task compelete!")

