from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import os

# bert = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# bert.save_pretrained("./bert-base-uncased")

# bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
# bert_tok.save_pretrained("./bert-origin")


# check outlier weight value 
# for k, v in bert.state_dict().items():
    
#     if torch.any(torch.isnan(v)):
#         print(k)


## dict item에 nan이 들어오기 시작하면 step수 print, save하지 말고 그냥 넘어가게 해야겠다 


def check_anomal(model_dict, ckpt):
    for k, v in model_dict.items():
        if torch.any(torch.isnan(v)):
            print(f"nan occurred in {ckpt}: \n")
            print(k)

    print("=================================================")
    
if __name__ == "__main__":
    
    a = {'a': 1, 'b': 2}
    
    print(a.values())
    
    quit()
    

    root = "/root/condenser_pretrain"
    ckpt_list = [item for item in os.listdir(root) if "checkpoint-" in item]

    for ckpt in ckpt_list:
        ckpt_path = os.path.join(root, ckpt)
        saved_model = AutoModelForMaskedLM.from_pretrained(
        ckpt_path,
        local_files_only=True)

        head_path = os.path.join(ckpt_path, "model.pt")
        head_weight = torch.load(head_path)
        print(head_weight.keys())
        load_result = saved_model.load_state_dict(head_weight, strict=False)
        
        check_anomal(saved_model.state_dict(), ckpt)
        

        # print(f"{k}:\n{v}")