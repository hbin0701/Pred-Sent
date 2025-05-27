import json

def ext(str_):
    return str_.split("=")[-1].replace(">>", "") 
    
# Don't edit test.
for file_name in ["train.json", "valid.json"]:
    
    li = json.load(open(file_name))
    all_elems = []
    
    for elem in li:
        if ext(elem['steps'][-1]) != elem['answer']:
            continue
        else:
            all_elems.append(elem)
    
    print("Before:", len(li), "After:", len(all_elems), "Percentage:", len(li) / len(all_elems) * 100)
    json.dump(all_elems, open(file_name, "w"), indent=4)