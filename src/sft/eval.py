import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

from huggingface_hub import login
import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
from tqdm.auto import tqdm  
# from utils_ans import extract_answer

def extract_answer(x):
    return x.strip().split("\n")[-1].replace("###", "").strip()

MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, temp=0.7):
    
    os.makedirs(args.result_file[:args.result_file.rfind("/")], exist_ok=True)

    gsm8k_ins = []
    gsm8k_answers = []
 
    # Check if it already exists.
    try:
        already_done = [json.loads(x) for x in open(args.result_file)]
    except:
        already_done = [] 

    with open(data_path,"r+", encoding="utf8") as f:
        for idx, elem in enumerate(json.load(f)):
            
            if idx < len(already_done):
                continue
            
            gsm8k_ins.append(elem["question"].strip() + "\n")
            
            if "###" not in elem['steps'][-1]:
                gsm8k_answers.append("\n".join(elem["steps"]) + "\n### " + elem["answer"])
            else:
                gsm8k_answers.append("\n".join(elem["steps"]) + "\n" + elem["answer"])
       
    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    # stop_tokens = ["\n\n", "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop_tokens = []
        
    if temp == 0.7:
        n = 100
    else:
        n = 1
    
    sampling_params = SamplingParams(temperature=temp, top_p=1, max_tokens=512, stop=stop_tokens, n=n)
    print('sampling =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size, enforce_eager=False, gpu_memory_utilization=0.2)
    result = []
    res_completions = []
    
    for idx, (prompt, prompt_answer) in tqdm(enumerate(zip(batch_gsm8k_ins, gsm8k_answers))):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)

        for num, output in enumerate(completions):
            prompt = output.prompt
            all_texts = [out.text for out in output.outputs]
            res_completions.append(all_texts)

            answer = gsm8k_answers[idx*batch_size + num]
            dict_ = {"prompt": prompt, "preds": all_texts, "answer": answer}

            with jsonlines.open(args.result_file, 'a') as writer:
                writer.write(dict_)            
    
    li = [json.loads(x) for x in open(args.result_file)]

    sa = [] # singgle acc

    for x in li:
        if 'answers' in x:
            lbl = str(x['answers'])
        else:
            lbl = str(x['answer'])

        # import pdb; pdb.set_trace()

        answers = [str(extract_answer(pred)) for pred in x['preds']]
        eq_answers = [ans == extract_answer(lbl) for ans in answers]
        sa.append(eq_answers.count(True) / len(eq_answers))
        # import pdb; pdb.set_trace()
    final_acc = sum(sa) / len(sa)
    print(args.result_file, "Final Acc:", final_acc)


    with open("results.txt", "a") as f:
        f.write(args.result_file + "\t" + str(round(final_acc, 4)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=1000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--result_file", type=str, default="./new_csqa_large.jsonl")  # tensor_parallel_size
    parser.add_argument("--temp", type=float, default=0.7) 

    return parser.parse_args()

if __name__ == "__main__":
    # Login First.
    # login(token="your_huggingface_token")

    args = parse_args()
    MODEL = args.model
    DATA = args.data_file
    
    BATCH_SIZE = 1000000000
    TEMP = 0
    args = parse_args()
    gsm8k_test(model=MODEL, data_path=DATA, start=0, end=1000000000, batch_size=BATCH_SIZE, temp=TEMP)
    