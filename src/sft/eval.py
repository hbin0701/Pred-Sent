import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

from huggingface_hub import login
import argparse
import json
import re
import jsonlines
import csv
import pathlib
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
from tqdm.auto import tqdm  
# from utils_ans import extract_answer

NEWLINE_TOKEN = "<|new_line|>"

def normalize_newlines_for_prompt(text: str) -> str:
    return text.replace("\n", NEWLINE_TOKEN)

def denormalize_newlines_from_model(text: str) -> str:
    return text.replace(NEWLINE_TOKEN, "\n")

def extract_answer(x):
    x = denormalize_newlines_from_model(x)
    # Prefer explicit "### <LETTER>" pattern if present
    m = re.search(r"###\s*([A-E])", x)
    if m:
        return m.group(1)
    # Fallback: take last segment and strip markers
    tail = x.strip().split("<|new_line|>")[-1].replace("###", "").strip()
    # Keep only a single capital MC letter if present
    m2 = re.search(r"\b([A-E])\b", tail)
    return m2.group(1) if m2 else tail

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
            prompt_q = normalize_newlines_for_prompt(elem["question"].strip())
            gsm8k_ins.append(prompt_q + NEWLINE_TOKEN)
            # Build a clean gold answer string: steps followed by exactly one "<|new_line|>### <LETTER>"
            steps_joined = NEWLINE_TOKEN.join(elem["steps"])
            # Remove any trailing answer-like suffix already present in steps (supports both tokens)
            steps_joined = re.sub(r'(?:<\|new_line\|>|<|new_line|>)+###\s*[A-E]\s*$', '', steps_joined)
            raw_gold = str(elem.get("answer", "")).strip()
            m = re.search(r"###\\s*([A-E])", raw_gold)
            if m:
                gold_letter = m.group(1)
            else:
                m2 = re.search(r"\\b([A-E])\\b", raw_gold)
                gold_letter = m2.group(1) if m2 else raw_gold[-1:]  # best-effort
            ans = f"{steps_joined}{NEWLINE_TOKEN}### {gold_letter}"
            gsm8k_answers.append(normalize_newlines_for_prompt(ans))
       
    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    # stop_tokens = ["<|new_line|><|new_line|>", "Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    stop_tokens = []
        
    if temp == 0.7:
        n = 100
    else:
        n = 1
    
    sampling_params = SamplingParams(
        temperature=temp,
        top_p=1,
        max_tokens=512,
        stop=stop_tokens,
        n=n,
        skip_special_tokens=False,  # keep <|new_line|> in outputs
        spaces_between_special_tokens=True
    )
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
    
    # Optionally write checkpoint accuracy to CSV
    if getattr(args, "output_csv", None):
        csv_path = args.output_csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
        file_exists = os.path.exists(csv_path)
        ckpt_name = os.path.basename(os.path.normpath(args.model))
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["checkpoint", "accuracy"])
            writer.writerow([ckpt_name, f"{final_acc:.6f}"])


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
    parser.add_argument("--output_csv", type=str, default="")  # path to append per-checkpoint accuracy

    return parser.parse_args()

if __name__ == "__main__":
    # Login First.
    # login(token="your_huggingface_token")
    args = parse_args()
    MODEL = args.model
    DATA = args.data_file
    
    args = parse_args()
    gsm8k_test(
        model=MODEL,
        data_path=DATA,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        temp=args.temp
    )
    