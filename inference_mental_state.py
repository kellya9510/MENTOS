from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import argparse
import random, json, os
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from peft import (
    PeftModel,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

is_first = True

def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

def create_instruction(dialog, mental_state_type):

    conversation = ""
    for d_idx, d in enumerate(dialog["conversation"][:-1]):
        num= d_idx + 1
        conversation += f"({num}) "+d["speaker"].replace("usr", "Client").replace("seeker", "Client").replace("sys", "Assistant").replace("supporter", "Assistant")+": "+d["text"].strip() + "\n"
    
    if  mental_state_type == "Belief":
        input_mental_states = ""
    else:
        input_mental_states = "[Assistant's Mental States]\n"    
        #print(dialog)
        dialog["mental_state"] = {"Belief": dialog["mental_state"]["Belief"]} if mental_state_type == "Emotion" else {"Belief": dialog["mental_state"]["Belief"], "Emotion": dialog["mental_state"]["Emotion"], "Desire": "The assistant's desire is to reduce the client’s emotional distress and help them cope with challenges."}
        
        for k, v in dialog["mental_state"].items():
            if k == "Emotion": v = "Rating from 0 (not present) to 3 (intense)\n"+v
            input_mental_states += "{}: {}\n".format(k, v.replace("’", "'"))
        input_mental_states += "\n"

    
    mental_state_question_dict = {
        "Belief": "What does the assistant believe about the client’s situation or internal state based on the client's last utterance?",
        "Emotion": "What emotional reaction might the assistant have after the client's last utterance, based on the assistant's belief and how the conversation has unfolded? Rate each basic emotion from 0 (not present) to 3 (intense). Then rate mixed emotions only if both contributing basic emotions are rated 2 or higher and are explicitly reflected in the assistant's wording.",
        "Intent": "What is the assistant's intent following the client's last utterance, based on the assistant's belief, emotional reaction, and desire?"
    }

    prompt = f"\n[Dialogue history]\n{conversation}\n\n{input_mental_states}[Question] {mental_state_question_dict[mental_state_type]}"
    
    return prompt
    

def make_prompt(dialog, mental_state_type, is_example = False):    
    system_prompt = "You are an emotional support assistant with expertise in client-centered, psychodynamic, and cognitive behavioral therapies."
    messages = [{"role": "system", "content": system_prompt.strip()}]

    user_prompt = "Given a dyadic dialogue clip between an assistant and a client who seeks help in relieving emotional distress, your task is to first infer the client's mental states in their last utterance. Then, infer the assistant's potential mental states that may arise in response to the client's last utterance. These mental states include Belief, Emotion, Desire, and Intent, and should be inferred in that order.\n\n"
    user_prompt += f"Clearly and concisely answer the assistant’s {mental_state_type} (no more than 40 words) of the following conversation clip. "
    user_prompt += f"The conversation clip is:\n\n" 

    instruction = create_instruction(dialog, mental_state_type)
    user_prompt += instruction.strip()
    
    if mental_state_type == "Emotion":
        user_prompt += "\nEmotion:\n[Basic] Sadness (opposite Joy): <score>, Disgust (opposite Trust): <score>, Anger (opposite Fear): <score>, Anticipation (opposite Surprise): <score>, Joy (opposite Sadness): <score>, Trust (opposite Disgust): <score>, Fear (opposite Anger): <score>, Surprise (opposite Anticipation): <score>\n[Mixed] Hopelessness (sadness + fear): <score>, Remorse (sadness + disgust): <score>, Disappointment (sadness + surprise): <score>, Sentimental (sadness + trust): <score>, Jealousy (sadness + anger): <score>, Pessimism (sadness + anticipation): <score>, Embarrassment (disgust + fear): <score>, Pride (anger + joy): <score>, Nervousness (anticipation + fear): <score>, Delight (joy + surprise): <score>, Gratitude/Love/Caring (joy + trust): <score>, Hope/Optimism (joy + anticipation): <score>, Guilt (joy + fear): <score>, Curiosity (surprise + trust): <score>\n"
    
    messages.append({"role": "user", "content": user_prompt})
    return messages

def test(args):
    args.check_point = str(args.check_point)
    ckpt_name = f"checkpoint-{args.check_point}"
    ckpt_dir  = os.path.join(args.model_dir, args.mental_state_type, ckpt_name)
    if args.output_file == "none":
        args.output_file = os.path.join(os.path.join(args.model_dir, args.mental_state_type, "baseline.jsonl" if args.is_init else f"output_{args.check_point}.jsonl"))
    else:
        args.output_file = os.path.join(os.path.join(args.model_dir, args.mental_state_type, args.output_file))
    
    if args.is_init: 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    if args.is_init:
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.eval()
    device = 'cuda' 
    model.to(device)

    if args.mental_state_type =='All':
        Mental_list = ['Belief', 'Emotion', 'Intent']
    else:
        Mental_list = [args.mental_state_type]
    
    
    for ms in Mental_list:
        args.mental_state_type = ms
        if args.mental_state_type == "Emotion":
            args.test_file = args.output_file.replace(args.mental_state_type, "Belief")
            print('test file: ', args.test_file)
            with open(args.test_file, "r") as f: 
                origin_dataset = [json.loads(d) for d in f]
        elif args.mental_state_type == "Intent":
            args.test_file = args.output_file.replace(args.mental_state_type, "Emotion")
            print('test file: ', args.test_file)
            with open(args.test_file, "r") as f: 
                origin_dataset = [json.loads(d) for d in f]
        else:
            if "json" == args.test_file.split(".")[-1]:
                with open(args.test_file, "r") as f: 
                    dataset = json.load(f) 
                origin_dataset = []
                for ex_idx, example in enumerate(dataset):
                    new_ex = {"data_idx": str(ex_idx), "conversation":[
                            {"text": d["text"],
                            "speaker": "supporter" if d["speaker"]=="sys" else "seeker"}
                            for d in example["dialog"]
                        ]}
                    origin_dataset.append(new_ex)

            elif "jsonl" == args.test_file.split(".")[-1]:
                with open(args.test_file, "r") as f: 
                    origin_dataset = [json.loads(d) for d in f]
                

        def preprocess_fn(example):
            args.mental_state_type = ms            
            if args.mental_state_type in ["Belief", "Emotion", "Intent"]:
                if "dialog" in example.keys():
                    conv = [
                        {"text": d["text"],
                        "speaker": "supporter" if d["speaker"]=="sys" else "seeker"}
                        for d in example["dialog"]
                    ]
                else:
                    conv = example["conversation"]
                messages = make_prompt(
                    {"conversation": conv} if args.mental_state_type == "Belief" else {"conversation": conv, "mental_state": example["mental_state"]},
                    args.mental_state_type,
                    args.is_example
                )
            else:
                messages = make_prompt(example,args.mental_state_type,is_example=False)
            # system+user only 
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            tokenized_input = tokenizer(prompt_str, truncation=True)
            global is_first
            if is_first:
                #print(conv)
                print(messages)
                #print(tokenized_input)
                is_first = False
            return tokenized_input
    
        test_dataset = load_dataset('json', data_files={'test': args.test_file})['test']
        test_data = test_dataset.map(preprocess_fn, remove_columns=test_dataset.column_names)
    
        collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

        dataloader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator
        )
        
        with open(args.output_file, "w") as fout:
            idx = 0
            for batch_idx, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc=f"create {args.mental_state_type}..."):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.no_grad():
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=400,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                        temperature= 0.5,
                        top_k=1,
                    )

                input_lens = attention_mask.sum(dim=1)
                for i in range(gen_ids.size(0)):
                    out_ids = gen_ids[i, input_lens[i]:]
                    text = tokenizer.decode(out_ids, skip_special_tokens=True)

                    data_idx = batch_idx * args.batch_size + i
                    
                    if data_idx == 0:
                        print(args.mental_state_type)
                        print("*****Input*****")
                        print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                        print("*****Output*****")
                        print(text)
                    
                    record = {"data_idx": origin_dataset[data_idx]["data_idx"], "mental_state": {}}
                    if args.mental_state_type in ["Emotion", "Intent"]:
                        record["mental_state"]["Belief"] = origin_dataset[data_idx]["mental_state"]["Belief"].strip()
                    if args.mental_state_type == "Intent":
                        record["mental_state"]["Emotion"] = origin_dataset[data_idx]["mental_state"]["Emotion"].strip()
                        if "Emotion:" in record["mental_state"]["Emotion"]: record["mental_state"]["Emotion"].split("Emotion:")[-1] 
                   
                    if args.mental_state_type == "Belief": 
                        record["mental_state"][args.mental_state_type] = text.split("Belief:")[1] if "Belief:" in text else text.strip()
                    elif args.mental_state_type == "Emotion":
                        record["mental_state"][args.mental_state_type] = text.split("Emotion:")[-1] if "Emotion:" in text else text.strip()
                    elif args.mental_state_type == "Intent":
                        record["mental_state"][args.mental_state_type] = text.split("Intent:")[1] if "Intent:" in text else text.strip()
                    else:
                        text  = text.split("Answer:")[1] if "Answer:" in text else text.strip()
                        if (len(text.split("\n\n")) > 1) and ("Now, generate one concise and relevant inference (no more than 40 words) of the" in text.split("\n\n")[1].strip()):
                            text = text.split("\n\n")[0].strip()
                        record["mental_state"][args.mental_state_type] = text.strip()

                    if args.mental_state_type in ["Belief", "Emotion", "Intent"]: 
                        record['correct'] = origin_dataset[data_idx]["mental_state"] if args.mental_state_type == "Belief" else origin_dataset[data_idx]['correct']
                    record['conversation'] = origin_dataset[data_idx]['conversation']
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    idx += 1

        print(f"Save: {args.output_file}\n")            
        

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--cache_dir", type=str, default='./models')
    cli_parser.add_argument("--is_init", type=bool, default=False)

    cli_parser.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    cli_parser.add_argument("--model_dir", type=str, default="Llama-2-7b-chat-hf/model/Full_FT")
    
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--mental_state_type", type=str, default='All') # ['Belief', 'Emotion', 'Intent', 'All']
    cli_parser.add_argument("--check_point", type=int, default=0) 
    cli_parser.add_argument("--output_file", type=str, default='none')
    
    cli_parser.add_argument("--test_file", type=str, default='data/test.jsonl')
    cli_parser.add_argument("--is_example", type=bool, default=False)
    
    cli_parser.add_argument("--batch_size", type=int, default=4)
    #cli_parser.add_argument("--num_gpu", type=int, default=0)
    
    args = cli_parser.parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu) 
    print("Device count:", torch.cuda.device_count())      
    print("Current device index:", torch.cuda.current_device())  
    print("Device name:", torch.cuda.get_device_name())     
    print("CUDA Version:", torch.version.cuda)
    print("is_BF16:",torch.cuda.is_bf16_supported())

    set_seed(args.seed)
    test(args)
    
    print(f"evaluate {args.output_file} done!")
    
    
