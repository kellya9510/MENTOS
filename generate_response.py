import torch
from datasets import load_dataset
from tqdm import tqdm
import argparse
import random, json, os
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator
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

def create_instruction(dialog, is_strategy=False, add_strategy=False, is_ablation=False, remove_for_ablation=None):
    conversation = ""
    for d_idx, d in enumerate(dialog["conversation"][:-1]):
        num= d_idx + 1
        conversation += f"({num}) "+d["speaker"].replace("usr", "Client").replace("seeker", "Client").replace("sys", "Assistant").replace("supporter", "Assistant")+": "+d["text"].strip() + "\n"
    
    info = ""
    if "comet" in dialog.keys():
        info = ["After posting the last utterance,"]
        if "xIntent" in dialog["comet"].keys()  and isinstance(dialog["comet"]["xIntent"], list):
            info.append("the client intent to {},".format(dialog["comet"]["xIntent"][0].strip()))
        elif "xNeed" in dialog["comet"].keys()  and isinstance(dialog["comet"]["xNeed"], list):
            info.append("the client need to {},".format(dialog["comet"]["xNeed"][0].strip()))
        elif "xWant" in dialog["comet"].keys()   and isinstance(dialog["comet"]["xWant"], list):
            info.append("the client want to {},".format(dialog["comet"]["xWant"][0].strip()))
        elif "xReact" in dialog["comet"].keys() and isinstance(dialog["comet"]["xReact"], list):
            info.append("the client may feel {},".format(dialog["comet"]["xReact"][0].strip()))
        elif "xEffect" in dialog["comet"].keys()  and isinstance(dialog["comet"]["xEffect"], list):
            info.append("the client would {},".format(dialog["comet"]["xEffect"][0].strip()))
        if info != "": 
            if len(info)>2: info = " ".join(info[:-1]) + " and "+info[-1][:-1] + "."
            elif len(info)==1: info = ""
            else: 
                info= " ".join(info)
                info = info[:-1]+"."
    elif "dialect" in dialog.keys():
        cause = dialog["dialect"]["Cause"]
        subev = dialog["dialect"]["SubEv"]
        react = dialog["dialect"]["React"]
        prere = dialog["dialect"]["Prere"]
        motiv = dialog["dialect"]["Motiv"]
        
        info = f"""The underlying cause of the last utterance (the reason contributing to the utterance stated by the client) is: {cause}

The subsequent event about the assistant that happens or could happen following the last the utterance stated by the client :{subev}

The prerequisite (or assumed prior state) that enables the last utterance stated by the client to occur is: {prere}

The underlying emotion or human drive that motivates the last utterance stated by the client is: {motiv}

The possible emotional reaction of the client in response to the last utterance stated by the client is : {react}"""
    elif "doctor" in dialog.keys():
        info = dialog["doctor"].replace("Person A", "Client").replace("Person B", "Assistant").replace("Subquestion", "Question").replace("Subanswer", "Answer").split("\n")
        if len(info) == 1:
            info = ""
        elif len(info) == 6:
            info = [i.strip() for i in info]
            info = "{}\n{}\n\n{}\n{}\n\n{}\n{}".format(info[0], info[1], info[2], info[3], info[4], info[5])
    elif "sibyl" in dialog.keys():
        info = []
        if dialog["sibyl"]["ChatGPT_cause"] != "":
            cause = dialog["sibyl"]["ChatGPT_cause"]
            info.append(f"The underlying cause of the client's last utterance (the reason contributing to the utterance stated by the client) is: {cause}")
        if dialog["sibyl"]["ChatGPT_subs"] != "":
            sub = dialog["sibyl"]["ChatGPT_subs"]
            info.append(f"The subsequent event about the assistant that happens or could happen following the last the utterance stated by the client : {sub}")
        if dialog["sibyl"]["ChatGPT_emo"] != "":
            emo = dialog["sibyl"]["ChatGPT_emo"]
            info.append(f"The possible emotional reaction of the client in response to the last utterance stated by the client is : {emo}")
        if dialog["sibyl"]["ChatGPT_intent"] != "":
            intent = dialog["sibyl"]["ChatGPT_intent"]
            info.append(f"The assistant's intent to post the last utterance according to the emotion reaction of the client is : {intent}")
        info = "\n\n".join(info)

    elif "mental_state" in dialog.keys():
        if is_ablation:
            if remove_for_ablation=="Belief":
                info = """The assistant’s possible emotional reaction following the client’s last utterance is as follows (rated from 0 to 3):\n{}\n\n{}""".format(dialog["mental_state"]["Emotion"].strip(), dialog["mental_state"]["Intent"].strip())
            elif remove_for_ablation=="Emotion":
                info = """{}\n\n{}""".format(dialog["mental_state"]["Belief"].strip(), dialog["mental_state"]["Intent"].strip())
            elif remove_for_ablation=="Intent":
                info = """{}\n\nThe assistant’s possible emotional reaction following the client’s last utterance is as follows (rated from 0 to 3):\n{}""".format(dialog["mental_state"]["Belief"].strip(), dialog["mental_state"]["Emotion"].strip())
        else:
            info = """{}\n\nThe assistant’s possible emotional reaction following the client’s last utterance is as follows (rated from 0 to 3):\n{}\n\n{}""".format(dialog["mental_state"]["Belief"].strip(), dialog["mental_state"]["Emotion"].strip(), dialog["mental_state"]["Intent"].strip())

    strategy_info = {
        "Question": "Asking for information related to the problem to help the client articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get specific information.",
        "Restatement or Paraphrasing": "A simple, more concise rephrasing of the client’s statements that could help them see their situation more clearly.",
        "Reflection of feelings": "Articulate and describe the client’s feelings.",
        "Self-disclosure": "Divulge similar experiences that you have had or emotions that you share with the client to express your empathy.",
        "Affirmation and Reassurance": "Affirm the client’s strengths, motivation, and capabilities and provide reassurance and encouragement.",
        "Providing Suggestions or Information": "Provide suggestions for change and useful information, such as data, facts, opinions, or resources. Be careful not to overstep or tell the client what to do.",
        "Greeting": "Politely open or close the conversation with a brief social exchange.",
        "Others": "Exchange pleasantries and use other support strategies that do not fall into the above categories.",
        }
    strategy_list = [k+": "+v for k, v in strategy_info.items()]
    numbering = "abcdefghijklmn"
    strategy_list = ["({}) ".format(numbering[i])+v for i, v in enumerate(strategy_list)]

    system_prompt = f"You are well aware that emotional support follows a three-stage process: exploration, providing comfort, and taking action. You possess the expertise to skillfully choose the appropriate strategy to gradually alleviate the negative emotions of those seeking help. There is a dyadic dialogue clip between an assistant and a client who seeks for help in relieving emotional distress.\nPlease generate a response that incorporates relevant common-sense knowledge: \n{info}"
    user_prompt = f"[Dialogue]\n{conversation}"
    user_prompt += f"\n\nHere's a possible the assistant's response in no more than 30 words: "        
    prompt = [system_prompt, user_prompt]    
        
    return prompt
    

def make_prompt(dialog, is_strategy, add_strategy=False, is_ablation=False, remove_for_ablation=None):    
    instruction = create_instruction(dialog, is_strategy=is_strategy, add_strategy=add_strategy, is_ablation=is_ablation, remove_for_ablation=remove_for_ablation)
    system_prompt = "You are an emotional support assistant with expertise in client-centered, psychodynamic, and cognitive behavioral therapies. "
    system_prompt += instruction[0].strip()
    
    messages = [{"role": "system", "content": system_prompt.strip()}]
    messages.append({"role": "user", "content": instruction[1].strip()})
    return messages


def test(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = 'cuda'
    model.to(device)
    
    def preprocess_fn(example):
        if "dialog" in example.keys():
            conv = [
                {"text": d["text"],
                "speaker": "supporter" if d["speaker"]=="sys" else "seeker"}
                for d in example["dialog"]
            ]
            example["conversation"] = conv

        if "baseline" in args.output_file:
            example = {"data_idx": example["data_idx"], "conversation": example["conversation"]}

        messages = make_prompt(example, 
                               is_strategy = args.is_strategy, add_strategy=args.add_strategy, 
                               is_ablation=args.is_ablation, remove_for_ablation=args.remove_for_ablation)
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        tokenized_input = tokenizer(prompt_str, max_length=4096, truncation=True, padding='max_length', return_tensors='pt')
        for key in tokenized_input.keys(): tokenized_input.update({key: list(tokenized_input[key][0])})
        
        global is_first
        if is_first:
            print("[Input]")
            decoded = tokenizer.decode([
                l for l in tokenized_input['input_ids'] if l != tokenizer.pad_token_id
            ])
            print(decoded)
            is_first = False
        return tokenized_input

    test_dataset = load_dataset('json', data_files={'test': args.test_file})['test']
    test_data = test_dataset.map(preprocess_fn, remove_columns=test_dataset.column_names)    
            
    dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    with open(args.test_file, "r") as f:
        if args.test_file.split(".")[-1] == 'jsonl':
            origin_dataset = [json.loads(d) for d in f]
        elif args.test_file.split(".")[-1] == 'json':
            dataset = json.load(f)
            origin_dataset = []
            for d_idx, data in enumerate(dataset):
                conv = [
                {"text": d["text"],
                "speaker": "supporter" if d["speaker"]=="sys" else "seeker"}
                for d in data["dialog"]
                ]
                origin_dataset.append({"data_idx": d_idx, "conversation": conv})
    
    with open(args.output_file, "w") as fout:
        idx = 0
        for batch_idx, batch in tqdm(enumerate(dataloader), total = len(dataloader), desc=f"generate..."):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.new_max_token,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    top_p=1,top_k=50,num_beams=1,
                    temperature= 1.0,
                    #top_k=1,
                )

            input_lens = attention_mask.sum(dim=1)
            for i in range(gen_ids.size(0)):
                out_ids = gen_ids[i, input_lens[i]:]
                text = tokenizer.decode(out_ids, skip_special_tokens=True)
                text = text.replace("\n", " ").strip()
                if idx+i == 0:
                    print("[Output]")
                    print(text)
                data_idx = batch_idx * args.batch_size + i
                record = {"data_idx": origin_dataset[data_idx]["data_idx"]}

                text = text.split("Assistant:")[-1] if "Assistant:" in text else text
                text = text.split("Here's a possible the assistant's response in no more than 30 words:")[-1] if "Here's a possible the assistant's response in no more than 30 words:" in text else text
                record["pred_response"] = text
                
                record['conversation'] = origin_dataset[data_idx]['conversation']
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                idx += 1

    print(f"Save: {args.output_file}\n")            


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--cache_dir", type=str, default='./models')
    cli_parser.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    cli_parser.add_argument("--seed", type=int, default=42)

    cli_parser.add_argument("--output_file", type=str, default="comet_response.jsonl")
    cli_parser.add_argument("--test_file", type=str, default='comet_test.jsonl')
    cli_parser.add_argument("--new_max_token", type=int, default=100)
    
    cli_parser.add_argument("--is_ablation", type=bool, default=False)
    cli_parser.add_argument("--remove_for_ablation", type=str, default=None)

    cli_parser.add_argument("--batch_size", type=int, default=4)
    #cli_parser.add_argument("--num_gpu", type=int, default=0)
    
    args = cli_parser.parse_args()
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)  
    print("Device count:", torch.cuda.device_count())       
    print("Current device index:", torch.cuda.current_device())  
    print("Device name:", torch.cuda.get_device_name())   
    print("CUDA Version:", torch.version.cuda)
    print("is_BF16:",torch.cuda.is_bf16_supported())

    set_seed(args.seed)
    test(args)
    
    print(f"evaluate {args.output_file} done!")
    
    
