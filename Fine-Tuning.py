import torch
import argparse 
import os 
import collections
import collections.abc
from attrdict import AttrDict 
import numpy as np
import random
for type_name in collections.abc.__all__:
    setattr(collections, type_name, getattr(collections.abc, type_name))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    default_data_collator
)
from datasets import load_dataset, concatenate_datasets
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

print_prompt_flag = 0

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


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def create_instruction(dialog, mental_state_type):
    answer = mental_state_type +": "+dialog["mental_state"][mental_state_type]
    conversation = ""
    for d_idx, d in enumerate(dialog["conversation"][:-1]):
        num= d_idx + 1
        conversation += f"({num}) "+d["speaker"].replace("usr", "Client").replace("seeker", "Client").replace("sys", "Assistant").replace("supporter", "Assistant")+": "+d["text"].strip() + "\n"

    if  mental_state_type == "Belief":
        input_mental_states = ""
    else:
        input_mental_states = "[Assistant's Mental States]\n"    
        dialog["mental_state"] = {"Belief": dialog["mental_state"]["Belief"]} if mental_state_type == "Emotion" else {"Belief": dialog["mental_state"]["Belief"], "Emotion": dialog["mental_state"]["Emotion"], "Desire": "The assistant's desire is to reduce the client’s emotional distress and help them cope with challenges."}
        
        for k, v in dialog["mental_state"].items():
            if k == "Emotion": v = "Rating from 0 (not present) to 3 (intense)\n"+v
            input_mental_states += "{}: {}\n".format(k, v.replace("’", "'"))
        input_mental_states += "\n"
    mental_state_question_dict = {
        "Belief": "What does the assistant believe about the client's situation and emotional state based on the client's last utterance?",
        "Emotion": "What emotional reaction might the assistant have after the client's last utterance, based on the assistant's belief and how the conversation has unfolded? Rate each basic emotion from 0 (not present) to 3 (intense). Then rate mixed emotions only if both contributing basic emotions are rated 2 or higher and are explicitly reflected in the assistant's wording.",
        "Intent": "What is the assistant's intent following the client's last utterance, based on the assistant's belief, emotional reaction, and desire?"
    }

    data = f"\n[Dialogue history]\n{conversation}\n\n{input_mental_states}[Question] {mental_state_question_dict[mental_state_type]}"
    return [data, answer]
    

def make_prompt(dialog, mental_state_type):    
    system_prompt = "You are an emotional support assistant with expertise in client-centered, psychodynamic, and cognitive behavioral therapies."
    messages = [{"role": "system", "content": system_prompt.strip()}]

    user_prompt = "Given a dyadic dialogue clip between an assistant and a client who seeks help in relieving emotional distress, your task is to first infer the client's mental states in their last utterance. Then, infer the assistant's potential mental states that may arise in response to the client's last utterance. These mental states include Belief, Emotion, Desire, and Intent, and should be inferred in that order.\n\n"
    user_prompt += f"Clearly and concisely answer the assistant’s {mental_state_type} (no more than 40 words) of the following conversation clip. "
    user_prompt += f"The conversation clip is:\n\n" 

    instruction = create_instruction(dialog, mental_state_type)
    user_prompt += instruction[0].strip()
    
    if mental_state_type == "Emotion":
        user_prompt += "\nEmotion:\n[Basic] Sadness (opposite Joy): <score>, Disgust (opposite Trust): <score>, Anger (opposite Fear): <score>, Anticipation (opposite Surprise): <score>, Joy (opposite Sadness): <score>, Trust (opposite Disgust): <score>, Fear (opposite Anger): <score>, Surprise (opposite Anticipation): <score>\n[Mixed] Hopelessness (sadness + fear): <score>, Remorse (sadness + disgust): <score>, Disappointment (sadness + surprise): <score>, Sentimental (sadness + trust): <score>, Jealousy (sadness + anger): <score>, Pessimism (sadness + anticipation): <score>, Embarrassment (disgust + fear): <score>, Pride (anger + joy): <score>, Nervousness (anticipation + fear): <score>, Delight (joy + surprise): <score>, Gratitude/Love/Caring (joy + trust): <score>, Hope/Optimism (joy + anticipation): <score>, Guilt (joy + fear): <score>, Curiosity (surprise + trust): <score>\n"
    
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": instruction[1].strip()})
    return messages


def apply_template(example, tokenizer, mental_state_type, response_template="assistant<|end_header_id|>", is_example = False):
    messages = make_prompt(example, mental_state_type = mental_state_type, is_example = is_example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    text = text.replace("\n", " ")
    if response_template not in text:
        print(f'"{response_template}" is not include prompt.\n\n[prompt]\n{text}')

    global print_prompt_flag
    if print_prompt_flag == 0:
        print(text)
        print_prompt_flag += 1
    
    return {"text": text}

def training(cli_args):
    args = AttrDict(vars(cli_args))
    args.device = "cuda"

    mental_state_type=args.mental_state_type    
    print(f"mental_state_type: {mental_state_type}")
    
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if len(args.model_name.split("/")) > 1: 
        model_name = args.model_name.split("/")[-1]

    output_dir = f"./{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리를 생성했습니다: {output_dir}")
    else:
        print(f"디렉토리가 이미 존재합니다: {output_dir}")
    
    output_dir = f"./{model_name}/model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리를 생성했습니다: {output_dir}")
    else:
        print(f"디렉토리가 이미 존재합니다: {output_dir}")

    output_dir = f"./{model_name}/model/{args.output_file}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리를 생성했습니다: {output_dir}")
    else:
        print(f"디렉토리가 이미 존재합니다: {output_dir}")

    output_dir = f"./{model_name}/model/{args.output_file}/{mental_state_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리를 생성했습니다: {output_dir}")
    else:
        print(f"디렉토리가 이미 존재합니다: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        add_eos_token=True, add_bos_token=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def generate_and_tokenize_prompt(data_point, mental_state_type, is_example):
        messages = make_prompt(data_point, mental_state_type, is_example)
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if args.is_padding:
            tokenized_full = tokenizer(full_prompt, padding="max_length", truncation=True, max_length = min(args.cutoff_len, args.max_length))
        else:
            tokenized_full = tokenizer(full_prompt, return_tensors="pt")
            for key in tokenized_full.keys(): tokenized_full.update({key: list(tokenized_full[key][0])})
        input_prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=False
        )
        input_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        if args.is_padding:
            tokenized_input = tokenizer(input_prompt, truncation=True)
        else:
            tokenized_input = tokenizer(input_prompt, return_tensors="pt")
            for key in tokenized_input.keys(): tokenized_input.update({key: list(tokenized_input[key][0])})
        
        input_len = len(tokenized_input["input_ids"])
        label_ids = [-100] * input_len + tokenized_full["input_ids"][input_len:]
        tokenized_full["labels"] = label_ids

        global print_prompt_flag
        if print_prompt_flag == 0:
            # print(messages)
            # print(input_prompt)
            decoded = tokenizer.decode([
                l for l in tokenized_full['input_ids'] if l != -100
            ])
            print('[Input]')
            print(decoded)
            decoded = tokenizer.decode([
                l for l in tokenized_full['labels'] if l != -100
            ])
            print('\n[Label]')
            print(decoded)
            print("<|eot_id|><|start_header_id|>assistant<|end_header_id|>:", tokenizer.tokenize("<|eot_id|><|start_header_id|>assistant<|end_header_id|>"))
            print("<|eot_id|>:", tokenizer.tokenize("<|eot_id|>"))
            print("<|start_header_id|>: ", tokenizer.tokenize("<|start_header_id|>"))
            print("<|end_header_id|>: ", tokenizer.tokenize("<|end_header_id|>"))
            print_prompt_flag += 1

        assert len(tokenized_full["input_ids"]) == len(tokenized_full["labels"])
        return tokenized_full

    if args.is_8bit_quantization:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=False
        )
        model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quant_config, device_map=device_map)
    elif args.is_4bit_quantization:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  
            bnb_4bit_quant_type="nf4",   # ["fp4", "nf4"]
            bnb_4bit_compute_dtype=torch.float16,   
            bnb_4bit_use_double_quant=False,   
        )
        model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quant_config, device_map=device_map)    
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32, device_map=device_map)
    

    train_dataset = load_dataset('json', data_files={'train': os.path.join(args.data_dir, 'train.jsonl')})['train']
    dev_dataset   = load_dataset('json', data_files={'dev': os.path.join(args.data_dir, 'dev.jsonl')})['dev']
    if args.is_use_SFTTrainer:
        if args.mental_state_type != "All":
            train_data = train_dataset.shuffle().map(lambda x: apply_template(x, tokenizer, mental_state_type = args.mental_state_type, response_template = args.response_template, is_example = args.is_example))
            dev_data = dev_dataset.shuffle().map(lambda x: apply_template(x, tokenizer, mental_state_type = args.mental_state_type, response_template = args.response_template, is_example = args.is_example))
        else:
            train_data = concatenate_datasets([
                train_dataset.shuffle().map(lambda x: apply_template(x, tokenizer, mental_state_type = ms_type, response_template = args.response_template, is_example = args.is_example)) 
                for ms_type in ["Belief", "Emotion", "Intent"]
                ]).shuffle(seed=args.seed)
            dev_data = concatenate_datasets([
                dev_dataset.shuffle().map(lambda x: apply_template(x, tokenizer, mental_state_type = ms_type, response_template = args.response_template, is_example = args.is_example)) 
                for ms_type in ["Belief", "Emotion", "Intent"]
                ]).shuffle(seed=args.seed)
    else:
        if args.mental_state_type != "All":
            train_data = train_dataset.shuffle().map(
                lambda x: generate_and_tokenize_prompt(x, args.mental_state_type, args.is_example),
                remove_columns=train_dataset.column_names
            )
            dev_data = dev_dataset.shuffle().map(
                lambda x: generate_and_tokenize_prompt(x, args.mental_state_type, args.is_example),
                remove_columns=dev_dataset.column_names
            )
        else:
            train_data = concatenate_datasets([
                train_dataset.shuffle().map(lambda x: generate_and_tokenize_prompt(x, ms_type, args.is_example),
                                remove_columns=train_dataset.column_names)
                for ms_type in ["Belief", "Emotion", "Intent"]
            ]).shuffle(seed=args.seed)
            dev_data = concatenate_datasets([
                dev_dataset.shuffle().map(lambda x: generate_and_tokenize_prompt(x, ms_type, args.is_example),
                                remove_columns=dev_dataset.column_names)
                for ms_type in ["Belief", "Emotion", "Intent"]
            ]).shuffle(seed=args.seed)
            
    lora_config = LoraConfig(
        target_modules=["q_proj", "v_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    if args.is_use_SFTTrainer:
        response_template = args.response_template
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer, mlm=False
        )
    else:
        # Train on completions only
        model = get_peft_model(model, lora_config)
        if args.is_padding:
            collator=default_data_collator
        else:
            collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    model.config.use_cache = False 
    model.config.pretraining_tp = 1 
    print_trainable_parameters(model)
    # --is_use_SFTTrainer false : 
    # trainable params: 3407872 || all params: 8033669120 || trainable%: 0.04241987003816259
    # --is_use_SFTTrainer true : 
    # trainable params: 1050939392 || all params: 8030261248 || trainable%: 13.087237881105608

    training_args = TrainingArguments(
        do_train=True,
        output_dir=os.path.join(model_name, 'model', args.output_file, args.mental_state_type),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=100,
        save_steps=500,
        save_total_limit=10,
        eval_strategy="epoch" if args.val_set_size>0 else "no",
        save_strategy="epoch",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=False if (args.is_8bit_quantization or args.is_4bit_quantization) else torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit" if (args.is_8bit_quantization or args.is_4bit_quantization) else "adamw_torch",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False if ddp else None,
        report_to=None,
        save_safetensors=False,
    )
    

    if args.is_use_SFTTrainer:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=dev_data,
            dataset_text_field="text",
            max_seq_length= min(args.cutoff_len, args.max_length),
            data_collator=collator,
            peft_config=lora_config,
            args=training_args,
            tokenizer=tokenizer,
            packing=False
        )        
    else:
        training_args.label_names=["labels"]  
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=dev_data,
            args=training_args,
            data_collator=collator,
        )
            
    for batch in trainer.get_train_dataloader():
        labels = batch["labels"]            # shape [bs, seq_len]
        valid_counts = (labels != -100).sum(dim=1)  # 배치별 유효 토큰 개수
        if (valid_counts == 0).any():
            print("⚠️ 유효 라벨이 0개인 샘플 발견")
            print("라벨 분포:", valid_counts)
            break

    def detect_nan(module, inp, out):
        # out이 Tensor면 튜플로, 튜플이면 그대로 사용
        outputs = out if isinstance(out, tuple) else (out,)
        for o in outputs:
            # o가 Tensor인지도 확인 (혹시 None일 수도 있으니)
            if isinstance(o, torch.Tensor):
                if torch.isnan(o).any() or torch.isinf(o).any():
                    print(f"⚠️ NaN/Inf detected in {module.__class__.__name__}")
                    return  # 한 번 감지되면 충분하면 종료

    for m in trainer.model.modules():
        m.register_forward_hook(detect_nan)

    torch.autograd.set_detect_anomaly(True)
    trainer.train()
    if args.is_use_SFTTrainer:    
        trainer.save_model(training_args.output_dir)
    else:
        model.save_pretrained(training_args.output_dir)
    print(
            "\n If there's a warning about missing keys above, please disregard :)"
        )

    return model    

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--data_dir", type=str, default='data')
    cli_parser.add_argument("--model_name", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    cli_parser.add_argument("--output_file", type=str, default='Full_FT')
    cli_parser.add_argument("--max_length", type=int, default=8192)

    cli_parser.add_argument("--is_8bit_quantization", type=bool, default=False)
    cli_parser.add_argument("--is_4bit_quantization", type=bool, default=False)
    
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--mental_state_type", type=str, default="Belief") # ["Belief", "Emotion", "Intent", "All"]
    cli_parser.add_argument("--is_example", type=bool, default=False)
    
    cli_parser.add_argument("--batch_size", type=int, default=4)
    cli_parser.add_argument("--micro_batch_size", type=int, default=2)
    cli_parser.add_argument("--num_epochs", type=int, default=5)
    cli_parser.add_argument("--learning_rate", type=float, default=3e-5)
    cli_parser.add_argument("--cutoff_len", type=int, default=4096)
    cli_parser.add_argument("--val_set_size", type=int, default=195)

    cli_parser.add_argument("--lora_r", type=int, default=8)
    cli_parser.add_argument("--lora_alpha", type=int, default=16)
    cli_parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    cli_parser.add_argument("--train_on_inputs", type=bool, default=False)
    cli_parser.add_argument("--group_by_lengths", type=bool, default=False)
        
    cli_parser.add_argument("--is_use_SFTTrainer", type=bool, default=False)
    cli_parser.add_argument("--response_template", type=str, default="<|start_header_id|>assistant<|end_header_id|>")
    cli_parser.add_argument("--is_padding", type=bool, default=False)

    # cli_parser.add_argument("--num_gpu", type=int, default=0)

    # Running Mode
    cli_args = cli_parser.parse_args()

    args = AttrDict(vars(cli_args))
    if len(args.model_name.split("/")) > 1: 
        model_name = args.model_name.split("/")[-1]

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)  
    print("Device count:", torch.cuda.device_count())     
    print("Current device index:", torch.cuda.current_device())  
    print("Device name:", torch.cuda.get_device_name())   
    print("CUDA Version:", torch.version.cuda)
    print("is_BF16:",torch.cuda.is_bf16_supported())

    set_seed(args.seed)
    
    
    training(cli_args)
    print(f"training {args.output_file} done!")
    
    





