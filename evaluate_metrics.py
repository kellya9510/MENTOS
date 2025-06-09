from tqdm import tqdm
import argparse
import random, json, os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import math
from metric import NLGEval 
from scipy.stats import ttest_rel
from collections import Counter
# import nltk
# nltk.download('wordnet')
# nltk.download('punkt')

def distinct_n_grams(texts, n=1):
    all_ngrams = [tuple(text.split()[i:i + n]) for text in texts for i in range(len(text.split()) - n + 1)]
    return len(set(all_ngrams)) / (len(all_ngrams) + 1e-10)

def compute_average_cosine(generated, references, model=None):
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    gen_emb = model.encode(generated, convert_to_tensor=True)
    ref_emb = model.encode(references, convert_to_tensor=True)
    return cosine_similarity(gen_emb.cpu().numpy(), ref_emb.cpu().numpy()).diagonal().mean()

def evaluate_emotion_predictions(golden_list, predict_list):
    total_mae = 0
    total_acc = 0
    total_tol_acc = 0
    total_count = 0
    y_true = []
    y_pred = []
    
    for golden, predict in zip(golden_list, predict_list):
        for category in ['Basic', 'Mixed']:
            for key in golden[category]:
                g = golden[category][key]
                p = predict[category].get(key, 0)  

                total_mae += abs(g - p)
                total_acc += int(g == p)
                total_tol_acc += int(abs(g - p) <= 1)
                total_count += 1
                y_true.append(g)
                y_pred.append(p)

    mae = total_mae / total_count
    acc = total_acc / total_count
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'MAE': mae,
        'Accuracy': acc,
        'Macro-F1': macro_f1
    }


def run_ttest(all_scores_model, all_scores_baseline, alpha=0.01):
    results = {}
    for metric in all_scores_model:
        if metric not in all_scores_baseline:
            continue
        model_scores = all_scores_model[metric]
        baseline_scores = all_scores_baseline[metric]

        if len(model_scores) != len(baseline_scores):
            print(f"⚠️ Skipping {metric} due to unequal lengths.")
            continue

        t_stat, p_value = ttest_rel(model_scores, baseline_scores)
        results[metric] = {
            't-statistic': t_stat,
            'p-value': p_value,
            'significant (p < {:.2g})'.format(alpha): p_value < alpha
        }

    return results

def convert_dict_floats(d):
    if isinstance(d, dict):
        return {k: convert_dict_floats(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_dict_floats(v) for v in d]
    elif isinstance(d, (np.float32, np.float64)):
        return float(d)
    elif isinstance(d, (np.int32, np.int64)):
        return int(d)
    else:
        return d
    
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--data_file", type=str, default='data/test.jsonl')
    cli_parser.add_argument("--check_point", type=int, default=0)
    cli_parser.add_argument("--out_file", type=str, default="Full_FT_Metric.jsonl")
    cli_parser.add_argument("--is_response", type=bool, default=False)
    cli_parser.add_argument("--is_tt_test", type=bool, default=False)
    cli_parser.add_argument("--t_alpha", type=float, default=0.05)
    cli_parser.add_argument("--num_turn", type=int, default=0)

    args = cli_parser.parse_args()
    if args.is_response:
        if ("sibyl" in args.data_file) or ("output" in args.data_file):    
            with open(f"{args.data_file}" + f"_response_{args.check_point}.jsonl", "r") as f:
                datas = [json.loads(d) for d in f]
        else:
            with open(f"{args.data_file}" + f"_response.jsonl", "r") as f:
                datas = [json.loads(d) for d in f]
        if args.num_turn > 0:
            if args.num_turn < 14: datas = [data for data in datas if int(data["data_idx"].split("_")[1])==args.num_turn]
            if args.num_turn > 13: datas = [data for data in datas if (int(data["data_idx"].split("_")[1])>13)]
            print(len(datas))

        my_datas = {"correct":[], "predict": []}
        with open("data/Sibyl_ESConv_test.json", "r") as f:
            correct_datas = json.load(f)
        
        for d_idx, data in enumerate(datas):
            my_datas["correct"].append([data["conversation"][-1]["text"]])
           
            # inference: llama-2
            data["pred_response"] = data["pred_response"].replace("[/INST]", "").replace("(xReact)", "").replace("(xIntent)", "").strip()
            if '\"' in data["pred_response"].strip(): 
                pred = data["pred_response"].strip().split('\"')
                if len(pred) == 3:
                    data["pred_response"] = pred[1].strip()

            if data["pred_response"].strip() == "": 
                data["pred_response"] = "I'm sorry, I can't answer."
            if len(my_datas["predict"]) == 10: print(my_datas["predict"])
            my_datas["predict"].append(data["pred_response"])
        
        # init NLGEval 
        evaluator = NLGEval(no_overlap=False, no_glove=False)
        overall_scores, all_scores_per_sample = evaluator.compute_metrics(my_datas["correct"], my_datas["predict"])
        
        # Distinct-1, Distinct-2
        dist1 = distinct_n_grams(my_datas["predict"], 1)
        dist2 = distinct_n_grams(my_datas["predict"], 2)
        dist3 = distinct_n_grams(my_datas["predict"], 3)
        
        overall_scores["DIST-1"] = dist1
        overall_scores["DIST-2"] = dist2
        overall_scores["DIST-3"] = dist3
        
        overall_scores["AvgCosSen"] = compute_average_cosine(generate=my_datas["predict"], golden=[i[0] for i in my_datas["correct"]])               
        
        print("📊 Overall Metrics:")
        for metric, score in overall_scores.items():
            #print(f"{metric}: {score:.4f}")
            print(f"{metric}: {score}")
        
        is_separate = 'true' if args.data_file == "Intent" else 'false'
        with open( args.out_file, "a") as wf:
            if args.num_turn>0:
                record = {"is_response": 'true', "num_turn": args.num_turn, "num_data": len(datas),
                      "data_file": args.data_file+f"_{args.check_point}" if ("sibyl" in args.data_file) or ("output" in args.data_file) else args.data_file, 
                      "metric": overall_scores}
            else:
                record = {"is_response": 'true' if args.is_response else 'false', "is_separate": is_separate, 
                      "data_file": args.data_file+f"_{args.check_point}" if ("sibyl" in args.data_file) or ("output" in args.data_file) else args.data_file, 
                      "metric": overall_scores}
            clean_record = convert_dict_floats(record)
            wf.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
            
    elif args.is_tt_test:
        if ("sibyl" in args.data_file) or ("output" in args.data_file):    
            with open(f"{args.data_file}" + f"_response_{args.check_point}.jsonl", "r") as f:
                datas1 = [json.loads(d) for d in f]
        else:
            with open(f"{args.data_file}" + f"_response.jsonl", "r") as f:
                datas1 = [json.loads(d) for d in f]
        
        with open(f"dialect_response.jsonl", "r") as f:
            datas2 = [json.loads(d) for d in f]
        
        my_datas_list = []
        for datas in [datas1, datas2]:
            my_datas = {"correct":[], "predict": []}
            with open("data/Sibyl_ESConv_test.json", "r") as f:
                correct_datas = json.load(f)
            for d_idx, data in enumerate(datas):
                if "Sibyl_ESConv" not in args.data_file: 
                    my_datas["correct"].append([data["conversation"][-1]["text"]])
                else:
                    my_datas["correct"].append([correct_datas[d_idx]["dialog"][-1]["text"]])
                # inference: llama-2
                data["pred_response"] = data["pred_response"].replace("[/INST]", "").replace("(xReact)", "").replace("(xIntent)", "").strip()
                if '\"' in data["pred_response"].strip(): 
                    pred = data["pred_response"].strip().split('\"')
                    if len(pred) == 3:
                        data["pred_response"] = pred[1].strip()
                # common
                if data["pred_response"].strip() == "": 
                    data["pred_response"] = "I'm sorry, I can't answer."
                if len(my_datas["predict"]) == 10: print(my_datas["predict"])
                my_datas["predict"].append(data["pred_response"])
            # init NLGEval 
            evaluator = NLGEval(no_overlap=False, no_glove=False)
            overall_scores, all_scores_per_sample = evaluator.compute_metrics(my_datas["correct"], my_datas["predict"])
            
            avg_cos = [
                    compute_average_cosine([pred], corr)
                    for pred, corr in tqdm(zip(my_datas["predict"], my_datas["correct"]),
                                        total=len(my_datas["predict"]))
                ]
            all_scores_per_sample["AvgCosSen"] = avg_cos
            
            my_datas_list.append(all_scores_per_sample)
                            
                                                    
        model_scores_per_sample = my_datas_list[0]
        dialect_scores_per_sample = my_datas_list[1]

        ttest_results = run_ttest(model_scores_per_sample, dialect_scores_per_sample, alpha = args.t_alpha)

        for metric, res in ttest_results.items():
            print(f"{metric} | t = {res['t-statistic']:.4f}, p = {res['p-value']:.4e}, significant: {res[f'significant (p < {args.t_alpha})']}")
    
    else:
        with open(f"Llama-2-7b-chat-hf/model/Full_FT/{args.data_file}_{args.check_point}.jsonl", "r") as f:
            datas = [json.loads(d) for d in f]
        
        if "Sibyl" in args.data_file.split("_"):
            with open("data/Sibyl_ESConv_test.json", "r") as f:
                correct_datas = json.load(f)
        else:
            with open("data/test.jsonl", "r") as f:
                correct_datas = [json.loads(d) for d in f]
        
        my_datas = {k: {"correct":[], "predict": []} for k in datas[0][args.data_file.split("_")[0] if args.data_file.split("_")[0] != "output" else "mental_state"].keys()}
        
        for data, correct in zip(datas, correct_datas):
            mental_states = {k: v.split(k)[1].strip().replace(":", "") if k in v else v.strip() for k , v in data[args.data_file.split("_")[0] if args.data_file.split("_")[0] != "output" else "mental_state"].items()}
            mental_states = {k: v if v != "" else "I'm sorry, I can't answer." for k, v in mental_states.items()}
            
            if "Emotion" in mental_states.keys():
                if "\nEmotion:" in mental_states["Emotion"]:
                    mental_states["Emotion"] = mental_states["Emotion"].split("\nEmotion:")[1]
                label_dict = {k: correct["mental_state"][k] for k in mental_states.keys()}
            
            if "Sibyl" in args.data_file.split("_"):
                label_dict = {k: correct["dialog"][-2][k] for k in mental_states.keys()}
            
            for k in my_datas.keys():
                if k in label_dict.keys(): my_datas[k]["correct"].append(label_dict[k])
                my_datas[k]["predict"].append(mental_states[k])
        
        
        if sum([1 if my_datas[k]["correct"] == [] else 0 for k in my_datas.keys()]) > 0:
            with open("./data/match_sibyl_test.jsonl", "r") as f:
                correct_datas = [json.loads(d) for d in f]
            for k in my_datas.keys():
                my_datas[k]["correct"] = [[correct_data["mental_state"][k]] for correct_data in correct_datas]
                
        evaluator = NLGEval(no_overlap=False, no_glove=False)
        output = {}
        for k in my_datas.keys():
            print(k)
            combined_metrics, _ = evaluator.compute_metrics(my_datas[k]["correct"], my_datas[k]["predict"])  
            if k == "Emotion":     
                golden_list=[] 
                predict_list=[]
                for correct ,predict in zip(my_datas["Emotion"]["correct"], my_datas["Emotion"]["predict"]):
                    emotions = [i.strip().split(",") for i in correct.replace("[Basic]", "").split("[Mixed]")]
                    correct_emotions = {
                    "Basic": {basic.strip().split(":")[0].strip(): int(basic.strip().split(":")[1].strip()[0]) for basic in emotions[0] if "<score>" not in basic.strip().split(":")[1].strip()}, 
                    "Mixed": {mixed.strip().split(":")[0].strip(): int(mixed.strip().split(":")[1].strip()[0]) for mixed in emotions[1] if (len(mixed.strip().split(":")) == 2) and ("<score>" not in mixed.strip().split(":")[1].strip())}
                    }
                    golden_list.append(correct_emotions)
                    
                    emotions = [i.strip().split(",") for i in predict.replace("[Basic]", "").split("[Mixed]")]
                    predict_emotions = {
                    "Basic": {basic.strip().split(":")[0].strip(): int(basic.strip().split(":")[1].strip()[0]) for basic in emotions[0] if "<score>" not in basic.strip().split(":")[1].strip()}, 
                    "Mixed": {mixed.strip().split(":")[0].strip(): int(mixed.strip().split(":")[1].strip()[0]) for mixed in emotions[1] if (len(mixed.strip().split(":")) == 2) and ("<score>" not in mixed.strip().split(":")[1].strip())}
                    }
                    predict_list.append(predict_emotions)
                emotion_metrics = evaluate_emotion_predictions(golden_list, predict_list)
                combined_metrics = {**combined_metrics, **emotion_metrics}
            output[k] = combined_metrics
            
            print("📊 Overall Metrics:")
            for metric, score in combined_metrics.items():
                print(f"{metric}: {score:.4f}")
