import os
from openai import OpenAI
import json
from tqdm import tqdm
import argparse
import re

def construct_messages(dialog, response, eval_type=None):
    if eval_type == "Supportiveness":
        system_prompt = f"""Your task is to rate the responses on one metric.
Please make sure you read and understand these instructions carefully. Keep the conversation history in mind while reviewing, and refer to it as needed.

Evaluation Criteria:
Supportiveness (1 - 3): Does the response help reduce the Client’s emotional distress and support them in coping with their challenges?

- A score of 1 (bad): The response fails to reduce emotional distress and does not support the Client in coping with their situation. It may feel dismissive, irrelevant, or emotionally unhelpful.
- A score of 2 (ok): The response somewhat reduces distress or offers partial support for coping. It may show some empathy or suggest vague reassurance but lacks clarity or effectiveness.
- A score of 3 (good): The response clearly reduces emotional distress and helps the Client cope with their challenges. It offers empathetic understanding, emotional reassurance, and/or helpful suggestions for managing the situation.

Evaluation Steps:
1. Read the conversation history between the Client and Assistant.
2. Read the Assistant’s potential next response.
3. Evaluate the response based on its emotional supportiveness and its usefulness in helping the Client cope with their situation.
4. Assign a score of 1, 2, or 3.

Please answer using the following format strictly:
Analysis: [your brief analysis]
Rating: [1|2|3]
"""
    elif eval_type == "Naturalness":
        system_prompt = f"""Your task is to rate the responses on one metric.
Please make sure you read and understand these instructions carefully. Keep the conversation history in mind while reviewing, and refer to it as needed.

Evaluation Criteria:
Naturalness (1-3) Is the response naturally written??
- A score of 1 (bad) means that the response is unnatural.
- A score of 2 (ok) means the response is strange, but not entirely unnatural.
- A score of 3 (good) means that the response is natural.

Evaluation Steps:
1. Read the conversation between the two individuals.
2. Read the potential response for the next turn in the conversation.
3. Evaluate the response based on its naturalness, using the provided criteria.
4. Assign a rating score of 1, 2, or 3 based on the evaluation.

Please answer using the following format strictly:
Analysis: [your brief analysis]
Rating: [1|2|3]
"""
    elif eval_type == "Coherence":
        system_prompt = f"""Your task is to rate the responses on one metric.
Please make sure you read and understand these instructions carefully. Keep the conversation history in mind while reviewing, and refer to it as needed.

Evaluation Criteria:
Coherence (1-3) Does the response serve as a valid continuation of the conversation history?
- A score of 1 (no) means that the response drastically changes topic orignores the conversation history.
- A score of 2 (somewhat) means the response refers to the conversation history in a limited capacity (e.g., in a generic way) and shifts the conversation topic.
- A score of 3 (yes) means the response is on topic and strongly acknowledges the conversation history.

Evaluation Steps:
1. Read the conversation history.
2. Read the potential response.
3. Evaluate the coherence of the response based on the conversation history.
4. Assign a score of 1, 2, or 3 for coherence.

Please answer using the following format strictly:
Analysis: [your brief analysis]
Rating: [1|2|3]
"""
    conversation = "\n".join(
        [d["speaker"].replace("usr", "Client").replace("seeker", "Client")
                     .replace("sys", "Assistant").replace("supporter", "Assistant") + ": " + d["text"] 
         for d in dialog]
    )

    user_prompt = f"""Conversation History:
{conversation}

Response:
{response}

Evaluation Form:
{eval_type}:"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

def gpt_api(api_key, model_type, response_type, eval_type, read_file):
    with open(read_file, "r", encoding="utf-8") as f:
        all_examples = [json.loads(d) for d in f]
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    write_file = os.path.join(model_type, read_file.replace("test", f"g_eval").replace("response", f"{eval_type}_{response_type}"))
    all_results = []          
    with open(write_file, "a", encoding="utf-8") as wf:
        for d_idx, data in tqdm(enumerate(all_examples), total = len(all_examples), desc = f"evaluate {response_type} using model {model_type} ..."):
            conversation = data["conversation"][:-1] # 후속 발화 제거
            response = data["response"][response_type]
            messages = construct_messages(dialog=conversation, response=response, eval_type=eval_type)
            if d_idx == 0:
                print(messages)
            
            total_samples = 20
            batch_size = 8
            sampled_scores = []
            
            while len(sampled_scores) < total_samples:
                n_call = min(batch_size, total_samples - len(sampled_scores))
                response = client.chat.completions.create(
                    model=model_type,
                    messages=messages,
                    temperature=1,
                    top_p=1,
                    n=n_call
                )

                for choice in response.choices:
                    content = choice.message.content
                    match = re.search(r"Rating:\s*([1-3])", content)
                    if match:
                        score = int(match.group(1))
                        sampled_scores.append(score)

            soft_score = sum(sampled_scores) / len(sampled_scores)
            output = {
                "data_idx": data["data_idx"],
                "response_type": response_type,
                "scores": sampled_scores,
                "soft_score": soft_score
            }
            if d_idx == 0:
                print(output)
            wf.write(json.dumps(output, ensure_ascii=False) + '\n')
            all_results.append(output)

    print()
    print(write_file + " saved!")
    print()

    # get average soft score 
    all_soft = [o["soft_score"] for o in all_results]
    print("Average soft score:", round(sum(all_soft)/len(all_soft), 4))

    return all_results     


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--api_key", type=str, default='OPENAI_API_KEY')
    cli_parser.add_argument("--read_file", type=str, default="test_response_200.jsonl")
    cli_parser.add_argument("--model_type", type=str, default="gpt-4o-mini-2024-07-18")

    args = cli_parser.parse_args()
    for eval_type in ["Supportiveness", "Naturalness", "Coherence"]:
        for response_type in ["baseline", "all", "dialect", "comet", "doctor", "sibyl"]:
            gpt_api(args.api_key, args.model_type, response_type, eval_type, args.read_file)



