import os
from openai import OpenAI
import json
from tqdm import tqdm
import argparse


def construct_messages_for_assistant(dialog, input_mental_states="", mental_state_type="Belief"):
    
    system_prompt = "You are an emotional support assistant with expertise in client-centered, psychodynamic, and cognitive behavioral therapies." 
    system_prompt += "Given a conversation between a client and an assistant, your task is to first infer the client’s mental states from the conversation. Then, infer the assistant’s mental states based on the inferred client mental states, the selected support strategy, and the assistant’s most recent utterance. These mental states include Belief, Emotion, Desire, and Intent, which should be inferred in that order."

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
    
    mental_state_question_dict = {
        "Belief": "What does the assistant believe about the client’s situation or internal state to generate the most recent response?",
        "Emotion": "What is the assistant’s emotional reaction in their most recent response, based on the client’s emotional state and how the conversation has unfolded? Rate each basic emotion from 0 (not present) to 3 (intense). Then rate mixed emotions only if both contributing basic emotions are rated 2 or higher and are explicitly reflected in the assistant’s wording. Avoid over-assigning positive emotions like Joy, Trust, Gratitude/Love/Caring, Hope/Optimism or Curiosity unless clearly expressed in the assistant’s wording, not just implied by supportiveness.",
        "Intent": "What is the assistant’s intent in the most recent response, based on how the assistant responded and the support strategy? Please do not directly mention the support strategy name or the assistant’s response text."
    }

    conversation = "\n".join(
        [d["speaker"].replace("usr", "Client").replace("seeker", "Client").replace("sys", "Assistant").replace("supporter", "Assistant")+": "+d["text"] for d in dialog]
        )
    strategy_define = strategy_info[dialog[-1]["strategy"]]
    response = dialog[-1]["text"]
    
    if input_mental_states != "": input_mental_states = "[ASSISTANT MENTAL STATE]\n" + input_mental_states
    if mental_state_type == "Intent": 
        input_mental_states += "Desire: The assistant's desire is to reduce the client’s emotional distress and help them cope with challenges.\n"
    
    mental_state_question = mental_state_question_dict[mental_state_type]

    user_prompt = """Based on the conversation below, infer the {0} of the assistant that underlie their most recent response.

    [CONVERSATION]
    {1}

    Below are the support strategy and assistant’s final response:  
    - Selected Support Strategy: {2}  
    - Assistant's Response: {3}

    First, silently infer the client’s mental states based on their last utterance. 
    These mental states include Belief, Emotion, Desire, and Intent, which should be inferred in that order but not included in the output.

    {4}

    {5} Your inferences should reflect the natural and evolving mental stance of the assistant across the conversation — not simply rationalize a pre-selected strategy or response.\n""".format(mental_state_type, conversation, strategy_define, response, input_mental_states, mental_state_question)

    if mental_state_type != "Emotion": 
        user_prompt += "Clearly and concisely answer the assistant’s {0} (max 40 words) based on the conversation leading up to, but not including, the assistant’s most recent response.".format(mental_state_type)
    else: 
        user_prompt += "If no emotion is apparent, write: Emotion: unbothered or oblivious.\n\nEmotion:\n"
        user_prompt += "[Basic] Sadness (opposite Joy): <score>, Disgust (opposite Trust): <score>, Anger (opposite Fear): <score>, Anticipation (opposite Surprise): <score>, Joy (opposite Sadness): <score>, Trust (opposite Disgust): <score>, Fear (opposite Anger): <score>, Surprise (opposite Anticipation): <score>\n"
        user_prompt += "[Mixed] Hopelessness (sadness + fear): <score>, Remorse (sadness + disgust): <score>, Disappointment (sadness + surprise): <score>, Sentimental (sadness + trust): <score>, Jealousy (sadness + anger): <score>, Pessimism (sadness + anticipation): <score>, Embarrassment (disgust + fear): <score>, Pride (anger + joy): <score>, Nervousness (anticipation + fear): <score>, Delight (joy + surprise): <score>, Gratitude/Love/Caring (joy + trust): <score>, Hope/Optimism (joy + anticipation): <score>, Guilt (joy + fear): <score>, Curiosity (surprise + trust): <score>\n" 

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    return messages



def gpt_api(api_key, model_type, read_file, write_file):
    with open(read_file, "r", encoding="utf-8") as f:
        all_examples = [json.loads(d) for d in f] 

    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    all_results = []          
    with open(write_file, "a", encoding="utf-8") as wf:
      for d_id, dialogue in enumerate(tqdm(all_examples, total=len(all_examples), desc="reading examples...")):
            for d_id_2 in range(2, len(dialogue["dialog"])//2 if len(dialogue["dialog"])%2 == 1 else (len(dialogue["dialog"])//2 + 1)):
                dialog = dialogue["dialog"][: d_id_2 * 2]
                
                mental_states={k:"" for k in ["Belief", "Emotion", "Intent"]}
                input_mental_states=""
                for mental_idx, mental_state_type in enumerate(["Belief", "Emotion", "Intent"]):
                    messages = construct_messages_for_assistant(dialog, input_mental_states=input_mental_states, mental_state_type=mental_state_type)
                    
                    response = client.chat.completions.create(
                    model=model_type,
                    messages=messages,
                    temperature=0
                    )
                    
                    result = response.choices[0].message.content
                    result = result.split(mental_state_type +":")[1] if len(result.split(mental_state_type +":")) > 1 else result.split(mental_state_type +":")[0]
                    mental_states[mental_state_type] = result.strip()
                    
                    result = mental_state_type +": "+result
                    result = result.replace(mental_state_type +": "+mental_state_type +":", mental_state_type +":")
                    if mental_state_type == "Emotion": 
                        result = result.replace(mental_state_type +":", mental_state_type +": Rating from 0 (not present) to 3 (intense) ")
                    
                    input_mental_states += result+"\n"
                
                results = {"data_idx": "{}_{}".format(dialogue["idx"], d_id_2), "mental_state": mental_states, "conversation": dialog}

                wf.write(json.dumps(results, ensure_ascii=False) + '\n')
                all_results.append(results)
    print()
    print(write_file + " saved!")
    print()

    return all_results

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--api_key", type=str, default='OPENAI_API_KEY')
    cli_parser.add_argument("--read_file", type=str, default="ESConv_train.jsonl")
    cli_parser.add_argument("--write_file", type=str, default="data/train.jsonl")
    cli_parser.add_argument("--model_type", type=str, default="gpt-4o-2024-11-20")

    args = cli_parser.parse_args()

    gpt_api(args.api_key, args.model_type, args.read_file, args.write_file)


