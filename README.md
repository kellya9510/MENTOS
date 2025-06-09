# ToMESC

## License

This dataset is derived from the [ESConv dataset](https://github.com/thu-coai/Emotional-Support-Conversation).  
  
The original ESConv dataset is licensed under the  
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.  
> © 2021 CoAI Group, Tsinghua University. All rights reserved.  
> Data and code are for academic research use only.

Accordingly, the ToMESC dataset is distributed under the same license and terms:  
**For academic research use only. Commercial use is strictly prohibited.**

For more information, see `LICENSE-ESConv`.  
All derived material in this repository is subject to the same non-commercial restriction.


## Data

We introduce **ToMESC**, a Theory of Mind-based dataset that models an assistant’s latent mental states—**Belief**, **Emotion**, **Desire**, and **Intent**—in a structured causal sequence.  
The dataset is publicly available at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.15624491).
Download ToMESC and place it in the `data/` folder before running any scripts.

---

## Dataset Construction

To construct the ToMESC dataset from ESConv, run the following after downloading ESConv:

  `python create_mental_state.py --api_key OPENAI_API_KEY --model_type gpt-4o-2024-11-20`

## Fine-tuning (SFT) using ToMESC

To fine-tune a model using the training split of ToMESC:

  `python Fine-Tuning.py --mental_state_type All --data_dir data --batch_size 4 -num_epochs 5 --learning_rate 3e-5 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05`

## Inference

Run inference for each mental state:

  `python inference_mental_state.py --mental_state_type Belief --check_point BestCheckPoint --test_file data/test.jsonl`
  `python inference_mental_state.py --mental_state_type Emotion --check_point BestCheckPoint --test_file data/test.json`
  `python inference_mental_state.py --mental_state_type Intent --check_point BestCheckPoint --test_file data/test.json`

## Zero-shot Response Generation

To generate responses using a zero-shot LLM:

  `python generate_response.py --model_name meta-llama/Llama-2-7b-chat-hf --test_file Meta-Llama-3.1-8B-Instruct/model/Full_FT/All/output_BestCheckPoint.jsonl --output_file output/output_All_response_BestCheckPoint.jsonl --new_max_token 100`

## Evaluation

Evaluate Inferred Mental States

  `python evaluate_metrics.py --data_dir Meta-Llama-3.1-8B-Instruct/model/Full_FT/All --check_point BestCheckPoint`

Evaluate Generated Responses

  `python evaluate_metrics.py --data_dir Meta-Llama-3.1-8B-Instruct/model/Full_FT/All --check_point BestCheckPoint --is_response true`

Evaluate with G-Eval
After post-processing the response outputs, ensure your file (e.g., test_response_200.jsonl) follows this format:

  `{
    "data_idx": "DialogueIdx_TurnIdx",
    "response": {
      "baseline": "...",
      "all": "...",
      "dialect": "...",
      "comet": "...",
      "doctor": "...",
      "sibyl": "..."
    },
    "conversation": [ ... ]
  }`



Then run:

  `python g_eval.py --read_file test_response_200.jsonl --model_type gpt-4o-mini-2024-07-18 --api_key OPENAI_API_KEY`
