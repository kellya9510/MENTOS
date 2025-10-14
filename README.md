# A Mental State Extraction Dataset for Theory-of-Mind-based Reasoning in Emotional Support Conversations

## License

This dataset is derived from the [ESConv dataset](https://github.com/thu-coai/Emotional-Support-Conversation).  
  
The original ESConv dataset is licensed under the  
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.  
> © 2021 CoAI Group, Tsinghua University. All rights reserved.  
> Data and code are for academic research use only.

Accordingly, the MENTOS dataset is distributed under the same license and terms:  
**For academic research use only. Commercial use is strictly prohibited.**

For more information, see `LICENSE-ESConv`.  
All derived material in this repository is subject to the same non-commercial restriction.

## MENTOS Dataset

<img src='DATA CONSTRUCTION.png' width='1000'>

We introduce **MENTOS**, a dataset that provides turn-level annotations of the assistant’s mental states (**Belief**, **Emotion**, **Desire**, and **Intent**), organized in a causal structure reflecting psychological principles.
This design integrates psychological principles of Theory of Mind (ToM) into commonsense reasoning.
A commonsense reasoning model trained on MENTOS predicts these mental states as intermediate reasoning signals that guide response generation, and these mental states were then injected into LLM-based response generators in a zero-shot setting.

### Dataset Construction

To construct the MENTOS dataset from ESConv, run the following command after downloading the ESConv dataset:

  `python create_mental_state.py --api_key OPENAI_API_KEY --model_type MODEL_TYPE`

**Mental State Extraction Prompt Components**

For each target mental state, the MENTOS dataset is constructed using the following components within the mental state extraction prompt:

(1) Dialogue history,

(2) Assistant response, including the supportive strategy description (strategy_info in `create_mental_state.py`).

(3) Assistant Mental State Component

(4) Question Component (mental_state_question_dict  in `create_mental_state.py`)

(5) Constraint Component

<p align="center"> <img src='Mental_State_Extraction_Prompt_GPT4o.png' width='1000'> </p>

These components together guide the model in generating structured annotations for each mental state (Belief, Emotion, Intent).
Among these, the Constraint Component for Emotion employs basic and mixed emotion categories, along with an intensity scale (0: None, 1: Low, 2: Medium, 3: High), grounded in psychological research ([Plutchik, 1982](https://is.muni.cz/el/1421/jaro2011/PSA_033/um/plutchik.pdf); [Sabour et al., 2024](https://aclanthology.org/2024.acl-long.326.pdf))

<p align="center">
  <img src="Emotion_Category.png" width="800">
</p>
<p align="center"><em>Image source: <a href="https://aclanthology.org/2024.acl-long.326.pdf">EmoBench: Evaluating the Emotional Intelligence of Large Language Models</a> (Sabour et al., ACL 2024)</em></p>


The dataset is built based on a turn-level annotation schema.
Each dialogue contains multiple turns, and for every turn (t), an independent data sample is created.
Each data sample consists of:

(1) Dialogue history up to the t-th client utterance

(2) The assistant’s response at the t-th turn

(3) The corresponding three mental state annotations (Belief, Emotion, and Intent)

Thus, a single dialogue yields as many data samples as there are turns

An example **MENTOS dataset sample** for a 2-turn dialogue is shown below:

```
{
  "data_idx": "example_id",
  "mental_state": {
    "Belief": "The assistant believes the client is feeling frustrated and struggling with the limitations of staying indoors, despite being an introvert, and is seeking understanding or validation for their emotional experience.",
    "Emotion": "[Basic] Sadness (opposite Joy): 2, Disgust (opposite Trust): 0, Anger (opposite Fear): 0, Anticipation (opposite Surprise): 0, Joy (opposite Sadness): 0, Trust (opposite Disgust): 2, Fear (opposite Anger): 0, Surprise (opposite Anticipation): 0\n[Mixed] Hopelessness (sadness + fear): 0, Remorse (sadness + disgust): 0, Disappointment (sadness + surprise): 0, Sentimental (sadness + trust): 2, Jealousy (sadness + anger): 0, Pessimism (sadness + anticipation): 0, Embarrassment (disgust + fear): 0, Pride (anger + joy): 0, Nervousness (anticipation + fear): 0, Delight (joy + surprise): 0, Gratitude/Love/Caring (joy + trust): 0, Hope/Optimism (joy + anticipation): 0, Guilt (joy + fear): 0, Curiosity (surprise + trust): 0",
    "Intent": "The assistant’s intent is to validate the client’s feelings of frustration, foster a sense of shared experience, and provide emotional reassurance by expressing empathy and understanding of the challenges of being confined indoors."
  },
  "conversation": [
    {"text": "Hello. How are you?", "speaker": "seeker"},
    {"text": "I'm doing well. How are you?", "speaker": "supporter", "strategy": "Reflection of feelings", "all_strategy": ["Reflection of feelings"]},
    {"text": "I am having a difficult time not being able to go out. I am an introvert and didn't think COVID-19 would be a problem but I find myself being short and impatient.", "speaker": "seeker"}, 
    {"text": "Yeah I'm in the same boat. It's tough having to be cooped up.", "speaker": "supporter", "strategy": "Affirmation and Reassurance", "all_strategy": ["Affirmation and Reassurance"]} # The assistant’s response at the 2nd turn.
  ]
}
```

### Evaluate the MENTOS quality

To assess the quality of MENTOS annotations, we conducted a human evaluation on 100 randomly sampled dialogues. Four annotators independently rated each assistant utterance across three mental state types (**Belief**, **Emotion**, and **Intent**) using four evaluation criteria per category, each on a 1–3 scale. To measure inter-annotator reliability, we report Gwet’s AC1, which is robust against prevalence and marginal distribution biases. Across all categories and criteria, AC1 values ranged from 0.6 to 0.8, indicating substantial agreement among annotators.

The evaluation was performed using the following command:

 `python evaluate_human_sample.py --read_file MENTOS_sample.jsonl`

## MENTOS-trained Commonsense Reasoning Model

### Fine-tuning (SFT) using MENTOS

To fine-tune a model using the training split of MENTOS:

  `python Fine-Tuning.py --mental_state_type All --data_dir data --batch_size 4 -num_epochs 5 --learning_rate 3e-5 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05`

For each target mental state, the MENTOS-trained model is fine-tuned using the following components within the prompt:

(1) Dialogue history

(2) Assistant Mental State Component

(3) Question Component

(4) Constraint Component

<p align="center"> <img src='SFT_prompt.png' width='1000'> </p>


### Inference

Run inference for each mental state:

  `python inference_mental_state.py --mental_state_type Belief --check_point BestCheckPoint --test_file data/test.jsonl`

  
  `python inference_mental_state.py --mental_state_type Emotion --check_point BestCheckPoint --test_file data/test.json`

  
  `python inference_mental_state.py --mental_state_type Intent --check_point BestCheckPoint --test_file data/test.json`

## Zero-shot Response Generation

To generate responses using a zero-shot LLM:

  `python generate_response.py --model_name meta-llama/Llama-2-7b-chat-hf --test_file Meta-Llama-3.1-8B-Instruct/model/Full_FT/All/output_BestCheckPoint.jsonl --output_file output/output_All_response_BestCheckPoint.jsonl --new_max_token 100`

For each target mental state, the response generator produces responses using the following components within the prompt:

(1) Dialogue history

(2) Commonsense Knowledge

<p align="center"> <img src='Response_Generation_prompt.png' width='1000'> </p>


## Evaluate Generated Responses

### (1) Automatic Evaluation Metrics

To use automatic evaluation metrics,

  `python evaluate_metrics.py --data_dir Meta-Llama-3.1-8B-Instruct/model/Full_FT/All --check_point BestCheckPoint --is_response true`

### (2) G-Eval

After post-processing the response outputs, ensure your file (e.g., test_response_200.jsonl) follows this format:

  ```
{
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
  }
```

Then run:

  `python g_eval.py --read_file test_response_200.jsonl --api_key OPENAI_API_KEY  --model_type MODEL_TYPE`

Using the following prompt:
<p align="center"> <img src='G-Eval.png' width='1000'> </p>


## Results of Automatic Evaluation Metrics

All assistant responses were generated using <img src="https://latex.codecogs.com/svg.latex?\text{Generator}_{\text{Llama2}}" />. Bold indicates the best performance.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="6">ESConv</th>
      <th colspan="6">ExTES</th>
    </tr>
    <tr>
      <th>B-4</th><th>MET</th><th>Dist-3</th><th>C_W</th><th>C_S</th><th>Greedy</th>
      <th>B-4</th><th>MET</th><th>Dist-3</th><th>C_W</th><th>C_S</th><th>Greedy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Llama2</td>
      <td>0.404</td><td>8.378</td><td>30.086</td><td>88.584</td><td>27.377</td><td>69.874</td>
      <td>2.052</td><td>10.476</td><td>16.232</td><td>93.856</td><td>48.396</td><td>74.564</td>
    </tr>
    <tr>
      <td>+COMET</td>
      <td>0.513</td><td>8.432</td><td>37.777</td><td>88.653</td><td>28.390</td><td>69.946</td>
      <td>2.368</td><td>10.951</td><td>19.799</td><td>94.206</td><td>49.751</td><td>75.353</td>
    </tr>
    <tr>
      <td>+DIALeCT</td>
      <td><b>0.571</b></td><td>8.643</td><td>46.210</td><td>88.888</td><td>30.516</td><td>70.459</td>
      <td>2.353</td><td>10.863</td><td>22.845</td><td>94.006</td><td>50.247</td><td>75.375</td>
    </tr>
    <tr>
      <td>+DOCTOR</td>
      <td>0.525</td><td>8.329</td><td>41.253</td><td>88.174</td><td>27.330</td><td>69.909</td>
      <td>2.216</td><td>10.524</td><td>23.107</td><td>93.618</td><td>46.909</td><td>75.010</td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{Sibyl}_{\text{Llama3.1}}" /></td>
      <td>0.448</td><td>8.165</td><td>48.380</td><td>87.832</td><td>29.189</td><td>69.124</td>
      <td>2.419</td><td>11.002</td><td><b>23.201</b></td><td>93.977</td><td>51.089</td><td>75.091</td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{Sibyl}_{\text{Llama2}}" /></td>
      <td>0.568</td><td>8.333</td><td><b>50.374</b></td><td>88.103</td><td>29.979</td><td>69.430</td>
      <td>2.322</td><td>11.016</td><td>22.950</td><td>94.053</td><td>50.246</td><td>75.059</td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama3.1}}" /></td>
      <td>0.489</td><td><b>9.167</b></td><td>46.461</td><td>89.264</td><td><b>31.053</b></td><td>70.763</td>
      <td>2.875</td><td>12.208</td><td>22.599</td><td>94.664</td><td>52.418</td><td>76.107</td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama2}}" /></td>
      <td>0.511</td><td>9.138</td><td>46.338</td><td><b>89.362</b></td><td>30.758</td><td><b>70.906</b></td>
      <td>2.955</td><td>12.225</td><td>21.789</td><td><b>94.761</b></td><td><b>53.014</b></td><td>76.254</td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Qwen3}}" /></td>
      <td>0.432</td><td>8.991</td><td>45.665</td><td>89.223</td><td>30.792</td><td>70.686</td>
      <td><b>2.958</b></td><td><b>12.231</b></td><td>21.739</td><td>94.684</td><td>52.846</td><td><b>76.259</b></td>
    </tr>
  </tbody>
</table>

All assistant responses were generated using <img src="https://latex.codecogs.com/svg.latex?\text{Generator}_{\text{Qwen3}}" />. Bold indicates the best performance.
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="6">ESConv</th>
      <th colspan="6">ExTES</th>
    </tr>
    <tr>
      <th>B-4</th><th>MET</th><th>Dist-3</th><th>C_W</th><th>C_S</th><th>Greedy</th>
      <th>B-4</th><th>MET</th><th>Dist-3</th><th>C_W</th><th>C_S</th><th>Greedy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen3</td>
      <td>    0.222  </td><td>   8.285  </td><td>   40.274  </td><td>   89.091  </td><td>   26.986  </td><td>   70.007  </td><td>   2.076  </td><td>   11.666  </td><td>   19.435  </td><td>   94.095  </td><td>   46.566  </td><td>   75.270  </td>
    </tr>
    <tr>
      <td>+COMET</td>
      <td>    0.359  </td><td>   8.297  </td><td>   39.418  </td><td>   89.332  </td><td>   27.052  </td><td>   70.667  </td><td>   1.975  </td><td>   11.509  </td><td>   18.334  </td><td>   94.485  </td><td>   45.730  </td><td>   75.594  </td>
    </tr>
    <tr>
      <td>+DIALeCT</td>
      <td> <b>0.482</b></td><td>   7.546  </td><td><b>55.222</b></td><td>   87.110  </td><td>   27.167  </td><td>   68.241  </td><td>   1.457  </td><td>   10.179  </td><td><b>25.451</b></td><td>   92.658  </td><td>   43.010  </td><td>   73.627  </td>
    </tr>
    <tr>
      <td>+DOCTOR</td>
      <td>    0.326  </td><td>   7.162  </td><td>   50.544  </td><td>   85.967  </td><td>   23.315  </td><td>   66.928  </td><td>   1.559  </td><td>   10.144  </td><td>   23.748  </td><td>   92.897  </td><td>   42.552  </td><td>   73.402  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{Sibyl}_{\text{Llama3.1}}" /></td>
      <td>    0.394  </td><td>   7.843  </td><td>   52.845  </td><td>   87.422  </td><td>   27.717  </td><td>   68.269  </td><td>   2.136  </td><td>   11.306  </td><td>   24.556  </td><td>   93.780  </td><td>   47.081  </td><td>   74.328  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{Sibyl}_{\text{Llama2}}" /></td>
      <td>    0.302  </td><td>   7.906  </td><td>   52.044  </td><td>   87.447  </td><td>   27.653  </td><td>   68.338  </td><td>   1.851  </td><td>   11.287  </td><td>   24.676  </td><td>   93.804  </td><td>   46.973  </td><td>   74.234  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama3.1}}" /></td>
      <td>    0.233  </td><td>   9.037  </td><td>   46.391  </td><td><b>89.585</b></td><td>   29.516  </td><td><b>70.827</b></td><td>   2.052  </td><td>   13.141  </td><td>   21.582  </td><td>   94.835  </td><td><b>50.221</b></td><td><b>76.110</b></td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama2}}" /></td>
      <td>    0.239  </td><td><b>9.059</b></td><td>   47.346  </td><td>   89.394  </td><td><b>29.681</b></td><td>   70.664  </td><td>   2.192  </td><td><b>13.195</b></td><td>   21.134  </td><td><b>95.045</b></td><td>   50.154  </td><td>   76.101  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Qwen3}}" /></td>
      <td>    0.247  </td><td>   8.984  </td><td>   47.774  </td><td>   89.390  </td><td>   29.564  </td><td>   70.624  </td><td><b>2.249</b></td><td>   12.918  </td><td>   21.988  </td><td>   94.685  </td><td>   49.442  </td><td>   75.765  </td>
    </tr>
  </tbody>
</table>


All assistant responses were generated using <img src="https://latex.codecogs.com/svg.latex?\text{Generator}_{\text{Llama3.2}}" />. Bold indicates the best performance.
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="6">ESConv</th>
      <th colspan="6">ExTES</th>
    </tr>
    <tr>
      <th>B-4</th><th>MET</th><th>Dist-3</th><th>C_W</th><th>C_S</th><th>Greedy</th>
      <th>B-4</th><th>MET</th><th>Dist-3</th><th>C_W</th><th>C_S</th><th>Greedy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Llama3.2</td>
      <td>    0.355  </td><td>   8.287  </td><td>   40.651  </td><td>   88.789  </td><td>   27.349  </td><td>   70.068  </td><td>   1.581  </td><td>   10.330  </td><td>   23.470  </td><td>   94.148  </td><td>   45.219  </td><td>   74.256  </td>
    <tr>
      <td>+COMET</td>
     <td>    0.368  </td><td>   8.235  </td><td>   41.565  </td><td>   88.736  </td><td>   27.310  </td><td>   70.289  </td><td>   1.800  </td><td>   10.608  </td><td>   23.089  </td><td>   94.148  </td><td>   45.147  </td><td>   74.707  </td>
    </tr>
    <tr>
      <td>+DIALeCT</td>
      <td>    0.431  </td><td>   8.602  </td><td>   41.406  </td><td>   89.507  </td><td>   28.493  </td><td><b>71.222</b></td><td>   1.845  </td><td>   10.644  </td><td>   22.569  </td><td>   94.452  </td><td>   45.981  </td><td>   75.321  </td>
    </tr>
    <tr>
      <td>+DOCTOR</td>
      <td>    0.430  </td><td>   8.402  </td><td>   40.353  </td><td>   89.185  </td><td>   27.634  </td><td>   70.618  </td><td>   1.908  </td><td>   10.593  </td><td>   22.426  </td><td>   94.264  </td><td>   45.209  </td><td>   74.927  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{Sibyl}_{\text{Llama3.1}}" /></td>
      <td>    0.394  </td><td>   7.843  </td><td>   52.845  </td><td>   87.422  </td><td>   27.717  </td><td>   68.269  </td><td>   2.136  </td><td>   11.306  </td><td>   24.556  </td><td>   93.780  </td><td>   47.081  </td><td>   74.328  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{Sibyl}_{\text{Llama2}}" /></td>
      <td>    0.454  </td><td>   8.606  </td><td>   42.547  </td><td>   89.463  </td><td>   28.873  </td><td>   71.187  </td><td>   1.851  </td><td>   11.287  </td><td><b>24.676</b></td><td>   93.804  </td><td>   46.973  </td><td>   74.234  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama3.1}}" /></td>
      <td> <b>0.457</b></td><td>   8.576  </td><td>   44.011  </td><td>   89.563  </td><td><b>29.614</b></td><td>   70.985  </td><td>   2.414  </td><td>   11.346  </td><td>   22.711  </td><td>   94.616  </td><td>   47.619  </td><td>   75.526  </td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama2}}" /></td>
      <td>    0.366  </td><td><b>8.609</b></td><td><b>44.711</b></td><td><b>89.658</b></td><td>   29.332  </td><td>   71.167  </td><td><b>2.418</b></td><td><b>11.384</b></td><td>   22.110  </td><td><b>94.623</b></td><td><b>47.874</b></td><td><b>75.528</b></td>
    </tr>
    <tr>
      <td>+<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Qwen3}}" /></td>
      <td>    0.352  </td><td>   8.587  </td><td>   44.885  </td><td>   89.360  </td><td>   28.966  </td><td>   70.895  </td><td>   2.160  </td><td>   11.254  </td><td>   22.917  </td><td>   94.555  </td><td>   47.707  </td><td>   75.364  </td>
    </tr>
  </tbody>
</table>


## Case Study
We demonstrate the effectiveness of the MENTOS-trained model through a representative ESC example, where the client expresses financial stress caused by COVID-19, a loss of self-confidence, and explicitly seeks experience-based encouragement from the assistant.

<p align="center"> <img src='Case_Study.png' width='1400'> </p>

In the w/o Knowledge setting, only the dialogue history is provided to <img src="https://latex.codecogs.com/svg.latex?\text{Generator}_{\text{Llama2}}" />, without any commonsense knowledge. The response fails to directly address the client’s request about the assistant’s personal experiences.

When using COMET, which relies only on the last client utterance, misclassifying the user's states (“Thanks. I appreciate that”). It misinterprets a complex emotional state as simply 'happy,' failing to consider the broader context of emotional vulnerability.

When using DOCTOR, the term "project" in the generated response is vague, and emotional cues about the client were omitted during multi-hop reasoning.
The inferred commonsense knowledge primarily focuses on the client (Spike) seeking help, while the xIntent type emphasizes networking rather than emotional support.
As a result, the <img src="https://latex.codecogs.com/svg.latex?\text{Generator}_{\text{Llama2}}" /> using DOCTOR fails to provide meaningful guidance for generating empathetic and problem-solving–oriented responses.

