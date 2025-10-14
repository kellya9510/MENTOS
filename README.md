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
The dataset is publicly available at [https://zenodo.org/doi/10.5281/zenodo.15624491](https://zenodo.org/doi/10.5281/zenodo.15624491).
Download MENTOS and place it in the `data/` folder before running any scripts.

---

### Dataset Construction

To construct the MENTOS dataset from ESConv, run the following after downloading ESConv:

  `python create_mental_state.py --api_key OPENAI_API_KEY --model_type gpt-4o-2024-11-20`

### Evaluate the MENTOS quality

To assess the quality of MENTOS annotations, we conducted a human evaluation on 100 randomly sampled dialogues. Four annotators independently rated each assistant utterance across three mental state categories—**Belief**, **Emotion**, and **Intent**—using four evaluation criteria per category, each on a 1–3 scale. To measure inter-annotator reliability, we report Gwet’s AC1, which is robust against prevalence and marginal distribution biases. Across all categories and criteria, AC1 values ranged from 0.6 to 0.8, indicating substantial agreement among annotators.

The evaluation was performed using the following command:

 `python evaluate_human_sample.py --read_file MENTOS_sample.jsonl`

## MENTOS-trained Commonsense Reasoning Model

### Fine-tuning (SFT) using MENTOS

To fine-tune a model using the training split of MENTOS:

  `python Fine-Tuning.py --mental_state_type All --data_dir data --batch_size 4 -num_epochs 5 --learning_rate 3e-5 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05`

### Inference

Run inference for each mental state:

  `python inference_mental_state.py --mental_state_type Belief --check_point BestCheckPoint --test_file data/test.jsonl`

  
  `python inference_mental_state.py --mental_state_type Emotion --check_point BestCheckPoint --test_file data/test.json`

  
  `python inference_mental_state.py --mental_state_type Intent --check_point BestCheckPoint --test_file data/test.json`

## Zero-shot Response Generation

To generate responses using a zero-shot LLM:

  `python generate_response.py --model_name meta-llama/Llama-2-7b-chat-hf --test_file Meta-Llama-3.1-8B-Instruct/model/Full_FT/All/output_BestCheckPoint.jsonl --output_file output/output_All_response_BestCheckPoint.jsonl --new_max_token 100`

## Evaluation

Evaluate Generated Responses

  `python evaluate_metrics.py --data_dir Meta-Llama-3.1-8B-Instruct/model/Full_FT/All --check_point BestCheckPoint --is_response true`


The detailed results are summarized below:

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
      <td>+Sibyl (Llama3.1)</td>
      <td>0.448</td><td>8.165</td><td>48.380</td><td>87.832</td><td>29.189</td><td>69.124</td>
      <td>2.419</td><td>11.002</td><td><b>23.201</b></td><td>93.977</td><td>51.089</td><td>75.091</td>
    </tr>
    <tr>
      <td>+Sibyl (Llama2)</td>
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
| Model                                                                               |   **B-4**  |  **MET**  | **Dist-3** |   **C_W**  |   **C_S**  | **Greedy** |  **B-4**  |   **MET**  | **Dist-3** |   **C_W**  |   **C_S**  | **Greedy** |
| :---------------------------------------------------------------------------------- | :--------: | :-------: | :--------: | :--------: | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|                                                                                     | **ESConv** |           |            |            |            |            | **ExTES** |            |            |            |            |            |
| Qwen3                                                                               |    0.222   |   8.285   |   40.274   |   89.091   |   26.986   |   70.007   |   2.076   |   11.666   |   19.435   |   94.095   |   46.566   |   75.270   |
| +COMET                                                                              |    0.359   |   8.297   |   39.418   |   89.332   |   27.052   |   70.667   |   1.975   |   11.509   |   18.334   |   94.485   |   45.730   |   75.594   |
| +DIALeCT                                                                            |  **0.482** |   7.546   | **55.222** |   87.110   |   27.167   |   68.241   |   1.457   |   10.179   | **25.451** |   92.658   |   43.010   |   73.627   |
| +DOCTOR                                                                             |    0.326   |   7.162   |   50.544   |   85.967   |   23.315   |   66.928   |   1.559   |   10.144   |   23.748   |   92.897   |   42.552   |   73.402   |
| +Sibyl (Llama3.1)                                                                   |    0.394   |   7.843   |   52.845   |   87.422   |   27.717   |   68.269   |   2.136   |   11.306   |   24.556   |   93.780   |   47.081   |   74.328   |
| +Sibyl (Llama2)                                                                     |    0.302   |   7.906   |   52.044   |   87.447   |   27.653   |   68.338   |   1.851   |   11.287   |   24.676   |   93.804   |   46.973   |   74.234   |
| +<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama3.1}}" /> |    0.233   |   9.037   |   46.391   | **89.585** |   29.516   | **70.827** |   2.052   |   13.141   |   21.582   |   94.835   | **50.221** | **76.110** |
| +<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama2}}" />   |    0.239   | **9.059** |   47.346   |   89.394   | **29.681** |   70.664   |   2.192   | **13.195** |   21.134   | **95.045** |   50.154   |   76.101   |
| +<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Qwen3}}" />    |    0.247   |   8.984   |   47.774   |   89.390   |   29.564   |   70.624   | **2.249** |   12.918   |   21.988   |   94.685   |   49.442   |   75.765   |


All assistant responses were generated using <img src="https://latex.codecogs.com/svg.latex?\text{Generator}_{\text{Llama3.2}}" />. Bold indicates the best performance.
| Model                                                                               |   **B-4**  |  **MET**  | **Dist-3** |   **C_W**  |   **C_S**  | **Greedy** |  **B-4**  |   **MET**  | **Dist-3** |   **C_W**  |   **C_S**  | **Greedy** |
| :---------------------------------------------------------------------------------- | :--------: | :-------: | :--------: | :--------: | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|                                                                                     | **ESConv** |           |            |            |            |            | **ExTES** |            |            |            |            |            |
| Llama3.2                                                                            |    0.355   |   8.287   |   40.651   |   88.789   |   27.349   |   70.068   |   1.581   |   10.330   |   23.470   |   94.148   |   45.219   |   74.256   |
| +COMET                                                                              |    0.368   |   8.235   |   41.565   |   88.736   |   27.310   |   70.289   |   1.800   |   10.608   |   23.089   |   94.148   |   45.147   |   74.707   |
| +DIALeCT                                                                            |    0.431   |   8.602   |   41.406   |   89.507   |   28.493   | **71.222** |   1.845   |   10.644   |   22.569   |   94.452   |   45.981   |   75.321   |
| +DOCTOR                                                                             |    0.430   |   8.402   |   40.353   |   89.185   |   27.634   |   70.618   |   1.908   |   10.593   |   22.426   |   94.264   |   45.209   |   74.927   |
| +Sibyl (Llama3.1)                                                                   |    0.430   |   8.571   |   42.916   |   89.352   |   28.991   |   71.077   |   2.136   |   11.306   |   24.556   |   93.780   |   47.081   |   74.328   |
| +Sibyl (Llama2)                                                                     |    0.454   |   8.606   |   42.547   |   89.463   |   28.873   |   71.187   |   1.851   |   11.287   | **24.676** |   93.804   |   46.973   |   74.234   |
| +<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama3.1}}" /> |  **0.457** |   8.576   |   44.011   |   89.563   | **29.614** |   70.985   |   2.414   |   11.346   |   22.711   |   94.616   |   47.619   |   75.526   |
| +<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Llama2}}" />   |    0.366   | **8.609** | **44.711** | **89.658** |   29.332   |   71.167   | **2.418** | **11.384** |   22.110   | **94.623** | **47.874** | **75.528** |
| +<img src="https://latex.codecogs.com/svg.latex?\text{MENTOS}_{\text{Qwen3}}" />    |    0.352   |   8.587   |   44.885   |   89.360   |   28.966   |   70.895   |   2.160   |   11.254   |   22.917   |   94.555   |   47.707   |   75.364   |



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

We illustrate the value of MENTOS using representative ESC examples (Example 1 and 2), where a user expresses emotional vulnerability and explicitly seeks experience-based encouragement from the assistant. Detailed comparisons are available [here](https://github.com/kellya9510/MENTOS/blob/main/Comparative%20Analysis%20of%20Commonsense%20Approaches%20in%20ESC.md).
