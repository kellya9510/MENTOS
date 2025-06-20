# Comparative Analysis of Commonsense Approaches in ESC

ToMESC is a structured commonsense knowledge resource designed to support emotionally and strategically appropriate responses in Emotional Support Conversations (ESC). Unlike prior approaches that rely on loosely connected or static knowledge, ToMESC models the assistant’s latent reasoning process by capturing key Theory of Mind (ToM) mental states—**Belief, Emotion, Desire, and Intent (BEDI)**—in a causally linked sequence. This structure allows LLMs to understand not just what to say, but why it should be said.

We illustrate the value of ToMESC using a representative ESC example, where a user expresses emotional vulnerability and explicitly seeks experience-based encouragement from the assistant.

## Example 
To evaluate the effectiveness of commonsense reasoning models in ESC, we examine a representative dialogue in which the user expresses financial stress due to COVID-19, diminished self-confidence, and explicitly seeks experience-based encouragement from the assistant.

### Dialogue History
User: Hello. My name is Spike. What is yours? Is anyone there? I feel kind of sad and rejected that no one is answering. Is this part of the task?

Assistant: oh no i'm sorry i left you waited. please tell me your problem

User: I'm trying to make money on MTurk and starting a YouTube channel. Money is a big concern for me.

Assistant: We are on the same boat. Do you have any other jobs?

User: No. That's my only income right now. It has led to a real crisis of self confidence to me.

Assistant: Have you considered applying for a job or trying uber drive/ eats?

User: I have. But I have health conditions that would mean Covid-19 would be very risky for me to get. So, I'm trying to work online. I'm trying to stay encouraged.

Assistant: I understand how you feel, it's tough not being able to work and earn much money. However, you are doing your best right now and should feel proud of yourself

User: Thanks. I appreciate that. What do you do to stay encouraged or confident?

Following-up Response]
Assistant: I try to spend more time outdoor and keep myself busy. I like to cook since it's very relaxing and helps keep my mind concentrated


### Response Without Commonsense
When prompted to generate a response based solely on the dialogue history—without any commonsense input—**Llama2** produces a generic and emotionally detached reply:
> *I see. It sounds like you're facing a tough situation. Let's explore some strategies to help you cope with your emotional distress and find ways to stay encouraged. Would you like that?*

While fluent, this response fails to directly address the user's question regarding the assistant’s personal experience, potentially weakening the emotional connection.

### Model-wise Analysis

1. COMET
COMET generates the following knowledge types based only on the final user utterance (“Thanks. I appreciate that”):
* `"xReact": ["happy"]`
* `"xIntent": ["to be successful"]`
* `"xNeed": ["to be confident"]`
* `"xWant": ["to be successful"]`
> *I see. It sounds like? you're facing some challenging circumstances. Let's work together to explore these feelings and find ways to help you stay encouraged and confident. Would you like to take a deep breath and continue sharing?*

These outputs highlight COMET’s limitation: it focuses only on the last utterance, misclassifying the user’s complex emotional state as simply “happy” and failing to consider the broader context of emotional vulnerability.

2. DIALeCT
DIALeCT generates more context-aware commonsense:
* `"Cause": "The listener asked the speaker what he does to stay motivated and confident."`
* `"SubEv": "The speaker asks the listener if he has any tips for staying motivated."`
* `"Prere": "Spike is doing his best to earn money online."`
* `"Motiv": "The speaker is curious to know what the listener is doing to stay motivated."`
* `"React": "The listener is happy to know that the speaker is encouraging him.“`
> *Ah, I see. It can be challenging to stay motivated when facing health concerns and financial struggles. Have you tried any mindfulness techniques or self-care activities to help manage your stress and anxiety?*

Although these outputs reflect the dialogue history more accurately, the model fails to capture the user’s request for the assistant’s own experience. The response generated from DIALeCT’s knowledge omits this aspect. While the `SubEv` type implies an experience-related exchange but lacks explicit phrasing, making it difficult for the LLM to infer the correct intent.


3. DOCTOR
DOCTOR produces the following:
* `"xWant": Person A wants to work on a project and is seeking help from Person B.`
* `"oReact": Person B is willing to help and offers to assist Person A in starting the project.`
* `"xIntent": Person B might want to support Person A and build a stronger relationship, which could potentially lead to more opportunities for collaboration or networking.`
> *I'm glad you're seeking help, Spike. I'm here to support you in any way I can. Let's work together to find a solution to your financial struggles.*

These outputs are misaligned with the ESC context. The term “project” is vague and irrelevant, and the assistant’s inferred motivation (`xIntent`) focuses on networking rather than emotional support. As a result, the model fails to provide useful guidance for generating empathetic responses.


4. Convosense
Convosense outputs include:
* `"cause": "a misunderstanding or miscommunication between the speaker and listener."`
* `"prerequisities": "the speaker and listener have a pre-existing relationship or connection."`
* `"motivation": "by a sense of responsibility to contribute to society."`
* `"subsequent": "the listener might ask the speaker if they have any plans for the weekend."`
* `"desire": "to discuss a potential project with the listener."`
* `"desire_o": "to ask speaker for advice on a personal matter."`
* `"react": "curious about the listener's response."`
* `"react_o": "curious about what speaker is going to say next."`
* `"attribute": "a good communicator."`
* `"constituents": "indicating that the speaker is the one who initiated the conversation."`
> *Of course, I'm here to help! It sounds like you're facing some challenges with your current situation. Can you tell me more about what's been going on and how you've been feeling?*

Many types (e.g., `cause`, `prerequisities`, and `subsequent`) are either irrelevant (e.g., “weekend plans”) or too generic (e.g., “a sense of responsibility”), making them unsuitable for guiding ESC response generation. This underscores the challenge of relying on heterogeneous, loosely coupled knowledge without filtering for contextual relevance.

5. Sibyl

Sibyl generates:

* `ChatGPT_cause: The supporter recognizes that the help seeker is struggling with low self-confidence due to financial difficulties and is seeking advice on how to stay encouraged and confident.`
* `ChatGPT_emo: Encouragement: The help seeker expresses encouragement and support towards the emotional supporter, acknowledging their efforts and praising their resilience in the face of financial challenges.`
* `ChatGPT_subs: The supporter may share their own strategies for staying encouraged and confident, such as engaging in hobbies or seeking support from friends and family.`
* `ChatGPT_intent: The supporter's intent is to reciprocate the encouragement and support given by the help seeker. By asking about the supporter's methods for staying encouraged or confident, the supporter aims to establish a connection and potentially learn from the help seeker's experiences.`
> *I see. It can be challenging to maintain confidence when facing financial difficulties. I engage in hobbies and seek support from friends and family to stay encouraged. Have you tried any of those strategies? *

While some of these are contextually valid (e.g., `ChatGPT_cause`, `ChatGPT_subs`), the lack of **causal connection** across types results in inconsistencies. For example, `ChatGPT_intent` incorrectly frames the assistant as learning from the user, contrary to the dialogue flow. This suggests that independent generation of knowledge types without structural grounding can undermine coherence.

6. ToMESC

ToMESC produces structured BEDI annotations based on the entire dialogue history and the assistant’s intended response:
* `Belief: The assistant believes the client is seeking encouragement and confidence strategies, feels uncertain about their ability to stay motivated, and values understanding and support in navigating their challenges.`
* `Emotion: `
    * `[Basic] Sadness (opposite Joy): 1, Disgust (opposite Trust): 0, Anger (opposite Fear): 0, Anticipation (opposite Surprise): 1, Joy (opposite Sadness): 1, Trust (opposite Disgust): 2, Fear (opposite Anger): 0, Surprise (opposite Anticipation): 0`
    * `[Mixed] Hopelessness (sadness + fear): 0, Remorse (sadness + disgust): 0, Disappointment (sadness + surprise): 0, Sentimental (sadness + trust): 2, Jealousy (sadness + anger): 0, Pessimism (sadness + anticipation): 0, Embarrassment (disgust + fear): 0, Pride (anger + joy): 0, Nervousness (anticipation + fear): 0, Delight (joy + surprise): 0, Gratitude/Love/Caring (joy + trust): 2, Hope/Optimism (joy + anticipation): 1, Guilt (joy + fear): 0, Curiosity (surprise + trust): 0`
* `Intent: The assistant’s intent is to empathize with the client by sharing a personal experience, fostering a sense of connection and understanding, and encouraging the client to stay motivated and hopeful despite their challenges.`
> *I understand, it can be challenging to stay motivated and confident when facing financial struggles. I've found that setting small, achievable goals for myself and celebrating my progress helps me stay encouraged. Have you tried that? *

ToMESC’s strengths lie in:

* **Causal grounding**: Unlike Sibyl, which generates each knowledge type independently, ToMESC models the assistant’s reasoning as a structured causal sequence—*Belief → Emotion → Intent*. This enables consistent and context-aware commonsense generation.

* **Pre-response mental state modeling**: In contrast to ConvoSense, which includes types that are not always relevant for guiding assistant responses (e.g., `subsequent`, `attribute`), ToMESC focuses exclusively on Belief, Emotion, and Intent—mental states that are essential for generating the assistant’s next utterance.

* **Emotion inference grounded in Belief**: Unlike DOCTOR, which omits the assistant’s affective stance or infers it abstractly, ToMESC explicitly derives Emotion based on the assistant’s interpretation of the user’s emotional state and situation (Belief for Emotion).

* **Dialogue-history alignment**: In contrast to COMET, which relies solely on the final utterance, ToMESC considers the full dialogue history, allowing it to model richer emotional and contextual cues.

* **Response-aware intent modeling**: Unlike DIALeCT, which does not take the assistant’s response into account during commonsense construction, ToMESC ensures that the annotated Intent is aligned with both the dialogue history and the assistant’s actual response—capturing, in this example, the user’s request for a personal strategy and the assistant’s decision to share their own experience.

Together, these features demonstrate how ToMESC overcomes key limitations of prior approaches by modeling the assistant’s reasoning as a causal sequence (Belief → Emotion → Intent). In Example 1, this enables the assistant to understand the user’s emotional need and respond with a relevant, emotionally grounded personal experience—something all prior models fail to achieve.

---


### Quantitative Comparison (G-Eval)

| Metric         | Llama2 | COMET | DIALeCT | DOCTOR | ConvoSense | Sibyl | ToMESC  |
| -------------- | ------ | ----- | ------- | ------ | ---------- | ----- | ------- |
| Supportiveness | 2.25   | 2.5   | 2.85    | 2.05   | 2.0        | 2.1   | **3.0** |
| Naturalness    | 2.25   | 2.25  | 2.95    | 2.1    | 2.25       | 2.75  | **3.0** |
| Coherence      | 2.9    | 2.85  | 2.95    | 2.0    | 2.0        | 3.0   | **3.0** |

ToMESC outperforms all baselines across metrics, confirming that structured ToM-based reasoning leads to responses that are more emotionally aligned, coherent, and supportive—essential qualities for effective ESC.



