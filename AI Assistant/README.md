# AI Assistant Module

<img width="2720" height="1280" alt="ai_assistant_module_logo" src="https://github.com/user-attachments/assets/5655fdd8-c18f-49d5-ae74-3de09529b1f9" />

Part of the flood detection AI framework. This module is an in-app assistant that answers user questions about how the other four modules work (Data Acquisition, Data Annotation, Training and Preprocessing, Damaged Building Detection). It is a support and documentation tool, not part of the flood detection model itself.

## What it does

A user types a question in plain English, for example "How do I draw an area of interest for downloading satellite imagery?", and the assistant returns a grounded answer pulled from the project's own knowledge base, rewritten in a more conversational tone. If the question is outside the project's scope, the assistant says so instead of guessing.

## Architecture: retrieve-and-rephrase (not generative RAG)

This is a constrained variant of RAG. A standard RAG pipeline retrieves context and lets the LLM generate a free-form answer from it. Here, the LLM is never allowed to generate the answer's content. It only rephrases an answer that was already written and verified by hand. This removes the main risk of RAG in a technical support setting: the model inventing or distorting a number, threshold, or parameter.

The pipeline has four stages:

```
User question
     |
     v
[1] Embed question  --------------->  all-MiniLM-L6-v2
     |
     v
[2] Retrieve nearest QA pair  ----->  cosine similarity over a fixed
     |                                 knowledge base, optionally
     |                                 filtered to one module
     v
[3] Threshold check  -------------->  below 0.55 -> fallback message,
     |                                 no LLM call
     v
[4] Rephrase  ---------------------->  Qwen2.5-1.5B-Instruct (GGUF)
     |                                 rewrites tone only, facts frozen
     v
Final answer + matched module + similarity score
```

### 1. Knowledge base

A curated set of 822 question-answer pairs (`qa_dataset.json`), each tagged with the module it belongs to:

| Module | QA pairs |
|---|---|
| Training and Preprocessing | 281 |
| Data Acquisition | 213 |
| Damaged Building Detection | 173 |
| Data Annotation | 155 |

Every pair is a single fact: a question phrased the way a user would actually ask it, and a short, precise answer. This dataset is the only source of truth the assistant is allowed to draw from.

### 2. Retrieval

Every question in the knowledge base is embedded once at startup with `sentence-transformers/all-MiniLM-L6-v2` (normalized embeddings, CPU inference). At query time the user's question is embedded the same way, and the best match is found by cosine similarity (dot product of normalized vectors) against the candidate pool.

Retrieval supports **module filtering**: if the calling UI already knows which module the user is in (because they asked the question from inside, say, the Data Annotation screen), the search space is restricted to that module's QA pairs only. This removes cross-module ambiguity (e.g. "threshold" means something different in annotation vs. damage detection) and makes retrieval faster and more precise. If no module is given, the assistant searches the full knowledge base.

### 3. Similarity threshold and fallback

A fixed threshold (`0.55`) decides whether the best match is good enough to answer from. If the top similarity score falls below this, the assistant returns a fixed fallback message and skips the LLM call entirely:

> "That question isn't covered in the flood detection project knowledge base."

This is what keeps the assistant from confidently answering questions it has no business answering, even if a module filter is set (e.g. asking for a cake recipe inside the Data Acquisition module still falls back).

### 4. Rephrasing

Once a record clears the threshold, its stored answer is rewritten by a small local LLM, **Qwen2.5-1.5B-Instruct** (GGUF, quantized, run via `llama-cpp-python`, 4 threads, 2048 context). The system prompt is deliberately narrow:

> Rephrase technical answers so they sound more conversational. Keep every fact, number, threshold, unit, and formula exactly as given. Do not add or remove information. Only adjust sentence structure and tone.

The model is given the verified answer as input and asked only to restyle it (low temperature, 0.3, to keep rewrites stable). It is never given the question, the knowledge base, or any instruction to "answer" anything — its only job is paraphrasing.

Rephrased answers are cached by record index, so the first time a given QA pair is matched it costs one LLM call, and every later hit on that same record is served instantly from the cache rather than calling the LLM again.

## Output

`answer_question(question, module=None)` returns a 3-tuple:

- the final answer (rephrased text, or the fallback message)
- the matched module (or `None` if the fallback fired)
- the similarity score of the best match, for logging/debugging

## Design notes

- **No hallucination by construction.** The LLM never sees the knowledge base or generates facts; it only restyles a pre-approved answer. Wrong answers can only come from bad retrieval, not invented content.
- **Module filtering** is optional but recommended: it cuts ambiguity between modules that reuse similar terminology and reduces the search space.
- **Threshold is a tunable knob.** Raising it makes the assistant stricter about admitting "I don't know"; lowering it makes it more willing to stretch a partial match.
- **Caching** means repeated questions across users (or the same user trying different phrasings that land on the same record) don't re-run the LLM.

## Stack

- `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
- `numpy` for similarity scoring
- `llama-cpp-python` + Qwen2.5-1.5B-Instruct GGUF (Q4_K_M) for rephrasing
- Plain JSON for the knowledge base, no vector database required at this scale
