# Full yoctoGPT Architecture — Stack View

**Purpose.** yoctoGPT is a minimal, readable, decoder-only GPT implementation in PyTorch. The architecture takes raw text, converts it into token IDs, maps those IDs into vectors, processes them through a stack of Transformer blocks, and produces logits for the next-token distribution. Sampling then turns these logits back into generated text.

---

## 1. Complete Flow

The full pipeline is:

```text
raw text
  → tokenizer
  → token IDs [B, T]
  → token embedding + position information [B, T, C]
  → L Transformer blocks
  → final normalization
  → LM head
  → logits [B, T, V]
  → probabilities
  → sampled next token
  → generated text
```

The core tensor shape after embedding is:

```text
[B, T, C]
```

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `T` | context length / `block_size` |
| `C` | embedding width / `n_embd` |
| `L` | number of Transformer blocks / `n_layer` |
| `H` | number of attention heads / `n_head` |
| `d_k` | per-head dimension, usually `C / H` |
| `V` | vocabulary size |

---

## 2. Raw Text

The model starts with plain text, for example:

```text
the cat sat on the mat
```

yoctoGPT supports both character-level and token-level modeling. The architectural pipeline is the same in both cases; what changes is the tokenizer and therefore the vocabulary.

---

## 3. Tokenization

### 3.1 Character-Level Tokenization

In character-level training, every character is a token.

Example:

```text
"cat" → ["c", "a", "t"]
```

Each unique character receives an integer ID:

```text
"c" → 12
"a" → 4
"t" → 23
```

The text becomes:

```text
[12, 4, 23]
```

**Role.** Character-level tokenization is simple and transparent. It is ideal for teaching because the complete data pipeline can be understood without additional tokenizer machinery.

**Trade-off.** The model must learn words, subwords, spelling, and longer patterns from very small units. This usually requires longer sequences and more training to reach fluent text.

### 3.2 BPE Tokenization

With BPE, frequent character sequences are merged into subword tokens.

Conceptually:

```text
"tokenization" → ["token", "ization"]
```

or:

```text
"financial markets" → ["financial", " markets"]
```

**Role.** BPE provides a more efficient representation than characters. It usually shortens sequences and lets the model learn at the subword level.

**Trade-off.** BPE adds tokenizer complexity, but it is closer to how modern GPT-style models are trained.

---

## 4. Token IDs `[B, T]`

After tokenization, the input is a matrix of integer token IDs:

```text
x.shape = [B, T]
```

where:

- `B` is the number of independent sequences processed in parallel.
- `T` is the context length, also called `block_size`.

Example with `B = 2` and `T = 8`:

```text
[
  [12,  4, 23, 18,  9,  7, 31,  2],
  [ 5, 11, 19,  3, 28,  6, 13,  8]
]
```

Each row is one training or inference sequence.

---

## 5. Context Length `T` / `block_size`

The block size controls how many previous tokens the model can see at once.

If:

```text
block_size = T = 128
```

then the model can condition each prediction on at most the previous 128 tokens.

### Effect of Larger `T`

A larger context length allows the model to use longer-range dependencies:

```text
earlier context → ... → current token
```

This helps with longer sentences, document structure, delayed dependencies, code indentation and scope, and domain-specific sequences where the relevant information appears earlier in the context.

### Cost of Larger `T`

Self-attention scales approximately quadratically in `T`:

```text
attention cost ∝ T²
```

Doubling the context length roughly quadruples the attention matrix size.

---

## 6. Token Embeddings

Token IDs are discrete integers. Neural networks operate on continuous vectors.

The token embedding table maps each token ID to a vector of length `C`:

```text
token_id → embedding vector in R^C
```

If the vocabulary size is `V`, the token embedding table has shape:

```text
[V, C]
```

After lookup:

```text
token IDs [B, T] → token embeddings [B, T, C]
```

### Role of `C = n_embd`

`C` is the width of the model. It controls how much information can be stored per token position.

A larger `C` gives the model more representational capacity but increases parameter count, compute cost, memory use, and overfitting risk on small datasets.

---

## 7. Position Information

A Transformer processes tokens largely in parallel. Without position information, it would not know whether a token appears early or late in the sequence.

Therefore, yoctoGPT adds position information to token embeddings.

Conceptually:

```text
x = token_embedding[token_id] + position_embedding[position]
```

The result is:

```text
x.shape = [B, T, C]
```

### Why Position Information Matters

The sequences:

```text
"the cat chased the dog"
"the dog chased the cat"
```

contain similar tokens but have different meanings because order matters. Position information lets the model distinguish these cases.

---

## 8. Decoder-Only GPT Architecture

yoctoGPT follows the decoder-only GPT pattern. The model processes the sequence from left to right and predicts the next token at every position.

For a sequence:

```text
[t0, t1, t2, t3]
```

the model is trained to predict:

```text
[t1, t2, t3, t4]
```

Each position predicts the following token.

---

## 9. Causal Masking

GPT models must not look into the future when predicting the next token.

At position `i`, the model may only attend to positions:

```text
0, 1, ..., i
```

It may not attend to:

```text
i + 1, i + 2, ...
```

This is enforced by a causal mask.

For a sequence length `T`, attention creates a `T × T` relationship matrix. Without masking, every token could attend to every other token. With causal masking, the matrix is lower triangular:

```text
position 0 sees: 0
position 1 sees: 0, 1
position 2 sees: 0, 1, 2
...
position T-1 sees: 0, ..., T-1
```

**Role.** Causal masking makes the model suitable for autoregressive generation.

---

## 10. Transformer Block Overview

The decoder stack consists of `L` repeated Transformer blocks.

Each block receives:

```text
x [B, T, C]
```

and returns:

```text
x [B, T, C]
```

The shape is preserved across blocks.

A simplified block is:

```text
x = x + causal_self_attention(norm(x))
x = x + mlp(norm(x))
```

The two main sublayers are:

1. causal multi-head self-attention,
2. feed-forward network / MLP.

Residual connections wrap both.

---

## 11. Normalization

Each block applies normalization before major transformations.

In the baseline model this is typically LayerNorm-like behavior. In improved variants such as `gpt_plus`, RMSNorm may be used.

### Role of Normalization

Normalization stabilizes training by keeping activations in a controlled range. It helps with gradient flow, numerical stability, faster convergence, and deeper stacks.

Without normalization, training can become unstable, especially as model depth increases.

---

## 12. Causal Self-Attention

Self-attention lets each token look at earlier tokens in the same sequence and decide which ones are relevant.

For every position, the model computes:

- a query vector,
- a key vector,
- a value vector.

Conceptually:

```text
Q = x W_Q
K = x W_K
V = x W_V
```

Attention scores are computed from queries and keys:

```text
scores = Q Kᵀ / sqrt(d_k)
```

After applying the causal mask and softmax:

```text
weights = softmax(masked_scores)
```

The output is a weighted combination of values:

```text
attention_output = weights V
```

### Role

Self-attention lets the model dynamically decide what earlier tokens matter for the current token.

For example, in:

```text
The bond price falls when yields rise because ...
```

a later token may need to attend to `bond`, `price`, `yields`, and `rise`. The attention mechanism learns these relationships from data.

---

## 13. Multi-Head Attention

Instead of performing one attention operation, GPT uses `H` parallel attention heads.

The embedding width `C` is split across heads:

```text
d_k = C / H
```

Each head works on a subspace of the representation.

For example, if:

```text
C = 256
H = 8
```

then each head has:

```text
d_k = 32
```

The input shape remains:

```text
[B, T, C]
```

but internally it is viewed as:

```text
[B, T, H, d_k]
```

or, depending on implementation:

```text
[B, H, T, d_k]
```

Each head computes its own attention pattern:

```text
head_1 = attention(Q_1, K_1, V_1)
head_2 = attention(Q_2, K_2, V_2)
...
head_H = attention(Q_H, K_H, V_H)
```

The heads are then concatenated:

```text
concat(head_1, ..., head_H) → [B, T, C]
```

and projected back into the model dimension.

---

## 14. Why Multiple Heads Instead of One?

A single attention head produces one attention pattern per layer.

Multiple heads allow the model to learn several different relationship types in parallel.

### With One Head

With `H = 1`:

```text
d_k = C
```

There is only one attention mechanism. It must use the full representation to capture all relationships.

This is simpler and useful for understanding the mechanics, but less flexible.

### With Multiple Heads

With `H > 1`, each head can specialize.

One head might focus on nearby tokens, syntax, repeated words, punctuation, longer-range dependencies, topic continuity, numerical patterns, or domain-specific phrases.

The model is not explicitly told to assign these roles. They emerge during training.

### Practical Effect

Multiple heads usually improve expressiveness because the model can represent different attention views at the same layer.

However, increasing `H` while keeping `C` fixed reduces the per-head dimension:

```text
d_k = C / H
```

So there is a trade-off:

- more heads provide more parallel attention patterns,
- too many heads make each head too narrow.

Good configurations balance `C` and `H`.

---

## 15. Attention Shape Summary

For input:

```text
x [B, T, C]
```

multi-head attention internally uses:

```text
Q, K, V [B, H, T, d_k]
```

where:

```text
d_k = C / H
```

For each head, the attention score matrix has shape:

```text
[B, H, T, T]
```

The `T × T` part represents which positions attend to which earlier positions.

After attention and concatenation, the output returns to:

```text
[B, T, C]
```

---

## 16. SDPA and FlashAttention in `gpt_fast`

The `gpt_fast` variant uses PyTorch's scaled dot-product attention path where available.

The mathematical operation is still causal self-attention. The difference is implementation efficiency:

- faster training,
- faster inference,
- lower memory use,
- better GPU utilization.

This is especially useful when `T`, `B`, or `C` become larger.

---

## 17. Residual Connections

A residual connection adds the input of a sublayer back to its output.

For attention:

```text
x = x + attention(norm(x))
```

For the MLP:

```text
x = x + mlp(norm(x))
```

### Role

Residual connections allow information to flow through the model even if some transformations are initially poor. They help with training deeper networks, gradient propagation, preserving useful information, and incremental refinement across layers.

Without residual connections, deep Transformer stacks would be much harder to train.

---

## 18. MLP / Feed-Forward Network

After attention, each token position is processed by an MLP.

Attention mixes information across positions. The MLP transforms information within each position.

Conceptually:

```text
[B, T, C] → [B, T, hidden] → [B, T, C]
```

The MLP is applied independently to each position but shares weights across positions.

### Role

The MLP gives the model nonlinear representational power. It helps turn the information gathered by attention into richer features.

---

## 19. GELU vs SwiGLU

The baseline GPT-style MLP often uses GELU.

The `gpt_plus` variant may use SwiGLU.

### GELU

GELU is a standard smooth nonlinear activation used in many Transformer models.

### SwiGLU

SwiGLU is a gated activation mechanism that can improve expressiveness by letting one part of the network modulate another.

Conceptually, instead of just:

```text
activation(linear(x))
```

SwiGLU uses a learned gate:

```text
activation(linear_1(x)) * linear_2(x)
```

### Role

SwiGLU can improve quality for similar model sizes, although it adds some architectural complexity.

---

## 20. What One Block Does

One Transformer block performs one round of contextual refinement.

Starting with:

```text
x [B, T, C]
```

the block:

1. normalizes `x`,
2. applies causal multi-head attention,
3. adds the result back through a residual connection,
4. normalizes again,
5. applies the MLP,
6. adds the result back through another residual connection.

In compact form:

```text
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

### Interpretation

A single block lets each token gather information from earlier tokens, process that information through nonlinear transformations, and pass the refined representation onward.

---

## 21. Why Use `L` Blocks Instead of One?

A single block performs one level of contextual processing.

Multiple blocks allow the model to refine representations repeatedly:

```text
block 1 → block 2 → ... → block L
```

Each block receives the output of the previous block.

### With One Block

With only one block, the model can attend to context once and apply one MLP transformation. This can work for tiny educational examples, but capacity is limited.

### With Multiple Blocks

With `L > 1`, the model performs multiple rounds of:

```text
attend → transform → refine
```

This usually improves modeling capacity.

Depth lets the model build hierarchical representations. Lower blocks may learn local patterns, middle blocks may combine phrase-level information, and higher blocks may represent broader context and next-token intent. This is not hard-coded; it emerges during training.

### Cost

More blocks increase parameter count, training time, inference latency, and memory use. Therefore, `L` is a major capacity and cost parameter.

---

## 22. Decoder Stack

The decoder stack is the repeated sequence of Transformer blocks.

```text
x_0 = embedding output

x_1 = block_1(x_0)
x_2 = block_2(x_1)
...
x_L = block_L(x_{L-1})
```

The shape remains:

```text
[B, T, C]
```

throughout the stack.

The stack is the main representation engine of the model.

---

## 23. Final Normalization

After the last Transformer block, the model applies a final normalization:

```text
x_final = norm(x_L)
```

### Role

The final norm stabilizes the representation before it is mapped to vocabulary logits. It makes the final LM head operate on a well-conditioned hidden state.

---

## 24. LM Head

The language modeling head maps the final hidden state to vocabulary logits.

Input:

```text
[B, T, C]
```

Output:

```text
[B, T, V]
```

where `V` is the vocabulary size.

The LM head is usually a linear projection:

```text
logits = x W_vocab
```

with:

```text
W_vocab.shape = [C, V]
```

Each position receives one score per vocabulary token.

---

## 25. Logits

Logits are raw, unnormalized scores.

For each batch item and each position:

```text
logits[b, t, :] → scores for all V possible next tokens
```

A higher logit means the model assigns a higher preference to that token before normalization.

Example:

```text
"the"     →  4.1
"market"  →  3.7
"banana"  → -1.2
"."       →  2.8
```

---

## 26. Training Objective

During training, the model predicts the next token at every position.

Input:

```text
x = [t0, t1, t2, ..., t_{T-1}]
```

Target:

```text
y = [t1, t2, t3, ..., t_T]
```

The logits have shape:

```text
[B, T, V]
```

The targets have shape:

```text
[B, T]
```

The loss is usually cross-entropy between predicted logits and the true next token.

### Role

Training adjusts the model weights so that the correct next token receives a higher probability.

---

## 27. From Logits to Probabilities

For generation, we usually use only the final position.

Given a current context:

```text
[t0, t1, ..., t_{T-1}]
```

the model produces logits:

```text
logits [B, T, V]
```

For next-token sampling, use:

```text
logits[:, -1, :]
```

This gives:

```text
[B, V]
```

Then apply softmax:

```text
probabilities = softmax(logits)
```

The result is a probability distribution over the vocabulary.

---

## 28. Temperature

Temperature controls how sharp or random the distribution is.

```text
scaled_logits = logits / temperature
```

### Low Temperature

A low temperature makes the distribution sharper. The model becomes more deterministic and conservative.

```text
temperature < 1
```

### High Temperature

A high temperature makes the distribution flatter. The model becomes more random and creative.

```text
temperature > 1
```

### Temperature 1

Temperature 1 leaves logits unchanged.

```text
temperature = 1
```

---

## 29. Top-k Sampling

Top-k sampling keeps only the `k` most likely tokens and removes the rest.

Example:

```text
top_k = 50
```

Only the 50 highest-probability tokens remain eligible.

### Role

Top-k prevents very unlikely tokens from being sampled while preserving controlled randomness. This often produces better text than unrestricted sampling.

---

## 30. Sampling the Next Token

After optional temperature scaling and top-k filtering:

```text
p = softmax(filtered_logits)
next_token = sample(p)
```

The sampled token is appended to the context:

```text
context = context + [next_token]
```

Then the process repeats.

---

## 31. Autoregressive Generation Loop

Text generation is autoregressive.

```text
for step in range(max_new_tokens):
    logits = model(context)
    next_logits = logits[:, -1, :]
    next_token = sample(next_logits)
    context = append(context, next_token)
```

At every step:

1. the model sees the current context,
2. predicts a distribution for the next token,
3. samples one token,
4. appends it,
5. repeats.

This is how generated text is produced.

---

## 32. Context Cropping During Sampling

If the generated context becomes longer than `block_size`, only the most recent `T` tokens are passed to the model.

```text
context_cond = context[:, -block_size:]
```

### Role

The model was trained with a maximum context length `T`. During generation, it cannot directly attend to more than `T` tokens at once. Older tokens fall out of the active context window.

---

## 33. Decoding Tokens Back to Text

After sampling token IDs, the tokenizer decodes them back into text.

For character-level models:

```text
[12, 4, 23] → "cat"
```

For BPE models:

```text
[token_id_1, token_id_2, ...] → generated text
```

The final output is ordinary text.

---

## 34. Model Variants

The visualization shows three model variants.

### 34.1 `gpt`

The baseline model.

Typical properties:

- minimal GPT implementation,
- learned position embeddings,
- standard normalization,
- standard MLP,
- optimized for readability and teaching.

Use this when the goal is to understand the core GPT architecture.

### 34.2 `gpt_plus`

The accuracy-focused variant.

It may include:

- RoPE,
- RMSNorm,
- biasless linear layers,
- SwiGLU,
- improved initialization or residual scaling.

**Role.** `gpt_plus` keeps the architecture compact but introduces modern Transformer improvements. It is useful for showing how a minimal GPT can be upgraded while staying readable.

### 34.3 `gpt_fast`

The speed-focused variant.

It uses optimized attention paths such as SDPA or FlashAttention where available.

**Role.** `gpt_fast` preserves the GPT idea but improves runtime performance. It is useful for scaling experiments and demonstrating implementation-level acceleration.

---

## 35. How the Main Hyperparameters Interact

### `T`: Context Length

Controls how far back the model can look.

Larger `T` improves long-context capability but increases attention cost roughly as `T²`.

### `C`: Embedding Width

Controls representation size per token.

Larger `C` increases model capacity, parameter count, and compute.

### `H`: Number of Heads

Controls how many parallel attention patterns are learned per block.

Larger `H` with fixed `C` increases the number of attention subspaces but decreases the per-head dimension `d_k = C / H`.

### `L`: Number of Blocks

Controls depth.

Larger `L` allows more refinement steps and increases capacity, but also increases compute and memory.

### `V`: Vocabulary Size

Controls how many possible tokens the model can predict.

Larger `V` supports richer tokenization but increases embedding and output projection sizes.

---

## 36. Minimal Mathematical Summary

Given token IDs:

```text
idx [B, T]
```

Embedding:

```text
x = token_embedding(idx) + position_embedding(pos)
x [B, T, C]
```

For each Transformer block:

```text
x = x + causal_multi_head_attention(norm(x))
x = x + mlp(norm(x))
```

After `L` blocks:

```text
x = final_norm(x)
```

Language model head:

```text
logits = lm_head(x)
logits [B, T, V]
```

Training loss:

```text
loss = cross_entropy(logits.view(B*T, V), targets.view(B*T))
```

Sampling:

```text
next_token ~ softmax(logits[:, -1, :] / temperature)
```

---

## 37. Conceptual Summary

yoctoGPT implements the essential GPT idea:

1. convert text into token IDs,
2. embed tokens and positions into vectors,
3. repeatedly refine those vectors through causal self-attention and MLP blocks,
4. map final vectors to vocabulary logits,
5. train by next-token prediction,
6. generate by sampling one token at a time.

The key architectural idea is repeated contextual refinement:

```text
tokens → embeddings → attention + MLP → deeper attention + MLP → logits → sampled text
```

The number of heads `H` controls how many parallel attention views each block has.

The number of blocks `L` controls how many times the representation is refined.

Together with context length `T` and embedding width `C`, these parameters define the model's capacity, computational cost, and learning behavior.

---

## 38. One-Sentence Interpretation

A yoctoGPT model is a compact decoder-only Transformer that learns to predict the next token by repeatedly letting every token look backward through causal multi-head attention, process the gathered information through an MLP, and finally score all possible next tokens through the language modeling head.
