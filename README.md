# Poetry Generator

This repository contains the final project for the course Neural Networks for Data Science at Sapienza. In the notebook, I implement a small language model and train it on Chinese Tang poetry, manually implementing the KV cache mechanism.

### Dataset
The dataset I chose for the project is a textual dataset composed of 57607 Chinese poems from the Tang Dynasty and is available at the following GitHub repository: https://github.com/chinese-poetry/chinese-poetry. My goal is to implement a small language model and to train it on Chinese poetry. For this, I will use a decoder-only transformer architecture, with the KV-cache mechanism built into the MHA layers.

### Pre-processing
Each poem is provided as a list of strings, which were concatenated into a single string during preprocessing. Since the plan was to tokenise at (Chinese) character-level, I looked at the distribution of poem lengths to see how long the maximum sequence length in the transformer would be. Most poems are short (about 64 characters on average), but due to the presence of some outliers with sequence length above 1000, I decided to keep only the poems with length under the 95th percentile of poem lengths. This choice was made to avoid overly long instances to require the `max_seq_len` parameter to be larger than necessary for a vast a majority of poems. After this step, I counted the occurences of each character in the remaining dataset. Since the embedding matrices depend on the vocabulary size, and dropping rare characters with less than 3 occurrences keeps 99.9046% of all character occurrences, I decided to replace these characters with the "unknown token" "`<UNK>`". After this, I encoded the poems, using a "`<BOS>`" (beginning of sequence) token at the start, "`<EOS>`" (end of sequence token) at the end of each poem, filling rare characters in with the "`<UNK>`" token, and using the "`<PAD>`" token to pad the remaining poem's encoding until the maximum sequence length is reached. Finally, the poems were sliced and divided into inputs and targets. The dataset was shuffled randomly and split into a training (90%) and validation (10%) set.

### **The transformer model**

For this project, I built transformer model named `TangPoetryTransformer` and trained it on the dataset of Chinese poetry. It relies on four classes:
- `TokenAndPositionEmbedding`: creates the token and position embeddings
- `MultiHeadAttentionWithCache`: implements the multi-head attention mechanism with a `use_cache` flag that, if set to true, enables the KV-cache
- `TransformerBlock`: a single transformer block, structured in this way:
  - Layer normalization 1
  - Multi-head attention (+ residual)
  - Linear layer 1
  - ReLU
  - Linear layer 2
  - Layer normalization 2 (+ residual)
- `TangPoetryTransformer`: uses the above classes and combines them into a unified transformer architecture

#### **Token and Positional Embeddings**

The token and positional embeddings were initialized with the `flax.nnx.Embed` method, using as embedding dimension the hyperparameter $e=$`dim_embed`. The total number of parameters to be learned in this layer are given by the total number of weights in the token lool-up table and the number of weights in the learned positional embedding matrix:

$$
N \cdot e + M \cdot e = e(N+M)
$$

with:
- $N=$ `vocab_size`, number of tokens in the vocabulary
- $M=$ `max_seq_len`, maximum sequence length

#### **Multi-head Attention**

A core component of transformers is the self-attention layer, which is able to capture long-range dependencies and compute relationships between all tokens in an input sequence simultaneously, by using the key-query-value mechanism. In Multi-head Attention (MHA), multiple attention operations run in parallel to model different types of dependencies.

For each attention head $\ell=1,\cdots,h$ of a layer (with $h=$ `num_heads` in the code), the following matrices are defined:

- $W_{k,\ell} \sim (e,k)$
- $W_{v,\ell} \sim (e,\nu)$
- $W_{q,\ell} \sim (e,k)$

Where $k=$ `dim_out_qk` and $\nu=$ `dim_out_v` are hyperparameters.

The embedded tokens are projected using these matrices, giving us the key, value and query tokens. If the input sequence has length $n=$ `seq_length`, the matrix of embedded input tokens $X$ will have shape $(n,e)$. We can write the projections as follows:

- $K_\ell = XW_{k,\ell} \sim (n,k)$
- $V_\ell = XW_{v,\ell} \sim (n,\nu)$
- $Q_\ell = XW_{q,\ell} \sim (n,k)$

The $\ell$-th self-attention layer is defined as

$$
\text{SA}_{\ell}(X) = \text{softmax}\left(\frac{Q_{\ell} K_{\ell}^T}{\sqrt{k}}\right) V_{\ell}
$$

The output of this layer has shape $(n,\nu)$. In the MHA step, the $\text{SA}$ outputs are concatenated and projected using a matrix $W_o \sim (h \nu, o)$, where $o$ is a hyperparameter called `dim_out_o` in the code.

$$
\text{MHA}(X) = [\text{SA}_1(X) \| \ldots \| \text{SA}_h(X)] W_o
$$

**The KV Cache**

The KV Cache is a mechanism used to avoid redundant re-computation of Key and Value matrices for tokens that were already processed. In particular, this is useful in step-by-step computation, where one output token is produced at a time.

In the code, a boolean `use_cache` can trigger the kv-cache mechanism. The buffers `k_cache` and `v_cache` were defined, with shape `(batch_size, num_heads, max_seq_len, head_dim)` where `head_dim` is `dim_out_qk` for the `k_cache` and `dim_out_v` for the `v_cache`. The caches are initialized as `nnx.Cache` objects and wrapped in `nnx.data()` because jax arrays are immutable, but in this way they can be stored as mutable attributes of the class. A `cache_index` is declared to track how many positions have been stored so far. During autoregressive generation, each call to the attention layer computes new queries, keys and values for the newest token(s). If `use_cache` is true, the new keys and values are written into the caches at the position given by the `cache_index` using `jax.lax.dynamic_update_slice`, before the index is incremented by the token sequence length. Then, the layer the entire cache until the current index, including the newly computed keys and values. The causal mask is not needed here because the cache doesn't contain information from future tokens by construction. Meanwhile in the training mode, a lower-triangular mask is needed because all positions are processed simultaneously. This means that each token's key and value are computed only once, the first time it is generated, and reused in every following step for the rest of the sequence. This turns a redundant $O(n^2)$ recomputation across $n$ sequential generation calls into $O(n)$.

A correctness test was executed to verify whether the logits generated with and without cache were close enough, resulting in a maximum absolute distance between logits of 1.07e-06. This confirms that the cache preserves the model's exact computation.

**References**

Below are some resources I consulted when working on the model:

- https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html
- https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html
- https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html

### Results

After training the model for 50 epochs, the following results were obtained:
- Best validation loss: 4.3501 at epoch 45
- Final training loss: 3.9134, final validation loss: 4.3646
- Best validation perplexity: 77.4865 at epoch 45
- Final training perplexity: 50.0671, final validation perplexity: 78.6175

<img width="1289" height="490" alt="image" src="https://github.com/user-attachments/assets/656b48c5-fcfc-49fa-a5ed-b1b984c423c0" />

Additionally, I let the model generate a few poems, from which we can see that the model learned the appropriate structure and vocabulary of the Tang Poetry, but occasionally struggles with consistency across lines.

Here are a few examples, with English translations:

**Example Poem 1**:

林棲不可望，夏漏忽相悲。羣閣多逸翮，儂家在重脂。碧雲生綠水，紅日落紅肌。唯有能歸去，悠悠方外期。

**Translation**:
```
The forest dwelling cannot be gazed upon;
summer's water-clock suddenly brings mutual sorrow.
The clustered pavilions hold such carefree wings;
my home lies amid layered rouge.
Jade-green clouds rise from the green water;
the red sun sets upon reddened flesh.
Only the ability to return remains —
vast and boundless, a meeting beyond this world.
```

**Example Poem 2**:
天開方朔朔，南陌重秋風。去日朝猶在，無人夜更通。春風分塞上，秋草帶河東。不得還鄉夢，無因免一功。

**Translation**:
```
Heaven opens, and Dongfang Shuo appears once more;
the southern lane meets autumn wind again.
The departing sun's morning light still lingers;
at night, no one passes through.
Spring wind divides along the frontier;
autumn grass lines the eastern riverbank.
I cannot even return home in dream —
there is no cause to be spared this one task.
```

**Example Poem 3**:
太平南風慘，難與長安客。旦夕降那歸，聞君成楚矣。乘流竟何處，志氣方晦跡。白雲抱幽慮，寂寂苔漫積。開理俱冥冥，氛氳換衣服。幽人感神和，萬象無窮極。又疑白日老，更憶青山曲。西閣鬱嵯峨，寥落空林雪。

**Translation**:
```
In great peace, the south wind still turns bleak;
hard to remain a guest in Chang'an.
Morning and evening, descending — where shall I return?
I hear that you have become a man of Chu.
Riding the current, where shall I end up?
My resolve now hides its own trace.
White clouds embrace hidden worries;
silent, moss piles ever deeper.
Understanding unfolds in obscurity;
mist and vapor exchange my very clothes.
The recluse feels harmony with the spirit;
the ten thousand forms are without end.
Again I suspect the bright sun itself grows old;
further still I recall the winding green mountains.
The western tower rises lofty and rugged;
desolate, snow fills the empty forest.
```



