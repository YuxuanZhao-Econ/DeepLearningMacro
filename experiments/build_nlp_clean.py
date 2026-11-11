"""Build the concise, self-contained Julia NLP notebook.

Runtime sections are embedded as executable notebook cells, never included from a
project source module at runtime. External files are data and parameter arrays only.
"""
from pathlib import Path
import re
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/"experiments/templates/nlp_clean_runtime.jl"
OUT=ROOT/"notebooks/Intro_English_GPT.ipynb"

def main():
    parts=re.split(r"^# %% (\w+)\s*$",RUNTIME.read_text(encoding="utf-8"),flags=re.M)
    blocks={parts[i]:parts[i+1].strip() for i in range(1,len(parts),2)}
    cells=[]
    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s.strip().replace("§",chr(96))))
    def code(key):
        c=nbf.v4.new_code_cell(blocks[key])
        c.metadata["course_section"]=key
        cells.append(c)
    def caption(n,s):
        md(f"*Figure {n}. {s}*")


    md(r"""
# Intro to NLP from Scratch in Julia

We will learn NLP through two complete projects: **classify IMDb movie reviews**, then
**predict and generate English text with a GPT-style Transformer**.
For each project, define the data and parameters, calculate a prediction, define the loss,
follow its derivatives, and train on the full prepared dataset.

The review classifier uses a 20,000-entry vocabulary, 64-dimensional embeddings, and
64 hidden units: 1,284,225 trainable parameters. The English GPT has 5,820,416 parameters.
We inspect real observations and slices of these models; we do not switch to a separate toy model.
All tokenizer, model, training, and plotting functions are defined in this notebook.

1. IMDb text → tokens → trainable word vectors.
2. Mean pooling → neural prediction → loss → parameter updates.
3. Fit the classifier and evaluate unseen reviews.
4. Next-token prediction → Q/K/V attention → a residual Transformer.
5. Train the English model, evaluate it, and generate text.
""")
    md(r"""
## Julia setup

Select the project's **Julia 1.12** kernel and run cells in order.
The notebook reads the supplied public data from §reference/datasets/§ and recorded
parameter arrays from §output/english_basics/§. It imports no project source module.

The default path trains the IMDb model on CPU and inspects the recorded English GPT.
Later, §TRAIN_LARGE_GPT=true§ starts a new GPT fit from random initialization on the GPU.
Packages, corpus files, BPE rules, and recorded weights remain external project assets.
""")
    code("setup")
    md(r"""
## 1. Start with the real task and data

### 1.1 What is observed, and what must be predicted?

**Natural Language Processing (NLP)** applies computational models to human language.
Here an observation is a complete movie review $s_n$ and its supplied label $y_n$:

$$y_n=\begin{cases}1&\text{positive review},\\0&\text{negative review}.\end{cases}$$

The index $n$ identifies a review. Our network has trainable parameters $\theta$,
which we will list explicitly below. Its output is one number:

$$\hat p_n=f_\theta(s_n)\in(0,1).$$

We interpret $\hat p_n$ as the model's estimate of $P(y_n=1\mid s_n)$.
The label belongs to the **whole review**. No one supplies a positive/negative label
for every word or a target vector for every word.

### 1.2 Tokenize the string and assign vocabulary IDs

A **token** is one unit produced by a tokenizer. Our first tokenizer removes HTML,
lowercases English text, and extracts words while preserving apostrophe contractions.
It retains words such as §not§. Punctuation is discarded in this classifier.
This simple ASCII rule can split accented spellings into fragments; it is a stated
limitation of our tokenizer, visible in the real review below.
""")
    code("tokenizer")
    md(r"""
Build a vocabulary $\mathcal V$ using **training text only**. Its size is
$V=|\mathcal V|=20{,}000$, including the special entry §<unk>§.
The vocabulary map $\iota:\mathcal V\to\{1,\ldots,V\}$ assigns each entry an integer ID.
Words outside the vocabulary map to the §<unk>§ ID.

For one review, the transformations are

$$s\ \xrightarrow{\text{tokenizer}}\ (w_1,\ldots,w_T)
  \ \xrightarrow{\iota}\ (x_1,\ldots,x_T),\qquad x_t=\iota(w_t).$$

$T$ is this review's number of tokens, $t$ is a token position,
$w_t$ is the token string, and $x_t$ is its integer ID.
**An ID is a table address**; adding two IDs or comparing their numerical distance
does not measure the words' meanings.

### 1.3 Prepare the IMDb splits

[IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) supplies 25,000 labeled training
reviews and 25,000 labeled test reviews. Our local preparation groups identical normalized
texts, retains one same-label copy, and discards conflicting-label groups.
It reserves a stratified 10% of the remaining training pool for validation.
It also removes test duplicates and exact overlaps with the original training pool.

| Prepared split | Reviews | Purpose |
|---|---:|---|
| Training | 22,411 | Build the vocabulary and update parameters |
| Validation | 2,490 | Choose the stopping checkpoint |
| Test | 24,675 | Evaluate the selected model |

The supplied TSVs contain source row IDs, labels, and hex-encoded UTF-8 text.
Source cards and hashes accompany the data; the dataset card lists the license as §other§.
This documented filtered split is not an unmodified leaderboard benchmark.

The code chooses the **first training review in source order with 80–160 tokens** for the
calculation walkthrough. This is one observation from the full project, chosen by length,
not by whether the model predicts it correctly. Every token in that review is used.
""")
    code("read_imdb")

    md(r"""
## 2. Define the network, one operation at a time

### 2.1 The embedding table is a parameter matrix

Choose an embedding dimension $d=64$. Each vocabulary entry has one trainable vector
$e_j\in\mathbb R^{64}$, where $j\in\{1,\ldots,20{,}000\}$ is a vocabulary ID.
Stack these vectors as **columns**:

$$E=\begin{bmatrix}e_1&e_2&\cdots&e_V\end{bmatrix}
  \in\mathbb R^{d\times V}=\mathbb R^{64\times20{,}000}.$$

The scalar $E_{rj}$ is coordinate $r$ of vocabulary entry $j$.
Thus $E[:,j]=e_j$, where the colon means “all coordinates.”
The table contains $64\times20{,}000=1{,}280{,}000$ trainable numbers.

We initialize those numbers randomly. Training changes their values; it does not change
the chosen dimension $d$. A coordinate is not assigned a meaning such as “emotion” or “noun.”
Predictive information can be spread across many coordinates.
The remaining parameter names are the hidden-layer weights $W\in\mathbb R^{64\times64}$,
hidden bias $b\in\mathbb R^{64}$, output weights $v\in\mathbb R^{64}$, and output bias
$c\in\mathbb R$. The following cell allocates all of them once, alongside $E$.
Sections 2.4–2.5 derive their operations separately.
""")
    code("classifier_parameters")
    code("embedding_figure")
    caption(1,"Selected entries from the actual 64 × 20,000 initial table. All 64 coordinates are shown; the displayed words are only a view into the full vocabulary.")
    md(r"""
### 2.2 Look up one vector for every token position

For the selected review, token position $t$ contains vocabulary ID $x_t$.
Define its looked-up vector as $\tilde e_t=e_{x_t}=E[:,x_t]$.
Collect the vectors in a new matrix:

$$H=\begin{bmatrix}\tilde e_1&\cdots&\tilde e_T\end{bmatrix}
  \in\mathbb R^{64\times T},\qquad H_{rt}=E_{r,x_t}.$$

$E$ stores a vector for **every vocabulary entry**.
$H$ contains the vectors selected by **this review**, in token order.
If the same token ID appears twice, two columns of $H$ refer to the same column of $E$.
The lookup is not a separate learned transformation: the entries of $E$ are the parameters.
""")
    code("embedding_lookup")
    md(r"""
### 2.3 Mean pooling: average across tokens, coordinate by coordinate

The neural classifier needs one fixed-length vector, although reviews have different $T$.
We choose **mean pooling**. Define the review vector $a\in\mathbb R^{64}$ by

$$a=\frac{1}{T}\sum_{t=1}^T\tilde e_t
    =\frac{1}{T}\sum_{t=1}^T E[:,x_t]
    =\frac{1}{T}H\mathbf1_T.$$

Here $\mathbf1_T$ is a $T$-vector of ones. Written one coordinate at a time,

$$a_r=\frac{E_{r,x_1}+E_{r,x_2}+\cdots+E_{r,x_T}}{T},
  \qquad r=1,\ldots,64.$$

So $a_1$ averages coordinate 1 over all tokens, $a_2$ averages coordinate 2, and so on.
We average **along the token-position axis**, keeping 64 separate coordinates.
Mean pooling is a modeling choice; it is not part of the definition of an embedding.

Repeated words can be counted instead of looked up repeatedly. For vocabulary ID $j$, define

$$n_j=\sum_{t=1}^T\mathbf1\{x_t=j\},\qquad
  f_j=\frac{n_j}{T},\qquad
  f=(f_1,\ldots,f_V)^\top\in\mathbb R^V.$$

$n_j$ is a count and $f_j$ is the fraction of review tokens with ID $j$.
Regrouping the same sum gives

$$a=\sum_{j=1}^V f_j e_j=Ef,\qquad \sum_{j=1}^V f_j=1.$$

For a batch of $B$ reviews, place their frequency vectors in
$F=[f^{(1)},\ldots,f^{(B)}]\in\mathbb R^{V\times B}$.
Then $EF\in\mathbb R^{64\times B}$ contains all their means.
$F$ is observed input data, not a trainable matrix. Most entries are zero, so sparse storage
makes this exact calculation efficient without discarding any words.
""")
    code("mean_pooling")
    code("pooling_figure")
    caption(2,"Left: all token vectors of the selected real review. Averaging across its columns produces the 64 values on the right. The numerical check confirms that lookup-and-average and E times the frequency vector are the same operation.")
    md(r"""
### 2.4 Transform the review vector with a hidden layer

Choose $m=64$ hidden units. Define a trainable weight matrix $W\in\mathbb R^{m\times d}$
and bias vector $b\in\mathbb R^m$. First compute the pre-activation vector $u$:

$$u=Wa+b\in\mathbb R^{64},\qquad
  u_k=\sum_{r=1}^{64}W_{kr}a_r+b_k,\quad k=1,\ldots,64.$$

Each hidden unit has its own weighted combination of all input coordinates.
Next apply the rectified linear unit (ReLU) to each coordinate:

$$h=\operatorname{ReLU}(u),\qquad h_k=\max(0,u_k).$$

$h\in\mathbb R^{64}$ is the hidden activation vector.
$d$ and $m$ are independent design choices; both happen to equal 64 here.
The nonlinearity lets the model learn a more flexible function than a single linear score.

### 2.5 Convert hidden activations into one probability

Define a trainable output vector $v\in\mathbb R^{64}$ and scalar bias $c\in\mathbb R$.
The output **logit** $z$ is

$$z=v^\top h+c=\sum_{k=1}^{64}v_kh_k+c\in\mathbb R.$$

It can be positive or negative and is not yet a probability. Apply the sigmoid function:

$$\hat p=\sigma(z)=\frac{1}{1+\exp(-z)}\in(0,1).$$

We predict “positive” when $\hat p\geq0.5$. The probability remains available even when
we do not convert it to a binary decision.
The code stores §v§ as a $1\times64$ row matrix, so §v*h§ implements $v^\top h$.
Each code column represents one review in a batch.
""")
    code("classifier_forward")
    code("forward_figure")
    caption(3,"The actual classifier has 64 mean-embedding coordinates and 64 hidden units. Eight representative nodes per layer make the connections readable; all 64 coordinates participate in the computation. The mean, hidden-layer, and output formulas appear directly in the figure.")
    md(r"""
### 2.6 Keep parameters and activations separate

| Object | Shape for one review | Role |
|---|---|---|
| $x_1,\ldots,x_T$, $y$ | $T$ IDs and one label | Observed data |
| $E$ | $64\times20{,}000$ | Trainable vocabulary embedding table |
| $W,b$ | $64\times64$, $64$ | Trainable hidden-layer weights and biases |
| $v,c$ | $64$, scalar | Trainable output weights and bias |
| $H,a,u,h,z,\hat p$ | As defined above | Activations recomputed for this input |

The complete parameter collection is $\theta=(E,W,b,v,c)$, with

$$\underbrace{64(20{,}000)}_{E}
+\underbrace{64(64)}_W+\underbrace{64}_b+\underbrace{64}_v+\underbrace{1}_c
=1{,}284{,}225\text{ parameters}.$$

The forward pass reads $\theta$ and produces activations. It does not change the parameters.
""")

    md(r"""
## 3. From prediction error to parameter updates

### 3.1 Define a loss for one observed review

For label $y\in\{0,1\}$ and predicted probability $\hat p$, define binary cross-entropy:

$$L(\theta;s,y)=-y\log\hat p-(1-y)\log(1-\hat p)
=\begin{cases}-\log\hat p&y=1,\\-\log(1-\hat p)&y=0.\end{cases}$$

The loss is small when the model assigns high probability to the observed label.
It becomes large when the prediction is confidently wrong.
The code uses the equivalent, numerically stable loss computed directly from logits.
""")
    code("loss_value")
    code("loss_figure")
    caption(4,"Left: how the probability affects cross-entropy. Right: the derivative with respect to the logit, derived next. The horizontal axis in both panels is the predicted probability.")
    md(r"""
### 3.2 Differentiate the output, without skipping the sigmoid

For an intermediate vector $q$, write $\delta_q=\partial L/\partial q$.
Begin with the two scalar derivatives:

$$\frac{\partial L}{\partial\hat p}
=-\frac{y}{\hat p}+\frac{1-y}{1-\hat p},
\qquad
\frac{\partial\hat p}{\partial z}=\hat p(1-\hat p).$$

The chain rule gives

$$\delta_z=\frac{\partial L}{\partial z}
=\left(-\frac{y}{\hat p}+\frac{1-y}{1-\hat p}\right)\hat p(1-\hat p)
=-y(1-\hat p)+(1-y)\hat p
=\hat p-y.$$

For a positive review, $\delta_z<0$, so a small descent step on $z$ would raise it.
For a negative review, $\delta_z>0$, so descent would lower it.
The optimizer actually changes parameters; the derivatives below tell it how each parameter affects $z$.

### 3.3 Continue backward through the output and hidden layer

Since $z=\sum_k v_kh_k+c$,

$$\frac{\partial L}{\partial v_k}=\delta_z h_k,\qquad
  \frac{\partial L}{\partial c}=\delta_z,\qquad
  (\delta_h)_k=\delta_z v_k.$$

For ReLU, a positive pre-activation has derivative 1 and a negative one has derivative 0.
We use derivative 0 at its kink, $u_k=0$. Therefore

$$\delta_u=\delta_h\odot\mathbf1_{\{u>0\}},$$

where $\odot$ means coordinate-by-coordinate multiplication.
Using $u_k=\sum_r W_{kr}a_r+b_k$, each derivative is

$$\frac{\partial L}{\partial W_{kr}}=(\delta_u)_k a_r,\qquad
  \frac{\partial L}{\partial b_k}=(\delta_u)_k,\qquad
  (\delta_a)_r=\sum_{k=1}^{64}W_{kr}(\delta_u)_k.$$

In matrix form:

$$\nabla_W L=\delta_u a^\top,\qquad
  \nabla_b L=\delta_u,\qquad
  \delta_a=W^\top\delta_u.$$

### 3.4 Continue through the average to the embedding table

Recall the exact scalar mean $a_r=\sum_j f_jE_{rj}$.
The frequencies $f_j$ are fixed for this review, so

$$\frac{\partial a_r}{\partial E_{rj}}=f_j=\frac{n_j}{T},\qquad
  \frac{\partial L}{\partial E_{rj}}=(\delta_a)_r\frac{n_j}{T}.$$

Across all coordinates and vocabulary entries,

$$\nabla_E L=\delta_a f^\top\in\mathbb R^{64\times20{,}000},
  \qquad \nabla_{E[:,j]}L=\frac{n_j}{T}\delta_a.$$

An unused word has $n_j=0$, hence zero gradient for this review.
A repeated word contributes through every occurrence.
Even a neutral word can move after one example; learning across many reviews separates
predictive patterns from incidental co-occurrence.
**There is no target embedding to copy. The prediction loss supplies its training signal.**

### 3.5 Apply one simultaneous parameter update

Let $\eta>0$ be the learning rate. Plain gradient descent applies

$$\theta_{\mathrm{new}}=\theta_{\mathrm{old}}
  -\eta\nabla_\theta L(\theta_{\mathrm{old}}).$$

For one embedding entry, this is explicitly

$$E_{rj}^{\mathrm{new}}
=E_{rj}^{\mathrm{old}}
-\eta\left[\frac{n_j}{T}(\delta_a)_r\right]_{\theta=\theta_{\mathrm{old}}}.$$

All derivatives are computed using the **old** parameters before any are updated.
Backpropagation computes gradients; the optimizer uses them to update parameters.
Activations such as $a$ and $h$ are recomputed on the next forward pass, not stored as learned parameters.

The cell below calculates every derivative by hand, checks it against Flux automatic
differentiation on the full model, and prints the actual numbers for one embedding update.
It leaves the original initialization intact for the training experiment.
""")
    code("manual_gradient")
    caption(5,"A real review produces gradients in the full 64 × 20,000 table. The displayed rows are its 16 most frequent vocabulary IDs; parentheses give occurrence counts. Repeated words receive proportionally larger contributions. The code also verifies zero gradients for every unused entry.")
    md(r"""
## 4. Train and evaluate the IMDb network

### 4.1 Average gradients across a batch

For a batch $\mathcal B$ of $B$ reviews, the objective and its derivative are

$$L_{\mathcal B}(\theta)=\frac1B\sum_{n\in\mathcal B}L(\theta;s_n,y_n),
\qquad
\nabla_\theta L_{\mathcal B}=\frac1B\sum_{n\in\mathcal B}\nabla_\theta L_n.$$

The training loop shuffles the training reviews, takes batches of 128, computes this mean loss,
backpropagates, and updates all parameters. One **epoch** visits every training review once.
The last batch can be smaller than 128.

We use Adam instead of plain gradient descent for the full fit. For update index $q$ and
batch gradient $g_q$, Adam keeps two running arrays, initially $m_0=v_0=0$:

$$m_q=\beta_1m_{q-1}+(1-\beta_1)g_q,\qquad
  v_q=\beta_2v_{q-1}+(1-\beta_2)(g_q\odot g_q).$$

With bias correction $\hat m_q=m_q/(1-\beta_1^q)$ and
$\hat v_q=v_q/(1-\beta_2^q)$, its update is

$$\theta_q=\theta_{q-1}
-\eta\frac{\hat m_q}{\sqrt{\hat v_q}+\epsilon}.$$

All operations in this update are coordinatewise; $\beta_1,\beta_2\in[0,1)$ control smoothing
and $\epsilon>0$ stabilizes the denominator. These optimizer arrays are distinct from the
classifier's output vector $v$. Adam changes the update rule, not the forward equations or chain rule.

### 4.2 Fit the same 1.28-million-parameter model

The fit starts from the exact initialization used above.
It uses eight epochs, learning rate 0.003, and all tokenized words in every review.
After each epoch, validation cross-entropy determines whether to keep that checkpoint.
Only after selection is the test set evaluated.

§TRAIN_SENTIMENT=true§ retrains the model; false loads its saved result.
The ten-minute fit budget is checked between updates; setup and final evaluation add time.
""")
    code("review_training")
    code("review_curves")
    caption(6,"Training and validation use different reviews. Falling training loss can coexist with rising validation loss; the dotted line identifies the selected checkpoint.")
    md(r"""
### 4.3 Measure performance and inspect mistakes

Let $\hat y_n=\mathbf1\{\hat p_n\geq0.5\}$ and let $N_{\rm test}$ be the number of test reviews.

$$\mathrm{Accuracy}=\frac1{N_{\rm test}}\sum_n\mathbf1\{\hat y_n=y_n\}.$$

A **true positive** (TP) is a positive review predicted positive.
A **false positive** (FP) is a negative review predicted positive.
A **false negative** (FN) is a positive review predicted negative.
Positive-class F1 combines these counts:

$$F_1=\frac{2TP}{2TP+FP+FN}.$$

The confusion matrix uses actual labels in rows and predicted labels in columns.
We also compare accuracy with the majority-label rule selected from training labels.
""")
    code("review_test")
    caption(7,"The matrix reports counts. Each probability histogram is normalized within its actual class, making confident errors and ambiguous predictions visible.")
    code("learned_embeddings")
    caption(8,"The same entries in the same full embedding table before and after fitting, with common color limits. All 64 coordinates are displayed.")
    code("own_reviews")
    md(r"""
### 4.4 What averaging cannot represent

The two probe sentences §The movie is good, not bad§ and §The movie is bad, not good§
have the same word counts. Therefore they have the same $f$, the same $a=Ef$,
and the same prediction, despite their different word order.
The code verifies this equality.

This is a limitation of this architecture, not a shortage of training epochs.
To model order and context, we need a representation that keeps a separate state at each position.
Short probes also differ from the full reviews used in training; their probabilities are not calibrated guarantees.
""")

    md(r"""
## 5. Change the target: predict the next token

### 5.1 The English corpus supplies its own labels

We now begin the second full project: a GPT-style language model on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
a synthetic simple-English corpus.
The prepared training subset contains 81,694 stories and about 17.66 million tokens.
Validation and test come from the source's held-out split.
Preparation uses bounded corpus prefixes, ASCII normalization, and exact deduplication.

This project has its **own** vocabulary and embedding table.
It uses 2,048 BPE tokens and model width 256; these are different choices from the
IMDb classifier's 20,000 words and width 64.
BPE (byte-pair encoding) starts with small symbols and repeatedly merges frequent adjacent pairs.
Its merge rules were fitted on this project's training text before neural training.
The supplied tokenizer is ASCII-only.

The complete encoder and decoder below convert between strings and IDs.
§<bos>§ and §<eos>§ mark beginning and end; a middle dot in a display makes a space visible.
Tokenizer decoding is a text/ID conversion, not a neural-network decoder.
""")
    code("bpe")
    md(r"""
Write a sequence of token IDs as $(i_1,\ldots,i_{T+1})$.
At position $t$, the input prefix is $(i_1,\ldots,i_t)$ and the observed target is
$y_t=i_{t+1}$. In a training window,

$$\underbrace{(i_1,\ldots,i_T)}_{\text{input IDs}}
\quad\longrightarrow\quad
\underbrace{(i_2,\ldots,i_{T+1})}_{\text{next-token targets}}.$$

No person needs to label every position: the text itself provides the targets.
This is **self-supervised learning**.
""")
    code("story_data")
    md(r"""
### 5.2 From a contextual state to a vocabulary distribution

For this project, let $V=2{,}048$ and $d=256$.
The network will produce a contextual state $h_t\in\mathbb R^d$ at each position.
Its trainable output matrix is $U\in\mathbb R^{V\times d}$.
Our implementation omits an output bias:

$$z_t=Uh_t\in\mathbb R^V,\qquad
  p_t[k]=\frac{\exp(z_t[k])}{\sum_{j=1}^V\exp(z_t[j])}.$$

$p_t[k]$ is the model's probability that vocabulary ID $k$ comes next.
For the observed target $y_t$, the position loss is

$$L_t=-\log p_t[y_t]
=-z_t[y_t]+\log\!\left(\sum_j\exp z_t[j]\right).$$

Differentiating this expression coordinate by coordinate gives

$$\frac{\partial L_t}{\partial z_t[k]}
=p_t[k]-\mathbf1\{k=y_t\}.$$

This is the multiclass counterpart of the classifier's $\hat p-y$ derivative.
For batch size $B$ and window length $T$, training minimizes

$$L_{\mathcal B}=\frac1{BT}\sum_{n=1}^B\sum_{t=1}^T L_{n,t}.$$

By the probability chain rule, the model defines
$P_\theta(i_1,\ldots,i_N)=\prod_{t=1}^N P_\theta(i_t\mid i_{<t})$,
with a beginning marker supplying the initial context.
Training uses observed prefixes (*teacher forcing*); generation will append its own chosen IDs.
""")

    md(r"""
## 6. Attention: define every matrix before multiplying

### 6.1 Token vectors, position vectors, and normalization

For the English model, define token parameters $E_{\rm LM}\in\mathbb R^{256\times2048}$
and position parameters $P\in\mathbb R^{256\times128}$.
For a window with $T\leq128$, its input vector and state matrix are

$$x_t=E_{\rm LM}[:,i_t]+P[:,t]\in\mathbb R^{256},\qquad
  X=[x_1,\ldots,x_T]\in\mathbb R^{256\times T}.$$

The two summands have the same dimension, so their sum is defined.
The token embedding is shared across occurrences; the position vector supplies order.

Our Transformer uses layer normalization before each sublayer.
For a single column $x\in\mathbb R^d$, define

$$\mu(x)=\frac1d\sum_r x_r,\qquad
  s^2(x)=\frac1d\sum_r(x_r-\mu(x))^2,$$

$$\operatorname{LN}(x)=
  \gamma\odot\frac{x-\mu(x)\mathbf1_d}{\sqrt{s^2(x)+\epsilon}}+\beta.$$

$\gamma,\beta\in\mathbb R^d$ are learned scale and shift; $\epsilon>0$ stabilizes division.
Each position is normalized over its own feature coordinates.
Write $\tilde X=\operatorname{LN}_1(X)$.

### 6.2 One head: query, key, and value projections

There are $H_{\rm heads}=8$ heads. Each has width $d_h=d/H_{\rm heads}=32$.
For one head, define six trainable objects:

$$W_Q,W_K,W_V\in\mathbb R^{32\times256},\qquad
  b_Q,b_K,b_V\in\mathbb R^{32}.$$

Their outputs are computed activations:

$$Q=W_Q\tilde X+b_Q\mathbf1_T^\top,\qquad
  K=W_K\tilde X+b_K\mathbf1_T^\top,\qquad
  \mathsf V=W_V\tilde X+b_V\mathbf1_T^\top.$$

All three outputs have shape $32\times T$.
Column $t$ contains $q_t$, $k_t$, or $v_t$.
A query supplies the matching request, a key supplies compatibility information,
and a value supplies the vector to be combined.
These descriptions interpret the operations; no correct query or attention map is supplied.
The matrix $\mathsf V$ denotes values, while scalar $V$ denotes vocabulary size.

### 6.3 Scores and the causal mask

For query position $i$ and key position $j$, calculate

$$S_{ji}=\frac{k_j^\top q_i}{\sqrt{32}}
=\frac1{\sqrt{32}}\sum_{r=1}^{32}K_{rj}Q_{ri}.$$

Therefore $S=K^\top Q/\sqrt{32}\in\mathbb R^{T\times T}$.
The scale factor controls dot-product size as head width grows.
To prevent a position from reading future tokens, define

$$M_{ji}=\begin{cases}0&j\leq i,\\-\infty&j>i.\end{cases}$$

Normalize the permitted scores for each query:

$$A_{ji}=\frac{\exp(S_{ji}+M_{ji})}
  {\sum_{\ell=1}^T\exp(S_{\ell i}+M_{\ell i})}.$$

Here **keys are rows, queries are columns, and each column sums to one**.
An excluded future key has weight exactly zero because $\exp(-\infty)=0$.
Books that store positions as rows use the transposed convention $QK^\top$.

### 6.4 Read a weighted combination of value vectors

For query $i$, calculate a new vector

$$o_i=\sum_{j=1}^T A_{ji}v_j\in\mathbb R^{32}.$$

Putting every query output into a matrix gives

$$O=\mathsf V A\in\mathbb R^{32\times T}.$$

Both the weights and the vectors being combined depend on the input.
This is different from the IMDb classifier's fixed equal weights $1/T$.
All positions can be calculated together during training, because the mask enforces the
prefix restriction. Generation still needs to select one new token before predicting the next.
""")

    md(r"""
## 7. Put attention inside the complete GPT network

### 7.1 Multiple heads, residual connections, and the MLP

Each of the eight heads has its own projections and produces
$O^{(h)}\in\mathbb R^{32\times T}$. Stack them along the feature axis:

$$C=\begin{bmatrix}O^{(1)}\\ \vdots\\ O^{(8)}\end{bmatrix}
  \in\mathbb R^{256\times T}.$$

With $W_O\in\mathbb R^{256\times256}$ and $b_O\in\mathbb R^{256}$, the multi-head output is

$$\operatorname{MHA}(\tilde X)=W_OC+b_O\mathbf1_T^\top
  \in\mathbb R^{256\times T}.$$

Now add the block input back to that output:

$$R=X+\operatorname{MHA}(\operatorname{LN}_1(X)).$$

This addition is a **residual connection**, also called a skip connection—the idea used
in ResNet. Both terms have shape $256\times T$, so addition is coordinate by coordinate.
Attention learns a change to the input state while the input also has a direct path forward.
The overall language model is a Transformer.

The block's ordinary neural-network sublayer uses
$W_1\in\mathbb R^{1024\times256}$, $b_1\in\mathbb R^{1024}$,
$W_2\in\mathbb R^{256\times1024}$, and $b_2\in\mathbb R^{256}$.
For a state matrix $Z\in\mathbb R^{256\times T}$, define

$$F(Z)=W_2\operatorname{GELU}(W_1Z+b_1\mathbf1_T^\top)+b_2\mathbf1_T^\top.$$

GELU is a smooth activation, $\operatorname{GELU}(u)=u\Phi(u)$, where $\Phi$ is the
standard normal cumulative distribution function (libraries may use a close approximation).
This MLP acts separately at every position: $256\to1024\to256$ coordinates.
Its second residual connection completes the block:

$$X_{\rm next}=R+F(\operatorname{LN}_2(R)).$$

Attention mixes positions; the MLP transforms coordinates within each position.
There are six such blocks, then a final layer normalization and the vocabulary output matrix $U$.

**Why does the skip path matter for gradients?** For a vector sublayer
$r=x+f(x)$, let $g=\partial L/\partial r$ and let $J_f$ be the Jacobian of $f$.
The chain rule gives

$$\frac{\partial L}{\partial x}
=(I+J_f)^\top g=g+J_f^\top g.$$

The first term carries a direct gradient path; the second passes through the learned transformation.
This can help optimization through many layers, though it does not guarantee gradients are always well behaved.
""")
    code("gpt_model")
    code("block_diagram")
    caption(9,"One full-width Transformer block. Orange paths implement the two additions in the formulas; they carry the original states around the learned sublayers. Every intermediate state at an addition has 256 coordinates per position.")
    md(r"""
### 7.2 Inspect one real prefix inside this full model

The next cells take the first 12 BPE input positions from the actual training corpus,
with the 13th token as the observed target. They inspect **head 1 of block 1** in the
complete 5.82-million-parameter model at random initialization.
The calculations use all 256 input coordinates and all 32 coordinates of that head.
No separate attention model is trained.
""")
    code("qkv")
    caption(10,"Actual arrays from the full model. The input panel displays coordinates 1–32 of 256 to keep its cells readable; all 256 participate in the multiplication. Each projected panel shows all 32 head coordinates. Rows in the display are token positions; Julia arrays store positions in columns.")
    code("attention")
    caption(11,"A is a key-row/query-column matrix, so the forbidden future-key region is below the diagonal. Query 9 reads positions 1–9 and assigns exactly zero weight to positions 10–12.")
    md(r"""
### 7.3 Backpropagate the full next-token loss through this head

The loss still uses all eight heads, all six blocks, and the vocabulary output layer.
Let $G_Z=\partial L/\partial Z$ for an intermediate matrix $Z$.
Automatic differentiation through the rest of the network supplies $G_O$ for this head.
We can then derive its local gradients explicitly.

From $O=\mathsf V A$,

$$G_{\mathsf V}=G_OA^\top,\qquad G_A=\mathsf V^\top G_O.$$

Within one query column $i$, the softmax Jacobian is
$\partial A_{ji}/\partial S_{\ell i}=A_{ji}(\mathbf1\{j=\ell\}-A_{\ell i})$.
Applying the chain rule and collecting terms gives

$$
(G_S)_{ji}
=\sum_\ell(G_A)_{\ell i}A_{\ell i}(\mathbf1\{\ell=j\}-A_{ji})
=A_{ji}\left[(G_A)_{ji}-\sum_\ell A_{\ell i}(G_A)_{\ell i}\right].
$$

Masked entries have $A_{ji}=0$, hence zero score gradient.
From $S=K^\top Q/\sqrt{32}$,

$$G_Q=KG_S/\sqrt{32},\qquad G_K=QG_S^\top/\sqrt{32}.$$

Finally, differentiate the affine projections:

$$
\begin{aligned}
G_{W_Q}&=G_Q\tilde X^\top,&G_{b_Q}&=G_Q\mathbf1_T,\\
G_{W_K}&=G_K\tilde X^\top,&G_{b_K}&=G_K\mathbf1_T,\\
G_{W_V}&=G_{\mathsf V}\tilde X^\top,&G_{b_V}&=G_{\mathsf V}\mathbf1_T.
\end{aligned}
$$

This head contributes the following gradient to the normalized input:

$$G_{\tilde X}^{(1)}
=W_Q^\top G_Q+W_K^\top G_K+W_V^\top G_{\mathsf V}.$$

The three terms add because the same input feeds all three projections.
The full network also sums contributions from the other heads and the residual path,
then differentiates normalization and the token-plus-position addition.
For that input addition, $x_t=E_{\rm LM}[:,i_t]+P[:,t]$, its gradient reaches both summands.
Repeated token IDs accumulate their contributions in the same embedding column.

The code checks these matrix derivatives, including biases, against automatic differentiation.
It also checks the head's query-weight gradient against backpropagation through the entire GPT.
A diagnostic update changes only this head's parameters; the input activation stays fixed.
The later training loop updates the complete model.
""")
    code("attention_gradient")
    caption(12,"Gradients of the actual head's three 32 × 256 projection matrices. Colors divide each matrix by its own largest absolute entry, stated in the title; compare those numbers for magnitude. Optimization uses the original, unscaled gradients.")
    md(r"""
**Encoder and decoder terminology.** An encoder can read the whole supplied input sequence.
In translation, a decoder can read an encoder's source representation and the target prefix.
GPT is decoder-only: its causal self-attention reads the available prefix.
These neural architectures differ from the tokenizer's text/ID encode and decode functions.
""")

    md(r"""
## 8. Train the full English model

The configuration stays fixed: vocabulary 2,048, width 256, six blocks, eight heads,
128-token training windows, and batches of eight windows.
The loss averages the $128\times8$ next-token targets in each training batch.

The complete loop below starts from random parameters, uses Adam and gradient clipping,
warms up the learning rate, and evaluates fixed training/validation windows.
It selects a checkpoint by validation loss. Test windows never select the checkpoint.
A 40-minute fit budget is checked between updates, with a reserve for evaluation.
This is not a hard process timeout.
""")
    code("gpt_training")
    md(r"""
The supplied model was trained from random initialization on this computer in about
38 minutes. Its saved arrays load into the exact model classes defined above.
The next cell checks that the restored model reproduces saved reference logits.

By default, §TRAIN_LARGE_GPT=false§ inspects that recorded experiment.
Set it to true to run a fresh GPU fit with the same full-sized architecture and corpus.
The new result is saved under §output/nlp_course/§.
This compact trainer's warm-up and time-based schedule can differ from the recorded run;
it starts a fresh experiment rather than resuming an optimizer state.
""")
    code("large_model")
    code("gpt_curves")
    caption(13,"Learning curves of the full English model, labeled as the recorded GPU experiment or a newly requested GPU fit. Both curves measure fixed corpus windows; they do not measure general conversational ability.")
    md(r"""
## 9. Evaluate and generate

### 9.1 Evaluate the same architecture before and after learning

We compare a newly initialized model with the trained model on the **same eight batches of
64-token test windows**, totaling 4,096 targets. Both have 5,820,416 parameters.
The random model is a same-architecture baseline, not a smaller model or an additional training run.

For the sampled test set $\mathcal T$, report mean cross-entropy and perplexity:

$$L_{\rm test}=-\frac1{|\mathcal T|}\sum_{(n,t)\in\mathcal T}\log p_{n,t}[y_{n,t}],
\qquad \operatorname{PPL}=\exp(L_{\rm test}).$$

These are sampled-window results, not an evaluation of every test token.
They are comparable because tokenizer, targets, and input windows are identical.
We also replace future input IDs and check that earlier logits remain unchanged.
""")
    code("language_evaluation")
    md(r"""
### 9.2 Select a token and feed it back

During generation, learned parameters stay fixed. At each step:

1. Encode the prompt and calculate logits for its final position.
2. Divide logits by a positive temperature $\tau$ and apply softmax.
3. Select a token ID, append it to the input, and calculate the next distribution.
4. Stop at §<eos>§ or the token limit, then decode the IDs.

The temperature-adjusted distribution is

$$p_\tau[k]=\frac{\exp(z[k]/\tau)}{\sum_j\exp(z[j]/\tau)}.$$

Lower $\tau$ concentrates probability on high scores; higher $\tau$ spreads it.
Greedy decoding selects the largest score.
For top-$k$ sampling, define $\mathcal K$ as the $k$ highest-scoring IDs and sample with

$$\tilde p[k]=\frac{\mathbf1\{k\in\mathcal K\}p_\tau[k]}
                   {\sum_{j\in\mathcal K}p_\tau[j]}.$$

Top-$p$ instead retains the smallest leading set whose cumulative probability reaches a chosen threshold.
Our code uses temperature and top-$k$.
A rolling window drops old tokens after the context fills; it does not extend the trained context length.
The displayed continuations are actual outputs, including repetitions and mistakes.
""")
    code("sampling")
    code("temperature")
    caption(14,"The twelve highest-scoring next tokens for a fixed prompt under two temperatures, before top-k truncation. Probabilities use the full eligible vocabulary; beginning and unknown markers are suppressed by the sampler. Omitted tokens retain probability mass.")
    code("learned_attention")
    caption(15,"Two measured first-layer heads in the trained full model. An attention weight is a mixture coefficient; by itself it is not a causal explanation of meaning.")
    md(r"""
## 10. Check your understanding

1. What distinguishes a token position, a token ID, and an embedding coordinate?
2. What are $E$, $H$, $f$, and $a$? Derive $a=H\mathbf1_T/T=Ef$.
3. Which objects does the optimizer change, and which are recomputed activations?
4. Derive $\partial L/\partial z=\hat p-y$, then $\nabla_E L=\delta_a f^\top$.
5. Why must two reviews with the same word counts receive the same classifier prediction?
6. What are the shapes of a head's $W_Q$, $Q$, $K^\top Q$, and $\mathsf V A$?
7. Which axis of $A$ sums to one, and which entries must be zero?
8. Why must residual branches have matching shapes, and where does the direct gradient term come from?
9. Why can training process many positions together while generation chooses tokens sequentially?

<details><summary>Answer notes</summary>

A position identifies where a token occurs; its ID selects a vocabulary entry; a coordinate
selects a component of its vector. $E$ is the vocabulary table and $H$ selects its columns
with repetition. Averaging those columns equals weighting each vocabulary column by its
observed frequency $f_j$. Parameters persist and are updated; activations are recomputed.
The sigmoid and binary-loss derivatives simplify to $\hat p-y$; differentiating $a=Ef$
then gives $\delta_a f^\top$. Reordering leaves $f$ unchanged.
For the GPT head, the four shapes are $32\times256$, $32\times T$, $T\times T$, and $32\times T$.
Attention columns sum to one and future-key entries are zero.
Residual addition requires matching coordinates and positions;
differentiating $x+f(x)$ gives the identity plus the sublayer Jacobian.
Teacher forcing supplies all observed prefixes; generation must first choose the next token.

</details>

For a controlled experiment, change one setting at a time and keep the data split fixed.
Use validation to select checkpoints and evaluate test performance afterward.
Fluent-looking continuations alone do not establish factual accuracy or reliable reasoning.
Restart the kernel after changing model type definitions.

### Further reading

[Keras: Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch/),
[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), and
[Dive into Deep Learning](https://d2l.ai/chapter_natural-language-processing-applications/index.html)
provide complementary treatments.
The architecture and corpus references are
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) and
[TinyStories](https://arxiv.org/abs/2305.07759).
The executable course uses Julia throughout.
""")
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"name":"julia-1.12","display_name":"Julia 1.12","language":"julia"},
        "language_info":{"name":"julia","version":"1.12.4","file_extension":".jl","mimetype":"application/julia"},
        "course":{"revision":"full-model-step-by-step","style_reference":["Intro_DL","RBC","KS1998"],
            "external_assets":["IMDb TSV data","TinyStories tokens and BPE rules","recorded GPT parameter arrays"],
            "external_project_source_imports":False,"review_vocabulary":20000,"review_embedding_dimension":64,
            "review_hidden_width":64,"review_parameters":1284225,"gpt_parameters":5820416}})
    nbf.validate(nb)
    nbf.write(nb,OUT)
    print(f"Wrote {len(cells)} cells, {sum(c.cell_type=='code' for c in cells)} Julia code cells")

if __name__=="__main__":
    main()
