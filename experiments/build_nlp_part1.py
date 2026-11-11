"""Build notebooks/Intro_NLP_Part1_Sentiment.ipynb from the classifier half of
notebooks/Intro_English_GPT.ipynb, then add the Part 1 revisions.

Run once:  python experiments/build_nlp_part1.py
Then execute the notebook with the julia-1.12 kernel to embed outputs.
"""
from pathlib import Path
import re
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "Intro_English_GPT.ipynb"
OUT = ROOT / "notebooks" / "Intro_NLP_Part1_Sentiment.ipynb"

src = nbformat.read(SRC, as_version=4).cells


def md(s):
    return new_markdown_cell(s.strip("\n"))


def code(s):
    return new_code_cell(s.strip("\n"))


def old(i):
    c = src[i]
    return md(c.source) if c.cell_type == "markdown" else code(c.source)


# ---------------------------------------------------------------- intro
TITLE = r"""
# Intro to NLP, Part 1: Text Classification from Scratch in Julia

We learn the building blocks of Natural Language Processing (NLP) by solving one complete task:
deciding whether an IMDb movie review is **positive or negative**.
Along the way we define every object—tokens, vocabulary IDs, word embeddings, a neural network,
and a loss—and follow the exact forward pass, derivatives, and parameter update
before training on 22,411 real reviews.

The model has a 20,000-word vocabulary, 64-dimensional embeddings, and 64 hidden units:
1,284,225 trainable parameters. It is a **classifier, not yet a language model**:
it reads a review and returns one probability.
Its main limitation—it ignores word order—motivates Part 2, where we train a small
language model with one attention head (queries, keys, values) and residual connections.

1. Text → tokens → vocabulary IDs.
2. IDs → trainable word vectors → one review vector → neural network → probability.
3. Loss → backpropagation → one parameter update, every step derived.
4. Train on the full dataset, evaluate unseen reviews, and inspect what was learned.
5. What averaging cannot see, and the bridge to attention.
"""

SETUP_MD = r"""
## Julia setup

Select the project's **Julia 1.12** kernel and run cells in order.
The notebook reads the prepared IMDb files in `reference/datasets/imdb/`
(create them with `experiments/prepare_imdb.py`). It imports no project source module:
every tokenizer, model, training, and plotting function is defined below.
The full fit runs on the CPU in well under a minute.
"""

setup_code = src[2].source.replace(
    'STORY_DIR=joinpath(ROOT,"reference","datasets","english_basics")\n', "")
setup_code = setup_code.replace(
    'println("Julia ",VERSION,',
    '''function flow_box!(p,x,y,title,sub;w=1.45,h=1.05,fill="#F4F8FA",edge=TEAL)
    plot!(p,Shape(x .+ [-w,w,w,-w]./2,y .+ [-h,-h,h,h]./2);
        fillcolor=fill,linecolor=edge,lw=1.5,label=false)
    annotate!(p,x,y+0.2,text(title,10,INK))
    annotate!(p,x,y-0.22,text(sub,10,GRAY))
end
flow_arrow!(p,x1,x2,y;color=GRAY)=plot!(p,[x1,x2],[y,y];arrow=true,color,lw=1.6,label=false)
println("Julia ",VERSION,''')

# Split the old 1.1/1.2 cell so the overview diagram sits between them.
s3 = src[3].source
cut = s3.index("### 1.2")
SEC11 = s3[:cut].rstrip() + r"""

NLP covers several kinds of prediction. They differ mainly in what the output is:

| Task | Input | Output | Examples |
|---|---|---|---|
| Text classification | a document | one label | sentiment, spam, topic — **this notebook** |
| Sequence labeling | a sentence | one label per token | parts of speech, named entities |
| Language modeling | a prefix | a distribution over the next token | autocomplete, GPT — **Part 2** |
| Sequence to sequence | a sequence | another sequence | translation, summarization |

Every one of these models first turns text into numbers.
Figure 1 shows the complete path for this notebook. Operations in **teal** are fixed rules
chosen before training; operations in **amber** contain trainable parameters.
Sections 1–2 define each box in order.
"""
SEC12 = s3[cut:]

PIPELINE = r"""
function pipeline_diagram()
    p=plot(xlims=(0,13.2),ylims=(-0.4,3.4),axis=false,grid=false,legend=false,size=(1150,330))
    stages=[("Review text","string s"),("Tokens","T word strings"),("IDs","T integers"),
            ("Token vectors","H: 64×T"),("Review vector","a: 64"),("Hidden layer","h: 64"),
            ("Prediction","p̂ ∈ (0,1)")]
    ops=[("tokenizer",false),("vocabulary ι",false),("lookup in E",true),
         ("mean over t",false),("W, b, ReLU",true),("v, c, sigmoid",true)]
    xs=collect(range(0.9,12.3;length=7));y=1.5
    for (k,(t,s)) in enumerate(stages)
        flow_box!(p,xs[k],y,t,s;edge=k==7 ? AMBER : TEAL)
    end
    for (k,(name,trained)) in enumerate(ops)
        flow_arrow!(p,xs[k]+0.75,xs[k+1]-0.75,y)
        annotate!(p,(xs[k]+xs[k+1])/2,y+0.85,text(name,9,trained ? AMBER : TEAL))
    end
    annotate!(p,6.6,-0.1,text("teal: fixed rule     amber: trainable parameters θ = (E, W, b, v, c)",10,GRAY))
    p
end
pipeline_diagram()
"""
CAP_PIPE = "*Figure 1. The path from a review string to a probability. Only the amber operations are learned; tokenization, ID lookup, and averaging are fixed rules.*"

# ---------------------------------------------------------------- section 2 additions
s11 = src[11].source.replace(
    "The lookup is not a separate learned transformation: the entries of $E$ are the parameters.",
    r"""The lookup is not a separate learned transformation: the entries of $E$ are the parameters.
Equivalently, let $o_j\in\mathbb R^V$ be the **one-hot vector** with a 1 in entry $j$ and zeros elsewhere.
Then $E\,o_{x_t}=E[:,x_t]$: multiplying by a one-hot vector picks one column.
Libraries implement the lookup as indexing because it gives the same result without forming $o_j$.""")

s13 = src[13].source.rstrip() + r"""

Because $a$ depends on a review only through its word frequencies $f$, this representation is a
**bag of words**: it records which words occur and how often, not their order.
Classical NLP used fixed bag-of-words features (counts or TF-IDF weights) with a linear model;
here the 64 features per word are learned.
"""

# ---------------------------------------------------------------- section 3: computation graph
GRAPH_MD = r"""
### 3.2 Draw the computation graph

The forward pass is a chain of simple operations. Backpropagation walks the same chain in reverse,
multiplying local derivatives. Figure 6 shows both directions; Sections 3.3–3.5 derive each
backward label, and the code checks every one against automatic differentiation.
"""

GRAPH = r"""
function backprop_diagram()
    p=plot(xlims=(0,13),ylims=(-1.3,4.6),axis=false,grid=false,legend=false,size=(1150,430))
    names=["a","u","h","z","p̂","L"];xs=collect(range(1.0,12.0;length=6));y=2.0
    for (k,n) in enumerate(names)
        scatter!(p,[xs[k]],[y];ms=22,color=k==6 ? AMBER : TEAL,markerstrokewidth=0)
        annotate!(p,xs[k],y,text(n,13,:white))
    end
    fwd=["Wa+b","ReLU","vᵀh+c","σ(z)","cross-entropy"]
    for k in 1:5
        flow_arrow!(p,xs[k]+0.45,xs[k+1]-0.45,y+0.12;color=TEAL)
        annotate!(p,(xs[k]+xs[k+1])/2,y+0.5,text(fwd[k],9,TEAL))
        plot!(p,[xs[k+1]-0.45,xs[k]+0.45],[y-0.55,y-0.55];arrow=true,color=AMBER,lw=1.6)
    end
    back=[L"\delta_a=W^\top\delta_u",L"\delta_u=\delta_h\odot 1\{u>0\}",
          L"\delta_h=v\,\delta_z",L"\delta_z=\hat{p}-y",""]
    for k in 1:4
        annotate!(p,(xs[k]+xs[k+1])/2,y-1.0,text(back[k],10,AMBER))
    end
    params=[(1,"E (via f)",L"\nabla_E L=\delta_a f^\top"),(2,"W, b",L"\nabla_W L=\delta_u a^\top"),
            (4,"v, c",L"\nabla_v L=\delta_z h^\top"),(6,"label y","")]
    for (k,name,grad) in params
        x=xs[k]
        plot!(p,[x,x],[y+1.75,y+0.42];arrow=true,color=GRAY,lw=1.2)
        annotate!(p,x,y+2.05,text(name,10,INK))
        grad=="" || annotate!(p,x,y+2.45,text(grad,9,AMBER))
    end
    annotate!(p,6.5,-1.05,text("teal: forward pass     amber: backward pass (gradients)",10,GRAY))
    p
end
backprop_diagram()
"""
CAP_GRAPH = "*Figure 5. Forward pass (teal, left to right) and backward pass (amber, right to left). Each backward label multiplies the incoming gradient by a local derivative; parameter gradients (top) branch off the chain.*"

s26 = src[26].source
for a, b in [("### 3.5", "### 3.6"), ("### 3.4", "### 3.5"), ("### 3.3", "### 3.4"), ("### 3.2", "### 3.3")]:
    s26 = s26.replace(a, b)

# ---------------------------------------------------------------- section 4: what was learned
LEARNED_MD = r"""
### 4.4 What did the word vectors learn?

Figure 10 compares the same table entries before and after fitting. Training changed $E$ only
through the sentiment loss, so the learned vectors encode whatever helps predict the label.
To read one word's learned vector, feed the network a "review" containing only word $j$.
Then $f=o_j$, $a=E o_j=e_j$, and the word's **single-word logit** is

$$s_j=v^\top\operatorname{ReLU}(We_j+b)+c.$$

Positive $s_j$ pushes a review toward "positive"; negative $s_j$ pushes it toward "negative".
We report words that appear in at least 100 training reviews, so that each estimate rests on many examples.
"""

WORD_SCORES = r"""
learned=sentiment.model
word_logit=vec(learned.v*max.(learned.W*learned.E .+ learned.b,0f0) .+ learned.c)
doc_freq=vec(sum(imdb.training.F .> 0;dims=2))
frequent=filter(j->j!=1 && doc_freq[j]>=100,1:V_review)
ranked=frequent[sortperm(word_logit[frequent])]
neg_ids,pos_ids=ranked[1:15],reverse(ranked[end-14:end])
@printf("%d frequent words; single-word logits range from %.2f to %.2f\n",
    length(frequent),word_logit[ranked[1]],word_logit[ranked[end]])
bp=bar(word_logit[pos_ids];orientation=:h,yticks=(1:15,imdb.vocab[pos_ids]),yflip=true,
    color=TEAL,linecolor=:transparent,legend=false,title="Most positive single-word logits",
    xlabel=L"s_j",grid=:x,ylims=(0.4,15.6),tickfontsize=11)
bn=bar(-word_logit[neg_ids];orientation=:h,yticks=(1:15,imdb.vocab[neg_ids]),yflip=true,
    color=AMBER,linecolor=:transparent,legend=false,title="Most negative single-word logits",
    xlabel=L"-s_j",grid=:x,ylims=(0.4,15.6),tickfontsize=11)
plot(bp,bn;layout=(1,2),size=(1120,600),left_margin=12Plots.mm)
"""
CAP_SCORES = "*Figure 10. Each bar is the trained network's logit for a review made of one word. No word was ever labeled; these scores come entirely from review-level labels and backpropagation through the mean.*"

NEIGHBORS_MD = r"""
Words with similar roles should end up with similar vectors. Measure closeness with **cosine similarity**,

$$\cos(e_i,e_j)=\frac{e_i^\top e_j}{\lVert e_i\rVert\,\lVert e_j\rVert}\in[-1,1],$$

which compares directions and ignores lengths. The cell lists each probe word's nearest frequent
neighbours in the random initial table and in the learned table.
"""

NEIGHBORS = r"""
function nearest_words(E,word;k=6,candidates=frequent)
    j=imdb.lookup[word];e=E[:,j]/norm(E[:,j])
    sims=[dot(e,E[:,i])/norm(E[:,i]) for i in candidates]
    [imdb.vocab[candidates[o]] for o in sortperm(sims;rev=true) if candidates[o]!=j][1:k]
end
for w in ["excellent","awful","boring","beautiful","plot"]
    @printf("%-10s random : %s\n",w,join(nearest_words(sentiment.initial.E,w),", "))
    @printf("%-10s learned: %s\n\n",w,join(nearest_words(learned.E,w),", "))
end
"""

AFTER_NEIGHBORS = r"""
After training, sentiment words cluster with words of the same polarity, while the random
table's neighbours are arbitrary. Even `plot` now sits among negative words: in IMDb reviews,
complaints tend to discuss the plot, and the loss only rewards information about sentiment.
The vector does not describe what a plot *is*. This is a general principle: **an embedding learns what its training
objective needs**. Word2vec-style embeddings and language models instead predict surrounding
or next words, so their vectors capture broader usage; that is the objective of Part 2.
"""

s39 = src[39].source.replace("### 4.4", "### 4.5").replace(
    "The code verifies this equality.",
    "This is the bag-of-words property from Section 2.3. The code below verifies the equality and lists two test-set mistakes.")

# ---------------------------------------------------------------- section 5: bridge to attention
BRIDGE_MD = r"""
## 5. From averaging to attention

### 5.1 Mean pooling is a fixed weighting

Write the review vector with explicit weights $\alpha_t$:

$$a=\sum_{t=1}^T\alpha_t\,\tilde e_t,\qquad \alpha_t\geq0,\qquad \sum_{t=1}^T\alpha_t=1.$$

Mean pooling fixes $\alpha_t=1/T$ for every position, whatever word sits there and whatever surrounds it.
That causes both limitations we observed:

1. Every token gets the same weight, so `not` cannot change how much `good` counts.
2. Each word's vector is the same in every context, so the pair (`not`, `good`) is just two separate terms in the sum.

### 5.2 Let the weights depend on the words

**Attention** computes the weights from the token vectors themselves. For each position $i$,
three learned matrices produce a query, a key, and a value:

$$q_i=W_Q\tilde e_i,\qquad k_j=W_K\tilde e_j,\qquad v_j=W_V\tilde e_j.$$

Position $i$ scores every position $j$ by $k_j^\top q_i$, turns those scores into weights with a softmax,
and reads a weighted combination of values:

$$\alpha_{ji}=\frac{\exp(k_j^\top q_i/\sqrt{d_k})}{\sum_{\ell}\exp(k_\ell^\top q_i/\sqrt{d_k})},
\qquad o_i=\sum_{j}\alpha_{ji}\,v_j.$$

Unlike $a$, which is one vector per review, $o_i$ is a **separate context vector for every position**,
so the representation of `good` can depend on a nearby `not`.
Part 2 defines every matrix in this formula, adds the causal mask and a residual connection
$r_i=\tilde e_i+o_i$, and trains the result as a small language model.

Figure 12 makes the contrast concrete on our traced review. The lower panel is **not learned attention**:
it is a hand-built rule, $\alpha_t\propto\exp(|s_{x_t}|/\tau)$ with $\tau=10$, that weights each token by the size of
its single-word logit. It only shows what input-dependent weights look like.
"""

BRIDGE = r"""
alpha_uniform=fill(1/T_review,T_review)
score_trace=abs.(word_logit[trace_ids])
tau=10f0
alpha_rule=exp.(score_trace./tau)./sum(exp.(score_trace./tau))
top=sortperm(alpha_rule;rev=true)[1:6]
wa=bar(1:T_review,alpha_uniform;color=GRAY,linecolor=:transparent,legend=false,
    ylims=(0,1.15maximum(alpha_rule)),title="Mean pooling: every weight equals 1/T",ylabel=L"\alpha_t",grid=:y)
wb=bar(1:T_review,alpha_rule;color=TEAL,linecolor=:transparent,legend=false,
    ylims=(0,1.15maximum(alpha_rule)),title="Hand-built input-dependent weights (illustration only)",
    xlabel="Token position in the traced review",ylabel=L"\alpha_t",grid=:y)
for t in top
    annotate!(wb,t,alpha_rule[t]+0.06maximum(alpha_rule),text(trace_tokens[t],9,INK))
end
@printf("Uniform weight per token: %.4f | largest hand-built weight: %.4f (%s)\n",
    1/T_review,maximum(alpha_rule),trace_tokens[top[1]])
plot(wa,wb;layout=(2,1),size=(1120,620))
"""
CAP_BRIDGE = "*Figure 11. Both panels weight the same 115 token vectors. Mean pooling spreads weight evenly; an input-dependent rule can concentrate it on informative tokens. Attention learns such weights from data, separately for every position.*"

QUIZ = r"""
## 6. Check your understanding

1. What distinguishes a token position, a token ID, and an embedding coordinate?
2. What are $E$, $H$, $f$, and $a$? Derive $a=H\mathbf1_T/T=Ef$.
3. Which objects does the optimizer change, and which are recomputed activations?
4. Derive $\partial L/\partial z=\hat p-y$, then $\nabla_E L=\delta_a f^\top$.
5. Why must two reviews with the same word counts receive the same prediction?
6. Write mean pooling as a weighted sum. What does attention change about the weights and the output?

<details><summary>Answer notes</summary>

A position identifies where a token occurs; its ID selects a vocabulary entry; a coordinate
selects a component of its vector. $E$ is the vocabulary table and $H$ selects its columns
with repetition. Averaging those columns equals weighting each vocabulary column by its
observed frequency $f_j$. Parameters persist and are updated; activations are recomputed.
The sigmoid and binary-loss derivatives simplify to $\hat p-y$; differentiating $a=Ef$
then gives $\delta_a f^\top$. Reordering leaves $f$ unchanged, so $a$ and $\hat p$ are unchanged.
Mean pooling is $a=\sum_t\alpha_t\tilde e_t$ with $\alpha_t=1/T$. Attention computes
$\alpha_{ji}$ from queries and keys, so the weights depend on the input, and it returns one output per position.

</details>

### Further reading

[Jurafsky & Martin, *Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/)
(chapters on classification and vector semantics),
[Keras: Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch/), and
[Dive into Deep Learning: sentiment analysis](https://d2l.ai/chapter_natural-language-processing-applications/index.html).
"""

# ---------------------------------------------------------------- assemble
cells = [md(TITLE), md(SETUP_MD), code(setup_code),
         md(SEC11), code(PIPELINE), md(CAP_PIPE), md(SEC12)]
cells += [old(i) for i in range(4, 11)]                 # 1.2 code … Figure 1 caption
cells += [md(s11)] + [old(12)] + [md(s13)] + [old(i) for i in range(14, 26)]
cells += [md(GRAPH_MD), code(GRAPH), md(CAP_GRAPH), md(s26)]
cells += [old(i) for i in range(27, 38)]                # manual gradient … Figure 8 caption
cells += [md(LEARNED_MD), code(WORD_SCORES), md(CAP_SCORES),
          md(NEIGHBORS_MD), code(NEIGHBORS), md(AFTER_NEIGHBORS)]
cells += [md(s39), old(38)]
cells += [md(BRIDGE_MD), code(BRIDGE), md(CAP_BRIDGE), md(QUIZ)]

# Renumber figure captions in order and fix in-text references.
mapping, n = {}, 0
for c in cells:
    m = re.match(r"\*Figure (\d+)\.", c.source)
    if c.cell_type == "markdown" and m:
        n += 1
        key = c.source[:60]
        mapping[key] = n
        c.source = re.sub(r"^\*Figure \d+\.", f"*Figure {n}.", c.source)

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Julia 1.12", "language": "julia", "name": "julia-1.12"},
    "language_info": {"name": "julia"},
}
nbformat.validate(nb)
nbformat.write(nb, OUT)
print(f"Wrote {OUT.name}: {len(cells)} cells, "
      f"{sum(c.cell_type == 'code' for c in cells)} code cells, {n} figure captions")
