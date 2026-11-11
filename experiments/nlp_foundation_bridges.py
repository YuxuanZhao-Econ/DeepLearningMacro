"""English Julia lessons prompted by the archived RAG/DAG reference dialogue."""


def token_boundaries(md):
    md(r"""
### Keep the tokenizer, the table, and the network separate

| Object | What it does | What learns in our standard pipeline? |
|---|---|---|
| Tokenizer | Text → pieces → integer IDs | BPE rules are fitted before neural training |
| Embedding table | ID → a column of numbers | Its numerical entries receive gradients |
| Context network | Sequence of vectors → contextual states | Its weights receive gradients |
| Output head | Hidden state → candidate scores | Its weights receive gradients |
| Tokenizer decode | Selected IDs → text pieces | Fixed reconstruction procedure |

An ID is an **address**, not a numerical feature. If $o_i$ is a $V$-dimensional
one-hot vector, $Eo_i=E[:,i]$. Lookup avoids constructing that mostly zero vector.
We differentiate selected entries of $E$, not spellings, integer addresses, or BPE merge decisions.
""")


def classifier_bridge(md, code, check):
    md(r"""
### 3a · A whole neural NLP model, small enough to audit

Before language generation, try a familiar **binary prediction** task: does a review
have a positive label? The tiny dataset is `really good → 1`, `very good → 1`,
`really bad → 0`, `very bad → 0`. These labels are supplied by the dataset.
In language modeling, the next token will supply the label instead.

This separate demonstration uses vocabulary `[really, very, good, bad]`.
For `really good`, IDs are `[1,3]`. Choose **two embedding coordinates and three hidden neurons**:

$$
a=\frac{E[:,1]+E[:,3]}2,\quad u=Wa+b,\quad
h=\operatorname{ReLU}(u),\quad z=v^\top h+c,\quad \hat p=\sigma(z).
$$

| Object | Shape | Role |
|---|---|---|
| $E$ | $2\times4$ | Trainable token table |
| $a$ | $2$ | Average of two selected vectors |
| $W,b$ | $3\times2,\ 3$ | Trainable hidden-layer weights and biases |
| $u,h$ | $3$ | Computed pre-activation and hidden state |
| $v,c$ | $3,\ 1$ | Trainable output weights and bias |
| $z,\hat p,y$ | Scalars | Logit, predicted probability, observed label |

Parameters $\theta=(E,W,b,v,c)$ contain **21 adjustable numbers**.
Intermediate quantities $a,u,h,z,\hat p$ are recomputed, not optimized independently.
One neuron combines both coordinates; it does not represent one word.

Why this architecture? Averaging supplies a fixed-size classifier input.
The nonlinearity adds flexibility: without ReLU, $v^\top(Wa+b)+c$ collapses to
one affine function of $a$. The last layer is a logistic classifier on jointly learned
features, much like a binary choice model whose regressors are learned too.
Averaging is our simple choice; GPT keeps a representation at every position.
""")
    code(r'''
foundation_words = ["really", "very", "good", "bad"]
foundation_parameters = (
    E=[0.2 -0.1 0.6 -0.3; 0.4 0.1 0.8 -0.2],
    W=[1.0 1.0; 1.0 -1.0; -1.0 2.0],
    b=zeros(3), v=[1.0, -1.0, 0.5], c=[0.0],
)
foundation_ids, foundation_y = [1, 3], 1.0
function foundation_forward(parameters, ids)
    a = vec(mean(parameters.E[:, ids]; dims=2))
    u = parameters.W * a + parameters.b
    h = max.(u, 0.0)
    z = dot(parameters.v, h) + only(parameters.c)
    probability = 1 / (1 + exp(-z))
    (; a, u, h, z, probability)
end
function foundation_loss(parameters, ids, y)
    p = foundation_forward(parameters, ids).probability
    # This hand-sized example stays away from 0/1. The GPT lab uses stable log-softmax.
    -y * log(p) - (1-y) * log1p(-p)
end
foundation_forward_before = foundation_forward(foundation_parameters, foundation_ids)
@assert foundation_parameters.E * [1.0, 0, 0, 0] == foundation_parameters.E[:, 1]
@assert sum(length, values(foundation_parameters)) == 21
for name in (:a, :u, :h, :z, :probability)
    println(name, " = ", getproperty(foundation_forward_before, name))
end
@printf("Loss for y=1: %.6f\n", foundation_loss(foundation_parameters, foundation_ids, foundation_y))

# Show the numerical values and actual neuron connectivity.
foundation_graph = plot(xlims=(0, 8), ylims=(0, 4.4), axis=false,
    grid=false, legend=false, size=(940, 370),
    title="Each hidden neuron combines coordinates of the mean vector")
foundation_input_y, foundation_hidden_y = [2.7, 1.3], [3.25, 2.0, 0.75]
for iy in foundation_input_y, hy in foundation_hidden_y
    plot!(foundation_graph, [1, 4], [iy, hy]; color="#b6c5d4", lw=2)
end
for hy in foundation_hidden_y
    plot!(foundation_graph, [4, 7], [hy, 2]; color="#b6c5d4", lw=2)
end
scatter!(foundation_graph, fill(1, 2), foundation_input_y;
    color="#e6effb", markersize=27, markerstrokecolor=blue)
scatter!(foundation_graph, fill(4, 3), foundation_hidden_y;
    color=["#e4f5ed", "#eeeeee", "#e4f5ed"], markersize=27, markerstrokecolor=green)
scatter!(foundation_graph, [7], [2]; color="#fff1dc", markersize=32, markerstrokecolor=gold)
for (y, label) in zip(foundation_input_y, ["0.4", "0.6"])
    annotate!(foundation_graph, 1, y, text(label, 11))
end
for (y, label) in zip(foundation_hidden_y, ["1.0", "0.0", "0.8"])
    annotate!(foundation_graph, 4, y, text(label, 11))
end
annotate!(foundation_graph, 7, 2, text("0.802", 11))
annotate!(foundation_graph, 1, 3.95, text("Mean embedding a\n2 coordinates", 10))
annotate!(foundation_graph, 4, 4.0, text("ReLU state h\n3 neurons", 10))
annotate!(foundation_graph, 7, 3.95, text("Sigmoid output\nP(positive)", 10))
annotate!(foundation_graph, 4, 0.22, text("The middle neuron is inactive for this input.", 9, "#666666"))
foundation_graph
''')
    md(r"""
### 3b · Follow the loss back to every embedding coordinate

The forward pass gave $a=(0.4,0.6)^\top$, $u=(1,-0.2,0.8)^\top$,
$h=(1,0,0.8)^\top$, $z=1.4$, $\hat p\approx0.8022$.
Binary cross-entropy is $L=-[y\log\hat p+(1-y)\log(1-\hat p)]$.
For $y=1$, the first chain-rule step is

$$
\frac{\partial L}{\partial z}
=\left(-\frac1{\hat p}\right)\hat p(1-\hat p)=\hat p-1\approx-0.1978.
$$

Increasing $z$ a little lowers this positive example's loss. Continue in reverse:

| Step | Chain rule | Approximate result |
|---|---|---|
| Output parameters | $\nabla_v L=(\hat p-y)h,\quad \partial_c L=\hat p-y$ | $(-0.1978,0,-0.1583)^\top,\ -0.1978$ |
| Hidden state | $\nabla_h L=(\hat p-y)v$ | $(-0.1978,0.1978,-0.0989)^\top$ |
| ReLU | $\nabla_u L=\nabla_h L\odot\mathbf1_{\{u>0\}}$ | $(-0.1978,0,-0.0989)^\top$ |
| Hidden parameters | $\nabla_W L=(\nabla_u L)a^\top,\quad\nabla_b L=\nabla_u L$ | A $3\times2$ matrix and a length-3 vector |
| Pooled input | $\nabla_a L=W^\top\nabla_u L$ | $(-0.0989,-0.3956)^\top$ |
| Each selected column | $\nabla_{E[:,1]}L=\nabla_{E[:,3]}L=\tfrac12\nabla_a L$ | $(-0.04945,-0.19782)^\top$ |

**Multiply along a path; add contributions from multiple paths.**
For example, $a_k$ influences every hidden neuron, so
$\partial L/\partial a_k=\sum_j W_{jk}\,\partial L/\partial u_j$.
The inactive ReLU transmits zero local gradient.
Equal word gradients come from our equal-weight average, not a general rule about embeddings.

Calculate **all** gradients at the old parameters, then simultaneously apply
$\theta_{\rm new}=\theta_{\rm old}-\eta\nabla_\theta L$.
One sample does not identify which word carries sentiment; other examples supply other gradients.
""")
    code(r'''
foundation_dz = foundation_forward_before.probability - foundation_y
foundation_dh = foundation_dz .* foundation_parameters.v
foundation_du = foundation_dh .* (foundation_forward_before.u .> 0)
foundation_da = foundation_parameters.W' * foundation_du
foundation_dE = zeros(size(foundation_parameters.E))
for id in foundation_ids
    foundation_dE[:, id] .+= foundation_da ./ length(foundation_ids)
end
foundation_manual = (
    E=foundation_dE, W=foundation_du * foundation_forward_before.a',
    b=foundation_du, v=foundation_dz .* foundation_forward_before.h, c=[foundation_dz],
)
foundation_auto = Flux.gradient(
    p -> foundation_loss(p, foundation_ids, foundation_y), foundation_parameters)[1]
foundation_gradient_error = maximum(
    maximum(abs.(getproperty(foundation_manual, key) .- getproperty(foundation_auto, key)))
    for key in keys(foundation_parameters))
@assert foundation_gradient_error < 1e-10
@assert all(iszero, foundation_auto.E[:, [2, 4]])  # words not looked up
@assert all(iszero, foundation_auto.W[2, :])       # inactive ReLU
foundation_eta = 0.1
foundation_updated = (
    E=foundation_parameters.E - foundation_eta * foundation_auto.E,
    W=foundation_parameters.W - foundation_eta * foundation_auto.W,
    b=foundation_parameters.b - foundation_eta * foundation_auto.b,
    v=foundation_parameters.v - foundation_eta * foundation_auto.v,
    c=foundation_parameters.c - foundation_eta * foundation_auto.c,
)
foundation_forward_after = foundation_forward(foundation_updated, foundation_ids)
foundation_losses = [
    foundation_loss(p, foundation_ids, foundation_y)
    for p in (foundation_parameters, foundation_updated)]
@assert foundation_losses[2] < foundation_losses[1]
@assert foundation_updated.E[:, [2, 4]] == foundation_parameters.E[:, [2, 4]]
@printf("Largest manual-versus-AD gradient error: %.3e\n", foundation_gradient_error)
println("Gradient of the good embedding: ", foundation_auto.E[:, 3])
println("good before / after: ", foundation_parameters.E[:, 3], " / ", foundation_updated.E[:, 3])
@printf("P(positive): %.6f -> %.6f | loss: %.6f -> %.6f\n",
    foundation_forward_before.probability, foundation_forward_after.probability,
    foundation_losses...)
foundation_grad_plot = heatmap(1:4, 1:2, foundation_auto.E;
    xticks=(1:4, foundation_words), yticks=(1:2, ["coordinate 1", "coordinate 2"]),
    color=:RdBu, clims=(-0.22, 0.22), colorbar_title="Derivative",
    title="Gradients reach the selected columns", xrotation=15)
foundation_loss_plot = bar(["Before update", "After update"], foundation_losses;
    color=[blue, green], label=false, ylabel="Binary cross-entropy",
    ylims=(0, 1.15maximum(foundation_losses)), title="One simultaneous update")
plot(foundation_grad_plot, foundation_loss_plot; layout=(1, 2), size=(1020, 330))
''')
    check("Parameters versus computed states",
          "Did the optimizer update integer IDs or the hidden vector h? Why did very and bad stay fixed?",
          "It updated E, W, b, v, and c. IDs are fixed addresses; h is recomputed. Unused embedding columns have no path to this example's loss. This assumes separate input/output weights and no extra regularization.")
    md(r"""
### 3c · Change the task, keep the learning logic

`I love cats .` supplies `I → love`, `I love → cats`, and `I love cats → .`.
The text supplies its own labels: **self-supervised learning**.
No "correct embedding" is supplied.

Replace the binary output by $V$ scores:

$$z=Uh+b_{\rm out}\in\mathbb R^V,\quad
p=\operatorname{softmax}(z),\quad L=-\log p_y,\quad \nabla_z L=p-o_y.$$

Each row of $U$ scores one candidate. Gradients return through $U$, the network
producing $h$, and selected embedding columns.
A shared prefix can have several valid next tokens: learn a conditional distribution,
rather than making every observed continuation certain.
The next scalar `bank → raises` example isolates this mechanism with fewer numbers.
It uses a new vocabulary and task, rather than continuing the review classifier.
""")


def model_sizes(md):
    md(r"""
### Three different sizes: vocabulary $V$, width $d$, and sequence length $T$

| Choice | Meaning | Real English model |
|---|---|---|
| $V$ | Number of distinct token IDs | 2,048 |
| $d$ | Numbers in each token vector | 256 |
| $T$ | Positions being processed in a window | At most 128 |

The table is $E\in\mathbb R^{d\times V}$; one embedded sequence is
$X\in\mathbb R^{d\times T}$. Token count is not embedding dimension.
The table has $Vd$ parameters: doubling $d$ doubles that table, while a $d\times d$
projection has four times as many weights.
Choose width using the task, data, runtime, and validation performance.
Gradient descent adjusts entries, not the number of coordinates.
""")


def recurrent_bridge(md, code, check):
    md(r"""
### 4a · A short bridge: averaging → recurrence → attention

Lookup preserves order as an ordered collection of columns.
**Averaging** discards it: `really good` and `good really` have the same mean.
No later classifier can recover information already discarded.

A recurrent neural network (**RNN**) replaces averaging by a state update:

$$h_t=\tanh(W_xe_{x_t}+W_hh_{t-1}+b),\qquad h_0=0.$$

Every position uses the **same** parameters; its input and state differ.
Equivalently, concatenate $e_{x_t}$ and $h_{t-1}$, then apply $[W_x\ W_h]$.
This is an ordinary neural layer reused along the sequence.
State width need not equal embedding width.

The example uses the earlier two-dimensional table and a **scalar** RNN state.
Hand-chosen weights expose order dependence; they are not trained language understanding.
After comparing word orders, a separate two-position, scalar-input example isolates
the backward derivative, so every chain-rule factor can be printed.
""")
    code(r'''
function foundation_rnn_final(E, ids; wx=[0.8, -0.3], wh=0.5)
    h = 0.0
    for id in ids
        h = tanh(dot(wx, E[:, id]) + wh*h)
    end
    h
end
foundation_order_a, foundation_order_b = [1, 3], [3, 1]
@assert mean(foundation_parameters.E[:, foundation_order_a]; dims=2) ≈
        mean(foundation_parameters.E[:, foundation_order_b]; dims=2)
foundation_order_states = [
    foundation_rnn_final(foundation_parameters.E, ids)
    for ids in (foundation_order_a, foundation_order_b)]
@assert abs(foundation_order_states[1] - foundation_order_states[2]) > 0.01
println("Identical means, different recurrent states: ", foundation_order_states)

# Only the second position has a loss: isolate its path to the first input.
foundation_e1, foundation_e2 = 0.2, 0.6
foundation_wx, foundation_wh = 0.8, 0.5
foundation_h1 = tanh(foundation_wx * foundation_e1)
foundation_h2 = tanh(foundation_wx * foundation_e2 + foundation_wh * foundation_h1)
foundation_late_loss(e1) = log1p(exp(-tanh(foundation_wx * foundation_e2 +
    foundation_wh * tanh(foundation_wx * e1))))  # y=1, logit=h2
foundation_late_dh2 = 1/(1+exp(-foundation_h2)) - 1
foundation_early_gradient = foundation_late_dh2 * (1-foundation_h2^2) *
    foundation_wh * (1-foundation_h1^2) * foundation_wx
foundation_early_auto = Flux.gradient(foundation_late_loss, foundation_e1)[1]
@assert isapprox(foundation_early_gradient, foundation_early_auto; atol=1e-12)
@printf("Later loss, earlier input gradient: manual %.6f | AD %.6f\n",
    foundation_early_gradient, foundation_early_auto)
bar(["really → good", "good → really"], foundation_order_states;
    color=[blue, green], label=false, ylabel="Final scalar state",
    ylims=(0, 1.15maximum(foundation_order_states)),
    title="Same vectors and weights, different reading order", size=(850, 300))
''')
    md(r"""
### 4b · How does a later error train an earlier token?

For a loss $L_2$ after two scalar states, the recurrent path gives

$$
\frac{\partial L_2}{\partial e_1}
=\frac{\partial L_2}{\partial h_2}
(1-h_2^2)w_h(1-h_1^2)w_x.
$$

That is **backpropagation through time**: the chain rule on an unrolled sequence.
If position 1 has its own loss, add its contribution.
Shared weights collect gradients from every occurrence before being updated once.
Long derivative products can shrink or grow, making long-distance credit assignment difficult.

An **LSTM** adds a gated memory path:

$$c_t=f_t\odot c_{t-1}+i_t\odot\widetilde c_t.$$

Forget and input gates are computed sigmoid outputs of trainable functions of the input
and previous hidden state. They are not independently optimized constants.
Holding gates fixed, the direct memory-path derivative is $\operatorname{diag}(f_t)$;
the full derivative can include other paths through the gates.

Take $c_{t-1}=0.8$, $\widetilde c_t=-0.3$, $f_t=0.9$, $i_t=0.2$: then $c_t=0.66$.
If the later network supplies $\partial L/\partial c_t=-2$, then
$\partial L/\partial f_t=-1.6$ and $\partial L/\partial i_t=0.6$.
Through sigmoid, pre-activation gradients are $-0.144$ and $0.096$.
Prediction error teaches the gate-producing weights what to preserve and write;
no explicit "remember this word" label is needed.
An improved memory path does not guarantee perfect long-term memory.

**Attention changes how history is accessed.** It reads a weighted combination of stored
position representations rather than relying only on a final recurrent state.
Transformer self-attention builds contextual states without RNN recurrence.
These are different representations for the same kind of prediction loss;
GPT does not run this RNN or LSTM example internally.

Optional reading: [D2L on RNNs](https://en.d2l.ai/chapter_recurrent-neural-networks/rnn.html)
and [Stanford CS224n on RNNs and gated models](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf).
""")
    code(r'''
foundation_gate_scores = [log(0.9/0.1), log(0.2/0.8)]
foundation_gate_local_loss(a) = -2 * (
    (1/(1+exp(-a[1]))) * 0.8 + (1/(1+exp(-a[2]))) * (-0.3))
foundation_gate_gradient = Flux.gradient(foundation_gate_local_loss, foundation_gate_scores)[1]
@assert isapprox(foundation_gate_gradient, [-0.144, 0.096]; atol=1e-12)
println("Forget/input gate pre-activation gradients: ", foundation_gate_gradient)
println("This local linear loss represents an upstream gradient of -2, not a trained LSTM.")
''')
    check("One set of weights, several gradient paths",
          "Does a longer sentence create more RNN parameters? Can its last prediction update an early embedding?",
          "The same matrices are reused, so no new parameters are created. The last loss depends on early states through recurrence, and its gradient can reach early embedding lookups. An early position's own loss contributes another gradient that must be added.")


def attention_learning(md, code, check):
    md(r"""
### 5a · Who teaches attention where to look?

No correct attention map was supplied in our hand-sized example.
The next-token loss trains the matrices producing queries, keys, and values.
For one query, let $r=\sum_j\alpha_jv_j$, $\alpha=\operatorname{softmax}(s)$,
and suppose the later network supplies $g=\nabla_r L$:

$$
\frac{\partial L}{\partial\alpha_j}=g^\top v_j,\qquad
\frac{\partial L}{\partial s_j}=\alpha_j\,g^\top(v_j-r).
$$

The score gradient compares a position's content with the **current mixture**.
Weights cannot change independently because they sum to one.
There is also a direct content-path gradient $\alpha_jg$ into $v_j$.
Score paths continue through $q^\top k_j$ into queries and keys; content paths reach values.
Both continue into projection matrices and embedding lookups.

The independent example below checks the score derivative.
$g^\top r$ is a local linear proxy for a supplied gradient, not an extra GPT objective.
In the full model, $g$ comes from the actual prediction loss.
""")
    code(r'''
foundation_values = [1.0 0 1; 0 1 1]
foundation_scores = [1.0, 1.0, 2.0]
foundation_attention = Flux.softmax(foundation_scores)
foundation_read = foundation_values * foundation_attention
foundation_upstream = [-0.7, 0.2]
foundation_score_manual = foundation_attention .* (
    foundation_values' * foundation_upstream .- dot(foundation_upstream, foundation_read))
foundation_score_auto = Flux.gradient(
    s -> dot(foundation_upstream, foundation_values * Flux.softmax(s)),
    foundation_scores)[1]
@assert maximum(abs.(foundation_score_manual-foundation_score_auto)) < 1e-12
@assert abs(sum(foundation_score_manual)) < 1e-12
println("Attention weights: ", foundation_attention)
println("Read vector: ", foundation_read)
println("Score gradients: ", foundation_score_manual)
foundation_ap = bar(["position 1", "position 2", "position 3"], foundation_attention;
    color=blue, label=false, ylabel="Attention weight", title="Current information mixture",
    ylims=(0, 0.7))
foundation_gp = bar(["score 1", "score 2", "score 3"], foundation_score_manual;
    color=[green, gold, green], label=false, ylabel="Local loss derivative",
    ylims=(-0.08, 0.15), title="A prediction gradient changes the routing")
hline!(foundation_gp, [0]; color="#777777", lw=1, label=false)
plot(foundation_ap, foundation_gp; layout=(1, 2), size=(1000, 310))
''')
    check("Attention learns through prediction",
          "Does a positive score gradient favor increasing that score? Are the attention weights permanent parameters?",
          "Gradient descent subtracts the gradient, so a positive derivative favors a lower score locally. Scores and weights are recomputed from inputs. Trainable query/key/value matrices receive gradients through those computations.")


def decoding_bridge(md, code, check):
    md(r"""
### 11a · Probabilities → one ID → readable text

The network outputs $V$ **scores**, not a word or an embedding to invert.
A generation rule selects an ID; tokenizer decoding reconstructs text from the IDs.
This reconstruction requires no second neural network.
For byte-based tokenizers, one token can contain only part of a Unicode character;
decode a sequence rather than assuming one token equals one displayed character.

For positive temperature $\tau$,

$$p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.$$

| Generation rule | What happens after the network supplies scores? |
|---|---|
| Greedy | Select $\arg\max_i z_i$ |
| Sampling | Draw an ID with probabilities $p_i(\tau)$ |
| Top-k | Retain the k highest scores, renormalize, then sample |
| Top-p / nucleus | Retain the smallest sorted prefix reaching cumulative mass p, renormalize, then sample |

Temperature changes concentration, not knowledge.
$\tau$ is distinct from sequence length $T$; a separate greedy option avoids division by zero.
Our teaching sampler offers greedy or temperature sampling.
The English sampler later uses **top-k=40**. Top-p is an isolated demonstration here.

Equivalently, a positive initial distribution transforms as $p_i(\tau)\propto p_i^{1/\tau}$.
Starting from $(0.8,0.2)$, temperatures $0.5,1,2$ give
approximately $(0.941,0.059)$, $(0.8,0.2)$, $(0.667,0.333)$.
Training uses the observed target directly in cross-entropy; it does not first sample
a candidate and backpropagate through that discrete draw.
""")
    code(r'''
foundation_candidate_words = ["rice", "noodles", "hotpot", "other"]
foundation_candidate_p = [0.40, 0.35, 0.20, 0.05]
foundation_candidate_logits = log.(foundation_candidate_p)
function foundation_top_p(probabilities, threshold)
    @assert 0 < threshold <= 1
    order = sortperm(probabilities; rev=true)
    count = min(searchsortedfirst(cumsum(probabilities[order]), threshold), length(order))
    keep = order[1:count]
    result = zeros(length(probabilities))
    result[keep] = probabilities[keep] ./ sum(probabilities[keep])
    result
end
foundation_nucleus = foundation_top_p(foundation_candidate_p, 0.9)
@assert findall(>(0), foundation_nucleus) == [1, 2, 3]
@assert isapprox(sum(foundation_nucleus), 1)
foundation_sampling_rng = MersenneTwister(42)
foundation_sampling_draws = [
    min(searchsortedfirst(cumsum(foundation_nucleus), rand(foundation_sampling_rng)), 4)
    for _ in 1:12]
println("Greedy ID/text: ", argmax(foundation_candidate_p), " / ",
    foundation_candidate_words[argmax(foundation_candidate_p)])
println("Sampled IDs: ", foundation_sampling_draws)
println("IDs decoded by vocabulary lookup: ",
    join(foundation_candidate_words[foundation_sampling_draws], " | "))
@assert all(in([1, 2, 3]), foundation_sampling_draws)
foundation_temperatures = [0.5, 1.0, 2.0]
foundation_temperature_plot = plot(xticks=(1:4, foundation_candidate_words),
    xlims=(0.4, 4.6), ylims=(0, 0.6), ylabel="Selection probability",
    title="Same scores, different generation settings", size=(900, 320))
for (index, tau) in enumerate(foundation_temperatures)
    bar!(foundation_temperature_plot, collect(1:4) .+ 0.24*(index-2),
        Flux.softmax(foundation_candidate_logits ./ tau);
        bar_width=0.22, label="τ=$(tau)", color=[blue, green, gold][index])
end
foundation_temperature_plot
''')
    check("Nucleus mass is a set threshold",
          "With probabilities 0.40, 0.35, 0.20, 0.05 and top-p=0.9, which candidates remain? Does each need probability 0.9?",
          "The first three remain: 0.40+0.35=0.75 and adding 0.20 reaches 0.95. The threshold applies to their combined original mass. Retained probabilities are then normalized to sum to one.")
    md(r"""
Greedy is locally optimal at each position, not necessarily optimal for the whole
sequence probability $\prod_t P(x_t\mid x_{<t})$.
High model probability also does not guarantee factual correctness.
For the research motivation behind nucleus sampling, see
[Holtzman et al.](https://arxiv.org/abs/1904.09751).
""")


def tokenizer_roundtrip(md, code):
    md(r"""
### 14a · Misspellings change pieces, not the reconstruction rule

Our English BPE can fall back to supported ASCII characters, so a new spelling need not
be a whole-word entry. Encoding `plya` still works; decoding restores `plya` rather
than silently correcting it. Whether the model infers "play" from context is a separate
question. This checks reconstruction, not typo understanding.

Chinese has no fixed characters-to-tokens conversion: a character can span byte pieces,
and several characters can share a token. Exact counts require the target tokenizer.
**The prepared English tokenizer here is ASCII-only**; this demo is not evidence of Chinese ability.
""")
    code(r'''
for foundation_spelling in ["I want to play football.", "I want to plya football."]
    foundation_spelling_ids = EnglishBasics.bpe_encode(tok, foundation_spelling)
    foundation_reconstructed = EnglishBasics.bpe_decode(tok, foundation_spelling_ids)
    @assert foundation_reconstructed == foundation_spelling
    println(repr(foundation_spelling), " → ", length(foundation_spelling_ids), " tokens")
    println("Pieces: ", repr.(tok.tokens[foundation_spelling_ids]))
    println("Decode: ", repr(foundation_reconstructed))
end
''')


def final_connections(md):
    md(r"""
## Connect the mechanism to larger LLM questions

| Question | Connection to the code you have run |
|---|---|
| Does ordinary generation keep training the model? | No optimizer step occurs in our samplers. Weights stay fixed while the sequence and states change. |
| Do intermediate reasoning steps require a new architecture? | The same autoregressive network can generate intermediate text and condition later tokens on it. This alone does not guarantee correct reasoning. |
| Why is exact arithmetic harder than plausible prose? | Cross-entropy rewards the observed token, rather than directly penalizing numerical distance. Approximate pattern fit provides no guarantee of an exact algorithm on unseen numbers. |
| Is larger context the same as wider embeddings? | Context changes the number of positions $T$; width changes coordinates per position $d$. |
| Can this English model read a million tokens at once? | No. Its learned position table and training setup support a 128-token window. The rolling sampler can write longer output while discarding earlier context; that does not enlarge the simultaneous window. |
| Why is full attention expensive with length? | Each head forms $T\times T$ scores. Doubling $T$ quadruples pairwise scores with width and heads fixed. |
| Does adding documents by RAG retrain embeddings? | Retrieval puts selected evidence into the input. Ordinary inference does not update this model's weights. |
| Is DAG another language model? | A directed acyclic graph describes dependencies, such as a workflow or computation graph. This is a different kind of concept from RAG. |

These are connections to the computation you have run, rather than claims about a closed
model's internal architecture or current product context limits.

The reference dialogue is archived at `reference/chat_records/rag_dag_transcript.md`.
Raw App pages and a six-page pagination verification accompany it.
The new explanations adapt its useful teaching sequence.
Historical product comparisons stay in the source record rather than being copied here as new facts.
""")
