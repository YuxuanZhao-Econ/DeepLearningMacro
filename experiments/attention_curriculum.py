"""Explicit column-oriented Q/K/V derivations and numerical gradient checks."""

def language_bridge(md,code,take,check):
    md('<a id="language-principles"></a>\n# Part C · Change the task: predict the next token')
    md(r"""
## 11 · A language model is a repeated classification problem

Our sentiment model used a supplied review label. A language model uses the **next observed token**
as its target. This is called self-supervised learning: the text itself provides training targets.
It does not require a sentiment annotation for every sentence.

| Ingredient | Review classification | Next-token language modeling |
|---|---|---|
| Observed input | Whole review | A prefix of token IDs |
| Target | One label, 0 or 1 | Next token ID, from $1,\ldots,V$ |
| Output | One logit → sigmoid | $V$ logits → softmax |
| Loss | Binary cross-entropy | Categorical cross-entropy |
| Parameters learned | Embeddings + classifier weights | Embeddings + context network + output head |
| Prediction after training | Report label/probability | Select a token, append it, and predict again |

For §I love cats .§, a boundary-aware training sequence supplies

| Input prefix | Target |
|---|---|
| §<bos>§ | I |
| §<bos> I§ | love |
| §<bos> I love§ | cats |
| §<bos> I love cats§ | . |
| §<bos> I love cats .§ | §<eos>§ |

The whole-sequence probability factors as a product of conditional probabilities:

$$P_\theta(x_1,\ldots,x_T)=\prod_{t=1}^T P_\theta(x_t\mid x_{<t}).$$

For a state $h_t$ representing the observed prefix,

$$z_t=Uh_t+b_{\rm out},\qquad
p_t[k]=\frac{\exp z_t[k]}{\sum_{j=1}^V\exp z_t[j]},\qquad
L_t=-\log p_t[y_t],\quad
\frac{\partial L_t}{\partial z_t}=p_t-o_{y_t}.$$

This is the multiclass counterpart of $\hat p-y$ in sentiment classification.
The total loss averages over target positions and batch examples.
We will train a **small GPT-style language model**; "large" describes scale, not a different loss formula.
This notebook's models do not have a frontier LLM's general capabilities.
""")
    take(20)
    take(21)
    take(22)
    take(23)
    md(r"""
## 12 · Preserve positions before mixing context

The review classifier's average cannot distinguish reordering. For a GPT, keep one state per
position, $X=[x_1,\ldots,x_T]\in\mathbb R^{d\times T}$, where
$x_t=E[:,i_t]+P[:,t]$. Here $i_t$ is the token ID, $E$ the token table, and $P$ the position table.

The same token selects the same initial embedding column. Its contextual state can differ because
its position and neighboring tokens differ. Position vectors are trainable too.
The position-aware sequence goes into attention; we do not average it into a single review vector.
""")
    take(26)
    take(27)
    md(r"""
## 13 · Q, K, V: three learned projections of the same states

Consider one sequence, one attention head, and **column vectors for positions**.
Use $d_k$ coordinates for queries/keys and $d_v$ coordinates for values:

$$
Q=W_QX,\qquad K=W_KX,\qquad \mathsf V=W_VX.
$$

| Object | Shape | What it is |
|---|---|---|
| $X$ | $d\times T$ | Current input states, one column per position |
| $W_Q,W_K$ | $d_k\times d$ each | Shared trainable parameter matrices |
| $W_V$ | $d_v\times d$ | Shared trainable parameter matrix |
| $Q,K$ | $d_k\times T$ each | Computed query and key vectors |
| $\mathsf V$ | $d_v\times T$ | Computed value vectors |

The script letter $\mathsf V$ means the **value matrix**; the scalar $V$ elsewhere is vocabulary size.
For one position, $q_i=W_Qx_i$ asks what to match; $k_j=W_Kx_j$ describes how to match position $j$;
$v_j=W_Vx_j$ is the information to mix. These phrases are intuition, not preassigned semantic labels.
The same $W_Q$ is reused at every position. $Q$ changes with the input; $W_Q$ changes only when trained.
We omit projection biases in this hand calculation. They are optional learned additions.

Take $d=d_k=d_v=2$, $T=3$:

$$X=\begin{bmatrix}1&0&1\\0&1&1\end{bmatrix},\
W_Q=\begin{bmatrix}1&0\\0&2\end{bmatrix},\
W_K=\begin{bmatrix}1&1\\0&1\end{bmatrix},\
W_V=\begin{bmatrix}1&0\\1&-1\end{bmatrix}.$$

For the third position, $x_3=(1,1)^\top$, giving $q_3=(1,2)^\top$,
$k_3=(2,1)^\top$, and $v_3=(1,0)^\top$.
The code computes every column from the projection matrices, rather than assuming Q/K/V are given.
""")
    code(r'''
attention_words=["position 1","position 2","position 3"]
qkv_X=[1.0 0 1;0 1 1]
qkv_WQ=[1.0 0;0 2]
qkv_WK=[1.0 1;0 1]
qkv_WV=[1.0 0;1 -1]
Q_demo=qkv_WQ*qkv_X
K_demo=qkv_WK*qkv_X
V_demo=qkv_WV*qkv_X
for (name,matrix) in [("X",qkv_X),("W_Q",qkv_WQ),("W_K",qkv_WK),
                      ("W_V",qkv_WV),("Q",Q_demo),("K",K_demo),("Values",V_demo)]
    println(name," =\n",matrix)
end
@assert Q_demo[:,3]==[1,2] && K_demo[:,3]==[2,1] && V_demo[:,3]==[1,0]
function qkv_panel(matrix,title)
    p=heatmap(1:size(matrix,2),1:size(matrix,1),matrix;color=:RdBu,clims=(-2,2),
        xticks=1:size(matrix,2),yticks=1:size(matrix,1),colorbar=false,
        xlabel="Position",ylabel="Coordinate",title=title,yflip=true)
    for j in axes(matrix,2),i in axes(matrix,1)
        annotate!(p,j,i,text(string(Int(matrix[i,j])),13,
            abs(matrix[i,j])>1.5 ? :white : :black))
    end
    p
end
plot(qkv_panel(qkv_X,"Input X"),qkv_panel(Q_demo,"Q = W_Q X"),
     qkv_panel(K_demo,"K = W_K X"),qkv_panel(V_demo,"Values = W_V X");
     layout=(1,4),size=(1120,300))
''')
    md(r"""
## 14 · Scores → probabilities over positions → weighted values

Compute all query/key comparisons:

$$S=\frac{K^\top Q}{\sqrt{d_k}}\in\mathbb R^{T\times T},\qquad
S_{ji}=\frac{k_j^\top q_i}{\sqrt{d_k}}.$$

**Rows are keys; columns are queries.** Dividing by $\sqrt{d_k}$ controls the scale
of dot products as the number of terms grows. Under a simple unit-variance,
independent-coordinate argument, the unscaled sum's variance grows with $d_k$.

Normalize each column over eligible keys:

$$A_{ji}=\frac{\exp(S_{ji}+M_{ji})}
{\sum_{\ell=1}^T\exp(S_{\ell i}+M_{\ell i})},\qquad
O=\mathsf V A,\qquad o_i=\sum_jA_{ji}v_j.$$

For now set $M=0$ so every position is eligible. For query 3, scores are
$(1,3,4)^\top/\sqrt 2$. Softmax creates nonnegative weights summing to one.
The output is a **weighted sum of value vectors**, not a sum of the IDs or of the scores.
The next section adds the causal restriction.

You may see $QK^\top$ in books: they often store positions as **rows**.
That formula and ours agree after consistently transposing the representation.
Do not mix the two conventions.
""")
    code(r'''
scores_demo=K_demo'*Q_demo/sqrt(2.0)
weights_demo=Flux.softmax(scores_demo;dims=1)
mixed_demo=V_demo*weights_demo
@assert vec(sum(weights_demo;dims=1)) ≈ ones(3)
@assert mixed_demo[:,3] ≈ sum(weights_demo[j,3].*V_demo[:,j] for j in 1:3)
println("Scores for query 3: ",round.(scores_demo[:,3];digits=4))
println("Weights for query 3: ",round.(weights_demo[:,3];digits=4))
println("Output at query 3: ",round.(mixed_demo[:,3];digits=4))
qkv_weights_plot=heatmap(1:3,1:3,weights_demo;clims=(0,1),color=:Blues,
    xticks=1:3,yticks=1:3,yflip=true,xlabel="Query position",ylabel="Key position",
    title="Each column is one distribution over keys")
qkv_query_plot=bar(attention_words,weights_demo[:,3];label=false,color=blue,
    ylabel="Attention weight",ylims=(0,1),title="The third query's mixture")
plot(qkv_weights_plot,qkv_query_plot;layout=(1,2),size=(1000,360))
''')
    check("Parameters versus computed matrices",
          "If I change the input sentence without training, which can change: W_Q, Q, A? Which axis must sum to one?",
          "Q and A can change immediately; W_Q stays fixed until an optimizer step. Keys are rows in this notebook, so every query column of A sums to one.")
    take(39,transform=lambda s:s.replace("## 6.","## 15 ·"))
    take(40)
    take(41)
    md(r"""
## 16 · How the next-token loss trains Q/K/V projections

No dataset needs to label "the correct attention weights."
The next-token prediction error flows backward through the output, attention, and projections.
Here is the full one-head chain. Let $G_Z=\partial L/\partial Z$ for any matrix $Z$,
and suppose $G_O$ is supplied by the later network:

$$
\begin{aligned}
G_{\mathsf V}&=G_OA^\top,&G_A&=\mathsf V^\top G_O,\\
(G_S)_{ji}&=A_{ji}\left[(G_A)_{ji}-\sum_\ell A_{\ell i}(G_A)_{\ell i}\right],\\
G_Q&=KG_S/\sqrt{d_k},&G_K&=QG_S^\top/\sqrt{d_k},\\
G_{W_Q}&=G_QX^\top,&G_{W_K}&=G_KX^\top,&G_{W_V}&=G_{\mathsf V}X^\top,\\
G_X&=W_Q^\top G_Q+W_K^\top G_K+W_V^\top G_{\mathsf V}.
\end{aligned}
$$

The middle line differentiates softmax **within each column**.
Forbidden entries have $A_{ji}=0$, so their score derivatives are zero.
The three terms in $G_X$ add because the same input contributes through three projections.
If $X[:,i]=E[:,x_i]+P[:,i]$, its derivative flows into both selected columns,
adding contributions when a token repeats.

For a numerical check, attach a three-candidate output head $z=Uo_3$ and target $y=2$.
Then $g_z=\operatorname{softmax}(z)-o_2$, $G_U=g_z o_3^\top$,
and only column 3 of $G_O$ is $U^\top g_z$.
We calculate every derivative by hand-coded matrix algebra and compare with Flux AD.
The fixed values below are an arithmetic example, not pretrained attention.
""")
    code(r'''
qkv_parameters=(X=qkv_X,WQ=qkv_WQ,WK=qkv_WK,WV=qkv_WV,
    U=[1.0 -1;0.5 1;-1 0.3])
qkv_mask=Float64.(causal_bias(3))
function qkv_next_token_loss(p)
    Q,K,values=p.WQ*p.X,p.WK*p.X,p.WV*p.X
    A=Flux.softmax(K'*Q/sqrt(2.0).+qkv_mask;dims=1)
    z=p.U*(values*A)[:,3]
    -Flux.logsoftmax(z)[2]
end
qkv_A=Flux.softmax(scores_demo.+qkv_mask;dims=1)
qkv_O=V_demo*qkv_A
qkv_z=qkv_parameters.U*qkv_O[:,3]
qkv_gz=Flux.softmax(qkv_z)-[0.0,1.0,0.0]
qkv_gO=zeros(2,3)
qkv_gO[:,3]=qkv_parameters.U'*qkv_gz
qkv_gV=qkv_gO*qkv_A'
qkv_gA=V_demo'*qkv_gO
qkv_gS=qkv_A.*(qkv_gA.-sum(qkv_A.*qkv_gA;dims=1))
qkv_gQ=K_demo*qkv_gS/sqrt(2.0)
qkv_gK=Q_demo*qkv_gS'/sqrt(2.0)
qkv_manual=(X=qkv_WQ'*qkv_gQ+qkv_WK'*qkv_gK+qkv_WV'*qkv_gV,
    WQ=qkv_gQ*qkv_X',WK=qkv_gK*qkv_X',WV=qkv_gV*qkv_X',
    U=qkv_gz*qkv_O[:,3]')
qkv_auto=Flux.gradient(qkv_next_token_loss,qkv_parameters)[1]
qkv_gradient_error=maximum(maximum(abs.(getproperty(qkv_manual,k).-getproperty(qkv_auto,k)))
    for k in keys(qkv_parameters))
@assert qkv_gradient_error<1e-10
@assert all(iszero,qkv_gS[tril(trues(3,3),-1)])
qkv_updated=map((p,g)->p.-0.01.*g,qkv_parameters,qkv_auto)
@assert qkv_next_token_loss(qkv_updated)<qkv_next_token_loss(qkv_parameters)
@printf("Manual vs AD max error: %.3e\n",qkv_gradient_error)
@printf("Next-token loss before / after: %.6f / %.6f\n",
    qkv_next_token_loss(qkv_parameters),qkv_next_token_loss(qkv_updated))
for key in (:WQ,:WK,:WV,:X)
    println("Gradient ",key," =\n",getproperty(qkv_manual,key))
end
qkv_grad_sizes=[norm(getproperty(qkv_auto,k)) for k in (:WQ,:WK,:WV,:X)]
bar(["W_Q","W_K","W_V","Input X"],qkv_grad_sizes;label=false,color=blue,
    ylims=(0,1.15maximum(qkv_grad_sizes)),
    ylabel="Gradient norm",title="One next-token error reaches all three projection matrices")
''')
    check("What is backpropagated?",
          "Do we send the unchanged loss number into W_Q, W_K, and W_V?",
          "No. We compute derivatives of the loss by the chain rule. The input gradient adds three paths. An optimizer uses parameter gradients to change W_Q/W_K/W_V; Q/K/values are recomputed on the next forward pass.")
    md(r"""
## 17 · From one head to a GPT block

With $H$ heads, usually choose $d_h=d/H$. Each head has separate learned projections and produces
$O^{(h)}\in\mathbb R^{d_h\times T}$. Stack those outputs along the coordinate axis and mix them:

$$\operatorname{MHA}(X)=W_O
\begin{bmatrix}O^{(1)}\\ \vdots\\ O^{(H)}\end{bmatrix},\qquad W_O\in\mathbb R^{d\times d}.$$

A **pre-normalization** Transformer block in our implementation is

$$
R=X+\operatorname{MHA}(\operatorname{LN}_1(X)),\qquad
X_{\rm next}=R+\operatorname{MLP}(\operatorname{LN}_2(R)).
$$

Layer normalization rescales coordinates within each position; residual addition preserves an
existing path through the network; the MLP applies learned nonlinear transformations at every position.
The MLP shares weights across positions but does not mix positions itself. Attention does that mixing.

Stack blocks, apply a final layer norm and vocabulary head, and compute next-token cross-entropy.
Gradients reach all blocks, the token/position embeddings, and the output head.
The code in Part D unfolds the batched multi-head shapes and runs this entire training loop.

For the original attention architecture, see
[Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762).
Our GPT uses causal self-attention and a pre-normalization block, rather than reproducing
the paper's full encoder–decoder translator.
""")
    take(42)
    take(43)
    take(44)
    take(45)
