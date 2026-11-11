"""Source cells for the sentiment-first English Julia curriculum."""

def opening(md, code, take, check, root):
    md(r"""
<div style="background:#142c43;color:white;padding:30px;border-radius:12px">
<div style="color:#8adecf;letter-spacing:2px;font-size:12px">AN ENGLISH JULIA SELF-STUDY COURSE</div>
<h1 style="color:white">From Sentiment Classification to Your Own GPT</h1>
<p style="font-size:18px">One small prediction. One complete gradient. Two real NLP projects.</p>
<p>Understand a review → train a classifier → predict tokens → build attention → train a language model</p>
</div>

**Start with a simpler question:** is this review positive or negative?
We will finish that entire learning problem before introducing next-token prediction.
All executable lesson cells use **Julia**; all neural models start from random parameters.
No previous NLP knowledge is required.
""")
    md(r"""
## Your self-study route

| Session | Work through | What you should be able to explain |
|---|---|---|
| [A · One review, one prediction](#sentiment-principles) | Tokenization, trainable embeddings, neural classifier, loss, full backpropagation | Which numbers are data, parameters, intermediate states, and gradients |
| [B · A real classification project](#sentiment-project) | Public IMDb reviews, train/validation/test, minibatches, training, mistakes | How a model learns from many labeled examples and how to evaluate it |
| [C · From classification to language modeling](#language-principles) | Shifted targets, positions, explicit Q/K/V products, causal attention, Transformer | How the same learning logic becomes a next-token model |
| [D · Train and inspect GPTs](#cpu-lab) | A transparent CPU training exercise, then the [real English project](#english-lab) | How to train from scratch, generate, and judge actual outputs |
| [Optional appendix](#recurrent-appendix) | RNN/LSTM and a field guide | Other ways to carry context, and what to try next |

Read the explanation before running the next cell. At each green question, answer in your own
words before opening the hint. Run cells in order; restart the kernel after changing type definitions.

**Default Run All:** train the IMDb classifier on CPU, train the tiny artificial-language GPT on CPU,
then inspect and sample from the supplied 5.82M-parameter English GPT.
Its separate 40-minute GPU run is controlled by §RUN_TRAINING=false§.
The IMDb flag §RUN_SENTIMENT_TRAINING=true§ can be switched off to inspect its saved run.
These are independent experiments, not successive fine-tuning stages.
""")
    md(r"""
## Setup · Julia throughout the learning workflow

Use the repository's **Julia 1.12** kernel, starting in the repository root or §notebooks/§.
The existing Flux/Plots environment is sufficient; no new Julia packages are required.
The prepared public datasets are included locally.

An offline Python/Arrow helper converts IMDb's public Parquet files into a simple text transport
format. The **tokenizer, vocabulary fitting, splitting, model, gradients, optimization, and evaluation
are Julia**. The existing English BPE corpus also has an optional offline preparation helper.
You do not need to write Python to study or train in this notebook.
""")
    take(3)
    code(r'''
if !isdefined(Main,:SentimentLab)
    include(joinpath(ROOT,"src","SentimentLab.jl"))
end
SENTIMENT_DATA=joinpath(ROOT,"reference","datasets","imdb")
SENTIMENT_OUT=joinpath(ROOT,"output","sentiment_imdb")
''')
    code(take(6,emit=False).split('flow_diagram(["Text',1)[0] + r'''
flow_diagram(["Review text\nreally good","Tokens → IDs\nreally, good\n[1, 3]",
              "Mean of learned\nword vectors","Neural network\none logit",
              "Output\nP(positive)"];
             title="Our first NLP task: one label for the whole review")
''')
    md('<a id="sentiment-principles"></a>\n# Part A · One review, one prediction')
    md(r"""
## 1 · Define the task before the network

**Natural Language Processing (NLP)** means computational work with human language.
Our first task is **binary sentiment classification**:

| Observed text $s$ | Supplied label $y$ |
|---|---|
| really good | 1: positive |
| very good | 1: positive |
| really bad | 0: negative |
| very bad | 0: negative |

The model predicts $\hat p=P_\theta(y=1\mid s)$, a probability between zero and one.
At a fixed threshold of 0.5, we report positive if $\hat p\geq0.5$, otherwise negative.
The label describes the **whole example**. We do not tell the tokenizer that "good" is positive
or "really" is neutral. The numerical relationship between words and labels must be learned.

These four artificial phrases make the mechanism legible. Part B uses real public movie reviews.
There is no neutral class in this particular task, and a review label is not a measure of the
writer's mental state or an objective judgment of the film.
""")
    md(r"""
## 2 · Tokenizer: a string becomes a sequence of addresses

A **token** is a unit chosen by a segmentation rule. It can be a word, word piece,
character, or byte. For this first classifier, choose lowercase words.
Keep negation such as §not§; discarding it would discard useful information.

For our four-word example, fix the vocabulary in this order:
$[\text{really},\text{very},\text{good},\text{bad}]$.
Then §Really good!§ becomes §["really","good"]§ and then IDs §[1,3]§.
Julia uses one-based indexing. ID 3 is a table address; it does not mean "more positive than ID 1."
The real classifier fits a larger vocabulary **on training reviews only**, with a reserved
§<unk>§ address for words outside that vocabulary.
""")
    code(r'''
sentiment_small_vocab=["really","very","good","bad"]
sentiment_small_lookup=Dict(w=>i for (i,w) in enumerate(sentiment_small_vocab))
sentiment_small_tokens=SentimentLab.words("Really good!")
sentiment_small_ids=[sentiment_small_lookup[w] for w in sentiment_small_tokens]
println("String: Really good!")
println("Tokens: ",sentiment_small_tokens)
println("IDs:    ",sentiment_small_ids)
@assert sentiment_small_ids==[1,3]
println("Negation remains a token: ",SentimentLab.words("This is not good."))
''')
    check("What has the tokenizer learned?",
          "Has the tokenizer assigned sentiment to good? Would a different integer ID change the meaning?",
          "No. It has only segmented and indexed text. Consistently relabeling IDs and table columns leaves the computation unchanged. The example-level labels train the neural parameters later.")
    md(r"""
## 3 · Embedding: the lookup table is part of what we train

Let vocabulary size be $V=4$ and choose width $d=2$.
An embedding table $E\in\mathbb R^{d\times V}$ contains one column per token:

$$E=\begin{bmatrix}0.2&-0.1&0.6&-0.3\\0.4&0.1&0.8&-0.2\end{bmatrix}.$$

The vector for token ID $j$ is $e_j=E[:,j]$. Thus §really good§ selects
$e_1=(0.2,0.4)^\top$ and $e_3=(0.6,0.8)^\top$, forming a $2\times2$ matrix.
We chose readable starting values here. The real project initializes these entries randomly.
**The labels never provide a "correct vector"; minimizing prediction loss will change the table.**

The width $d$ is a design choice, not something inferred automatically by looking at a token.
Coordinates can jointly encode useful patterns, but we do not label coordinate 1 "emotion"
and coordinate 2 "noun." Individual axes need not have stable human meanings.

Three distinct sizes: $V$ counts vocabulary entries, $d$ counts coordinates per token,
and $T$ counts tokens in this input. The input matrix has shape $d\times T$.
""")
    code(r'''
sentiment_small_E=[0.2 -0.1 0.6 -0.3;0.4 0.1 0.8 -0.2]
sentiment_small_H=sentiment_small_E[:,sentiment_small_ids]
println("Selected vectors, one column per token:\n",sentiment_small_H)
println("Mean embedding: ",vec(mean(sentiment_small_H;dims=2)))
heatmap(1:4,1:2,sentiment_small_E;xticks=(1:4,sentiment_small_vocab),
    yticks=(1:2,["coordinate 1","coordinate 2"]),clims=(-0.8,0.8),color=:RdBu,
    yflip=true,colorbar_title="",title="Four trainable vectors, not four sentiment labels")
''')
    md(r"""
## 4 · Forward pass: vectors → neural network → probability

Reviews have different lengths. Our simple classifier **averages** their vectors:

$$a=\frac1T\sum_{t=1}^T E[:,x_t]\in\mathbb R^d.$$

Now apply a hidden layer and a scalar output:

$$u=Wa+b,\quad h=\operatorname{ReLU}(u),\quad
z=v^\top h+c,\quad \hat p=\sigma(z)=\frac{1}{1+\exp(-z)}.$$

The logit $z$ is an unrestricted score; sigmoid turns it into a probability.
The following complete example keeps the same four words and the same embedding table.
""")
    take(14, transform=lambda s:s.replace("### 3a · A whole neural NLP model, small enough to audit",
        "### A complete numerical forward pass").replace(
        "In language modeling, the next token will supply the label instead.","")
        .replace("Averaging is our simple choice; GPT keeps a representation at every position.",
                 "Averaging is our simple choice. It discards word order, a limitation we will test."))
    take(15)
    md(r"""
## 5 · Loss: measure the error in a probabilistic prediction

For one observed label $y\in\{0,1\}$, **binary cross-entropy** is

$$L(\theta;s,y)=-y\log\hat p-(1-y)\log(1-\hat p).$$

For a positive review it is $-\log\hat p$; for a negative one it is $-\log(1-\hat p)$.
Confidently predicting the wrong label is expensive. A thresholded 0/1 accuracy score is useful
for reporting, but its flat regions make it unsuitable as this network's gradient-based loss.
For $N$ reviews, minimize $\mathcal L(\theta)=N^{-1}\sum_n L(\theta;s_n,y_n)$.

Our example has $y=1$, $\hat p\simeq0.8022$, and loss $\simeq0.2204$.
The scalar loss summarizes the prediction error. It is **derivatives of that loss**,
not the unchanged loss number, that propagate backward through the network.
In the practical code, §logitbinarycrossentropy§ computes the equivalent stable expression
directly from logits, so we do not take the log of rounded probabilities.
""")
    code(r'''
sentiment_probability_grid=collect(0.01:0.01:0.99)
sentiment_loss_plot=plot(sentiment_probability_grid,-log.(sentiment_probability_grid);
    label="Observed positive: y = 1",color=green,xlabel="Predicted P(positive)",
    ylabel="Binary cross-entropy",title="A wrong confident prediction receives a large loss")
plot!(sentiment_loss_plot,sentiment_probability_grid,-log1p.(-sentiment_probability_grid);
    label="Observed negative: y = 0",color=gold)
scatter!(sentiment_loss_plot,[foundation_forward_before.probability],
    [foundation_loss(foundation_parameters,foundation_ids,foundation_y)];
    color=blue,label="Our forward pass")
''')
    md(r"""
## 6 · Backpropagation: compute derivatives, then update parameters

Write $\delta_z=\partial L/\partial z=\hat p-y$.
For the positive example, $\delta_z\simeq-0.1978$.
The negative sign means increasing the logit would locally reduce the loss.

Backpropagation is repeated use of the **chain rule**. At each node we combine
an upstream derivative with that node's local derivative. If a parameter influences
the loss through several paths or repeated occurrences, add those contributions.
""")
    take(16, transform=lambda s:s.replace("### 3b ·","###"))
    take(17)
    take(18)
    md(r"""
### From one update to learning across examples

For token $j$ appearing $n_j$ times in a review of length $T$,

$$\frac{\partial L}{\partial E[:,j]}=
\frac{n_j}{T}\,W^\top\big[(\hat p-y)v\odot\mathbf1_{\{u>0\}}\big].$$

An unused column has zero gradient **for this example and this plain objective**.
Both §really§ and §good§ move in our positive example. Why does "really" not simply become
positive forever? It also appears with "bad"; gradients accumulate across differently labeled
examples. Shared parameters must compromise across the dataset.

For a minibatch $\mathcal B$, use the mean gradient
$g=|\mathcal B|^{-1}\sum_{n\in\mathcal B}\nabla_\theta L_n$.
Plain SGD applies $\theta\leftarrow\theta-\eta g$.
Our real run uses **Adam**, which keeps moving summaries of gradients and squared gradients
to scale parameter updates. The forward pass and chain-rule gradients are the same;
the update rule changes.

No labels, token IDs, or hidden activations are optimizer parameters. Activations are recomputed
on the next forward pass. Ordinary prediction after training performs no optimizer update.
""")
    code(r'''
flow_diagram(["Forward\nIDs → p","Loss\ncompare p with y",
              "Backward\nchain rule","Optimizer\nupdate parameters",
              "Repeat\nnext minibatch"];
    title="Training repeats prediction, differentiation, and a parameter update")
''')

def practical(md,code,take,check,root):
    md('<a id="sentiment-project"></a>\n# Part B · Train a real IMDb sentiment classifier')
    md(r"""
## 7 · A public dataset and an honest experimental split

The [Stanford IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
provides **25,000 labeled training reviews and 25,000 labeled test reviews**, balanced in
the original splits. Labels are negative (0) or positive (1), for strongly polarized English
movie reviews. We use the labeled splits; we do not use the extra unlabeled corpus.
The source card and download hashes are retained in §reference/datasets/imdb/§.
The card lists the license as §other§; the source is not described here as public domain.

Our preparation is reproducible and conservative:

1. Normalize with the classifier's lowercase word tokenizer; group identical token sequences.
2. Keep one copy of same-label duplicates; discard groups with conflicting labels.
3. Hold out 10% of the remaining official training pool, stratified by label, as **validation**.
4. Fit the 12,000-entry vocabulary on the training portion only. Map unseen words to §<unk>§.
5. Remove duplicate test sequences and any exact normalized overlap with the original training pool.

Training reviews update weights. Validation loss chooses the epoch. The resulting fixed model
is then scored on the filtered test set. Because we filter duplicates, our reported metric is
on this documented local split, not a directly comparable official benchmark score.
Exact deduplication does not prove absence of near duplicates or shared movie-specific patterns.
""")
    code(r'''
sentiment_data=SentimentLab.prepare_data(SENTIMENT_DATA)
for name in (:train,:validation,:test)
    split=getproperty(sentiment_data,name)
    @printf("%-10s %6d reviews | positive %5.1f%% | unknown tokens %5.2f%%\n",
        string(name),length(split.rows),100mean(split.y),100split.unknown_rate)
end
println("Vocabulary entries: ",length(sentiment_data.vocab))
println("Excluded from source train/test: ",
    sentiment_data.metadata.excluded_train," / ",sentiment_data.metadata.excluded_test)
println("Split seed: ",sentiment_data.metadata.seed)
''')
    check("Which split may choose the epoch?",
          "If test accuracy falls after epoch 5, may we use that fact to select epoch 4?",
          "Use validation to select an epoch. Repeatedly selecting with test results turns test into another validation set. Our loop selects by validation cross-entropy before computing its test score.")
    md(r"""
### Inspect one review before fitting the model

The label is attached to the full review, not to individual words.
The next cell prints a short excerpt, its token sequence, and IDs.
The classifier uses the full tokenized review: there is no truncation or padding in this mean-pooling baseline.
""")
    code(r'''
sentiment_example=sentiment_data.train.rows[1]
println("Observed label: ",sentiment_example.y==1 ? "positive" : "negative")
println("Excerpt: ",first(replace(sentiment_example.text,r"<[^>]*>"=>" "),220))
println("First tokens: ",first(SentimentLab.words(sentiment_example.text),16))
println("First IDs:    ",first(sentiment_data.train.encoded[1],16))
''')
    md(r"""
## 8 · The same classifier, computed efficiently for a batch

Use $V=12{,}000$, embedding width $d=48$, and 32 hidden neurons.
The parameters are still $(E,W,b,v,c)$; there are no pretrained embeddings.

Let $F_{jn}=n_{jn}/T_n$ be token $j$'s frequency in review $n$ divided by its length.
Most entries are zero, so we store $F$ as a **sparse matrix**.
Then

$$A=EF,\quad H=\operatorname{ReLU}(WA+b\mathbf1^\top),\quad
Z=v^\top H+c\mathbf1^\top.$$

This is exactly the mean of the looked-up embeddings, reorganized into a fast multiplication.
It is not an additional learned feature extractor and it does not add word order.
For a batch of $B$ reviews, shapes are $F:V\times B$, $A:d\times B$, $H:32\times B$,
and $Z:1\times B$. The implementation stores the output weight as a $1\times32$ row.
""")
    model=(root/"src/SentimentLab.jl").read_text(encoding="utf-8").split("# BEGIN MODEL",1)[1]
    model=model.split("\n",1)[1].split("# END MODEL",1)[0]
    code(model.replace("initialize(","review_initialize(").replace("logits(","review_logits(")
         .replace("loss(m,F,y) =","review_loss(m,F,y) =").strip())
    code(r'''
sentiment_initial=review_initialize(length(sentiment_data.vocab))
sentiment_probe_ids=sentiment_data.train.encoded[1]
sentiment_lookup_mean=vec(mean(sentiment_initial.E[:,sentiment_probe_ids];dims=2))
sentiment_sparse_mean=vec(sentiment_initial.E*sentiment_data.train.X[:,1:1])
@assert isapprox(sentiment_lookup_mean,sentiment_sparse_mean;atol=1f-6)
@assert review_logits(sentiment_initial,sentiment_data.train.X[:,1:4]) ≈
        SentimentLab.logits(sentiment_initial,sentiment_data.train.X[:,1:4])
println("Parameters: ",sum(length,values(sentiment_initial)))
println("Four review logits: ",vec(review_logits(sentiment_initial,sentiment_data.train.X[:,1:4])))
println("Sparse multiplication matches explicit lookup-and-mean.")
''')
    md(r"""
### One real minibatch: inspect before updating

This cell differentiates 32 real reviews and checks that gradients reach the embedding table.
The demonstration copy takes one plain SGD step; the full training run below starts fresh and uses Adam.
""")
    code(r'''
sentiment_batch_F=sentiment_data.train.X[:,1:32]
sentiment_batch_y=sentiment_data.train.y[:,1:32]
sentiment_batch_loss,sentiment_batch_gradient=Flux.withgradient(
    m->review_loss(m,sentiment_batch_F,sentiment_batch_y),sentiment_initial)
@assert norm(sentiment_batch_gradient[1].E)>0
sentiment_one_step=map((p,g)->p.-0.1f0.*g,sentiment_initial,sentiment_batch_gradient[1])
@printf("Batch CE before / after one SGD step: %.6f / %.6f\n",
    sentiment_batch_loss,review_loss(sentiment_one_step,sentiment_batch_F,sentiment_batch_y))
@printf("Embedding gradient norm: %.6f\n",norm(sentiment_batch_gradient[1].E))
''')
    md(r"""
## 9 · Train: minibatch gradients, Adam, validation, and a saved checkpoint

Read the loop before running it. Each epoch shuffles the training reviews.
For each minibatch, §withgradient§ computes the forward loss and its parameter derivatives;
§update!§ uses those derivatives. Evaluation makes predictions without updating weights.

We plan eight epochs, batch size 128, and learning rate 0.003.
There is a ten-minute **fit-function budget checked between updates**;
compilation, dataset preparation, and final evaluation can add time.
The measured duration is reported below rather than assumed from the budget.
The best validation-loss checkpoint is retained even if later training loss keeps improving.
""")
    source=(root/"src/SentimentLab.jl").read_text(encoding="utf-8")
    training=source.split("# BEGIN TRAINING",1)[1].split("\n",1)[1].split("# END TRAINING",1)[0]
    training=training.replace("function fit(","function train_review_classifier(")
    training=training.replace("initial=initialize(","initial=review_initialize(")
    training=training.replace("evaluate(","SentimentLab.evaluate(").replace("m->loss(","m->review_loss(")
    code(training.strip())
    code(r'''
RUN_SENTIMENT_TRAINING=true
if RUN_SENTIMENT_TRAINING
    sentiment_run=train_review_classifier(sentiment_data)
    SentimentLab.save_run(sentiment_run,SENTIMENT_OUT)
else
    sentiment_run=deserialize(joinpath(SENTIMENT_OUT,"run.jls"))
end
@assert sentiment_run.vocab==sentiment_data.vocab
@assert sentiment_run.data_metadata==sentiment_data.metadata
@printf("Selected epoch %d | fit time %.1f seconds | optimizer updates %d\n",
    sentiment_run.best_epoch,sentiment_run.elapsed_seconds,sentiment_run.steps)
''')
    md(r"""
## 10 · Evaluate performance, and look at errors

**Accuracy** is the fraction of labels predicted correctly at the fixed 0.5 threshold.
The **confusion matrix** counts actual labels in rows and predictions in columns.
**Positive-class F1** is $2TP/(2TP+FP+FN)$; unlike loss, it depends on the threshold.
A majority-label rule is a simple baseline. It learns its chosen label from training only.
""")
    code(r'''
sentiment_hist=sentiment_run.history
sentiment_epochs=[h.epoch for h in sentiment_hist]
sentiment_curve=plot(sentiment_epochs,[h.train_loss for h in sentiment_hist];
    label="Train",color=blue,xlabel="Epoch",ylabel="Binary cross-entropy",
    title="Select the checkpoint with validation loss")
plot!(sentiment_curve,sentiment_epochs,[h.validation_loss for h in sentiment_hist];
    label="Validation",color=green)
vline!(sentiment_curve,[sentiment_run.best_epoch];label="Selected epoch",color=gold,linestyle=:dash)
sentiment_acc_plot=plot(sentiment_epochs,100 .* [h.train_accuracy for h in sentiment_hist];
    label="Train",color=blue,xlabel="Epoch",ylabel="Accuracy (%)",ylims=(40,100),
    title="Better training fit need not mean better generalization")
plot!(sentiment_acc_plot,sentiment_epochs,100 .* [h.validation_accuracy for h in sentiment_hist];
    label="Validation",color=green)
plot(sentiment_curve,sentiment_acc_plot;layout=(1,2),size=(1100,380))
''')
    code(r'''
sentiment_test=sentiment_run.test
sentiment_majority=mean(sentiment_data.train.y)>=0.5
sentiment_baseline=mean((vec(sentiment_data.test.y).>=0.5).==sentiment_majority)
@printf("Filtered test reviews: %d\n",length(sentiment_data.test.rows))
@printf("Training-majority baseline: %.2f%% | classifier: %.2f%% | positive F1: %.4f\n",
    100sentiment_baseline,100sentiment_test.accuracy,sentiment_test.f1)
@printf("Test cross-entropy: %.4f\n",sentiment_test.loss)
sentiment_cm=sentiment_test.confusion
sentiment_cm_plot=heatmap(1:2,1:2,sentiment_cm;xticks=(1:2,["negative","positive"]),
    yticks=(1:2,["negative","positive"]),yflip=true,color=:Blues,
    xlabel="Predicted label",ylabel="Actual label",title="Held-out confusion matrix",
    colorbar_title="",size=(650,420))
for actual in 1:2,predicted in 1:2
    annotate!(sentiment_cm_plot,predicted,actual,text(string(sentiment_cm[actual,predicted]),
        14,sentiment_cm[actual,predicted]>maximum(sentiment_cm)/2 ? :white : :black))
end
sentiment_cm_plot
''')
    md(r"""
### Read fixed-order successes and failures

The next cell selects the **first two correctly classified and first two incorrectly classified**
test reviews in source order. It prints short excerpts, not cherry-picked model demonstrations.
The displayed probability describes the full review, which can contain mixed opinions.
""")
    code(r'''
sentiment_correct=(sentiment_test.probabilities.>=0.5).==(vec(sentiment_data.test.y).>=0.5)
for outcome in (true,false)
    println(outcome ? "\nCORRECT predictions" : "\nINCORRECT predictions")
    for i in first(findall(==(outcome),sentiment_correct),2)
        row=sentiment_data.test.rows[i]
        @printf("Source test row %d | label %d | P(positive) %.3f\n",
            row.id,row.y,sentiment_test.probabilities[i])
        println(first(replace(row.text,r"<[^>]*>"=>" "),250),"\n")
    end
end
''')
    md(r"""
### Try your own reviews; then expose a known limitation

Edit the strings below. These short instructor-written probes are not a second benchmark.
Their style differs from long IMDb reviews, so the classifier may be unreliable on them.
The final pair contains **the same words in different orders**:
§The movie is good, not bad.§ and §The movie is bad, not good.§
Our mean-pooling classifier must give them the same probability, despite the different meanings.
Probabilities are not calibrated confidence guarantees; displayed values are rounded.
""")
    code(r'''
sentiment_try=["I loved this wonderful movie. The acting was excellent.",
    "This movie was boring and disappointing. I hated it.",
    "The movie is good, not bad.","The movie is bad, not good."]
for review in sentiment_try
    result=SentimentLab.predict_review(sentiment_run,review)
    @printf("P(positive)=%.3f | %-8s | unknown %.1f%% | %s\n",
        result.probability_positive,result.label,100result.unknown_rate,review)
end
@assert SentimentLab.predict_review(sentiment_run,sentiment_try[3]).probability_positive ≈
        SentimentLab.predict_review(sentiment_run,sentiment_try[4]).probability_positive
sentiment_words_to_show=["good","bad","excellent","boring","not","movie"]
sentiment_ids_to_show=[findfirst(==(w),sentiment_run.vocab) for w in sentiment_words_to_show]
@assert all(!isnothing,sentiment_ids_to_show)
sentiment_movement=[norm(sentiment_run.model.E[:,i]-sentiment_run.initial.E[:,i])
    for i in sentiment_ids_to_show]
@assert maximum(sentiment_movement)>0
bar(sentiment_words_to_show,sentiment_movement;label=false,color=blue,
    ylims=(0,1.15maximum(sentiment_movement)),
    ylabel="Distance from random initial vector",title="Review labels trained the embedding table")
''')
    check("What did we actually achieve?",
          "Does good test accuracy show that the model understands word order? Did it learn by predicting the next token?",
          "No to both. It learned review-level labels using a permutation-invariant mean of embeddings. The identical-word-order pair proves a limitation even if test accuracy is good. Next-token prediction is the new task in Part C.")
    md(r"""
### Your first completed NLP project

You now have the full workflow: define an observed target, split data, fit a tokenizer vocabulary,
initialize a network, compute a differentiable loss, update parameters, select on validation,
and inspect test errors. Keep that structure as we change the prediction task.

The saved run is in §output/sentiment_imdb/§. Reproduce it outside the notebook with
§julia --startup-file=no --project=. experiments/train_sentiment.jl§.
To experiment, change one setting such as embedding width or vocabulary size and inspect validation
behavior. Reserve a fresh final evaluation if you repeatedly redesign the model after seeing test results.
""")
