# %% setup
using Pkg
function course_root(start=pwd())
    dir=abspath(start)
    while !isfile(joinpath(dir,"Project.toml"))
        parent=dirname(dir)
        parent==dir && error("Start in the supplied project, or set ROOT to its location.")
        dir=parent
    end
    dir
end
ROOT=course_root()
if Base.active_project()===nothing || normpath(Base.active_project())!=joinpath(ROOT,"Project.toml")
    Pkg.activate(ROOT)
end
ENV["GKSwstype"]="100"
using Flux, Random, Statistics, LinearAlgebra, SparseArrays, Serialization, Printf, SHA, Plots, LaTeXStrings
BLAS.set_num_threads(1)
IMDB_DIR=joinpath(ROOT,"reference","datasets","imdb")
STORY_DIR=joinpath(ROOT,"reference","datasets","english_basics")
RESULTS=joinpath(ROOT,"output","nlp_course")
mkpath(RESULTS)
INK,TEAL,AMBER,GRAY="#253744","#287C95","#C27B38","#89949C"
DIV=cgrad(["#245B78","#FAFAF8","#B87532"])
SEQ=cgrad(["#F7FAFC","#AACDD9","#245B78"])
default(fontfamily="Helvetica",size=(980,440),dpi=160,linewidth=2.4,
    titlefontsize=13,guidefontsize=11,tickfontsize=10,legendfontsize=10,
    foreground_color=INK,background_color=:white,framestyle=:axes,
    grid=:y,gridalpha=0.10,legend_foreground_color=:transparent,
    left_margin=7Plots.mm,right_margin=7Plots.mm,top_margin=6Plots.mm,bottom_margin=9Plots.mm)
function matrix_panel(M;title="",rowlabels=nothing,xlabel="Coordinate",limit=maximum(abs,M),
                      signed=true,colorbar=true)
    ticks=rowlabels===nothing ? :auto : (1:length(rowlabels),rowlabels)
    heatmap(1:size(M,2),1:size(M,1),M;title,xlabel,yticks=ticks,yflip=true,
        color=signed ? DIV : SEQ,clims=signed ? (-max(limit,1e-8),max(limit,1e-8)) : (0,limit),
        colorbar,grid=false,framestyle=:box,foreground_color_axis=GRAY,tick_direction=:out)
end
println("Julia ",VERSION," · Flux ",pkgversion(Flux)," · all course functions are defined below")

# %% tokenizer
word_tokens(s)=String[m.match for m in eachmatch(r"[a-z]+(?:'[a-z]+)?",
    lowercase(replace(s,r"<[^>]*>"=>" ")))]
word_ids(text,lookup)=begin
    words=word_tokens(text)
    isempty(words) ? [1] : [get(lookup,w,1) for w in words]
end

# %% read_imdb
function read_reviews(path)
    [begin
        id,y,s=split(line,'\t')
        (;id=parse(Int,id),y=parse(Int,y),text=String(hex2bytes(s)))
    end for line in eachline(path)]
end
function prepare_imdb(dir;max_vocab=20000,seed=31415)
    original=read_reviews(joinpath(dir,"train.tsv"))
    ws=word_tokens.(getproperty.(original,:text))
    groups=Dict{String,Vector{Int}}()
    for (i,key) in enumerate(join.(ws," "))
        push!(get!(groups,key,Int[]),i)
    end
    kept=sort([first(g) for g in values(groups) if length(unique(original[i].y for i in g))==1])
    pool,pool_ws=original[kept],ws[kept]
    train_at,val_at=Int[],Int[];rng=MersenneTwister(seed)
    for label in (0,1)
        ids=shuffle(rng,findall(r->r.y==label,pool))
        n=round(Int,0.1length(ids))
        append!(val_at,ids[1:n]);append!(train_at,ids[n+1:end])
    end
    sort!(train_at);sort!(val_at)
    counts=Dict{String,Int}()
    for i in train_at,w in pool_ws[i]
        counts[w]=get(counts,w,0)+1
    end
    ranked=sort(collect(keys(counts));by=w->(-counts[w],w))
    vocab=vcat(["<unk>"],ranked[1:min(max_vocab-1,length(ranked))])
    lookup=Dict(w=>i for (i,w) in enumerate(vocab))
    function pack(rows,tokenlists)
        ids=[isempty(w) ? [1] : [get(lookup,t,1) for t in w] for w in tokenlists]
        (;rows,ids,y=reshape(Float32[r.y for r in rows],1,:))
    end
    training=pack(pool[train_at],pool_ws[train_at])
    validation=pack(pool[val_at],pool_ws[val_at])
    test_original=read_reviews(joinpath(dir,"test.tsv"))
    test_ws=word_tokens.(getproperty.(test_original,:text))
    test_groups=Dict{String,Vector{Int}}()
    for (i,key) in enumerate(join.(test_ws," "))
        push!(get!(test_groups,key,Int[]),i)
    end
    seen=Set(join.(ws," "))
    test_at=sort([first(g) for (k,g) in test_groups if k ∉ seen &&
                  length(unique(test_original[i].y for i in g))==1])
    test=pack(test_original[test_at],test_ws[test_at])
    @assert isempty(intersect(Set(join.(pool_ws[train_at]," ")),Set(join.(pool_ws[val_at]," "))))
    (;training,validation,test,vocab,lookup,seed)
end
imdb=prepare_imdb(IMDB_DIR)
for name in (:training,:validation,:test)
    data=getproperty(imdb,name)
    @printf("%-11s %6d reviews | positive %.1f%%\n",string(name),length(data.rows),100mean(data.y))
end

V_review=length(imdb.vocab)
@assert V_review==20000
# Choose by length in source order, never by the model's prediction.
trace_at=findfirst(i->80<=length(imdb.training.ids[i])<=160,
    eachindex(imdb.training.rows))
trace_row=imdb.training.rows[trace_at]
trace_ids=imdb.training.ids[trace_at]
trace_tokens=word_tokens(trace_row.text)
T_review=length(trace_ids)
@printf("\nTraining-source row %d | %d tokens | supplied label %d\n",
    trace_row.id,T_review,trace_row.y)
excerpt=join(first(split(replace(trace_row.text,r"<[^>]*>"=>" ")),30)," ")
println("Review excerpt (all tokens are still used): ",excerpt," …")
println("\nFirst 16 tokens and vocabulary IDs (the model uses all tokens):")
for t in 1:16
    @printf("%2d  %-18s  ID %5d  table entry: %s\n",
        t,trace_tokens[t],trace_ids[t],imdb.vocab[trace_ids[t]])
end

# %% classifier_parameters
function review_model(V;d=64,hidden=64,seed=2718,T=Float32)
    rng=MersenneTwister(seed)
    (;E=T(0.1).*randn(rng,T,d,V),
      W=sqrt(T(2)/d).*randn(rng,T,hidden,d),b=zeros(T,hidden),
      v=randn(rng,T,1,hidden)./sqrt(T(hidden)),c=zeros(T,1))
end

classifier_initial=review_model(V_review;d=64,hidden=64)
@printf("Embedding table E: %s = %d trainable entries\n",
    string(size(classifier_initial.E)),length(classifier_initial.E))
@assert sum(length,values(classifier_initial))==1284225

# %% embedding_lookup
trace_H=classifier_initial.E[:,trace_ids]
@assert size(trace_H)==(64,T_review)
println("H has shape ",size(trace_H),": 64 coordinates × ",T_review," token positions.")
println("First five coordinates of the first three token vectors:")
for t in 1:3
    @printf("%-12s ID=%5d  %s\n",trace_tokens[t],trace_ids[t],
        string(round.(trace_H[1:5,t];digits=4)))
end

# %% embedding_figure
shown_words=["movie","film","story","cast","characters","ending","wonderful","beautiful",
    "clever","excellent","thoughtful","warm","funny","boring","terrible","dull",
    "confusing","predictable","awkward","careless","the","with","and","not"]
shown_words=filter(w->haskey(imdb.lookup,w),shown_words)
shown_ids=[imdb.lookup[w] for w in shown_words]
matrix_panel(classifier_initial.E[:,shown_ids]';rowlabels=shown_words,
    title="The 64-coordinate embedding table · selected vocabulary entries",
    xlabel="Embedding coordinate") |> p->plot(p;size=(980,650))

# %% mean_pooling
function word_frequencies(sequences,V;T=Float32)
    rows,cols,vals=Int[],Int[],T[]
    for (j,ids) in enumerate(sequences)
        append!(rows,ids);append!(cols,fill(j,length(ids)))
        append!(vals,fill(one(T)/length(ids),length(ids)))
    end
    sparse(rows,cols,vals,V,length(sequences))
end

function add_frequencies(data,V)
    (;data...,F=word_frequencies(data.ids,V))
end
imdb=merge(imdb,(training=add_frequencies(imdb.training,V_review),
    validation=add_frequencies(imdb.validation,V_review),
    test=add_frequencies(imdb.test,V_review)))
trace_f=imdb.training.F[:,trace_at:trace_at]
trace_mean=vec(mean(trace_H;dims=2))
@assert trace_mean≈vec(classifier_initial.E*trace_f)
@assert isapprox(sum(trace_f),1f0;atol=1f-6)
coordinate=7
@printf("Coordinate %d: sum across %d tokens = %.6f; divide by %d → %.6f\n",
    coordinate,T_review,sum(trace_H[coordinate,:]),T_review,trace_mean[coordinate])
@printf("f shape %s | E*f shape %s | all training means: 64 × %d\n",
    string(size(trace_f)),string(size(classifier_initial.E*trace_f)),length(imdb.training.rows))

# %% pooling_figure
hp=heatmap(1:T_review,1:64,trace_H;color=DIV,
    clims=(-maximum(abs,trace_H),maximum(abs,trace_H)),colorbar=true,yflip=true,
    xlabel="Token position · every token in the review",ylabel="Embedding coordinate",
    title="H · one column per token",grid=false)
mp=bar(1:64,trace_mean;label=false,color=TEAL,linecolor=:transparent,
    xlabel="Embedding coordinate",ylabel="Mean value",
    title="a · one mean for each coordinate")
hline!(mp,[0];color=GRAY,lw=1,label=false)
plot(hp,mp;layout=grid(1,2,widths=[0.62,0.38]),size=(1120,430))

# %% classifier_forward
function review_forward(m,F)
    a=m.E*F
    u=m.W*a.+m.b
    h=max.(u,zero(eltype(u)))
    z=m.v*h.+m.c
    (;a,u,h,z,p=Flux.sigmoid.(z))
end
review_loss(m,F,y)=Flux.logitbinarycrossentropy(review_forward(m,F).z,y)

trace_target=imdb.training.y[:,trace_at:trace_at]
trace=review_forward(classifier_initial,trace_f)
@assert vec(trace.a)≈trace_mean
@assert trace.u[1]≈sum(classifier_initial.W[1,r]*trace_mean[r] for r in 1:64)+classifier_initial.b[1]
@assert only(trace.z)≈sum(classifier_initial.v[1,k]*trace.h[k] for k in 1:64)+only(classifier_initial.c)
println("a → u → h → z → probability shapes:")
println(size(trace.a)," → ",size(trace.u)," → ",size(trace.h)," → ",size(trace.z)," → ",size(trace.p))
@printf("Hidden unit 1: weighted sum + bias = %.6f; ReLU → %.6f\n",trace.u[1],trace.h[1])
@printf("Output: v' * h + c = %.6f; sigmoid → P(positive) = %.6f\n",only(trace.z),only(trace.p))

# %% forward_figure
function classifier_diagram(d=64,hidden=64)
    p=plot(xlims=(0,10),ylims=(-1.6,8.8),axis=false,grid=false,legend=false,size=(1100,520))
    xs=[1.2,4.4,7.5];ys=collect(range(1.4,6.6;length=8))
    for a in ys,b in ys
        plot!(p,[xs[1],xs[2]],[a,b];color=GRAY,alpha=0.22,lw=0.7)
    end
    for a in ys
        plot!(p,[xs[2],xs[3]],[a,4];color=GRAY,alpha=0.45,lw=1)
    end
    scatter!(p,fill(xs[1],8),ys;ms=8,color=TEAL,markerstrokewidth=0)
    scatter!(p,fill(xs[2],8),ys;ms=8,color=TEAL,markerstrokewidth=0)
    scatter!(p,[xs[3]],[4];ms=11,color=AMBER,markerstrokewidth=0)
    annotate!(p,1.2,7.8,text("Mean embedding",12,INK))
    annotate!(p,4.4,7.8,text("Hidden layer",12,INK))
    annotate!(p,7.5,7.8,text("Prediction",12,INK))
    annotate!(p,1.2,0.4,text("$d coordinates",11,INK))
    annotate!(p,4.4,0.4,text("$hidden ReLU units",11,INK))
    annotate!(p,7.5,0.4,text("One logit → sigmoid",11,INK))
    annotate!(p,1.2,-0.85,text(L"a=\frac{1}{T}\sum_{t=1}^{T}E[:,x_t]",11,INK))
    annotate!(p,4.4,-0.85,text(L"h=\mathrm{ReLU}(Wa+b)",11,INK))
    annotate!(p,8.9,4,text("P(positive)",11,INK))
    annotate!(p,7.5,-0.85,text(L"\hat{p}=\sigma(v^\top h+c)",11,INK))
    p
end

classifier_diagram()

# %% loss_value
L_review=review_loss(classifier_initial,trace_f,trace_target)
manual_L=-only(trace_target)*log(only(trace.p))-(1-only(trace_target))*log1p(-only(trace.p))
@assert isapprox(L_review,manual_L;atol=1f-6)
@printf("Observed y = %d; predicted p = %.6f; cross-entropy = %.6f\n",
    trace_row.y,only(trace.p),L_review)

# %% loss_figure
prob_grid=collect(0.01:0.005:0.99)
lp=plot(prob_grid,-log.(prob_grid);label="Positive label",color=TEAL,
    xlabel="Predicted P(positive)",ylabel="Binary cross-entropy",title="The loss")
plot!(lp,prob_grid,-log1p.(-prob_grid);label="Negative label",color=AMBER,linestyle=:dash)
gp=plot(prob_grid,prob_grid.-1;label="Positive label",color=TEAL,
    xlabel="Predicted P(positive)",ylabel="Derivative with respect to logit",title="The starting derivative")
plot!(gp,prob_grid,prob_grid;label="Negative label",color=AMBER,linestyle=:dash)
hline!(gp,[0];color=GRAY,label=false,lw=1)
plot(lp,gp;layout=(1,2),size=(1120,420))

# %% manual_gradient
delta_z=trace.p-trace_target
delta_h=classifier_initial.v'*delta_z
delta_u=delta_h.*(trace.u.>0)
delta_a=classifier_initial.W'*delta_u
manual_gradient=(E=Matrix(delta_a*trace_f'),W=delta_u*trace.a',
    b=vec(delta_u),v=delta_z*trace.h',c=vec(delta_z))
ad_gradient=Flux.gradient(m->review_loss(m,trace_f,trace_target),classifier_initial)[1]
gradient_error=maximum(maximum(abs.(getproperty(manual_gradient,k).-getproperty(ad_gradient,k)))
    for k in keys(classifier_initial))
@assert gradient_error<2f-6
unused=setdiff(1:V_review,trace_ids)
@assert all(iszero,ad_gradient.E[:,unused])
eta=0.1f0
one_step=map((p,g)->p.-eta.*g,classifier_initial,ad_gradient)
@assert review_loss(one_step,trace_f,trace_target)<L_review
word_counts=Dict(j=>count(==(j),trace_ids) for j in unique(trace_ids))
gradient_ids=sort(collect(keys(word_counts));by=j->(-word_counts[j],j))[1:16]
focus_id=first(gradient_ids);r=7
@printf("Manual / AD maximum difference: %.2e\n",gradient_error)
@printf("One step on the full model: loss %.6f → %.6f\n",
    L_review,review_loss(one_step,trace_f,trace_target))
@printf("Token '%s', ID %d, occurs %d/%d times\n",
    imdb.vocab[focus_id],focus_id,word_counts[focus_id],T_review)
@printf("E[7,%d]: %.7f − %.1f × (%.7f) = %.7f\n",
    focus_id,classifier_initial.E[r,focus_id],eta,ad_gradient.E[r,focus_id],one_step.E[r,focus_id])
@printf("%d unused vocabulary entries have exactly zero embedding gradient for this review.\n",length(unused))
gradient_labels=[string(imdb.vocab[j],"  (",word_counts[j],")") for j in gradient_ids]
matrix_panel(ad_gradient.E[:,gradient_ids]';rowlabels=gradient_labels,
    title="Embedding gradients · the 16 most frequent tokens in this review",
    xlabel="Embedding coordinate") |> p->plot(p;size=(980,550))

# %% review_training
function evaluate_reviews(m,data;batch_size=512)
    probabilities=Float32[];total=0.0
    for start in 1:batch_size:length(data.rows)
        idx=start:min(start+batch_size-1,length(data.rows))
        trace=review_forward(m,data.F[:,idx])
        append!(probabilities,vec(trace.p))
        total+=Float64(Flux.logitbinarycrossentropy(trace.z,data.y[:,idx]))*length(idx)
    end
    truth=vec(data.y).==1;pred=probabilities.>=0.5
    cm=[sum((truth.==a).&(pred.==p)) for a in (false,true),p in (false,true)]
    (;loss=total/length(truth),accuracy=mean(truth.==pred),probabilities,confusion=cm,
      f1=2cm[2,2]/max(2cm[2,2]+cm[1,2]+cm[2,1],1))
end
function train_reviews(data;d=64,hidden=64,epochs=8,batch_size=128,lr=0.003f0,budget=600)
    started=time();initial=deepcopy(classifier_initial)
    @assert size(initial.E)==(d,length(data.vocab)) && size(initial.W)==(hidden,d)
    m=deepcopy(initial);opt=Flux.setup(Flux.Adam(lr),m);rng=MersenneTwister(2719)
    tr,va=evaluate_reviews(m,data.training),evaluate_reviews(m,data.validation)
    history=[(;epoch=0,train_loss=tr.loss,val_loss=va.loss,train_acc=tr.accuracy,val_acc=va.accuracy)]
    best,best_loss,best_epoch=deepcopy(m),va.loss,0
    for epoch in 1:epochs
        order=randperm(rng,length(data.training.rows))
        for start in 1:batch_size:length(order)
            time()-started>budget && break
            idx=order[start:min(start+batch_size-1,length(order))]
            F,y=data.training.F[:,idx],data.training.y[:,idx]
            loss,g=Flux.withgradient(p->review_loss(p,F,y),m)
            @assert isfinite(loss)
            Flux.update!(opt,m,g[1])
        end
        tr,va=evaluate_reviews(m,data.training),evaluate_reviews(m,data.validation)
        push!(history,(;epoch,train_loss=tr.loss,val_loss=va.loss,train_acc=tr.accuracy,val_acc=va.accuracy))
        if va.loss<best_loss
            best,best_loss,best_epoch=deepcopy(m),va.loss,epoch
        end
        @printf("Epoch %d | train %.3f | validation %.3f | val accuracy %.1f%%\n",
            epoch,tr.loss,va.loss,100va.accuracy)
        time()-started>budget && break
    end
    test=evaluate_reviews(best,data.test)
    (;model=best,initial,history,best_epoch,test,seconds=time()-started,
      vocab=data.vocab,config=(;d,hidden,epochs,batch_size,lr))
end
TRAIN_SENTIMENT=true
sentiment=TRAIN_SENTIMENT ? train_reviews(imdb) : deserialize(joinpath(RESULTS,"sentiment.jls"))
@assert sentiment.vocab==imdb.vocab
@assert sentiment.initial==classifier_initial
TRAIN_SENTIMENT && serialize(joinpath(RESULTS,"sentiment.jls"),sentiment)
@printf("%s parameters | selected epoch %d | fit %.1f seconds\n",
    string(sum(length,values(sentiment.model))),sentiment.best_epoch,sentiment.seconds)

# %% review_curves
epochs=[h.epoch for h in sentiment.history]
p1=plot(epochs,[h.train_loss for h in sentiment.history];label="Training",color=TEAL,
    xlabel="Epoch",ylabel="Binary cross-entropy",title="Loss and checkpoint selection")
plot!(p1,epochs,[h.val_loss for h in sentiment.history];label="Validation",color=AMBER,linestyle=:dash)
vline!(p1,[sentiment.best_epoch];label="Selected epoch",color=GRAY,linestyle=:dot)
p2=plot(epochs,100 .* [h.train_acc for h in sentiment.history];label="Training",color=TEAL,
    xlabel="Epoch",ylabel="Accuracy (%)",ylims=(40,101),title="Training and validation accuracy")
plot!(p2,epochs,100 .* [h.val_acc for h in sentiment.history];label="Validation",color=AMBER,linestyle=:dash)
plot(p1,p2;layout=(1,2),size=(1120,430))

# %% review_test
cm=sentiment.test.confusion
@printf("Filtered test: %d reviews | accuracy %.2f%% | positive F1 %.4f | loss %.4f\n",
    length(imdb.test.rows),100sentiment.test.accuracy,sentiment.test.f1,sentiment.test.loss)
majority=mean(imdb.training.y)>=0.5
@printf("Training-majority baseline: %.2f%%\n",100mean((vec(imdb.test.y).>=0.5).==majority))
cp=heatmap(1:2,1:2,cm;color=SEQ,colorbar=false,yflip=true,grid=false,framestyle=:box,
    xticks=(1:2,["Negative","Positive"]),yticks=(1:2,["Negative","Positive"]),
    xlabel="Predicted label",ylabel="Actual label",title="Test confusion matrix")
for a in 1:2,p in 1:2
    annotate!(cp,p,a,text(string(cm[a,p]),16,cm[a,p]>maximum(cm)/2 ? :white : INK))
end
neg=sentiment.test.probabilities[vec(imdb.test.y).==0]
pos=sentiment.test.probabilities[vec(imdb.test.y).==1]
hp=histogram(neg;bins=0:0.04:1,normalize=:probability,color=AMBER,alpha=0.55,
    linecolor=:white,label="Negative reviews",xlabel="Predicted P(positive)",
    ylabel="Fraction within each class",title="Confidence and mistakes")
histogram!(hp,pos;bins=0:0.04:1,normalize=:probability,color=TEAL,alpha=0.55,
    linecolor=:white,label="Positive reviews")
plot(cp,hp;layout=(1,2),size=(1120,440))

# %% learned_embeddings
real_words=filter(w->haskey(imdb.lookup,w),shown_words)
real_ids=[imdb.lookup[w] for w in real_words]
E0,E1=sentiment.initial.E[:,real_ids]',sentiment.model.E[:,real_ids]'
scale=max(maximum(abs,E0),maximum(abs,E1))
ep0=matrix_panel(E0;rowlabels=real_words,title="Random initialization",limit=scale)
ep1=matrix_panel(E1;rowlabels=real_words,title="Learned from review labels",limit=scale)
plot(ep0,ep1;layout=(1,2),size=(1120,640))

# %% own_reviews
function sentiment_probability(text,run=sentiment)
    lookup=Dict(w=>i for (i,w) in enumerate(run.vocab))
    ids=word_ids(text,lookup)
    only(review_forward(run.model,word_frequencies([ids],length(run.vocab))).p)
end
my_reviews=["The actors were wonderful and the story kept me interested.",
    "The plot was confusing and the dialogue was painfully dull.",
    "The movie is good, not bad.","The movie is bad, not good."]
for s in my_reviews
    @printf("P(positive) = %.4f | %s\n",sentiment_probability(s),s)
end
@assert sentiment_probability(my_reviews[3])≈sentiment_probability(my_reviews[4])
mistakes=findall((sentiment.test.probabilities.>=0.5).!=(vec(imdb.test.y).>=0.5))
println("\nFirst two errors in source order (short excerpts):")
for i in first(mistakes,2)
    @printf("label=%d, P(positive)=%.3f: %s…\n",imdb.test.rows[i].y,
        sentiment.test.probabilities[i],first(replace(imdb.test.rows[i].text,r"<[^>]*>"=>" "),130))
end

# %% bpe
struct BPETokenizer
    tokens::Vector{String}
    characters::Dict{Char,Int}
    merges::Dict{Tuple{Int,Int},Tuple{Int,Int}}
    bos::Int
    eos::Int
    unk::Int
end
function read_bpe(dir)
    tokens=[String(hex2bytes(s)) for s in readlines(joinpath(dir,"tokens_hex.txt"))]
    chars=Dict(only(s)=>i for (i,s) in enumerate(tokens) if length(s)==1)
    merges=Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (rank,line) in enumerate(readlines(joinpath(dir,"merges.tsv")))
        a,b,c=parse.(Int,split(line,'\t'));@assert tokens[a]*tokens[b]==tokens[c]
        merges[(a,b)]=(rank,c)
    end
    BPETokenizer(tokens,chars,merges,findfirst(==("<bos>"),tokens),
        findfirst(==("<eos>"),tokens),findfirst(==("<unk>"),tokens))
end
function encode_piece(tok,text)
    ids=[get(tok.characters,c,0) for c in text]
    any(iszero,ids) && error("This prepared tokenizer supports ASCII English.")
    while length(ids)>1
        rank,at,new=typemax(Int),0,0
        for i in 1:length(ids)-1
            candidate,next=get(tok.merges,(ids[i],ids[i+1]),(typemax(Int),0))
            if candidate<rank
                rank,at,new=candidate,i,next
            end
        end
        at==0 && break
        ids[at]=new;deleteat!(ids,at+1)
    end
    ids
end
bpe_encode(tok,text)=vcat([encode_piece(tok,m.match) for m in
    eachmatch(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+",text)]...)
bpe_decode(tok,ids)=join(tok.tokens[i] for i in ids if i ∉ (tok.bos,tok.eos,tok.unk))
visible_token(s)=replace(s," "=>"·","\n"=>"↵")
tok=read_bpe(STORY_DIR)
sample_text="The little cat found a new friend."
piece_ids=bpe_encode(tok,sample_text)
@assert bpe_decode(tok,piece_ids)==sample_text
@printf("%d vocabulary entries | %d tokens in this sentence\n",length(tok.tokens),length(piece_ids))
println(visible_token.(tok.tokens[piece_ids]))
sequence=vcat(tok.bos,piece_ids,tok.eos)
println("\n input token       → next-token target")
for (x,y) in zip(sequence[1:end-1],sequence[2:end])
    @printf(" %-17s → %s\n",visible_token(tok.tokens[x]),visible_token(tok.tokens[y]))
end
for line in eachline(joinpath(STORY_DIR,"tokenizer_probes.tsv"))
    text_hex,ids=split(line,'\t')
    @assert bpe_encode(tok,String(hex2bytes(text_hex)))==parse.(Int,split(ids,','))
end

# %% story_data
load_token_file(path)=collect(reinterpret(UInt16,read(path)))
story_tokens=Dict(name=>load_token_file(joinpath(STORY_DIR,name*".bin"))
    for name in ("train","validation","test"))
for name in ("train","validation","test")
    @printf("%-10s %10d BPE tokens\n",name,length(story_tokens[name]))
end
function token_batch(ids,T,B,rng)
    starts=rand(rng,1:length(ids)-T,B)
    X=[Int(ids[s+t-1]) for t in 1:T,s in starts]
    Y=[Int(ids[s+t]) for t in 1:T,s in starts]
    X,Y
end
batch_X,batch_Y=token_batch(story_tokens["train"],64,8,MersenneTwister(51))
@assert batch_X[2:end,:]==batch_Y[1:end-1,:]
println("One training window:\n",bpe_decode(tok,batch_X[:,1]))
println("Input/target shapes: ",size(batch_X)," / ",size(batch_Y))

# %% gpt_model
causal_mask(T)=Float32[j<=i ? 0 : -Inf for j in 1:T,i in 1:T]
struct CausalAttention{Q,K,V,O}
    query::Q;key::K;value::V;output::O;heads::Int
end
Flux.@layer CausalAttention trainable=(query,key,value,output)
CausalAttention(d,heads)=CausalAttention(Dense(d=>d),Dense(d=>d),Dense(d=>d),Dense(d=>d),heads)
function split_heads(Z,heads)
    d,T,B=size(Z)
    reshape(permutedims(reshape(Z,d÷heads,heads,T,B),(1,3,2,4)),d÷heads,T,heads*B)
end
function (a::CausalAttention)(X,mask)
    d,T,B=size(X);dh=d÷a.heads
    q,k,v=split_heads(a.query(X),a.heads),split_heads(a.key(X),a.heads),split_heads(a.value(X),a.heads)
    scores=Flux.NNlib.batched_mul(permutedims(k,(2,1,3)),q)/sqrt(Float32(dh))
    weights=Flux.softmax(scores.+reshape(mask,T,T,1);dims=1)
    mixed=Flux.NNlib.batched_mul(v,weights)
    joined=reshape(permutedims(reshape(mixed,dh,T,a.heads,B),(1,3,2,4)),d,T,B)
    a.output(joined)
end
struct DecoderBlock{N1,A,N2,F}
    norm1::N1;attention::A;norm2::N2;feedforward::F
end
Flux.@layer DecoderBlock
DecoderBlock(d,heads)=DecoderBlock(LayerNorm(d),CausalAttention(d,heads),LayerNorm(d),
    Chain(Dense(d=>4d,Flux.gelu),Dense(4d=>d)))
function (b::DecoderBlock)(X,mask)
    R=X+b.attention(b.norm1(X),mask)
    R+b.feedforward(b.norm2(R))
end
struct TinyGPT{TE,PE,BL,N,O}
    tokens::TE;positions::PE;blocks::BL;finalnorm::N;head::O
end
Flux.@layer TinyGPT
function TinyGPT(V;d=256,heads=8,layers=6,context=128)
    @assert d%heads==0
    TinyGPT(Embedding(0.02f0.*randn(Float32,d,V)),Embedding(0.02f0.*randn(Float32,d,context)),
        [DecoderBlock(d,heads) for _ in 1:layers],LayerNorm(d),Dense(d=>V;bias=false))
end
function gpt_logits(m,ids,mask,positions)
    T,B=size(ids);d,max_context=size(m.positions.weight)
    @assert T<=max_context
    X=m.tokens(ids).+reshape(m.positions(positions),d,T,1)
    for block in m.blocks
        X=block(X,mask)
    end
    m.head(m.finalnorm(X))
end
parameter_count(m)=sum(length,Flux.trainables(m))
function restore_weights!(model,weights)
    arrays=Flux.trainables(model)
    @assert length(arrays)==length(weights)
    for (a,b) in zip(arrays,weights)
        @assert size(a)==size(b)
        copyto!(a,b)
    end
    model
end
cpu_copy(m)=Flux.fmap(x->x isa AbstractArray ? copy(Array(x)) : x,m)

Random.seed!(314159)
gpt_initial=TinyGPT(length(tok.tokens);d=256,layers=6,heads=8,context=128)
@assert parameter_count(gpt_initial)==5820416
println("One English model: ",parameter_count(gpt_initial)," parameters, 6 blocks, 8 heads, width 256.")

# %% block_diagram
function block_diagram()
    p=plot(xlims=(0,10),ylims=(0,10),axis=false,grid=false,legend=false,size=(960,560))
    x=4.5
    ys=[1.1,2.4,3.7,5.0,6.3,7.6,8.9]
    labels=["Input states","Layer norm","Multi-head attention","+","Layer norm","MLP","+"]
    for (i,(y,label)) in enumerate(zip(ys,labels))
        if label=="+"
            scatter!(p,[x],[y];ms=11,color=:white,markerstrokecolor=TEAL,markerstrokewidth=1.6)
            annotate!(p,x,y,text("+",14,TEAL))
        else
            plot!(p,Shape([x-1.3,x+1.3,x+1.3,x-1.3],[y-.33,y-.33,y+.33,y+.33]);
                fillcolor=:white,linecolor=TEAL,lw=1.3)
            annotate!(p,x,y,text(label,11,INK))
        end
        i<length(ys) && plot!(p,[x,x],[y+.37,ys[i+1]-.38];color=GRAY,lw=1.3,arrow=true)
    end
    plot!(p,[x-1.3,1.8,1.8,x-.25],[1.1,1.1,5.0,5.0];color=AMBER,lw=1.6,arrow=true)
    plot!(p,[x+.25,7.3,7.3,x+.25],[5.0,5.0,8.9,8.9];color=AMBER,lw=1.6,arrow=true)
    annotate!(p,1.25,3.0,text("Skip",10,AMBER))
    annotate!(p,8.0,7.0,text("Skip",10,AMBER))
    annotate!(p,4.5,9.7,text("One pre-normalization Transformer block",13,INK))
    annotate!(p,8.1,3.7,text("Mix positions",10,GRAY))
    annotate!(p,1.6,7.6,text("Mix coordinates",10,GRAY))
    p
end
block_diagram()

# %% qkv
# A real corpus prefix; inspect head 1 of the full, randomly initialized GPT.
attention_sequence=Int.(story_tokens["train"][1:13])
attention_ids=attention_sequence[1:12]
target_id=attention_sequence[13]
attention_T=length(attention_ids)
attention_labels=[string(t," ",visible_token(tok.tokens[id])) for (t,id) in enumerate(attention_ids)]
input_states=gpt_initial.tokens(reshape(attention_ids,:,1)).+
    reshape(gpt_initial.positions(1:attention_T),256,attention_T,1)
first_block=gpt_initial.blocks[1]
attention_layer=first_block.attention
normalized_states=first_block.norm1(input_states)
head_coordinates=1:32
projection=(X=dropdims(normalized_states;dims=3),
    WQ=attention_layer.query.weight[head_coordinates,:],bQ=attention_layer.query.bias[head_coordinates],
    WK=attention_layer.key.weight[head_coordinates,:],bK=attention_layer.key.bias[head_coordinates],
    WV=attention_layer.value.weight[head_coordinates,:],bV=attention_layer.value.bias[head_coordinates])
Q=projection.WQ*projection.X.+projection.bQ
K=projection.WK*projection.X.+projection.bK
att_values=projection.WV*projection.X.+projection.bV
@assert size(Q)==size(K)==size(att_values)==(32,12)
println("Corpus prefix: ",bpe_decode(tok,attention_ids))
println("Observed next token: ",visible_token(tok.tokens[target_id]))
@printf("Normalized input %s | each W %s | each projected head %s\n",
    string(size(projection.X)),string(size(projection.WQ)),string(size(Q)))
# Plot a readable feature slice; all 256 input coordinates were used above.
lim=maximum(maximum(abs,M) for M in (projection.X[1:32,:],Q,K,att_values))
qplots=[matrix_panel(M';title=label,rowlabels=attention_labels,limit=lim)
    for (M,label) in [(projection.X[1:32,:],"Normalized input · coordinates 1–32 of 256"),
                     (Q,"Q · all 32 head coordinates"),(K,"K · all 32 head coordinates"),
                     (att_values,"Values · all 32 head coordinates")]]
plot(qplots...;layout=(2,2),size=(1120,780),titlefontsize=11)

# %% attention
scores=K'*Q/sqrt(32f0)
attention_mask=causal_mask(attention_T)
A=Flux.softmax(scores.+attention_mask;dims=1)
O=att_values*A
@assert vec(sum(A;dims=1))≈ones(attention_T)
@assert all(iszero,A[tril(trues(attention_T,attention_T),-1)])
@assert O[:,end]≈sum(A[j,end].*att_values[:,j] for j in 1:attention_T)
ap=heatmap(1:attention_T,1:attention_T,A;color=SEQ,clims=(0,1),colorbar=false,
    xticks=(1:attention_T,1:attention_T),yticks=(1:attention_T,attention_labels),
    yflip=true,grid=false,framestyle=:box,xlabel="Query position",title="Causal attention")
query_at=9
bp=bar(1:attention_T,A[:,query_at];orientation=:horizontal,color=TEAL,
    linecolor=:transparent,label=false,yticks=(1:attention_T,attention_labels),yflip=true,
    xlabel="Attention weight",title="What position 9 reads",xlims=(0,1.15maximum(A[:,query_at])))
plot(ap,bp;layout=(1,2),size=(1120,510))

# %% attention_gradient
# Keep all eight heads and the rest of the full GPT in the loss calculation.
all_q=split_heads(attention_layer.query(normalized_states),8)
all_k=split_heads(attention_layer.key(normalized_states),8)
all_v=split_heads(attention_layer.value(normalized_states),8)
all_scores=Flux.NNlib.batched_mul(permutedims(all_k,(2,1,3)),all_q)/sqrt(32f0)
all_weights=Flux.softmax(all_scores.+reshape(attention_mask,attention_T,attention_T,1);dims=1)
all_outputs=Flux.NNlib.batched_mul(all_v,all_weights)
mixed_heads=reshape(permutedims(reshape(all_outputs,32,attention_T,8,1),(1,3,2,4)),256,attention_T)
@assert mixed_heads[1:32,:]≈O
function loss_after_first_attention(mixed)
    # Continue the actual first block, then every remaining block and output head.
    R=input_states+attention_layer.output(reshape(mixed,256,attention_T,1))
    X=R+first_block.feedforward(first_block.norm2(R))
    for block in gpt_initial.blocks[2:end]
        X=block(X,attention_mask)
    end
    logits=gpt_initial.head(gpt_initial.finalnorm(X))[:,end,1]
    -Flux.logsoftmax(logits)[target_id]
end
g_mixed=Flux.gradient(loss_after_first_attention,mixed_heads)[1]
g_O=g_mixed[1:32,:]
g_values=g_O*A';g_A=att_values'*g_O
g_scores=A.*(g_A.-sum(A.*g_A;dims=1))
g_Q=K*g_scores/sqrt(32f0);g_K=Q*g_scores'/sqrt(32f0)
qkv_manual=(X=projection.WQ'*g_Q+projection.WK'*g_K+projection.WV'*g_values,
    WQ=g_Q*projection.X',bQ=vec(sum(g_Q;dims=2)),
    WK=g_K*projection.X',bK=vec(sum(g_K;dims=2)),
    WV=g_values*projection.X',bV=vec(sum(g_values;dims=2)))
function loss_through_head(p)
    q=p.WQ*p.X.+p.bQ;k=p.WK*p.X.+p.bK;v=p.WV*p.X.+p.bV
    weights=Flux.softmax(k'*q/sqrt(32f0).+attention_mask;dims=1)
    loss_after_first_attention(vcat(v*weights,mixed_heads[33:end,:]))
end
qkv_ad=Flux.gradient(loss_through_head,projection)[1]
qkv_error=maximum(maximum(abs.(getproperty(qkv_manual,k).-getproperty(qkv_ad,k)))
    for k in keys(projection))
@assert qkv_error<1f-5
function full_prefix_loss(model)
    logits=gpt_logits(model,reshape(attention_ids,:,1),attention_mask,collect(1:attention_T))[:,end,1]
    -Flux.logsoftmax(logits)[target_id]
end
full_gradient=Flux.gradient(full_prefix_loss,gpt_initial)[1]
@assert loss_through_head(projection)≈full_prefix_loss(gpt_initial)
@assert isapprox(qkv_manual.WQ,full_gradient.blocks[1].attention.query.weight[1:32,:];atol=1f-5)
# Only parameters move in this diagnostic step; the activation X stays fixed.
step_size=0.001f0
projection_updated=merge(projection,(
    WQ=projection.WQ-step_size*qkv_ad.WQ,bQ=projection.bQ-step_size*qkv_ad.bQ,
    WK=projection.WK-step_size*qkv_ad.WK,bK=projection.bK-step_size*qkv_ad.bK,
    WV=projection.WV-step_size*qkv_ad.WV,bV=projection.bV-step_size*qkv_ad.bV))
@assert loss_through_head(projection_updated)<loss_through_head(projection)
@printf("Manual / AD attention-gradient difference: %.2e\n",qkv_error)
@printf("Full-GPT next-token loss, updating only head 1: %.6f → %.6f\n",
    loss_through_head(projection),loss_through_head(projection_updated))
println("Head-1 query gradients also match backpropagation through the entire GPT.")
gps=map([(:WQ,"W_Q"),(:WK,"W_K"),(:WV,"W_V")]) do (key,name)
    gradient=getproperty(qkv_ad,key);magnitude=maximum(abs,gradient)
    title=@sprintf("%s · max |gradient| = %.2e",name,magnitude)
    matrix_panel(gradient/magnitude;title,limit=1.0,xlabel="Input coordinate · all 256")
end
plot(gps...;layout=(3,1),size=(980,740))

# %% gpt_training
function fit_gpt(train_ids,val_ids;V=2048,d=256,layers=6,heads=8,context=128,
                 batch_size=8,steps=100000,budget_seconds=2400,lr=0.0006f0,
                 seed=314159,device=identity,eval_every=50)
    started=time();Random.seed!(seed)
    config=(;V,d,layers,heads,context,batch_size,steps,budget_seconds,lr,seed)
    m=Flux.fmap(device,TinyGPT(V;d,layers,heads,context))
    rng=MersenneTwister(seed);eval_rng=MersenneTwister(991)
    positions=device(collect(1:context));mask=device(causal_mask(context))
    function batch(ids,rng)
        X,Y=token_batch(ids,context,batch_size,rng)
        device(X),device(Float32.(Flux.onehotbatch(vec(Y),1:V)))
    end
    batch_loss(model,X,Y)=Flux.logitcrossentropy(
        reshape(gpt_logits(model,X,mask,positions),V,:),Y)
    train_probe=[batch(train_ids,eval_rng) for _ in 1:3]
    val_probe=[batch(val_ids,eval_rng) for _ in 1:6]
    evaluate(model,bs)=mean(Float64(batch_loss(model,X,Y)) for (X,Y) in bs)
    tr,va=evaluate(m,train_probe),evaluate(m,val_probe)
    history=[(;step=0,seconds=time()-started,train_loss=tr,validation_loss=va)]
    best,best_loss,best_step=cpu_copy(m),va,0
    opt=Flux.setup(Flux.OptimiserChain(Flux.ClipNorm(1f0),Flux.Adam(lr)),m)
    last_step=0
    for step in 1:steps
        # Reserve time for final evaluation; budget is checked between updates.
        time()-started>=max(1,budget_seconds-30) && break
        fraction=clamp((time()-started)/max(1,budget_seconds-30),0,1)
        eta=Float32(lr*min(step/30,1)*(0.1+0.9*(1+cos(pi*fraction))/2))
        Flux.Optimisers.adjust!(opt,eta)
        X,Y=batch(train_ids,rng)
        value,g=Flux.withgradient(model->batch_loss(model,X,Y),m)
        @assert isfinite(value)
        Flux.update!(opt,m,g[1]);last_step=step
        if step%eval_every==0 || step==steps
            tr,va=evaluate(m,train_probe),evaluate(m,val_probe)
            push!(history,(;step,seconds=time()-started,train_loss=tr,validation_loss=va))
            if va<best_loss
                best,best_loss,best_step=cpu_copy(m),va,step
            end
            @printf("Step %d | %.1f s | train %.3f | validation %.3f\n",step,time()-started,tr,va)
        end
    end
    if history[end].step<last_step
        tr,va=evaluate(m,train_probe),evaluate(m,val_probe)
        push!(history,(;step=last_step,seconds=time()-started,train_loss=tr,validation_loss=va))
        if va<best_loss
            best,best_loss,best_step=cpu_copy(m),va,last_step
        end
    end
    (;weights=deepcopy(Flux.trainables(best)),config,history,best_step,seconds=time()-started)
end

# %% large_model
saved=deserialize(joinpath(ROOT,"output","english_basics","notebook_weights.jls"))
c=saved.config
english_gpt=restore_weights!(TinyGPT(length(tok.tokens);d=c.d,layers=c.layers,
    heads=c.heads,context=c.context),saved.weights)
@assert saved.vocab==tok.tokens
probe=saved.probe
probe_T=size(probe.probe,1)
probe_error=maximum(abs.(gpt_logits(english_gpt,probe.probe,causal_mask(probe_T),
    collect(1:probe_T))-probe.logits))
@assert probe_error<1f-4
println("Saved English GPT: ",parameter_count(english_gpt)," parameters")
@printf("Loaded-weight verification: maximum logit difference %.2e\n",probe_error)
TRAIN_LARGE_GPT=false
if TRAIN_LARGE_GPT
    @eval using CUDA
    CUDA.allowscalar(false)
    CUDA.math_mode!(CUDA.FAST_MATH;precision=:TensorFloat32)
    fresh_run=fit_gpt(story_tokens["train"],story_tokens["validation"];
        d=256,layers=6,heads=8,context=128,steps=100000,budget_seconds=2400,
        lr=0.0006f0,device=CUDA.cu,eval_every=1000)
    serialize(joinpath(RESULTS,"fresh_large_gpt.jls"),fresh_run)
    english_gpt=restore_weights!(TinyGPT(length(tok.tokens);d=256,layers=6,
        heads=8,context=128),fresh_run.weights)
end

# %% gpt_curves
function language_curve(history,title;time_axis=false)
    x=time_axis ? [h.seconds/60 for h in history] : [h.step for h in history]
    p=plot(x,[h.train_loss for h in history];label="Training windows",color=TEAL,
        xlabel=time_axis ? "Elapsed minutes" : "Optimizer updates",ylabel="Cross-entropy per BPE token",title)
    plot!(p,x,[h.validation_loss for h in history];label="Validation windows",color=AMBER,linestyle=:dash)
    p
end
large_history=TRAIN_LARGE_GPT ? fresh_run.history : saved.history

plot(language_curve(large_history,TRAIN_LARGE_GPT ?
    "5.82M GPT · fresh GPU run" : "5.82M GPT · recorded GPU run";time_axis=true);size=(980,440))

# %% language_evaluation
function evaluate_language(m,batches)
    V=length(tok.tokens)
    mean(begin
        T=size(X,1)
        z=reshape(gpt_logits(m,X,causal_mask(T),collect(1:T)),V,:)
        Float64(Flux.logitcrossentropy(z,Flux.onehotbatch(vec(Y),1:V)))
    end for (X,Y) in batches)
end
test_rng=MersenneTwister(27182)
test_windows=[token_batch(story_tokens["test"],64,8,test_rng) for _ in 1:8]
initial_test_loss=evaluate_language(gpt_initial,test_windows)
large_test_loss=evaluate_language(english_gpt,test_windows)
@printf("%-14s %12s %12s %12s\n","Model","Parameters","Test loss","Perplexity")
@printf("%-14s %12d %12.4f %12.2f\n","Random start",parameter_count(gpt_initial),initial_test_loss,exp(initial_test_loss))
@printf("%-14s %12d %12.4f %12.2f\n","Trained GPT",parameter_count(english_gpt),large_test_loss,exp(large_test_loss))
# Earlier logits must not change when only future inputs are replaced.
check_ids=copy(test_windows[1][1][:,1:1])
base=gpt_logits(english_gpt,check_ids,causal_mask(64),collect(1:64))
changed=copy(check_ids);changed[33:end,:].=tok.bos
altered=gpt_logits(english_gpt,changed,causal_mask(64),collect(1:64))
@assert maximum(abs.(base[:,1:32,:]-altered[:,1:32,:]))<1f-4
println("Future-token isolation check passed.")

# %% sampling
function generate_text(m,tok;prompt,seed=8,temperature=0.65,top_k=40,max_new=90)
    @assert temperature>0 && top_k>=1
    rng=MersenneTwister(seed)
    ids=vcat(tok.bos,bpe_encode(tok,prompt));generated=Int[]
    max_context=size(m.positions.weight,2);ended=false
    for _ in 1:max_new
        window=ids[max(1,length(ids)-max_context+1):end];T=length(window)
        z=vec(gpt_logits(m,reshape(window,:,1),causal_mask(T),collect(1:T))[:,end,1])
        z[tok.bos]=-Inf32;z[tok.unk]=-Inf32
        keep=partialsortperm(z,1:min(top_k,length(z));rev=true)
        p=Flux.softmax(z[keep]./Float32(temperature))
        next=keep[min(searchsortedfirst(cumsum(p),rand(rng)),length(keep))]
        if next==tok.eos
            ended=true;break
        end
        push!(generated,next);push!(ids,next)
    end
    (;prompt,continuation=bpe_decode(tok,generated),ended,tokens=length(generated))
end
prompts=["Once upon a time, there was a little",
    "The cat looked at the dog and said,","One day, a little girl found a"]
samples=[generate_text(english_gpt,tok;prompt,seed=7+i) for (i,prompt) in enumerate(prompts)]
for s in samples
    println("\nPROMPT: ",s.prompt,"\nCONTINUATION: ",s.continuation)
    println("Tokens generated: ",s.tokens," | ended at EOS: ",s.ended)
end
serialize(joinpath(RESULTS,"english_samples.jls"),samples)

# %% temperature
prefix=vcat(tok.bos,bpe_encode(tok,"The little cat was"))
T_prefix=length(prefix)
next_scores=vec(gpt_logits(english_gpt,reshape(prefix,:,1),causal_mask(T_prefix),
    collect(1:T_prefix))[:,end,1])
next_scores[tok.bos]=-Inf32;next_scores[tok.unk]=-Inf32
top_ids=partialsortperm(next_scores,1:12;rev=true)
token_labels=visible_token.(tok.tokens[top_ids])
temperatures=(0.65f0,1.2f0)
temperature_xmax=min(1.0,1.1maximum(maximum(Flux.softmax(next_scores./temp)) for temp in temperatures))
tps=map(temperatures) do temp
    p=Flux.softmax(next_scores./temp)
    bar(1:12,p[top_ids];orientation=:horizontal,label=false,color=TEAL,
        linecolor=:transparent,yticks=(1:12,token_labels),yflip=true,
        xlabel="Probability over the full vocabulary",xlims=(0,temperature_xmax),
        title="Temperature = $(temp)")
end
plot(tps...;layout=(1,2),size=(1120,480))

# %% learned_attention
view_ids=vcat(tok.bos,bpe_encode(tok,"The little cat looked at the dog and wanted to play."))[1:12]
N=length(view_ids)
H=english_gpt.tokens(reshape(view_ids,:,1)).+reshape(english_gpt.positions(1:N),256,N,1)
first_block=english_gpt.blocks[1];a=first_block.attention
normalized=first_block.norm1(H)
q=split_heads(a.query(normalized),a.heads);k=split_heads(a.key(normalized),a.heads)
att_scores=Flux.NNlib.batched_mul(permutedims(k,(2,1,3)),q)/sqrt(32f0)
actual_attention=Flux.softmax(att_scores.+reshape(causal_mask(N),N,N,1);dims=1)
labels=[string(i," ",visible_token(tok.tokens[t])) for (i,t) in enumerate(view_ids)]
headplots=[heatmap(1:N,1:N,actual_attention[:,:,h];color=SEQ,clims=(0,1),colorbar=true,
    grid=false,framestyle=:box,yflip=true,xticks=(1:N,1:N),yticks=(1:N,labels),
    xlabel="Query position",title="Trained first layer · head $h") for h in 1:2]
plot(headplots...;layout=(1,2),size=(1120,510))
