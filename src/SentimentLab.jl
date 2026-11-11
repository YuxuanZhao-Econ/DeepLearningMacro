module SentimentLab
using Flux, Random, Statistics, LinearAlgebra, SparseArrays, Serialization, SHA, Printf
export words, read_reviews, prepare_data, frequencies, initialize, logits, loss,
       evaluate, fit, predict_review, save_run

"Lowercase word tokenizer. Keep not and contractions; remove HTML tags."
words(text::AbstractString) = [String(m.match) for m in
    eachmatch(r"[a-z]+(?:'[a-z]+)?", lowercase(replace(text,r"<[^>]*>"=>" ")))]

function read_reviews(path)
    [begin
        id,label,encoded = split(line,'\t')
        (;id=parse(Int,id),y=parse(Int,label),text=String(hex2bytes(encoded)))
    end for line in eachline(path)]
end

function frequencies(encoded,V)
    ii,jj,values = Int[],Int[],Float32[]
    for (j,ids) in enumerate(encoded)
        isempty(ids) && error("An encoded review must contain at least <unk>.")
        append!(ii,ids)
        append!(jj,fill(j,length(ids)))
        append!(values,fill(1f0/length(ids),length(ids)))
    end
    sparse(ii,jj,values,V,length(encoded)) # repeated tokens sum within a column
end

function prepare_data(dir;max_vocab=12000,seed=31415,validation_fraction=0.1)
    # Deduplicate on the exact normalized token sequence, BEFORE the validation split.
    # If two identical normalized texts disagree on the label, discard that group.
    source = read_reviews(joinpath(dir,"train.tsv"))
    groups = Dict{String,Vector{Int}}()
    tokens = words.(getproperty.(source,:text))
    keys_train = join.(tokens," ")
    for (i,key) in enumerate(keys_train)
        push!(get!(groups,key,Int[]),i)
    end
    kept = sort([first(g) for g in values(groups)
                 if length(unique(source[i].y for i in g))==1])
    pool = source[kept]
    pool_words = tokens[kept]
    rng = MersenneTwister(seed)
    train_at,validation_at = Int[],Int[]
    for label in (0,1)
        group = shuffle(rng,findall(r->r.y==label,pool))
        n = round(Int,validation_fraction*length(group))
        append!(validation_at,group[1:n])
        append!(train_at,group[n+1:end])
    end
    sort!(train_at); sort!(validation_at)
    counts = Dict{String,Int}()
    for i in train_at, word in pool_words[i]
        counts[word] = get(counts,word,0)+1
    end
    ranked = sort(collect(keys(counts));by=w->(-counts[w],w))
    vocab = vcat(["<unk>"],ranked[1:min(max_vocab-1,length(ranked))])
    lookup = Dict(w=>i for (i,w) in enumerate(vocab))
    encode(ws) = isempty(ws) ? [1] : [get(lookup,w,1) for w in ws]
    function make(rows,tokenlists)
        encoded=encode.(tokenlists)
        (;rows,encoded,X=frequencies(encoded,length(vocab)),
          y=reshape(Float32[r.y for r in rows],1,:),
          unknown_rate=sum(count(==(1),ids) for ids in encoded)/sum(length,encoded))
    end
    tr = make(pool[train_at],pool_words[train_at])
    va = make(pool[validation_at],pool_words[validation_at])
    # Exclude test overlaps with ALL original training texts, even discarded groups.
    test_raw = read_reviews(joinpath(dir,"test.tsv"))
    test_words = words.(getproperty.(test_raw,:text))
    test_groups = Dict{String,Vector{Int}}()
    for (i,ws) in enumerate(test_words)
        push!(get!(test_groups,join(ws," "),Int[]),i)
    end
    seen_train = Set(keys_train)
    test_keep = sort([first(g) for (k,g) in test_groups if k ∉ seen_train &&
                      length(unique(test_raw[i].y for i in g))==1])
    te = make(test_raw[test_keep],test_words[test_keep])
    @assert isempty(intersect(Set(join.(pool_words[train_at]," ")),
                              Set(join.(pool_words[validation_at]," "))))
    @assert all(abs.(vec(sum(tr.X;dims=1)).-1) .< 1e-4)
    metadata=(;seed,max_vocab,validation_fraction,source_train=length(source),
        source_test=length(test_raw),train=length(tr.rows),validation=length(va.rows),
        test=length(te.rows),excluded_train=length(source)-length(pool),
        excluded_test=length(test_raw)-length(te.rows),
        train_tsv_sha256=bytes2hex(sha256(read(joinpath(dir,"train.tsv")))),
        test_tsv_sha256=bytes2hex(sha256(read(joinpath(dir,"test.tsv")))))
    (;train=tr,validation=va,test=te,vocab,lookup,metadata)
end

# BEGIN MODEL -- this block is also shown verbatim in the notebook.
function initialize(V;d=48,hidden=32,seed=2718)
    rng = MersenneTwister(seed)
    (;E=0.1f0 .* randn(rng,Float32,d,V),
      W=sqrt(2f0/d).*randn(rng,Float32,hidden,d),b=zeros(Float32,hidden),
      v=randn(rng,Float32,1,hidden)./sqrt(Float32(hidden)),c=zeros(Float32,1))
end
function logits(m,F)
    a = m.E * F                       # d × batch: mean embeddings
    h = max.(m.W*a .+ m.b,0f0)        # hidden × batch
    m.v*h .+ m.c                     # 1 × batch: no sigmoid before this loss
end
loss(m,F,y) = Flux.logitbinarycrossentropy(logits(m,F),y)
# END MODEL

function evaluate(m,data;batch_size=512)
    probabilities=Float32[]
    total_loss=0.0
    for start in 1:batch_size:length(data.rows)
        idx=start:min(start+batch_size-1,length(data.rows))
        z=logits(m,data.X[:,idx])
        append!(probabilities,vec(Flux.sigmoid.(z)))
        total_loss+=Float64(Flux.logitbinarycrossentropy(z,data.y[:,idx]))*length(idx)
    end
    truth=vec(data.y).==1
    predicted=probabilities.>=0.5
    confusion=[sum((truth.==actual).&(predicted.==guess))
               for actual in (false,true), guess in (false,true)]
    tp,fp,fn=confusion[2,2],confusion[1,2],confusion[2,1]
    (;loss=total_loss/length(truth),accuracy=mean(predicted.==truth),
      f1=2tp/max(2tp+fp+fn,1),confusion,probabilities)
end

# BEGIN TRAINING -- the exact loop is printed as editable Julia source in the notebook.
function fit(data;epochs=8,batch_size=128,learning_rate=0.003f0,
             d=48,hidden=32,seed=2718,budget_seconds=600)
    started=time()
    initial=initialize(length(data.vocab);d,hidden,seed)
    model=deepcopy(initial)
    opt=Flux.setup(Flux.Adam(learning_rate),model)
    rng=MersenneTwister(seed+1)
    initial_train=evaluate(model,data.train)
    initial_validation=evaluate(model,data.validation)
    history=[(;epoch=0,seconds=time()-started,train_loss=initial_train.loss,
        validation_loss=initial_validation.loss,train_accuracy=initial_train.accuracy,
        validation_accuracy=initial_validation.accuracy)]
    best,best_loss,best_epoch=deepcopy(model),initial_validation.loss,0
    steps=0
    stopped=false
    for epoch in 1:epochs
        order=randperm(rng,length(data.train.rows))
        for start in 1:batch_size:length(order)
            if time()-started>=budget_seconds
                stopped=true
                break
            end
            idx=order[start:min(start+batch_size-1,length(order))]
            F,y=data.train.X[:,idx],data.train.y[:,idx]
            value,grad=Flux.withgradient(m->loss(m,F,y),model)
            @assert isfinite(value)
            Flux.update!(opt,model,grad[1])
            steps+=1
        end
        tr,va=evaluate(model,data.train),evaluate(model,data.validation)
        push!(history,(;epoch,seconds=time()-started,train_loss=tr.loss,
            validation_loss=va.loss,train_accuracy=tr.accuracy,
            validation_accuracy=va.accuracy))
        if va.loss < best_loss
            best,best_loss,best_epoch=deepcopy(model),va.loss,epoch
        end
        @printf("epoch %d | %.1f s | train CE %.4f | val CE %.4f | val accuracy %.2f%%\n",
                epoch,time()-started,tr.loss,va.loss,100va.accuracy)
        flush(stdout)
        stopped && break
    end
    # Test labels are used only after validation has selected the checkpoint.
    test=evaluate(best,data.test)
    config=(;epochs,batch_size,learning_rate,d,hidden,seed,budget_seconds)
    (;model=best,initial,history,best_epoch,test,steps,config,
      elapsed_seconds=time()-started,stopped_at_budget=stopped,
      vocab=data.vocab,data_metadata=data.metadata)
end
# END TRAINING

function predict_review(run,text)
    lookup=Dict(w=>i for (i,w) in enumerate(run.vocab))
    ws=words(text)
    ids=isempty(ws) ? [1] : [get(lookup,w,1) for w in ws]
    F=frequencies([ids],length(run.vocab))
    p=only(Flux.sigmoid.(logits(run.model,F)))
    (;probability_positive=p,label=p>=0.5 ? "positive" : "negative",
      unknown_rate=count(==(1),ids)/length(ids))
end

function save_run(run,out)
    mkpath(out)
    serialize(joinpath(out,"run.jls"),run)
    open(joinpath(out,"metrics.csv"),"w") do io
        println(io,"epoch,seconds,train_loss,validation_loss,train_accuracy,validation_accuracy")
        for h in run.history
            println(io,join(values(h),","))
        end
    end
    open(joinpath(out,"summary.txt"),"w") do io
        for (k,v) in pairs(run.data_metadata)
            println(io,k," = ",v)
        end
        println(io,"parameters = ",sum(length,Flux.trainables(run.model)))
        println(io,"best_epoch = ",run.best_epoch)
        println(io,"elapsed_seconds = ",run.elapsed_seconds)
        println(io,"test_accuracy = ",run.test.accuracy)
        println(io,"test_cross_entropy = ",run.test.loss)
        println(io,"test_positive_f1 = ",run.test.f1)
        println(io,"confusion_rows_actual_columns_predicted = ",run.test.confusion)
    end
end
end
