# From scratch, TinyStories only. Run from the repository root with --project=.
const ENGLISH_PROCESS_STARTED = time() # includes package loading and first-update compilation
using CUDA, Flux, Random, LinearAlgebra, Statistics, Printf, Serialization, SHA
if !isdefined(@__MODULE__,:JokesGPT)
    include(joinpath(@__DIR__,"..","src","JokesGPT.jl"))
end
if !isdefined(@__MODULE__,:EnglishBasics)
    include(joinpath(@__DIR__,"..","src","EnglishBasics.jl"))
end
using .JokesGPT, .EnglishBasics
CUDA.allowscalar(false)
CUDA.math_mode!(CUDA.FAST_MATH;precision=:TensorFloat32)
BLAS.set_num_threads(1)

function train_english(;budget_seconds=40*60,max_steps=100000,context=128,batch_size=8,
                       d=256,layers=6,heads=8,eval_every=1000)
    started = ENGLISH_PROCESS_STARTED
    root = normpath(joinpath(@__DIR__,".."))
    data = joinpath(root,"reference","datasets","english_basics")
    out = joinpath(root,"output","english_basics")
    mkpath(out)
    tok = load_bpe(data)
    @printf("Julia/reference BPE agreement: %d probes passed.\n",check_bpe(tok,data))
    corpus = Dict(name=>load_tokens(joinpath(data,name*".bin")) for name in ("train","validation","test"))
    V = length(tok.tokens)
    config = (;d,layers,heads,context,batch_size,seed=314159,max_steps,budget_seconds,
               peak_learning_rate=0.0006f0,warmup_steps=150,precision="Float32 arrays with TF32 matrix products")
    Random.seed!(config.seed)
    rng = MersenneTwister(config.seed)
    initial = TinyGPT(V;d,layers,heads,context)
    model = Flux.fmap(CUDA.cu,initial)
    @assert all(p isa CUDA.CuArray for p in Flux.trainables(model))
    cpu_copy(m) = Flux.fmap(x -> x isa CUDA.CuArray ? Array(x) : x,m)
    positions,mask = CUDA.cu(collect(1:context)),CUDA.cu(causal_mask(context))
    function batch(ids,rng)
        X,Y = make_batch(ids,context,batch_size,rng)
        CUDA.cu(Int.(X)),CUDA.cu(Float32.(Flux.onehotbatch(vec(Y),1:V)))
    end
    function evaluate(m,batches)
        mean(Float64(Flux.logitcrossentropy(reshape(predict(m,X,mask,positions),V,:),Y)) for (X,Y) in batches)
    end
    eval_rng = MersenneTwister(20261028)
    train_eval = [batch(corpus["train"],eval_rng) for _ in 1:8]
    val_eval = [batch(corpus["validation"],eval_rng) for _ in 1:16]
    opt = Flux.setup(Flux.OptimiserChain(Flux.ClipNorm(1f0),Flux.Adam(config.peak_learning_rate)),model)
    history = NamedTuple[]
    metadata = Dict{String,Any}("gpu"=>CUDA.name(CUDA.device()),"julia"=>string(VERSION),
        "objective"=>"next BPE token on TinyStories; ALL parameters initialized randomly",
        "parameters"=>parameter_count(initial),"source"=>"https://huggingface.co/datasets/roneneldan/TinyStories",
        "fine_tuning"=>false,"jokes_in_training"=>false,
        "tokenizer_sha256"=>bytes2hex(sha256(read(joinpath(data,"tokenizer.json")))))
    for name in ("train","validation","test")
        metadata[name*"_sha256"] = bytes2hex(sha256(read(joinpath(data,name*".bin"))))
    end
    save_checkpoint(joinpath(out,"initial.jls"),initial,tok.tokens,config,copy(history),copy(metadata))
    initial_train,initial_val = evaluate(model,train_eval),evaluate(model,val_eval)
    push!(history,(;step=0,seconds=time()-started,train_loss=initial_train,validation_loss=initial_val,learning_rate=0.0))
    best_val,best_step,stale,last_step = initial_val,0,0,0
    save_checkpoint(joinpath(out,"best.jls"),initial,tok.tokens,config,copy(history),copy(metadata))
    open(joinpath(out,"metrics.csv"),"w") do f
        println(f,"step,seconds,train_loss,validation_loss,learning_rate")
        @printf(f,"0,%.6f,%.6f,%.6f,0\n",time()-started,initial_train,initial_val)
    end
    @printf("GPU %s | %d parameters | %d training BPE tokens | initial val %.4f\n",
            metadata["gpu"],parameter_count(model),length(corpus["train"]),initial_val)
    flush(stdout)
    stop_reason = "maximum steps"
    timed_steps = Float64[]
    last_eta = 0f0
    for step in 1:max_steps
        elapsed = time()-started
        if elapsed > budget_seconds-120
            stop_reason = "wall-clock budget; 120 seconds reserved for final evaluation/saving"
            break
        end
        # Warm up, then decay by elapsed budget rather than assuming a GPU speed.
        fraction = clamp(elapsed/(budget_seconds-120),0.0,1.0)
        eta = Float32(config.peak_learning_rate*min(step/config.warmup_steps,1)*(0.1+0.9*0.5*(1+cos(pi*fraction))))
        last_eta = eta
        Flux.Optimisers.adjust!(opt,eta)
        tick = time()
        X,Y = batch(corpus["train"],rng)
        loss,grads = Flux.withgradient(model) do candidate
            Flux.logitcrossentropy(reshape(predict(candidate,X,mask,positions),V,:),Y)
        end
        @assert isfinite(loss) "Non-finite training loss."
        Flux.update!(opt,model,grads[1])
        last_step = step
        if 11 <= step <= 60
            CUDA.synchronize()
            push!(timed_steps,time()-tick)
        end
        if step==1
            CUDA.synchronize()
            @printf("First complete GPU update passed after %.1f seconds, including startup/compilation.\n",time()-started)
            flush(stdout)
        elseif step==60
            metadata["early_median_step_seconds"] = median(timed_steps)
            metadata["early_tokens_per_second"] = context*batch_size/median(timed_steps)
            @printf("Measured real-text training: median %.4f s/step | %.0f BPE tokens/s.\n",
                    median(timed_steps),metadata["early_tokens_per_second"])
            flush(stdout)
        end
        if step==200 || step%eval_every==0 || step==max_steps
            tr,va = evaluate(model,train_eval),evaluate(model,val_eval)
            seconds = time()-started
            push!(history,(;step,seconds,train_loss=tr,validation_loss=va,learning_rate=Float64(eta)))
            open(joinpath(out,"metrics.csv"),"a") do f
                @printf(f,"%d,%.6f,%.6f,%.6f,%.8f\n",step,seconds,tr,va,eta)
            end
            if va < best_val
                best_val,best_step,stale = va,step,0
                best_meta = merge(metadata,Dict("best_step"=>step,"best_validation_loss"=>va))
                save_checkpoint(joinpath(out,"best.jls"),cpu_copy(model),tok.tokens,config,copy(history),best_meta)
            else
                stale += 1
            end
            @printf("step %6d | %6.1f s | train %.4f | validation %.4f | best %d\n",step,seconds,tr,va,best_step)
            flush(stdout)
            if step%5000==0
                samples = [sample_text(model,tok;prompt,seed=7+i,max_new=90,device=CUDA.cu)
                    for (i,prompt) in enumerate(["Once upon a time, there was a little",
                        "The cat looked at the dog and said,", "One day, a little girl found a"])]
                serialize(joinpath(out,"samples_step_$(step).jls"),samples)
                open(joinpath(out,"samples_step_$(step).txt"),"w") do f
                    for s in samples
                        println(f,"PROMPT: ",s.prompt,"\nCONTINUATION: ",s.continuation,"\n")
                    end
                end
            end
            if stale>=10
                stop_reason="validation early stopping: ten evaluations without improvement"
                break
            end
        end
    end
    # If the budget lands between scheduled evaluations, evaluate the current model
    # on validation only, to avoid discarding the final interval of useful training.
    if last_step > history[end].step
        tr,va = evaluate(model,train_eval),evaluate(model,val_eval)
        push!(history,(;step=last_step,seconds=time()-started,train_loss=tr,validation_loss=va,learning_rate=Float64(last_eta)))
        if va < best_val
            best_val,best_step = va,last_step
            save_checkpoint(joinpath(out,"best.jls"),cpu_copy(model),tok.tokens,config,copy(history),copy(metadata))
        end
    end
    best = load_checkpoint(joinpath(out,"best.jls"))
    best_gpu = Flux.fmap(CUDA.cu,best.model)
    test_rng = MersenneTwister(991)
    test_batches = [batch(corpus["test"],test_rng) for _ in 1:16]
    test_loss = evaluate(best_gpu,test_batches) # one test evaluation, after selection
    samples = [sample_text(best_gpu,tok;prompt,seed=7+i,max_new=110,device=CUDA.cu)
        for (i,prompt) in enumerate(["Once upon a time, there was a little",
            "The cat looked at the dog and said,", "One day, a little girl found a",
            "\"Hello,\" said Tom. \"How are you?"])]
    serialize(joinpath(out,"samples.jls"),samples)
    open(joinpath(out,"samples.txt"),"w") do f
        for s in samples
            println(f,"PROMPT: ",s.prompt,"\nCONTINUATION: ",s.continuation,
                    "\nStopped at EOS: ",s.ended_at_eos,"\n")
        end
    end
    probe = reshape(vcat(tok.bos,bpe_encode(tok,"The little cat was happy.")),:,1)
    T = size(probe,1)
    probe_logits = predict(best.model,probe,causal_mask(T),collect(1:T))
    serialize(joinpath(out,"verification_probe.jls"),(;probe,logits=probe_logits))
    final_meta = merge(metadata,Dict("best_step"=>best_step,"last_step"=>last_step,
        "best_validation_loss"=>best_val,"test_loss"=>test_loss,"elapsed_seconds"=>time()-started,
        "tokens_processed"=>last_step*context*batch_size,"stop_reason"=>stop_reason))
    save_checkpoint(joinpath(out,"best.jls"),best.model,tok.tokens,config,history,final_meta)
    serialize(joinpath(out,"run_summary.jls"),final_meta)
    open(joinpath(out,"run_summary.txt"),"w") do f
        for key in sort(collect(keys(final_meta)))
            println(f,key," = ",final_meta[key])
        end
    end
    @printf("Finished: %.2f minutes | best step %d | test CE %.4f.\n",(time()-started)/60,best_step,test_loss)
    flush(stdout)
    best
end
if abspath(PROGRAM_FILE)==@__FILE__
    train_english()
end
