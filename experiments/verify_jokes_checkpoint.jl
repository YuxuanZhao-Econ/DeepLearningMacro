# Fresh-process checkpoint verification and matched-window baseline calculation.
# Run after training: julia --startup-file=no --project=. experiments/verify_jokes_checkpoint.jl
using Flux, Random, LinearAlgebra, Serialization, SHA, Printf
include(joinpath(@__DIR__,"..","src","JokesGPT.jl"))
using .JokesGPT
BLAS.set_num_threads(1)
root = normpath(joinpath(@__DIR__,".."))
data = joinpath(root,"reference","datasets","short_jokes")
out = joinpath(root,"output","jokes_gpt")
@assert isfile(joinpath(out,"run_summary.txt")) "Training must finish first."
best = load_checkpoint(joinpath(out,"best.jls"))
initial = load_checkpoint(joinpath(out,"initial.jls"))
vocab, corpus = load_corpus(data)
@assert best.vocab == vocab
for name in ("train","validation","test")
    @assert best.metadata[name*"_sha256"] == bytes2hex(sha256(read(joinpath(data,name*".txt"))))
end
@assert best.metadata["best_step"] > 0
@assert all(p isa Array for p in Flux.trainables(best.model))
@assert all(p -> all(isfinite,p),Flux.trainables(best.model))
@assert norm(best.model.tokens.weight-initial.model.tokens.weight) > 0

# The first local trainer scored the bigram on the full held-out streams.
# Score it on exactly the GPT's fixed windows for the notebook comparison.
function cpu_batch(ids,rng)
    X,Y = make_batch(ids,best.config.context,best.config.batch_size,rng)
    X,Float32.(Flux.onehotbatch(vec(Y),1:length(vocab)))
end
eval_rng = MersenneTwister(2026)
for _ in 1:8; cpu_batch(corpus["train"],eval_rng); end
val_batches = [cpu_batch(corpus["validation"],eval_rng) for _ in 1:16]
test_rng = MersenneTwister(991)
test_batches = [cpu_batch(corpus["test"],test_rng) for _ in 1:16]
meta = copy(best.metadata)
meta["validation_bigram_loss"] = bigram_window_loss(corpus["train"],val_batches,length(vocab))
meta["test_bigram_loss"] = bigram_window_loss(corpus["train"],test_batches,length(vocab))
meta["baseline_evaluation"] = "same fixed X/Y evaluation windows as GPT"
save_checkpoint(joinpath(out,"best.jls"),best.model,vocab,best.config,best.history,meta)
serialize(joinpath(out,"run_summary.jls"),meta)
open(joinpath(out,"run_summary.txt"),"w") do f
    for key in sort(collect(keys(meta)))
        println(f,key," = ",meta[key])
    end
    println(f,"parameters = ",parameter_count(best.model))
end

# Serialize/reload and require identical CPU predictions, not merely a readable file.
probe = reshape(encode("Why did the cat?",vocab),:,1)
T = size(probe,1)
logits = predict(best.model,probe,causal_mask(T),collect(1:T))
reloaded = load_checkpoint(joinpath(out,"best.jls"))
logits_reloaded = predict(reloaded.model,probe,causal_mask(T),collect(1:T))
@assert logits == logits_reloaded
changed = copy(probe)
changed[6:end,1] .= encode("Z",vocab)[1]
future_logits = predict(reloaded.model,changed,causal_mask(T),collect(1:T))
future_effect = maximum(abs.(logits[:,1:5,:]-future_logits[:,1:5,:]))
@assert future_effect < 1f-5
serialize(joinpath(out,"verification_probe.jls"),(;probe,logits))
open(joinpath(out,"verification.txt"),"w") do f
    println(f,"Fresh Julia process loaded finite CPU checkpoint parameters; data hashes matched.")
    println(f,"Serialize/reload prediction difference = ",maximum(abs.(logits-logits_reloaded)))
    println(f,"Changing future characters: earlier-logit effect = ",future_effect)
    println(f,"Embedding movement norm = ",norm(best.model.tokens.weight-initial.model.tokens.weight))
    println(f,"Matched-window validation bigram CE = ",meta["validation_bigram_loss"])
    println(f,"Matched-window test bigram CE = ",meta["test_bigram_loss"])
end
println(read(joinpath(out,"verification.txt"),String))
for (i,prompt) in enumerate(["Why did the computer","What do you call a","Why did the chicken"])
    println(i,". ",generate(reloaded.model,vocab;prompt,seed=7+i,temperature=0.8,top_k=30,max_new=150))
end
