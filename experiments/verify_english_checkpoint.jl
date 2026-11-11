# Fresh CPU process: weights, tokenizer agreement, data provenance, serialization, causality.
using Flux, LinearAlgebra, Serialization, SHA, Printf
include(joinpath(@__DIR__,"..","src","JokesGPT.jl"))
include(joinpath(@__DIR__,"..","src","EnglishBasics.jl"))
using .JokesGPT, .EnglishBasics
BLAS.set_num_threads(1)
root=normpath(joinpath(@__DIR__,".."))
data=joinpath(root,"reference","datasets","english_basics")
out=joinpath(root,"output","english_basics")
@assert isfile(joinpath(out,"run_summary.txt")) "Complete the training run before final verification."
tok=load_bpe(data)
probes=check_bpe(tok,data)
best=load_checkpoint(joinpath(out,"best.jls"))
initial=load_checkpoint(joinpath(out,"initial.jls"))
@assert tok.tokens==best.vocab
@assert best.metadata["tokenizer_sha256"]==bytes2hex(sha256(read(joinpath(data,"tokenizer.json"))))
for name in ("train","validation","test")
    @assert best.metadata[name*"_sha256"]==bytes2hex(sha256(read(joinpath(data,name*".bin"))))
end
@assert best.metadata["fine_tuning"]==false
@assert all(p isa Array for p in Flux.trainables(best.model))
@assert all(p -> all(isfinite,p),Flux.trainables(best.model))
movement=norm(best.model.tokens.weight-initial.model.tokens.weight)
@assert movement>0
reference=deserialize(joinpath(out,"verification_probe.jls"))
probe=reference.probe
T=size(probe,1)
L=predict(best.model,probe,causal_mask(T),collect(1:T))
reload_error=maximum(abs.(L-reference.logits))
@assert reload_error<1f-5
again=load_checkpoint(joinpath(out,"best.jls"))
@assert L==predict(again.model,probe,causal_mask(T),collect(1:T))
changed=copy(probe)
changed[4:end,1].=tok.eos
Lc=predict(best.model,changed,causal_mask(T),collect(1:T))
Lp=predict(best.model,probe[1:3,:],causal_mask(3),collect(1:3))
future_effect=maximum(abs.(L[:,1:3,:]-Lc[:,1:3,:]))
prefix_error=maximum(abs.(L[:,1:3,:]-Lp))
@assert future_effect<1f-5
@assert prefix_error<1f-4
open(joinpath(out,"verification.txt"),"w") do f
    println(f,"Fresh CPU Julia process; no GPU or pretrained weights used for verification.")
    println(f,"Julia/reference BPE agreement: ",probes," cases")
    println(f,"Vocabulary and training/validation/test hashes matched.")
    println(f,"Fresh-process CPU reference error: ",reload_error)
    println(f,"Repeated checkpoint loading: identical CPU logits.")
    println(f,"Future-input effect: ",future_effect)
    println(f,"Full vs separate prefix difference: ",prefix_error)
    println(f,"Embedding movement norm: ",movement)
end
println(read(joinpath(out,"verification.txt"),String))
