module EnglishBasics
using Flux, Random, Statistics
using ..JokesGPT: TinyGPT, predict, causal_mask
export BPETokenizer, load_bpe, bpe_encode, bpe_decode, load_tokens, sample_text, check_bpe

struct BPETokenizer
    tokens::Vector{String}
    bytext::Dict{String,Int}
    characters::Dict{Char,Int}
    merges::Dict{Tuple{Int,Int},Tuple{Int,Int}}
    bos::Int
    eos::Int
    unk::Int
end

function load_bpe(dir)
    tokens = [String(hex2bytes(s)) for s in readlines(joinpath(dir,"tokens_hex.txt"))]
    bytext = Dict(s=>i for (i,s) in enumerate(tokens))
    characters = Dict(only(s)=>i for (i,s) in enumerate(tokens) if length(s)==1)
    merges = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (rank,line) in enumerate(readlines(joinpath(dir,"merges.tsv")))
        left,right,new = parse.(Int,split(line,'\t'))
        @assert tokens[left]*tokens[right] == tokens[new]
        merges[(left,right)] = (rank,new)
    end
    BPETokenizer(tokens,bytext,characters,merges,bytext["<bos>"],bytext["<eos>"],bytext["<unk>"])
end

function bpe_encode(tok::BPETokenizer,text)
    ids = Int[]
    for matched in eachmatch(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+",text)
        append!(ids,bpe_encode_piece(tok,matched.match))
    end
    ids
end
function bpe_encode_piece(tok::BPETokenizer,text)
    ids = [get(tok.characters,c,0) for c in text]
    any(iszero,ids) && throw(ArgumentError("Use ASCII English text with this teaching tokenizer."))
    # Highest-priority available adjacent merge, with left-to-right tie breaking.
    while length(ids)>1
        best_rank, best_at, best_new = typemax(Int),0,0
        for i in 1:length(ids)-1
            rank,new = get(tok.merges,(ids[i],ids[i+1]),(typemax(Int),0))
            if rank < best_rank
                best_rank,best_at,best_new = rank,i,new
            end
        end
        best_at==0 && break
        ids[best_at] = best_new
        deleteat!(ids,best_at+1)
    end
    ids
end
bpe_decode(tok,ids) = join(tok.tokens[i] for i in ids if i ∉ (tok.bos,tok.eos,tok.unk))
load_tokens(path) = collect(reinterpret(UInt16,read(path))) # little-endian x86 local files

function check_bpe(tok,dir)
    lines = readlines(joinpath(dir,"tokenizer_probes.tsv"))
    for line in lines
        text_hex,id_text = split(line,'\t')
        text = String(hex2bytes(text_hex))
        ids = parse.(Int,split(id_text,','))
        @assert bpe_encode(tok,text) == ids "Julia and reference BPE encoders disagree."
        @assert bpe_decode(tok,ids) == text
    end
    length(lines)
end

"Sample a continuation; prompt is supplied text, not model-generated text."
function sample_text(model,tok;prompt="Once upon a time, there was a little",seed=8,
                     temperature=0.65,top_k=40,max_new=100,device=identity)
    @assert temperature>0
    rng = MersenneTwister(seed)
    ids = vcat(tok.bos,bpe_encode(tok,prompt))
    generated = Int[]
    ended = false
    max_context = size(model.positions.weight,2)
    for step in 1:max_new
        window = ids[max(1,length(ids)-max_context+1):end]
        T = length(window)
        logits = vec(Array(predict(model,device(reshape(window,:,1)),device(causal_mask(T)),
                                    device(collect(1:T)))[:,end,1]))
        logits[tok.bos] = -Inf32
        logits[tok.unk] = -Inf32
        keep = partialsortperm(logits,1:min(top_k,length(logits));rev=true)
        p = Flux.softmax(logits[keep]./Float32(temperature))
        next = keep[min(searchsortedfirst(cumsum(p),rand(rng)),length(keep))]
        if next==tok.eos
            ended=true
            break
        end
        push!(generated,next)
        push!(ids,next)
    end
    (;prompt,continuation=bpe_decode(tok,generated),text=prompt*bpe_decode(tok,generated),
       ended_at_eos=ended,new_tokens=length(generated),seed,temperature)
end
end
