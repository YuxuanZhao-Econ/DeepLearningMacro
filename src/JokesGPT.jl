module JokesGPT

using Flux, Random, Statistics, LinearAlgebra, Serialization, Printf
export TinyGPT, causal_mask, predict, encode, decode, generate, parameter_count,
       load_corpus, make_batch, bigram_loss, bigram_window_loss, save_checkpoint, load_checkpoint

# The same transparent GPT-style building blocks as Intro_NLP.ipynb.
# Arrays supplied to predict must all live on the same device as the model.
struct CausalAttention{Q,K,V,O}
    query::Q
    key::K
    value::V
    output::O
    heads::Int
end
Flux.@layer CausalAttention trainable=(query, key, value, output)
CausalAttention(d, heads) = CausalAttention(Dense(d => d), Dense(d => d),
    Dense(d => d), Dense(d => d), heads)

function split_heads(Z, heads)
    d, T, B = size(Z)
    reshape(permutedims(reshape(Z, d ÷ heads, heads, T, B), (1,3,2,4)), d ÷ heads, T, heads*B)
end

function (a::CausalAttention)(X, mask)
    d, T, B = size(X)
    dh = d ÷ a.heads
    Q, K, V = split_heads(a.query(X), a.heads), split_heads(a.key(X), a.heads), split_heads(a.value(X), a.heads)
    scores = Flux.NNlib.batched_mul(permutedims(K, (2,1,3)), Q) / sqrt(Float32(dh))
    # Rows = keys, columns = queries. Each query sees its current/past keys.
    weights = Flux.softmax(scores .+ reshape(mask, T, T, 1); dims=1)
    mixed = Flux.NNlib.batched_mul(V, weights)
    joined = reshape(permutedims(reshape(mixed, dh, T, a.heads, B), (1,3,2,4)), d, T, B)
    a.output(joined)
end

struct DecoderBlock{N1,A,N2,F}
    norm1::N1
    attention::A
    norm2::N2
    feedforward::F
end
Flux.@layer DecoderBlock
DecoderBlock(d, heads) = DecoderBlock(LayerNorm(d), CausalAttention(d, heads),
    LayerNorm(d), Chain(Dense(d => 4d, Flux.gelu), Dense(4d => d)))
function (block::DecoderBlock)(H, mask)
    U = H + block.attention(block.norm1(H), mask)
    U + block.feedforward(block.norm2(U))
end

struct TinyGPT{TE,PE,BL,N,O}
    tokens::TE
    positions::PE
    blocks::BL
    finalnorm::N
    head::O
end
Flux.@layer TinyGPT
function TinyGPT(vocab_size; d=128, heads=4, layers=4, context=192)
    @assert d % heads == 0
    TinyGPT(Embedding(0.02f0 .* randn(Float32, d, vocab_size)),
        Embedding(0.02f0 .* randn(Float32, d, context)),
        [DecoderBlock(d, heads) for _ in 1:layers], LayerNorm(d),
        Dense(d => vocab_size; bias=false))
end

causal_mask(T) = Float32[key <= query ? 0 : -Inf for key in 1:T, query in 1:T]
function predict(m::TinyGPT, ids, mask, position_ids)
    T, B = size(ids)
    d, max_context = size(m.positions.weight)
    @assert T <= max_context
    H = m.tokens(ids) .+ reshape(m.positions(position_ids), d, T, 1)
    for block in m.blocks
        H = block(H, mask)
    end
    m.head(m.finalnorm(H))
end

parameter_count(m) = sum(length, Flux.trainables(m))
encode(text, vocab) = [findfirst(==(c), vocab)::Int for c in text]
decode(ids, vocab) = join(vocab[ids])

function load_corpus(data_dir)
    vocab = collect(read(joinpath(data_dir, "vocab.txt"), String))
    ids = Dict(name => encode(read(joinpath(data_dir, name * ".txt"), String), vocab)
               for name in ("train", "validation", "test"))
    vocab, ids
end

function make_batch(ids, context, batch_size, rng)
    starts = rand(rng, 1:length(ids)-context, batch_size)
    X = [ids[s+t-1] for t in 1:context, s in starts]
    Y = [ids[s+t] for t in 1:context, s in starts]
    X, Y
end

function bigram_loss(train, eval_ids, V; smoothing=0.1)
    counts = fill(Float64(smoothing), V, V)
    for i in 1:length(train)-1
        counts[train[i+1], train[i]] += 1
    end
    p = counts ./ sum(counts; dims=1)
    mean(-log(p[eval_ids[i+1], eval_ids[i]]) for i in 1:length(eval_ids)-1)
end

"Score the exact same X/one-hot-Y windows as the GPT, on CPU."
function bigram_window_loss(train, batches, V; smoothing=0.1)
    counts = fill(Float64(smoothing), V, V)
    for i in 1:length(train)-1
        counts[train[i+1],train[i]] += 1
    end
    p = counts ./ sum(counts;dims=1)
    total, n = 0.0, 0
    for (X, onehot_Y) in batches
        inputs = vec(Array(X))
        targets = Flux.onecold(Array(onehot_Y),1:V)
        total -= sum(log.(p[targets .+ V .* (inputs .- 1)]))
        n += length(targets)
    end
    total/n
end

"""Sample one joke on CPU; newline is the end-of-joke marker, not a chat role."""
function generate(m::TinyGPT, vocab; prompt="Why did the", seed=7, temperature=0.8,
                  top_k=30, max_new=180)
    @assert temperature > 0
    rng = MersenneTwister(seed)
    ids = encode("\n" * prompt, vocab)
    context = size(m.positions.weight, 2)
    generated = Int[]
    for _ in 1:max_new
        window = ids[max(1, length(ids)-context+1):end]
        T = length(window)
        logits = vec(predict(m, reshape(window, :, 1), causal_mask(T), collect(1:T))[:,end,1])
        keep = partialsortperm(logits, 1:min(top_k, length(vocab)); rev=true)
        p = Flux.softmax(logits[keep] ./ Float32(temperature))
        r = rand(rng)
        next = keep[min(searchsortedfirst(cumsum(p), r), length(keep))]
        push!(ids, next)
        vocab[next] == '\n' && break
        push!(generated, next)
    end
    prompt * decode(generated, vocab)
end

function save_checkpoint(path, cpu_model, vocab, config, history, metadata)
    # Portable within this recorded Julia/Flux project; not a pretrained download.
    mkpath(dirname(path))
    temp = path * ".tmp"
    serialize(temp, (; model=cpu_model, vocab, config, history, metadata))
    mv(temp, path; force=true)
end
load_checkpoint(path) = deserialize(path)

end # module
