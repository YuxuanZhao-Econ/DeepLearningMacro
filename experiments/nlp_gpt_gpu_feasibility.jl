# GPU feasibility / throughput benchmark, not a language-training experiment.
# Run from the repository root:
#   julia --startup-file=no --project=. experiments/nlp_gpt_gpu_feasibility.jl
# Uses random fixed tokens and the same GPT-style components as Intro_NLP.ipynb.
# Explicit CUDA array adaptation avoids a silent fallback to CPU. Position IDs
# and the fixed-length causal mask stay on the device. No corpus is downloaded.
# First-step compilation, evaluation, data loading, and checkpoint overhead must
# be distinguished from the steady-state training-only extrapolation.
using CUDA, Flux, Random, LinearAlgebra, Statistics, Printf
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
Random.seed!(42)
const BENCH_T = 128
const V = 65
const POSITION_IDS = CUDA.cu(collect(1:BENCH_T))
const MASK = CUDA.cu([key <= query ? 0f0 : -Inf32 for key in 1:BENCH_T, query in 1:BENCH_T])
causal_bias(T) = (@assert T == BENCH_T; MASK)
struct CausalAttention{Q,K,VV,O}
    query::Q
    key::K
    value::VV
    output::O
    heads::Int
end
Flux.@layer CausalAttention trainable=(query, key, value, output)

function CausalAttention(d::Int, heads::Int)
    @assert d % heads == 0
    CausalAttention(Dense(d => d), Dense(d => d), Dense(d => d), Dense(d => d), heads)
end

function split_heads(Z, heads)
    d, T, B = size(Z)
    dh = d ÷ heads
    reshape(permutedims(reshape(Z, dh, heads, T, B), (1,3,2,4)), dh, T, heads*B)
end

function attention_forward(a::CausalAttention, X)
    d, T, B = size(X)
    dh = d ÷ a.heads
    Q = split_heads(a.query(X), a.heads)
    K = split_heads(a.key(X), a.heads)
    VV = split_heads(a.value(X), a.heads)
    scores = Flux.NNlib.batched_mul(permutedims(K, (2,1,3)), Q) / sqrt(Float32(dh))
    weights = Flux.softmax(scores .+ reshape(causal_bias(T), T, T, 1); dims=1)
    mixed = Flux.NNlib.batched_mul(VV, weights)
    joined = reshape(permutedims(reshape(mixed, dh, T, a.heads, B), (1,3,2,4)), d, T, B)
    a.output(joined), weights
end
(a::CausalAttention)(X) = first(attention_forward(a, X))

struct DecoderBlock{N1,A,N2,F}
    norm1::N1
    attention::A
    norm2::N2
    feedforward::F
end
Flux.@layer DecoderBlock
DecoderBlock(d, heads) = DecoderBlock(LayerNorm(d), CausalAttention(d, heads),
    LayerNorm(d), Chain(Dense(d => 4d, Flux.gelu), Dense(4d => d)))
function (block::DecoderBlock)(H)
    U = H + block.attention(block.norm1(H))
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
function TinyGPT(vocab_size; d=32, heads=2, layers=2, context=16)
    TinyGPT(Embedding(0.02f0 .* randn(Float32, d, vocab_size)),
            Embedding(0.02f0 .* randn(Float32, d, context)),
            [DecoderBlock(d, heads) for _ in 1:layers], LayerNorm(d),
            Dense(d => vocab_size; bias=false))
end

function initial_states(m::TinyGPT, ids)
    T, B = size(ids)
    d, max_context = size(m.positions.weight)
    @assert 1 <= T <= max_context "Prefix exceeds the model's context length."
    m.tokens(ids) .+ reshape(m.positions(POSITION_IDS), d, T, 1)
end
function hidden_states(m::TinyGPT, ids)
    H = initial_states(m, ids)
    for block in m.blocks
        H = block(H)
    end
    m.finalnorm(H)
end
(m::TinyGPT)(ids) = m.head(hidden_states(m, ids))


function run_benchmark()
    batch_size = 16
    m = Flux.fmap(CUDA.cu, TinyGPT(V; d=128, heads=4, layers=4, context=BENCH_T))
    @assert all(parameter isa CUDA.CuArray for parameter in Flux.trainables(m))
    parameters = sum(length, Flux.trainables(m))
    X = CUDA.cu(rand(1:V, BENCH_T, batch_size))
    Y = rand(1:V, BENCH_T, batch_size)
    targets = CUDA.cu(Float32.(Flux.onehotbatch(vec(Y), 1:V)))
    opt = Flux.setup(Flux.Adam(0.001f0), m)
    before = Array(m.tokens.weight)
    function step!()
        loss, grads = Flux.withgradient(m) do candidate
            Flux.logitcrossentropy(reshape(candidate(X), V, :), targets)
        end
        Flux.update!(opt, m, grads[1])
        loss
    end
    @printf("Benchmark: %d parameters; width 128, 4 layers, 4 heads, context 128, batch 16\n", parameters)
    println("Timing data are RANDOM tokens: this is a throughput test, not language training.")
    flush(stdout)
    warmup = @elapsed CUDA.@sync first_loss = step!()
    @printf("First forward/backward/update passed; warmup including compilation: %.2f s\n", warmup)
    flush(stdout)
    CUDA.@sync for _ in 1:4; step!(); end
    samples = Float64[]
    for _ in 1:20
        duration = @elapsed CUDA.@sync step!()
        push!(samples, duration)
    end
    @assert all(isfinite, Array(m.tokens.weight))
    movement = norm(Array(m.tokens.weight) - before)
    @assert movement > 0
    @printf("Median step: %.4f s; mean: %.4f s; 20-step range: %.4f-%.4f s\n", median(samples), mean(samples), minimum(samples), maximum(samples))
    @printf("Approx steady-state tokens/sec: %.0f\n", BENCH_T * batch_size / median(samples))
    @printf("5000-step TRAINING-ONLY extrapolation: %.2f min\n", 5000median(samples)/60)
    @printf("Input-embedding update norm: %.5f | free GPU memory after test: %.2f GiB\n", movement, CUDA.available_memory()/2.0^30)
    println("GPU forward, backward, and optimizer update all passed.")
end
run_benchmark()
