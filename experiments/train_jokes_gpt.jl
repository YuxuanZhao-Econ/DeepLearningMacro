# Run: julia --startup-file=no --project=. experiments/train_jokes_gpt.jl
# The trained object is a character language model, not an instruction/chat model.
using CUDA, Flux, Random, LinearAlgebra, Statistics, Printf, SHA, Serialization
if !isdefined(@__MODULE__, :JokesGPT)
    include(joinpath(@__DIR__, "..", "src", "JokesGPT.jl"))
end
using .JokesGPT
CUDA.allowscalar(false)
BLAS.set_num_threads(1)

function train_jokes(; max_steps=16000, budget_seconds=45*60, eval_every=500,
                      context=192, batch_size=16, d=128, layers=4, heads=4)
    started = time()
    root = normpath(joinpath(@__DIR__, ".."))
    data = joinpath(root, "reference", "datasets", "short_jokes")
    out = joinpath(root, "output", "jokes_gpt")
    mkpath(out)
    vocab, corpus = load_corpus(data)
    V = length(vocab)
    config = (; d, layers, heads, context, batch_size, seed=42, max_steps,
               learning_rate=0.0006f0, budget_seconds)
    Random.seed!(42)
    rng = MersenneTwister(42)
    cpu_initial = TinyGPT(V; d, layers, heads, context)
    # Flux.gpu can silently select CPU in this local environment: assert the device.
    model = Flux.fmap(CUDA.cu, cpu_initial)
    @assert all(p isa CUDA.CuArray for p in Flux.trainables(model))
    mask = CUDA.cu(causal_mask(context))
    positions = CUDA.cu(collect(1:context))
    opt = Flux.setup(Flux.OptimiserChain(Flux.ClipNorm(1f0), Flux.Adam(config.learning_rate)), model)
    cpu_copy(m) = Flux.fmap(x -> x isa CUDA.CuArray ? Array(x) : x, m)
    to_batch(ids, rng) = begin
        X, Y = make_batch(ids, context, batch_size, rng)
        CUDA.cu(X), CUDA.cu(Float32.(Flux.onehotbatch(vec(Y), 1:V)))
    end
    # Fixed evaluation windows make checkpoint comparisons reproducible.
    eval_rng = MersenneTwister(2026)
    train_eval = [to_batch(corpus["train"], eval_rng) for _ in 1:8]
    val_eval = [to_batch(corpus["validation"], eval_rng) for _ in 1:16]
    evaluate(m, batches) = mean(Float64(Flux.logitcrossentropy(
        reshape(predict(m, X, mask, positions), V, :), Y)) for (X,Y) in batches)
    history = NamedTuple[]
    metadata = Dict("source_sha256" => bytes2hex(sha256(read(joinpath(data, "shortjokes.csv")))),
        "train_sha256" => bytes2hex(sha256(read(joinpath(data, "train.txt")))),
        "validation_sha256" => bytes2hex(sha256(read(joinpath(data, "validation.txt")))),
        "test_sha256" => bytes2hex(sha256(read(joinpath(data, "test.txt")))),
        "manifest_sha256" => bytes2hex(sha256(read(joinpath(data, "manifest.json")))),
        "gpu" => CUDA.name(CUDA.device()), "julia" => string(VERSION),
        "objective" => "next character; all trainable parameters initialized randomly")
    save_checkpoint(joinpath(out, "initial.jls"), cpu_initial, vocab, config, history, metadata)
    initial_train, initial_val = evaluate(model, train_eval), evaluate(model, val_eval)
    push!(history, (; step=0, seconds=time()-started, train_loss=initial_train, validation_loss=initial_val))
    best_val, best_step, stale = initial_val, 0, 0
    save_checkpoint(joinpath(out, "best.jls"), cpu_initial, vocab, config, copy(history),
        merge(metadata,Dict("best_step"=>0,"best_validation_loss"=>best_val)))
    baseline = bigram_window_loss(corpus["train"], val_eval, V)
    open(joinpath(out, "metrics.csv"), "w") do f
        println(f, "step,seconds,train_loss,validation_loss")
        @printf(f, "0,%.6f,%.6f,%.6f\n", time()-started, initial_train, initial_val)
    end
    @printf("GPU: %s | %d parameters | %d training characters\n", metadata["gpu"], parameter_count(model), length(corpus["train"]))
    @printf("Initial validation CE %.4f | bigram validation CE %.4f\n", initial_val, baseline)
    flush(stdout)
    stop_reason = "maximum steps"
    last_step = 0
    for step in 1:max_steps
        if time()-started > budget_seconds-60
            stop_reason = "wall-clock budget (60 seconds reserved for saving)"
            break
        end
        X, targets = to_batch(corpus["train"], rng)
        loss, grads = Flux.withgradient(model) do candidate
            Flux.logitcrossentropy(reshape(predict(candidate, X, mask, positions), V, :), targets)
        end
        @assert isfinite(loss) "Non-finite training loss."
        Flux.update!(opt, model, grads[1])
        last_step = step
        if step == 1
            CUDA.synchronize()
            @printf("First complete GPU update passed after %.1f s (includes compilation).\n", time()-started)
            flush(stdout)
        end
        if step % eval_every == 0 || step == max_steps
            train_loss, validation_loss = evaluate(model, train_eval), evaluate(model, val_eval)
            seconds = time()-started
            push!(history, (; step, seconds, train_loss, validation_loss))
            open(joinpath(out, "metrics.csv"), "a") do f
                @printf(f, "%d,%.6f,%.6f,%.6f\n", step, seconds, train_loss, validation_loss)
            end
            if validation_loss < best_val
                best_val, best_step, stale = validation_loss, step, 0
                best_meta = merge(metadata, Dict("best_step" => step, "best_validation_loss" => best_val))
                save_checkpoint(joinpath(out, "best.jls"), cpu_copy(model), vocab, config, copy(history), best_meta)
            else
                stale += 1
            end
            @printf("step %5d | %6.1f s | train %.4f | validation %.4f | best %d\n", step, seconds, train_loss, validation_loss, best_step)
            flush(stdout)
            if stale >= 6
                stop_reason = "validation early stopping: six evaluations without improvement"
                break
            end
        end
    end
    @assert isfile(joinpath(out, "best.jls")) "No evaluated checkpoint was produced."
    best = load_checkpoint(joinpath(out, "best.jls"))
    # The test set is touched only once, after validation has selected the checkpoint.
    test_rng = MersenneTwister(991)
    test_batches = [to_batch(corpus["test"], test_rng) for _ in 1:16]
    restored = Flux.fmap(CUDA.cu, best.model)
    test_loss = evaluate(restored, test_batches)
    test_bigram = bigram_window_loss(corpus["train"], test_batches, V)
    # Refresh the notebook's CPU reference whenever a fresh run replaces the model.
    probe = reshape(encode("Why did the cat?",vocab),:,1)
    probe_T = size(probe,1)
    probe_logits = predict(best.model,probe,causal_mask(probe_T),collect(1:probe_T))
    serialize(joinpath(out,"verification_probe.jls"),(;probe,logits=probe_logits))
    final_meta = merge(metadata, Dict("best_step" => best_step, "last_step" => last_step,
        "best_validation_loss" => best_val, "test_loss" => test_loss,
        "validation_bigram_loss" => baseline, "test_bigram_loss" => test_bigram,
        "elapsed_seconds" => time()-started, "stop_reason" => stop_reason,
        "tokens_processed" => last_step*context*batch_size))
    save_checkpoint(joinpath(out, "best.jls"), best.model, vocab, config, history, final_meta)
    serialize(joinpath(out, "run_summary.jls"), final_meta)
    open(joinpath(out, "run_summary.txt"), "w") do f
        for key in sort(collect(keys(final_meta)))
            println(f, key, " = ", final_meta[key])
        end
        println(f, "parameters = ", parameter_count(best.model))
    end
    println("Saved best.jls and run_summary.txt; training finished in $(round((time()-started)/60; digits=2)) minutes.")
    flush(stdout)
    best
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_jokes()
end
