using Serialization
using Dates
using Printf

@assert @isdefined(result) "Run the training cell before saving a checkpoint."

parameter_snapshot = (;
    (name => getfield(par, name) for name in fieldnames(typeof(par)))...,
)
training_periods = length(result.loss_history)
result_device = hasproperty(result, :device) ? result.device : :cpu_legacy
result_elapsed_seconds = hasproperty(result, :elapsed_seconds) ?
    result.elapsed_seconds : nothing
saved_simulation = @isdefined(simulation) ? deepcopy(simulation) : nothing
saved_aggregation = @isdefined(aggregation) ? deepcopy(aggregation) : nothing

checkpoint = (;
    format_version=1,
    model="KS1998 deep-learning Euler method",
    saved_at=string(now()),
    julia_version=string(VERSION),
    parameters=parameter_snapshot,
    steady_state=deepcopy(ss),
    training_config=(;
        periods=training_periods,
        batch_size=10,
        update_every=2,
        learning_rate=1e-3,
        seed=2026,
        device=result_device,
        elapsed_seconds=result_elapsed_seconds,
    ),
    result=deepcopy(result),
    simulation=saved_simulation,
    aggregation=saved_aggregation,
)

mkpath(@__DIR__)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
checkpoint_name =
    "KS1998_N$(par.agents)_H$(par.hidden)_K$(par.shock_draws)_P$(training_periods)_seed2026_$(result_device)_$(timestamp).jls"
checkpoint_path = joinpath(@__DIR__, checkpoint_name)
serialize(checkpoint_path, checkpoint)

loaded_checkpoint = deserialize(checkpoint_path)
@assert loaded_checkpoint.result.theta == result.theta
@assert loaded_checkpoint.result.updates == result.updates
@assert length(loaded_checkpoint.result.loss_history) == training_periods
if saved_aggregation !== nothing
    @assert loaded_checkpoint.aggregation.r2 == aggregation.r2
end

@printf(
    "checkpoint saved and verified: %s (%.2f MiB)\n",
    checkpoint_path,
    filesize(checkpoint_path) / 2.0^20,
)

checkpoint_path
