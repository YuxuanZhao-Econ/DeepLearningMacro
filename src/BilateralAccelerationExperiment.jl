module BilateralAccelerationExperiment

using Flux
using Random
using Statistics
using Printf

export benchmark_population_architectures,
       train_price_predictor,
       benchmark_price_solver,
       run_bilateral_acceleration_experiment

const F32 = Float32

sigmoid_fast(x) = inv(one(x) + exp(-x))

"""Median wall time in seconds after one warm-up call."""
function median_seconds(f; repeats::Int=5)
    f()
    GC.gc()
    times = Vector{Float64}(undef, repeats)
    for repetition in 1:repeats
        times[repetition] = @elapsed f()
    end
    return median(times)
end

parameter_count(model) = sum(length, Flux.trainables(model); init=0)

struct NaiveBilateralPolicy{H,F}
    household_net::H
    firm_net::F
end
Flux.@layer NaiveBilateralPolicy

struct SharedBilateralPolicy{EH,EF,DH,DF}
    household_encoder::EH
    firm_encoder::EF
    household_decoder::DH
    firm_decoder::DF
end
Flux.@layer SharedBilateralPolicy

function make_naive_policy(population_size::Int; hidden::Int=32)
    input_dimension = 2 + 4 * population_size + 1
    household_net = Chain(Dense(input_dimension => hidden, tanh), Dense(hidden => 1))
    firm_net = Chain(Dense(input_dimension => hidden, tanh), Dense(hidden => 1))
    return NaiveBilateralPolicy(household_net, firm_net)
end

function make_shared_policy(; hidden::Int=32)
    household_encoder = Chain(Dense(2 => hidden, tanh))
    firm_encoder = Chain(Dense(2 => hidden, tanh))
    decoder_input = 2 + 2 * hidden + 1
    household_decoder = Chain(Dense(decoder_input => hidden, tanh), Dense(hidden => 1))
    firm_decoder = Chain(Dense(decoder_input => hidden, tanh), Dense(hidden => 1))
    return SharedBilateralPolicy(
        household_encoder,
        firm_encoder,
        household_decoder,
        firm_decoder,
    )
end

function population_state(rng::AbstractRNG, population_size::Int)
    household_states = randn(rng, F32, 2, population_size)
    firm_states = randn(rng, F32, 2, population_size)
    price = F32[0.05]
    return household_states, firm_states, price
end

function policy_outputs(
    model::NaiveBilateralPolicy,
    household_states,
    firm_states,
    price,
)
    population_size = size(household_states, 2)
    full_distribution = vcat(vec(household_states), vec(firm_states), price)
    repeated_distribution = repeat(full_distribution, 1, population_size)
    household_inputs = vcat(household_states, repeated_distribution)
    firm_inputs = vcat(firm_states, repeated_distribution)
    return model.household_net(household_inputs), model.firm_net(firm_inputs)
end

function policy_outputs(
    model::SharedBilateralPolicy,
    household_states,
    firm_states,
    price,
)
    population_size = size(household_states, 2)
    household_embedding = mean(model.household_encoder(household_states); dims=2)
    firm_embedding = mean(model.firm_encoder(firm_states); dims=2)
    common_state = vcat(household_embedding, firm_embedding, reshape(price, :, 1))
    repeated_common_state = repeat(common_state, 1, population_size)
    household_inputs = vcat(household_states, repeated_common_state)
    firm_inputs = vcat(firm_states, repeated_common_state)
    return model.household_decoder(household_inputs), model.firm_decoder(firm_inputs)
end

function policy_loss(model, household_states, firm_states, price)
    household_policy, firm_policy = policy_outputs(
        model,
        household_states,
        firm_states,
        price,
    )
    return mean(abs2, household_policy) + mean(abs2, firm_policy)
end

function gradient_evaluation(model, household_states, firm_states, price)
    return Flux.gradient(
        candidate -> policy_loss(candidate, household_states, firm_states, price),
        model,
    )[1]
end

"""
Compare a full-distribution-per-agent MLP with a shared measure encoder.

The naive architecture reproduces the main scaling pathology in the current KS
experiment: each agent receives the full empirical distribution. The shared
architecture encodes each population once and broadcasts a common state.
"""
function benchmark_population_architectures(
    ;
    population_sizes=(50, 100, 200, 400),
    hidden::Int=32,
    repeats::Int=5,
    seed::Int=20260806,
)
    rows = NamedTuple[]
    for population_size in population_sizes
        rng = MersenneTwister(seed + population_size)
        household_states, firm_states, price = population_state(rng, population_size)
        naive_model = make_naive_policy(population_size; hidden)
        shared_model = make_shared_policy(; hidden)

        naive_forward = median_seconds(
            () -> policy_outputs(naive_model, household_states, firm_states, price);
            repeats,
        )
        shared_forward = median_seconds(
            () -> policy_outputs(shared_model, household_states, firm_states, price);
            repeats,
        )
        naive_gradient = median_seconds(
            () -> gradient_evaluation(naive_model, household_states, firm_states, price);
            repeats,
        )
        shared_gradient = median_seconds(
            () -> gradient_evaluation(shared_model, household_states, firm_states, price);
            repeats,
        )

        push!(rows, (;
            population_size,
            naive_parameters=parameter_count(naive_model),
            shared_parameters=parameter_count(shared_model),
            naive_forward_ms=1_000 * naive_forward,
            shared_forward_ms=1_000 * shared_forward,
            forward_speedup=naive_forward / shared_forward,
            naive_gradient_ms=1_000 * naive_gradient,
            shared_gradient_ms=1_000 * shared_gradient,
            gradient_speedup=naive_gradient / shared_gradient,
        ))
    end
    return rows
end

# -----------------------------------------------------------------------------
# Synthetic two-sided labor market used for the amortized price experiment.
# -----------------------------------------------------------------------------

struct BilateralDataset{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray}
    households::A  # 2 × N_H × B: wealth and labor efficiency
    firms::B       # 2 × N_F × B: capital and productivity
    log_wages::C   # 1 × B exact equilibrium log wages
end

function generate_populations(
    rng::AbstractRNG,
    population_size::Int,
    batch_size::Int,
)
    households = Array{F32}(undef, 2, population_size, batch_size)
    firms = Array{F32}(undef, 2, population_size, batch_size)
    for batch in 1:batch_size
        wealth_location = rand(rng, F32) * F32(1.2) - F32(0.5)
        wealth_scale = F32(0.15) + rand(rng, F32) * F32(0.75)
        efficiency_location = rand(rng, F32) * F32(0.5) - F32(0.25)
        efficiency_scale = F32(0.10) + rand(rng, F32) * F32(0.35)
        capital_location = rand(rng, F32) * F32(1.0) - F32(0.35)
        capital_scale = F32(0.15) + rand(rng, F32) * F32(0.65)
        productivity_location = rand(rng, F32) * F32(0.6) - F32(0.3)
        productivity_scale = F32(0.10) + rand(rng, F32) * F32(0.40)

        households[1, :, batch] .= exp.(
            wealth_location .+ wealth_scale .* randn(rng, F32, population_size),
        )
        households[2, :, batch] .= exp.(
            efficiency_location .+ efficiency_scale .* randn(rng, F32, population_size),
        )
        firms[1, :, batch] .= exp.(
            capital_location .+ capital_scale .* randn(rng, F32, population_size),
        )
        firms[2, :, batch] .= exp.(
            productivity_location .+ productivity_scale .* randn(rng, F32, population_size),
        )
    end
    return households, firms
end

"""Aggregate household effective labor supply and its wage derivative."""
function household_labor_supply_and_derivative(households, wage)
    wealth = @view households[1, :]
    efficiency = @view households[2, :]
    index = @. -F32(0.8) + F32(1.7) * log(wage) -
        F32(0.18) * log1p(wealth) + F32(0.35) * log(efficiency)
    participation = sigmoid_fast.(index)
    effective_labor = efficiency .* participation
    derivative = @. efficiency * participation * (1 - participation) * F32(1.7) / wage
    return mean(effective_labor), mean(derivative)
end

"""Aggregate firm labor demand and its wage derivative from Cobb-Douglas FOCs."""
function firm_labor_demand_and_derivative(firms, wage; capital_share=F32(0.35))
    capital = @view firms[1, :]
    productivity = @view firms[2, :]
    labor_demand = @. (
        (1 - capital_share) * productivity * capital^capital_share / wage
    )^(1 / capital_share)
    aggregate_demand = mean(labor_demand)
    derivative = -(one(wage) / capital_share) * aggregate_demand / wage
    return aggregate_demand, derivative
end

function labor_market_residual_and_derivative(households, firms, wage)
    supply, supply_derivative = household_labor_supply_and_derivative(households, wage)
    demand, demand_derivative = firm_labor_demand_and_derivative(firms, wage)
    return supply - demand, supply_derivative - demand_derivative
end

function equilibrium_wage(
    households,
    firms;
    lower=F32(0.03),
    upper=F32(8.0),
    tolerance=F32(1e-6),
    max_iterations::Int=80,
)
    low = lower
    high = upper
    low_residual, _ = labor_market_residual_and_derivative(households, firms, low)
    high_residual, _ = labor_market_residual_and_derivative(households, firms, high)
    low_residual < 0 || error("Lower wage does not imply excess labor demand")
    high_residual > 0 || error("Upper wage does not imply excess labor supply")
    evaluations = 2
    wage = (low + high) / 2
    residual = F32(Inf)
    for _ in 1:max_iterations
        wage = (low + high) / 2
        residual, _ = labor_market_residual_and_derivative(households, firms, wage)
        evaluations += 1
        if abs(residual) <= tolerance
            break
        elseif residual > 0
            high = wage
        else
            low = wage
        end
    end
    return wage, residual, evaluations
end

function make_dataset(
    rng::AbstractRNG,
    population_size::Int,
    batch_size::Int,
)
    households, firms = generate_populations(rng, population_size, batch_size)
    log_wages = Array{F32}(undef, 1, batch_size)
    for batch in 1:batch_size
        wage, _, _ = equilibrium_wage(
            @view(households[:, :, batch]),
            @view(firms[:, :, batch]),
        )
        log_wages[1, batch] = log(wage)
    end
    return BilateralDataset(households, firms, log_wages)
end

struct PricePredictor{EH,EF,D}
    household_encoder::EH
    firm_encoder::EF
    decoder::D
end
Flux.@layer PricePredictor

function make_price_predictor(; hidden::Int=16)
    household_encoder = Chain(Dense(2 => hidden, tanh), Dense(hidden => hidden, tanh))
    firm_encoder = Chain(Dense(2 => hidden, tanh), Dense(hidden => hidden, tanh))
    decoder = Chain(Dense(2 * hidden => 2 * hidden, tanh), Dense(2 * hidden => 1))
    return PricePredictor(household_encoder, firm_encoder, decoder)
end

function encode_batched_population(encoder, population)
    _, population_size, batch_size = size(population)
    encoded_flat = encoder(reshape(population, 2, :))
    hidden = size(encoded_flat, 1)
    encoded = reshape(encoded_flat, hidden, population_size, batch_size)
    return dropdims(mean(encoded; dims=2); dims=2)
end

function predict_log_wages(model::PricePredictor, households, firms)
    household_embedding = encode_batched_population(model.household_encoder, households)
    firm_embedding = encode_batched_population(model.firm_encoder, firms)
    return model.decoder(vcat(household_embedding, firm_embedding))
end

function train_price_predictor(
    ;
    population_size::Int=128,
    training_states::Int=192,
    validation_states::Int=64,
    hidden::Int=16,
    epochs::Int=250,
    learning_rate::Real=3e-3,
    seed::Int=20260806,
)
    rng = MersenneTwister(seed)
    data_seconds = @elapsed training_data = make_dataset(
        rng,
        population_size,
        training_states,
    )
    validation_data = make_dataset(rng, population_size, validation_states)
    Random.seed!(seed)
    model = make_price_predictor(; hidden)
    optimizer_state = Flux.setup(Flux.Adam(F32(learning_rate)), model)

    loss_history = Float64[]
    training_seconds = @elapsed begin
        for epoch in 1:epochs
            loss_value, gradients = Flux.withgradient(model) do candidate
                prediction = predict_log_wages(
                    candidate,
                    training_data.households,
                    training_data.firms,
                )
                mean(abs2, prediction .- training_data.log_wages)
            end
            Flux.update!(optimizer_state, model, gradients[1])
            if epoch == 1 || epoch % 25 == 0 || epoch == epochs
                push!(loss_history, Float64(loss_value))
            end
        end
    end

    validation_prediction = predict_log_wages(
        model,
        validation_data.households,
        validation_data.firms,
    )
    validation_rmse = sqrt(mean(abs2, validation_prediction .- validation_data.log_wages))
    validation_mape = mean(
        abs.(exp.(validation_prediction) .- exp.(validation_data.log_wages)) ./
        exp.(validation_data.log_wages),
    )
    return (;
        model,
        training_data,
        validation_data,
        loss_history,
        data_seconds,
        training_seconds,
        validation_log_rmse=Float64(validation_rmse),
        validation_price_mape=Float64(validation_mape),
    )
end

function exact_batch_wages(households, firms)
    batch_size = size(households, 3)
    wages = Vector{F32}(undef, batch_size)
    residuals = Vector{F32}(undef, batch_size)
    evaluations = Vector{Int}(undef, batch_size)
    for batch in 1:batch_size
        wages[batch], residuals[batch], evaluations[batch] = equilibrium_wage(
            @view(households[:, :, batch]),
            @view(firms[:, :, batch]),
        )
    end
    return wages, residuals, evaluations
end

function corrected_batch_wages(model, households, firms; correction_steps::Int=2)
    predicted_log_wages = predict_log_wages(model, households, firms)
    wages = vec(exp.(predicted_log_wages))
    batch_size = length(wages)
    residuals = Vector{F32}(undef, batch_size)
    evaluations = zeros(Int, batch_size)
    for batch in 1:batch_size
        wage = wages[batch]
        residual = F32(Inf)
        for _ in 1:correction_steps
            residual, derivative = labor_market_residual_and_derivative(
                @view(households[:, :, batch]),
                @view(firms[:, :, batch]),
                wage,
            )
            evaluations[batch] += 1
            wage = clamp(wage - residual / derivative, F32(0.03), F32(8.0))
        end
        residual, _ = labor_market_residual_and_derivative(
            @view(households[:, :, batch]),
            @view(firms[:, :, batch]),
            wage,
        )
        evaluations[batch] += 1
        wages[batch] = wage
        residuals[batch] = residual
    end
    return wages, residuals, evaluations
end

function benchmark_price_solver(
    model;
    population_size::Int=5_000,
    batch_size::Int=32,
    correction_steps::Int=2,
    repeats::Int=5,
    seed::Int=20260807,
)
    rng = MersenneTwister(seed)
    households, firms = generate_populations(rng, population_size, batch_size)

    exact_wages, exact_residuals, exact_evaluations = exact_batch_wages(households, firms)
    corrected_wages, corrected_residuals, corrected_evaluations = corrected_batch_wages(
        model,
        households,
        firms;
        correction_steps,
    )

    exact_seconds = median_seconds(
        () -> exact_batch_wages(households, firms);
        repeats,
    )
    corrected_seconds = median_seconds(
        () -> corrected_batch_wages(model, households, firms; correction_steps);
        repeats,
    )

    raw_wages = vec(exp.(predict_log_wages(model, households, firms)))
    raw_mape = mean(abs.(raw_wages .- exact_wages) ./ exact_wages)
    corrected_mape = mean(abs.(corrected_wages .- exact_wages) ./ exact_wages)
    return (;
        population_size,
        batch_size,
        correction_steps,
        exact_seconds,
        corrected_seconds,
        online_speedup=exact_seconds / corrected_seconds,
        raw_price_mape=Float64(raw_mape),
        corrected_price_mape=Float64(corrected_mape),
        exact_max_residual=Float64(maximum(abs, exact_residuals)),
        corrected_max_residual=Float64(maximum(abs, corrected_residuals)),
        exact_mean_evaluations=mean(exact_evaluations),
        corrected_mean_evaluations=mean(corrected_evaluations),
    )
end

function print_scaling_rows(rows)
    println("N | forward ms naive/shared | speedup | gradient ms naive/shared | speedup")
    for row in rows
        @printf(
            "%4d | %9.3f / %8.3f | %7.2fx | %9.3f / %8.3f | %7.2fx\n",
            row.population_size,
            row.naive_forward_ms,
            row.shared_forward_ms,
            row.forward_speedup,
            row.naive_gradient_ms,
            row.shared_gradient_ms,
            row.gradient_speedup,
        )
    end
end

function run_bilateral_acceleration_experiment(
    ;
    scaling_sizes=(50, 100, 200, 400),
    scaling_repeats::Int=5,
    training_population::Int=128,
    training_states::Int=192,
    validation_states::Int=64,
    epochs::Int=250,
    benchmark_population::Int=5_000,
    benchmark_states::Int=32,
    correction_steps::Int=2,
    benchmark_repeats::Int=5,
)
    scaling_rows = benchmark_population_architectures(
        ;
        population_sizes=scaling_sizes,
        repeats=scaling_repeats,
    )
    print_scaling_rows(scaling_rows)

    training = train_price_predictor(
        ;
        population_size=training_population,
        training_states,
        validation_states,
        epochs,
    )
    @printf(
        "price predictor: data %.3fs, training %.3fs, validation MAPE %.3f%%\n",
        training.data_seconds,
        training.training_seconds,
        100 * training.validation_price_mape,
    )

    price_benchmark = benchmark_price_solver(
        training.model;
        population_size=benchmark_population,
        batch_size=benchmark_states,
        correction_steps,
        repeats=benchmark_repeats,
    )
    @printf(
        "price solve: %.3fs exact vs %.3fs predictor+corrector = %.2fx speedup\n",
        price_benchmark.exact_seconds,
        price_benchmark.corrected_seconds,
        price_benchmark.online_speedup,
    )
    @printf(
        "raw/corrected price MAPE %.3f%% / %.6f%%; corrected max residual %.3e\n",
        100 * price_benchmark.raw_price_mape,
        100 * price_benchmark.corrected_price_mape,
        price_benchmark.corrected_max_residual,
    )

    savings_per_batch = price_benchmark.exact_seconds - price_benchmark.corrected_seconds
    offline_seconds = training.data_seconds + training.training_seconds
    break_even_batches = savings_per_batch > 0 ? offline_seconds / savings_per_batch : Inf
    @printf(
        "offline cost %.3fs; break-even at approximately %.1f comparable batches\n",
        offline_seconds,
        break_even_batches,
    )
    return (; scaling_rows, training, price_benchmark, offline_seconds, break_even_batches)
end

end # module
