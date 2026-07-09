using Plots

function _diagram_layer_y(n::Integer; spacing::Real=0.55)
    n == 1 && return [0.0]
    center = (n - 1) / 2
    return [(center - i) * spacing for i in 0:(n - 1)]
end

function _diagram_edges!(plt, from_x, from_y, to_x, to_y; color=:gray70, alpha=0.75, lw=0.8)
    for y1 in from_y, y2 in to_y
        plot!(plt, [from_x, to_x], [y1, y2], color=color, alpha=alpha, lw=lw, label=false)
    end
    return plt
end

function _diagram_nodes!(plt, x, ys; color, markersize=12)
    scatter!(plt, fill(x, length(ys)), ys;
        marker=:circle,
        markersize=markersize,
        markercolor=color,
        markerstrokecolor=:white,
        markerstrokewidth=1.6,
        label=false)
    return plt
end

function _diagram_label!(plt, x, y, label; size=9, color=:black)
    annotate!(plt, x, y, text(label, size, color, :center))
    return plt
end

"""
    plot_policy_network_diagram(; hidden=8, xi_min=0.02, xi_max=0.98)

Draw the neural-network policy rule used in `RBC.ipynb`.

The diagram follows the actual code:

1. Normalize the input state `(k, z)` into a two-dimensional vector `x`.
2. Pass `x` through one hidden layer with `tanh` activation.
3. Compute one scalar score `a`.
4. Use a scaled `sigmoid` to obtain the consumption share `xi`.
5. Convert `xi` into `c` and `k'` using the resource constraint.
"""
function plot_policy_network_diagram(; hidden::Integer=8, xi_min::Real=0.02, xi_max::Real=0.98)
    input_x = 0.0
    hidden_x = 2.0
    score_x = 4.0
    share_x = 5.6
    policy_x = 7.4

    input_y = [0.45, -0.45]
    hidden_y = _diagram_layer_y(hidden)
    score_y = [0.0]
    share_y = [0.0]
    policy_y = [0.45, -0.45]

    plt = plot(
        xlim=(-0.75, 9.05),
        ylim=(-2.75, 2.75),
        size=(1150, 520),
        axis=false,
        grid=false,
        legend=false,
        background_color=:white,
        foreground_color=:black,
        margin=8Plots.mm,
    )

    _diagram_edges!(plt, input_x, input_y, hidden_x, hidden_y; color=:gray70, alpha=0.9, lw=1.0)
    _diagram_edges!(plt, hidden_x, hidden_y, score_x, score_y; color=:gray70, alpha=0.9, lw=1.0)
    _diagram_edges!(plt, score_x, score_y, share_x, share_y; color=:gray55, alpha=0.9, lw=1.2)
    _diagram_edges!(plt, share_x, share_y, policy_x, policy_y; color=:gray55, alpha=0.9, lw=1.2)

    _diagram_nodes!(plt, input_x, input_y; color=RGB(0.43, 0.68, 0.36), markersize=15)
    _diagram_nodes!(plt, hidden_x, hidden_y; color=RGB(0.24, 0.62, 0.72), markersize=12)
    _diagram_nodes!(plt, score_x, score_y; color=RGB(0.95, 0.67, 0.25), markersize=15)
    _diagram_nodes!(plt, share_x, share_y; color=RGB(0.86, 0.38, 0.47), markersize=15)
    _diagram_nodes!(plt, policy_x, policy_y; color=RGB(0.47, 0.45, 0.75), markersize=15)

    _diagram_label!(plt, input_x, 2.25, "Input state", size=12)
    _diagram_label!(plt, hidden_x, 2.25, "Hidden layer", size=12)
    _diagram_label!(plt, score_x, 2.25, "Linear output", size=12)
    _diagram_label!(plt, share_x, 2.25, "Output activation", size=12)
    _diagram_label!(plt, policy_x, 2.25, "Policy objects", size=12)

    _diagram_label!(plt, input_x - 0.25, input_y[1], "x₁ = normalized log k", size=8)
    _diagram_label!(plt, input_x - 0.25, input_y[2], "x₂ = normalized z", size=8)

    _diagram_label!(plt, hidden_x, -2.25, "h = tanh(W₁x + b₁)", size=10)
    _diagram_label!(plt, score_x, -0.55, "a = W₂h + b₂", size=9)
    _diagram_label!(plt, share_x, -0.65, "ξ = ξmin + (ξmax - ξmin)σ(a)\nξ ∈ ($(xi_min), $(xi_max))", size=9)
    _diagram_label!(plt, policy_x + 0.6, policy_y[1], "c = ξw(k,z)", size=9)
    _diagram_label!(plt, policy_x + 0.6, policy_y[2], "k' = (1 - ξ)w(k,z)", size=9)
    _diagram_label!(plt, 6.55, -1.55, "w(k,z) = exp(z)k^α + (1 - δ)k", size=9, color=:gray30)

    return plt
end

"""
    plot_intro_network_diagram(; hidden=6)

Draw the simple one-input, one-hidden-layer neural network used in `Intro_DL.ipynb`.

The notebook trains

    z1 = W1 * x + b1
    h = tanh.(z1)
    yhat = W2 * h + b2

and minimizes mean squared error.
"""
function plot_intro_network_diagram(; hidden::Integer=6)
    input_x = 0.0
    hidden_x = 2.0
    output_x = 4.0
    loss_x = 5.8

    input_y = [0.0]
    hidden_y = _diagram_layer_y(hidden)
    output_y = [0.0]
    loss_y = [0.0]

    plt = plot(
        xlim=(-0.75, 6.85),
        ylim=(-2.25, 2.25),
        size=(960, 430),
        axis=false,
        grid=false,
        legend=false,
        background_color=:white,
        foreground_color=:black,
        margin=8Plots.mm,
    )

    _diagram_edges!(plt, input_x, input_y, hidden_x, hidden_y; color=:gray70, alpha=0.9, lw=1.0)
    _diagram_edges!(plt, hidden_x, hidden_y, output_x, output_y; color=:gray70, alpha=0.9, lw=1.0)
    _diagram_edges!(plt, output_x, output_y, loss_x, loss_y; color=:gray55, alpha=0.9, lw=1.2)

    _diagram_nodes!(plt, input_x, input_y; color=RGB(0.43, 0.68, 0.36), markersize=15)
    _diagram_nodes!(plt, hidden_x, hidden_y; color=RGB(0.24, 0.62, 0.72), markersize=12)
    _diagram_nodes!(plt, output_x, output_y; color=RGB(0.95, 0.67, 0.25), markersize=15)
    _diagram_nodes!(plt, loss_x, loss_y; color=RGB(0.86, 0.38, 0.47), markersize=15)

    _diagram_label!(plt, input_x, 1.85, "Input", size=12)
    _diagram_label!(plt, hidden_x, 1.85, "Hidden layer", size=12)
    _diagram_label!(plt, output_x, 1.85, "Output", size=12)
    _diagram_label!(plt, loss_x, 1.85, "Loss", size=12)

    _diagram_label!(plt, input_x - 0.2, input_y[1] - 0.35, "x", size=10)
    _diagram_label!(plt, hidden_x, -1.85, "h = tanh(W1 x + b1)", size=10)
    _diagram_label!(plt, output_x, -0.55, "yhat = W2 h + b2", size=9)
    _diagram_label!(plt, loss_x + 0.1, -0.55, "MSE(yhat, y)", size=9)

    _diagram_label!(plt, 1.0, 1.25, "W1, b1", size=9, color=:gray30)
    _diagram_label!(plt, 3.0, 1.25, "W2, b2", size=9, color=:gray30)
    _diagram_label!(plt, 4.9, 0.45, "compare with y", size=9, color=:gray30)

    return plt
end
