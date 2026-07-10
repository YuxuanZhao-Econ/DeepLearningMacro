using Plots
using LaTeXStrings

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

function _diagram_box!(plt, x, y, width, height; fillcolor=RGB(0.95, 0.98, 1.0), linecolor=RGB(0.20, 0.45, 0.62), lw=1.2)
    xs = [x - width / 2, x + width / 2, x + width / 2, x - width / 2]
    ys = [y - height / 2, y - height / 2, y + height / 2, y + height / 2]
    plot!(plt, Shape(xs, ys), fillcolor=fillcolor, linecolor=linecolor, lw=lw, label=false)
    return plt
end

function _diagram_arrow!(plt, x1, y1, x2, y2; color=:gray45, lw=1.2)
    plot!(plt, [x1, x2], [y1, y2], color=color, lw=lw, arrow=true, label=false)
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

    _diagram_label!(plt, input_x - 0.25, input_y[1], L"x_1=\mathrm{normalized}\ \log k", size=8)
    _diagram_label!(plt, input_x - 0.25, input_y[2], L"x_2=\mathrm{normalized}\ z", size=8)

    _diagram_label!(plt, hidden_x, -2.25, L"h=\tanh(W_1x+b_1)", size=10)
    _diagram_label!(plt, score_x, -0.55, L"a=W_2h+b_2", size=9)
    _diagram_label!(plt, share_x, -0.55, L"\xi=\xi_{\min}+(\xi_{\max}-\xi_{\min})\sigma(a)", size=8)
    _diagram_label!(plt, share_x, -0.85, "ξ ∈ ($(xi_min), $(xi_max))", size=8)
    _diagram_label!(plt, policy_x + 0.6, policy_y[1], L"c=\xi w(k,z)", size=9)
    _diagram_label!(plt, policy_x + 0.6, policy_y[2], L"k'=(1-\xi)w(k,z)", size=9)
    _diagram_label!(plt, 6.55, -1.55, L"w(k,z)=\exp(z)k^\alpha+(1-\delta)k", size=9, color=:gray30)

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
    _diagram_label!(plt, hidden_x, -1.85, L"h=\tanh(W_1x+b_1)", size=10)
    _diagram_label!(plt, output_x, -0.55, L"\hat y=W_2h+b_2", size=9)
    _diagram_label!(plt, loss_x + 0.1, -0.55, L"\mathrm{MSE}(\hat y,y)", size=9)

    _diagram_label!(plt, 1.0, 1.25, L"W_1,b_1", size=9, color=:gray30)
    _diagram_label!(plt, 3.0, 1.25, L"W_2,b_2", size=9, color=:gray30)
    _diagram_label!(plt, 4.9, 0.45, L"\mathrm{compare\ with}\ y", size=9, color=:gray30)

    return plt
end

"""
    plot_fully_connected_network_diagram(; input=5, hidden_layers=[7, 7, 7], output=3)

Draw a generic feedforward fully-connected neural network.

Each neuron in one layer is connected to every neuron in the next layer. This is the
basic architecture behind the small example in `Intro_DL.ipynb`, except the example
uses one input, one hidden layer, and one output.
"""
function plot_fully_connected_network_diagram(;
        input::Integer=5,
        hidden_layers::AbstractVector{<:Integer}=[7, 7, 7],
        output::Integer=3)
    layer_sizes = [input; collect(hidden_layers); output]
    n_layers = length(layer_sizes)
    xs = collect(range(0.0, 2.2 * (n_layers - 1), length=n_layers))
    layer_ys = [_diagram_layer_y(n; spacing=0.42) for n in layer_sizes]

    max_nodes = maximum(layer_sizes)
    y_margin = max(2.2, 0.42 * max_nodes / 2 + 1.0)

    plt = plot(
        xlim=(-0.75, xs[end] + 0.75),
        ylim=(-y_margin, y_margin),
        size=(1100, 480),
        axis=false,
        grid=false,
        legend=false,
        background_color=:white,
        foreground_color=:black,
        margin=8Plots.mm,
    )

    for j in 1:(n_layers - 1)
        _diagram_edges!(plt, xs[j], layer_ys[j], xs[j + 1], layer_ys[j + 1];
            color=:gray70, alpha=0.8, lw=0.8)
    end

    colors = [
        RGB(0.43, 0.68, 0.36);
        fill(RGB(0.24, 0.62, 0.72), length(hidden_layers));
        RGB(0.86, 0.38, 0.47)
    ]

    for j in 1:n_layers
        markersize = j == 1 || j == n_layers ? 13 : 11
        _diagram_nodes!(plt, xs[j], layer_ys[j]; color=colors[j], markersize=markersize)
    end

    _diagram_label!(plt, xs[1], y_margin - 0.35, "Input layer", size=12)
    for j in 2:(n_layers - 1)
        _diagram_label!(plt, xs[j], y_margin - 0.35, "Hidden layer $(j - 1)", size=12)
    end
    _diagram_label!(plt, xs[end], y_margin - 0.35, "Output layer", size=12)

    _diagram_label!(plt, xs[1] - 0.25, layer_ys[1][1], "x", size=10)
    _diagram_label!(plt, xs[end] + 0.35, layer_ys[end][1], L"\hat y", size=10)
    _diagram_label!(plt, xs[2], -y_margin + 0.62, L"a_1=W_1x+b_1", size=9)
    _diagram_label!(plt, xs[2], -y_margin + 0.28, L"h_1=\phi(a_1)", size=9)

    if n_layers > 3
        _diagram_label!(plt, xs[3], -y_margin + 0.62, L"a_2=W_2h_1+b_2", size=9)
        _diagram_label!(plt, xs[3], -y_margin + 0.28, L"h_2=\phi(a_2)", size=9)
    end

    _diagram_label!(plt, xs[end], -y_margin + 0.45, "output = final layer\nactivation depends on task", size=9)

    return plt
end

"""
    plot_ha_manifold_flow_diagram()

Draw the distribution-input learning example used in Section 14 of `Intro_DL.ipynb`.

The diagram contrasts a hand-picked moment approximation with a neural network that
uses the full histogram input.
"""
function plot_ha_manifold_flow_diagram()
    plt = plot(
        xlim=(-0.3, 9.2),
        ylim=(-2.25, 2.35),
        size=(1180, 520),
        axis=false,
        grid=false,
        legend=false,
        background_color=:white,
        foreground_color=:black,
        margin=8Plots.mm,
    )

    box_w = 1.55
    box_h = 0.62
    x_z = 0.7
    x_mu = 2.55
    x_target = 5.0
    x_mom = 4.65
    x_ridge = 7.05
    x_nn = 4.65
    x_out = 7.05

    y_z = 0.0
    y_mu = 0.0
    y_target = 1.35
    y_mom = 0.0
    y_ridge = 0.0
    y_nn = -1.35
    y_out = -1.35

    blue = RGB(0.92, 0.97, 1.0)
    green = RGB(0.93, 0.98, 0.92)
    orange = RGB(1.0, 0.96, 0.88)
    purple = RGB(0.97, 0.94, 1.0)

    for (x, y, w, fill) in [
            (x_z, y_z, box_w, blue),
            (x_mu, y_mu, box_w, green),
            (x_target, y_target, 2.1, orange),
            (x_mom, y_mom, 2.0, purple),
            (x_ridge, y_ridge, 1.75, purple),
            (x_nn, y_nn, 2.0, blue),
            (x_out, y_out, 1.75, blue)]
        _diagram_box!(plt, x, y, w, box_h; fillcolor=fill)
    end

    _diagram_arrow!(plt, x_z + box_w / 2, y_z, x_mu - box_w / 2, y_mu)
    _diagram_arrow!(plt, x_mu + box_w / 2, y_mu + 0.12, x_target - 1.05, y_target - box_h / 2)
    _diagram_arrow!(plt, x_mu + box_w / 2, y_mu, x_mom - 1.0, y_mom)
    _diagram_arrow!(plt, x_mom + 1.0, y_mom, x_ridge - 0.875, y_ridge)
    _diagram_arrow!(plt, x_mu + box_w / 2, y_mu - 0.12, x_nn - 1.0, y_nn + box_h / 2)
    _diagram_arrow!(plt, x_nn + 1.0, y_nn, x_out - 0.875, y_out)

    _diagram_label!(plt, x_z, y_z + 0.12, "Latent state", size=10)
    _diagram_label!(plt, x_z, y_z - 0.13, L"z\in\mathbb{R}^3", size=10)

    _diagram_label!(plt, x_mu, y_mu + 0.12, "Distribution input", size=10)
    _diagram_label!(plt, x_mu, y_mu - 0.13, L"\mu\in\mathbb{R}^{60}", size=10)

    _diagram_label!(plt, x_target, y_target + 0.12, "Target from shape", size=10)
    _diagram_label!(plt, x_target, y_target - 0.13, L"y=\sin(5s_1)+s_2^2-0.8s_3+\varepsilon", size=9)

    _diagram_label!(plt, x_mom, y_mom + 0.12, "Moment features", size=10)
    _diagram_label!(plt, x_mom, y_mom - 0.13, L"f(\mu)=(1,m,v,s,m^2,\ldots,vs)'", size=9)

    _diagram_label!(plt, x_ridge, y_ridge + 0.12, "Ridge regression", size=10)
    _diagram_label!(plt, x_ridge, y_ridge - 0.13, L"\hat y=\beta f(\mu)", size=10)

    _diagram_label!(plt, x_nn, y_nn + 0.12, "Neural network", size=10)
    _diagram_label!(plt, x_nn, y_nn - 0.13, L"h=\tanh(W_1x+b_1)", size=10)

    _diagram_label!(plt, x_out, y_out + 0.12, "Output", size=10)
    _diagram_label!(plt, x_out, y_out - 0.13, L"\hat y=W_2h+b_2", size=10)

    _diagram_label!(plt, 5.85, 0.55, "hand-picked compression", size=9, color=:gray35)
    _diagram_label!(plt, 5.85, -0.80, "learned features from full histogram", size=9, color=:gray35)
    _diagram_label!(plt, 2.0, 0.45, L"z\mapsto\mu", size=9, color=:gray35)

    return plt
end

"""
    plot_ha_nn_architecture_diagram(; shown_inputs=7, shown_hidden=9)

Draw the neural-network architecture used in the Section 14 distribution-input
example. The actual input has 60 histogram bins and the hidden layer has 64
neurons; the diagram shows a smaller readable subset.
"""
function plot_ha_nn_architecture_diagram(; shown_inputs::Integer=7, shown_hidden::Integer=9)
    input_x = 0.0
    hidden_x = 2.25
    output_x = 4.55

    input_y = _diagram_layer_y(shown_inputs; spacing=0.33)
    hidden_y = _diagram_layer_y(shown_hidden; spacing=0.30)
    output_y = [0.0]

    y_margin = 2.25

    plt = plot(
        xlim=(-0.85, 5.65),
        ylim=(-y_margin, y_margin),
        size=(1050, 460),
        axis=false,
        grid=false,
        legend=false,
        background_color=:white,
        foreground_color=:black,
        margin=8Plots.mm,
    )

    _diagram_edges!(plt, input_x, input_y, hidden_x, hidden_y; color=:gray75, alpha=0.75, lw=0.75)
    _diagram_edges!(plt, hidden_x, hidden_y, output_x, output_y; color=:gray75, alpha=0.85, lw=0.85)

    _diagram_nodes!(plt, input_x, input_y; color=RGB(0.43, 0.68, 0.36), markersize=12)
    _diagram_nodes!(plt, hidden_x, hidden_y; color=RGB(0.24, 0.62, 0.72), markersize=11)
    _diagram_nodes!(plt, output_x, output_y; color=RGB(0.95, 0.67, 0.25), markersize=15)

    _diagram_label!(plt, input_x, y_margin - 0.25, "Input layer", size=12)
    _diagram_label!(plt, hidden_x, y_margin - 0.25, "Hidden layer", size=12)
    _diagram_label!(plt, output_x, y_margin - 0.25, "Output layer", size=12)

    _diagram_label!(plt, input_x - 0.33, 0.0, L"x=60\mu\in\mathbb{R}^{60}", size=9)
    _diagram_label!(plt, hidden_x + 0.25, hidden_y[1] + 0.15, "64 neurons", size=9, color=:gray30)
    _diagram_label!(plt, output_x + 0.42, 0.0, L"\hat y\in\mathbb{R}", size=10)

    _diagram_label!(plt, hidden_x, -1.65, L"a_1=W_1x+b_1", size=10)
    _diagram_label!(plt, hidden_x, -1.98, L"h=\tanh(a_1)", size=10)
    _diagram_label!(plt, output_x, -1.10, L"\hat y=W_2h+b_2", size=10)
    _diagram_label!(plt, output_x, -1.43, "identity output activation", size=9, color=:gray30)

    _diagram_label!(plt, 1.12, 1.35, L"W_1\in\mathbb{R}^{64\times 60},\ b_1\in\mathbb{R}^{64}", size=8, color=:gray30)
    _diagram_label!(plt, 3.40, 1.35, L"W_2\in\mathbb{R}^{1\times 64},\ b_2\in\mathbb{R}", size=8, color=:gray30)
    _diagram_label!(plt, input_x, -1.70, "60 histogram bins\nshown schematically", size=8, color=:gray35)

    return plt
end
