# Linear analysis of a feedback loop

In this tutorial, we generate the linearization of a feedback loop and analyze it with
ControlSystemsBase.jl, then linearize a nonlinear circuit with an algebraic state at
several operating points. The functions used are described in
[Linearize a model and order its equations](@ref).

```@example linear_analysis
using CasADiCodegen
using ControlSystemsBase
nothing # hide
```

## The closed loop with a perturbed signal

The loop has a plant of the first order and a proportional controller,

```math
P(s) = \frac{k}{\tau s + 1}, \qquad c = K y,
```

with ``k = 1``, ``\tau = 1`` and ``K = -1``, which carries the sign of the negative
feedback. The package has no notion of a block-diagram signal, so the perturbation ``w``
at the plant input is an input of the model, and its outputs ``u = c + w``, ``-c`` and
``y`` give the sensitivity functions ``S`` and ``T`` and the transfer function ``P S``:

```@example linear_analysis
x, w, p = SX.sym("x"), SX.sym("w"), SX.sym("p", 3)        # p = [k, τ, K]
k, τ, K = p[1], p[2], p[3]
plant(x, u) = (k * u - x) / τ
c = K * x                                                  # the controller output, with y = x
u = c + w
f = CasadiFunction("closed_f", [x, w, p], [plant(x, u)])
h = CasadiFunction("closed_h", [x, w, p], [vcat(u, -c, x)])
lin, _ = CasADiCodegen.linearization(f, h)
nothing # hide
```

We linearize at ``x = 0`` and index the `StateSpace` system by output and input; the
sensitivity function at the plant input is

```@example linear_analysis
A, B, C, D = lin([0.0], [0.0], [1.0, 1.0, -1.0])
closed = ss(A, B, C, D)
S, T, PS = closed[1, 1], closed[2, 1], closed[3, 1]
tf(S)
```

and the complementary sensitivity function is

```@example linear_analysis
tf(T)
```

The feedback moves the pole from ``-1`` to ``-2``. The sum `S + T` keeps the states of
both systems, so `minreal` cancels the repeated pole before the sum shows as one:

```@example linear_analysis
minreal(tf(S + T))
```

Both agree with `sensitivity` and `comp_sensitivity` of ControlSystemsBase.jl, which
assume negative feedback and thus take the controller gain ``1``:

```@example linear_analysis
Pcs = tf(1.0, [1.0, 1.0])
hinfnorm(S - sensitivity(Pcs, 1.0))[1], hinfnorm(T - comp_sensitivity(Pcs, 1.0))[1]
```

Half of a constant disturbance at the plant input reaches the output with the loop
closed:

```@example linear_analysis
dcgain(PS)[1]
```

## The loop transfer function

For a loop of one input and one output, the loop transfer function ``L`` is the same at
the plant input and output, and ``S = 1 / (1 - L)`` with the sign of the feedback in ``L``.
To open the loop, we change the model so that an input ``e`` replaces ``y`` as the argument
of the controller:

```@example linear_analysis
e = SX.sym("e")
f_open = CasadiFunction("open_f", [x, e, p], [plant(x, K * e)])
h_open = CasadiFunction("open_h", [x, e, p], [x])
lin_open, _ = CasADiCodegen.linearization(f_open, h_open)
L = ss(lin_open([0.0], [0.0], [1.0, 1.0, -1.0])...)
tf(L)
```

Here ``L = K P = -P``, and ControlSystemsBase.jl, which assumes negative feedback, takes
``-L``. The gain margin is infinite, and the phase margin is 180 degrees at the crossover
at zero frequency:

```@example linear_analysis
margins = margin(-L)
margins.gm[1], margins.pm[1]
```

## A nonlinear model with an algebraic state

We take a circuit in which a source of voltage ``u`` drives a capacitor ``C_p``, loaded by a resistor ``R_L``,
through a resistor ``R`` and a diode. The diode current ``i`` follows the Shockley law and
is an algebraic state whose equation has no solution in elementary functions:

```math
\begin{aligned}
C_p v' &= i - v / R_L, \\
0 &= i - I_s \left(e^{(u - v - R i) / V_t} - 1\right), \\
y &= v.
\end{aligned}
```

[`CasADiCodegen.linearization`](@ref) takes the three equations as the functions `f`, `g`
and `h` (see [Linearize a model with algebraic states](@ref)), and the linear model has the
capacitor voltage as its only state:

```@example linear_analysis
v, i, u, q = SX.sym("v"), SX.sym("i"), SX.sym("u"), SX.sym("q", 5)    # q = [Cp, R, RL, Is, Vt]
Cp, R, RL, Is, Vt = q[1], q[2], q[3], q[4], q[5]
f = CasadiFunction("f", [v, i, u, q], [(i - v / RL) / Cp])
g = CasadiFunction("g", [v, i, u, q], [i - Is * (exp((u - v - R * i) / Vt) - 1)])
h = CasadiFunction("h", [v, i, u, q], [v])
lin_circuit, _ = CasADiCodegen.linearization(f, g, h)
nothing # hide
```

The caller must give a point where the algebraic equation holds. At a steady state with
the voltage ``v``, the current is ``i = v / R_L`` and the diode law gives the input
``u = v + R i + V_t \log(1 + i / I_s)``. We linearize at three voltages and compare the
pole with ``-(1 / (R + r_d) + 1 / R_L) / C_p`` of the small-signal circuit, in which the
diode is a resistor ``r_d = V_t / (i + I_s)``:

```@example linear_analysis
q0 = [1e-3, 10.0, 1000.0, 1e-12, 0.025]                           # Cp, R, RL, Is, Vt
function circuit(v)
    i = v / q0[3]
    u = v + q0[2] * i + q0[5] * log1p(i / q0[4])
    ss(lin_circuit([v], [i], [u], q0)...)
end
function small_signal_pole(v)
    rd = q0[5] / (v / q0[3] + q0[4])
    -(1 / (q0[2] + rd) + 1 / q0[3]) / q0[1]
end
[(v, real(poles(circuit(v))[1]) - small_signal_pole(v), small_signal_pole(v), dcgain(circuit(v))[1]) for v in (0.5, 1.0, 2.0)]
```

Each tuple gives the voltage, the difference from the small-signal pole, which is at the
level of rounding, the pole and the steady-state gain. As the voltage grows, the pole moves
away from the origin and the gain approaches ``R_L / (R + R_L) \approx 0.99``.
