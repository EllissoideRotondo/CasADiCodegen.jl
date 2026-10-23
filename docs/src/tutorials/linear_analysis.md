# Linear analysis of a feedback loop

The design of a controller and the analysis of its robustness work on linear models: the
transfer functions of a loop about an operating point, among which the sensitivity function
``S``, which tells how much of a disturbance at a signal remains with the loop closed, and the
complementary sensitivity function ``T = 1 - S``. A nonlinear model gives these
functions through its linearization, whose matrices ``A``, ``B``, ``C`` and ``D`` depend on
the operating point and on which signals are taken as the input and the output. In this
tutorial, we write a feedback loop as CasADi functions with an explicit input at the signal
that we perturb, generate the code of its linearization with CasADiCodegen.jl, and analyze
the result with ControlSystemsBase.jl: the sensitivity functions, the loop transfer function
and its margins. We then linearize a nonlinear circuit with an algebraic state at several
operating points, where the algebraic state is eliminated in the generated code. The
functions themselves, their arguments and their sparse form are described in
[Linearize a model and order its equations](@ref), which this tutorial does not repeat.

The examples need CasADiCodegen.jl and ControlSystemsBase.jl.

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

with the gain ``k = 1``, the time constant ``\tau = 1`` and the controller gain ``K = -1``,
which carries the sign of the negative feedback. The plant input is the controller output
``c`` plus a perturbation ``w``, which is the input of the linear model. The sensitivity
function at the plant input is the transfer function from ``w`` to the plant input
``u = c + w``, and the complementary sensitivity function is the one from ``w`` to ``-c``,
so that ``S + T = 1`` follows from ``u = c + w``; the transfer function from ``w`` to the
plant output ``y`` is ``P S``. We therefore give the model three outputs, ``u``, ``-c`` and
``y``, and one input, ``w``. The package has no notion of a signal of a block diagram, so the
place where the loop is perturbed is written into the model as this input:

```@example linear_analysis
x, w, p = SX.sym("x"), SX.sym("w"), SX.sym("p", 3)        # p = [k, τ, K]
k, τ, K = p[1], p[2], p[3]
plant(x, u) = (k * u - x) / τ
c = K * x                                                  # the controller output, with y = x
u = c + w
f = CasadiFunction("closed_f", [x, w, p], [plant(x, u)])
h = CasadiFunction("closed_h", [x, w, p], [vertcat(u, -c, x)])
lin, _ = CasADiCodegen.linearization(f, h)
nothing # hide
```

The loop is linear, so any point gives the same matrices; we take the steady state
``x = 0`` with no perturbation. `ss` of ControlSystemsBase.jl makes a `StateSpace` system of
the four matrices, and indexing it by an output and an input selects one transfer
function. The sensitivity function at the plant input is then

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

The feedback moves the pole of the plant from ``-1`` to ``-2``. The two functions add up to
one. The sum of two state-space systems keeps the states of both, so `S + T` has two
copies of the pole at ``-2``, which `minreal` cancels against its zeros:

```@example linear_analysis
minreal(tf(S + T))
```

They also agree with `sensitivity` and `comp_sensitivity` of ControlSystemsBase.jl, which
take the plant and the controller in the convention of negative feedback of
ControlSystemsBase.jl, where the controller gain is ``1``; the largest difference over all frequencies, `hinfnorm` of the
difference, is zero:

```@example linear_analysis
Pcs = tf(1.0, [1.0, 1.0])
hinfnorm(S - sensitivity(Pcs, 1.0))[1], hinfnorm(T - comp_sensitivity(Pcs, 1.0))[1]
```

The transfer function from the perturbation to the plant output, ``P S = 1 / (s + 2)``,
has the steady-state gain one half, which is the fraction of a constant disturbance at the
plant input that reaches its output with the loop closed:

```@example linear_analysis
dcgain(PS)[1]
```

## The loop transfer function

The loop transfer function at the plant output is the transfer function of the loop opened
there; for a loop of one input and one output, as here, it is also the loop transfer
function at the plant input, so that ``S = 1 / (1 - L)``, with the sign of the feedback in
``L``. When the loop is opened, the signal that enters the controller becomes the input
``e``, and the plant output, which no longer returns to the controller, is the output. Opening the loop is again a change
of the model, a new pair of functions in which ``e`` replaces ``y`` as the argument of the
controller:

```@example linear_analysis
e = SX.sym("e")
f_open = CasadiFunction("open_f", [x, e, p], [plant(x, K * e)])
h_open = CasadiFunction("open_h", [x, e, p], [x])
lin_open, _ = CasADiCodegen.linearization(f_open, h_open)
L = ss(lin_open([0.0], [0.0], [1.0, 1.0, -1.0])...)
tf(L)
```

With the sign of the feedback inside the controller, ``L = K P = -P``. The analysis
functions of ControlSystemsBase.jl assume negative feedback, so they take ``-L``, which is
``P`` with the controller gain ``1``. Its phase stays between ``0`` and ``-90`` degrees and
never reaches ``-180`` degrees, so the gain margin is infinite. Its gain is below one at
every positive frequency and equal to one at zero frequency, which `margin` takes as the
crossover; the phase there is zero, so the phase margin it returns is 180 degrees:

```@example linear_analysis
margins = margin(-L)
margins.gm[1], margins.pm[1]
```

## A nonlinear model with an algebraic state

The matrices of a nonlinear model depend on the operating point, and many models have
algebraic states, of which no derivative appears in the equations; when the equation of
such a state has no solution in elementary functions, the state cannot be substituted by
hand. As
an example, we take a circuit in which a source of voltage ``u`` drives a capacitor ``C_p``,
loaded by a resistor ``R_L``, through a resistor ``R`` and a diode. The current ``i`` of the
diode follows the Shockley law with the saturation current ``I_s`` and the thermal voltage
``V_t``. No derivative of ``i`` appears, so ``i`` is an algebraic state, and its equation
is implicit:

```math
\begin{aligned}
C_p v' &= i - v / R_L, \\
0 &= i - I_s \left(e^{(u - v - R i) / V_t} - 1\right), \\
y &= v.
\end{aligned}
```

We write the three equations as the CasADi functions `f`, `g` and `h` of the capacitor
voltage, the current, the source voltage and the parameters, the form that
[`CasADiCodegen.linearization`](@ref) takes for a model with algebraic states (see
[Linearize a model with algebraic states](@ref)); the linear model has the capacitor
voltage as its only state:

```@example linear_analysis
v, i, u, q = SX.sym("v"), SX.sym("i"), SX.sym("u"), SX.sym("q", 5)    # q = [Cp, R, RL, Is, Vt]
Cp, R, RL, Is, Vt = q[1], q[2], q[3], q[4], q[5]
f = CasadiFunction("f", [v, i, u, q], [(i - v / RL) / Cp])
g = CasadiFunction("g", [v, i, u, q], [i - Is * (exp((u - v - R * i) / Vt) - 1)])
h = CasadiFunction("h", [v, i, u, q], [v])
lin_circuit, _ = CasADiCodegen.linearization(f, g, h)
nothing # hide
```

The matrices are those of the model at a point where the algebraic equations hold, which
need not be a steady state, and finding such a point is the task of the caller. Here a steady state at the capacitor
voltage ``v`` has the current ``i = v / R_L``, and the diode law, solved for the source
voltage, gives the input ``u = v + R i + V_t \log(1 + i / I_s)`` that keeps it there. We
linearize at three voltages and compare the pole with that of the small-signal circuit, in
which the diode is a resistor ``r_d = V_t / (i + I_s)``, so that the pole is
``-(1 / (R + r_d) + 1 / R_L) / C_p``:

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

Each tuple gives the voltage, the difference between the pole of the generated
linearization and that of the small-signal circuit, which is at the level of rounding, the
pole itself and the steady-state gain. The diode conducts better at a larger current, so the
pole moves away from the origin as the voltage grows, and the steady-state gain from the
source to the capacitor approaches ``R_L / (R + R_L) \approx 0.99``.
