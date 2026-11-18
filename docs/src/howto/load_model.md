# Load a model from another language

A CasADi function from Python, MATLAB or C++ becomes Julia code through CasADi's file
format. In the other language, save the function:

```python
f.save("model.casadi")        # Python; in MATLAB the call is the same
```

In Julia, load it with `CasadiFunction.load` and generate code; here a function made
in Julia stands for the other model:

```@example load
using CasADiCodegen

path = joinpath(tempdir(), "model.casadi")
x = SX.sym("x", 2)
other = CasadiFunction("f", [x], [x[1] * sin(x[2])])  # stands for the other model
other.save(path)

f = CasadiFunction.load(path)
g, g! = build_function(f; expression = Val{false})
g([2.0, 0.5])
```

The file must come from CasADi 3.8.2 or older. A model that holds a CasADi solver
(`cvodes`, `nlpsol`, ...) also needs [Run CasADi solver objects with SciML solvers](@ref).
