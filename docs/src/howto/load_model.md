# Load a model from another language

A CasADi function from Python, MATLAB or C++ becomes Julia code through the file
format of CasADi. In the other language, save the function:

```python
f.save("model.casadi")        # Python; in MATLAB the call is the same
```

In Julia, load it with `CasadiFunction.load` and generate code; here a function made
in Julia stands for the other model:

```@example load
using CasADiCodegen

x = SX.sym("x", 2)                                    # stands for the model of the other language
CasadiFunction("f", [x], [x[1] * sin(x[2])]).save(joinpath(tempdir(), "model.casadi"))

f = CasadiFunction.load(joinpath(tempdir(), "model.casadi"))
g, g! = build_function(f; expression = Val{false})
g([2.0, 0.5])
```

The file must come from CasADi 3.8.2 or older. A model that holds a solver of CasADi
(`cvodes`, `nlpsol`, ...) also needs [Replace the solvers of CasADi](@ref).
