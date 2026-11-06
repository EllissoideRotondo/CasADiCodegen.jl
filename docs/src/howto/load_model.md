# Load a model from another language

A model that is written with CasADi in Python, MATLAB or C++ becomes Julia code
without a translation by hand, because CasADi saves a function to a file that the
Julia front end reads. This page saves a function, loads it in Julia and generates
code for it.

In the other language, we save the function with the method `save` of CasADi:

```python
f.save("model.casadi")        # Python; in MATLAB the call is the same
```

In Julia, we load the file with `CasadiFunction.load`. `build_function` and
`generate_julia` take the result as any other function. In the example, a function
made in Julia stands for the model of the other language:

```@example load
using CasADiCodegen

x = SX.sym("x", 2)                                    # stands for the model of the other language
CasadiFunction("f", [x], [x[1] * sin(x[2])]).save(joinpath(tempdir(), "model.casadi"))

f = CasadiFunction.load(joinpath(tempdir(), "model.casadi"))
g, g! = build_function(f; expression = Val{false})
g([2.0, 0.5])
```

The file must come from CasADi 3.8.2, the version of the package, or from an older
version: CasADi reads the files of its own version and of older versions only. A model that holds a solver of CasADi (`cvodes`,
`nlpsol`, ...) also needs [Replace the solvers of CasADi](@ref).
