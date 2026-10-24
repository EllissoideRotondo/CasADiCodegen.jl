using Test

# Each group of tests runs in a Julia process of its own, and several processes
# run at the same time, because most of the time of a group is compilation that
# the groups do not share. `Pkg.test("CasADiCodegen"; test_args = ["mx", "root"])` runs
# the given groups only, and the environment variable CASADI_TEST_JOBS gives the
# number of processes at the same time (the default is 6, or half of the
# processor threads when that is less). With CASADI_TEST_VERBOSE set, each group
# prints the time of each of its test sets. With CASADI_TEST_EXHAUSTIVE set, the
# groups also run the checks of the exhaustive tier (`EXHAUSTIVE` in helpers.jl):
# slow cases of behaviors that the default run checks in a faster case.
# The slowest groups come first, so that they do not start last.
const GROUPS = ("integrator", "mx", "forms", "cache", "casadi", "emit", "nodes", "solvers", "switches", "optimization", "callback", "sciml", "bifurcation", "root", "api", "rules", "traced", "symbolics")

"""
    run_group(group) -> (passed, seconds, output)

Run one group of tests in a new Julia process with the options and the
environment of this one. `passed` is true only when the process ended with no failure.
"""
function run_group(group)
    @assert group in GROUPS "\"$group\" is not a group of tests; the groups are $(join(GROUPS, ", "))"
    output = IOBuffer()
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) --group $group`
    seconds = @elapsed process = run(pipeline(ignorestatus(command); stdout = output, stderr = output))
    (success(process), seconds, String(take!(output)))
end

"""
The number of tests that passed, from the summary that a group printed; zero when there is none.
The colors of a terminal (`--color=yes`, as in CI) are removed first.
"""
function passes(output, group)
    plain = replace(output, r"\e\[[0-9;]*m" => "")
    found = match(Regex("^test_$group\\.jl\\s*\\|\\s*(\\d+)", "m"), plain)
    found === nothing ? 0 : parse(Int, found[1])
end

if length(ARGS) == 2 && ARGS[1] == "--group"
    include("helpers.jl")
    @testset verbose = haskey(ENV, "CASADI_TEST_VERBOSE") "test_$(ARGS[2]).jl" begin
        include("test_$(ARGS[2]).jl")
    end
else
    groups = isempty(ARGS) ? collect(GROUPS) : ARGS
    jobs = parse(Int, get(ENV, "CASADI_TEST_JOBS", string(min(6, max(1, Sys.CPU_THREADS ÷ 2)))))
    @assert jobs >= 1 "CASADI_TEST_JOBS must be 1 or more"
    # Each group prints its time, so that a slow group is seen when it is added.
    done = asyncmap(groups; ntasks = jobs) do group
        passed, seconds, output = run_group(group)
        (passed && !haskey(ENV, "CASADI_TEST_VERBOSE")) || println(output)
        println(rpad("test_$group.jl", 22), lpad(passes(output, group), 5), " tests  ", round(seconds, digits = 1), " s")
        (group, passed, passes(output, group))
    end
    println("Tests passed in all groups: ", sum(last, done; init = 0))
    # The checks are in the main task, because a test set does not record the tests of other tasks on each version of Julia.
    @testset "CasADiCodegen.jl" begin
        # A group with no passed test ran nothing, or its summary was not read.
        @testset "test_$group.jl" for (group, passed, count) in done
            @test passed && count > 0
        end
    end
end
