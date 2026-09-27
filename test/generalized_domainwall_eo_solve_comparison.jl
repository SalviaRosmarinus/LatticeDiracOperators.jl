# Run in an environment containing the v1 packages:
#   JACC_BACKEND=Threads julia --project=<v1-environment> \
#       test/generalized_domainwall_eo_solve_comparison.jl
# Optional: LDO_EO_BENCH_N=4 LDO_EO_BENCH_L5=4 LDO_EO_BENCH_REPEATS=5
# Single-process comparison of the raw systems D*x=b and D†*x=b.

import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Test

function _eo_comparison_timed_solve!(solution, D, source)
    # Every sample starts from zero. Setup, clearing, and pending backend work
    # are excluded; EO reduction/reconstruction and CG RHS formation are not.
    clear_fermion!(solution)
    JACC.synchronize()
    elapsed = @elapsed begin
        solve_DinvX!(solution, D, source)
        JACC.synchronize()
    end
    return elapsed
end

function _eo_comparison_relative_difference!(work, left, right)
    substitute_fermion!(work, left)
    axpby!(-1, right, 1, work)
    return sqrt(real(dot(work, work)) / max(real(dot(right, right)), eps(Float64)))
end

function _eo_comparison_residual!(work, D, solution, source)
    mul!(work, D, solution)
    axpby!(-1, source, 1, work)
    return sqrt(real(dot(work, work)) / real(dot(source, source)))
end

function _eo_comparison_median(values)
    ordered = sort(values)
    midpoint = fld(length(ordered), 2) + 1
    return isodd(length(ordered)) ? ordered[midpoint] :
           (ordered[midpoint - 1] + ordered[midpoint]) / 2
end

@testset "Generalized domain-wall EO/non-EO solve comparison" begin
    N = parse(Int, get(ENV, "LDO_EO_BENCH_N", "4"))
    L5 = parse(Int, get(ENV, "LDO_EO_BENCH_L5", "4"))
    repeats = parse(Int, get(ENV, "LDO_EO_BENCH_REPEATS", "5"))
    N >= 2 && iseven(N) || throw(ArgumentError("LDO_EO_BENCH_N must be positive and even"))
    L5 >= 2 || throw(ArgumentError("LDO_EO_BENCH_L5 must be at least 2"))
    repeats >= 1 || throw(ArgumentError("LDO_EO_BENCH_REPEATS must be positive"))

    gsize = (N, N, N, N)
    U = gauge_configuration(
        gsize; colors=3, halo=1, start=:hot, seed=UInt64(47),
        process_grid=(1, 1, 1, 1), comm=SerialCommunicator(), verbose=0)
    source = Initialize_pseudofermion_fields(U[1], "GeneralizedDomainwall"; L5)
    gauss_distribution_fermion!(source; seed=42)
    solution_full, solution_eo = similar(source), similar(source)
    work = similar(source)
    references = (similar(source), similar(source))
    tolerance = 1e-8

    parameters = Dict{String,Any}(
        "Dirac_operator" => "GeneralizedDomainwall",
        "L5" => L5, "mass" => 0.15, "M" => -1.0,
        "as" => collect(range(0.9, 1.1; length=L5)),
        "bs" => collect(range(1.7, 0.9; length=L5)),
        "cs" => collect(range(0.6, 0.1; length=L5)),
        "eps_CG" => 1e-20, "MaxCGstep" => 3000, "verbose_level" => 0)

    @info "EO solve comparison setup (single process; setup/warmup excluded)" lattice=gsize L5 repeats threads=Threads.nthreads() storage=typeof(source.f.A) eps_CG=parameters["eps_CG"] tolerance

    for method in ("cg", "bicg", "bicgstab")
        @testset "$method" begin
            full_parameters = merge(parameters, Dict("method_CG" => method, "evenodd" => false))
            eo_parameters = merge(parameters, Dict("method_CG" => method, "evenodd" => true))
            full_outer = Dirac_operator(U, source, full_parameters)
            eo_outer = Dirac_operator(U, source, eo_parameters)
            D_full, D_eo = full_outer.D5DW, eo_outer.D5DW

            for adj in (false, true)
                direction = adj ? "adjoint" : "forward"
                @testset "$direction" begin
                    full_operator = adj ? D_full' : D_full
                    eo_operator = adj ? D_eo' : D_eo
                    reference = references[adj ? 2 : 1]

                    # Warm the same helper and both paths before sampling.
                    _eo_comparison_timed_solve!(solution_full, full_operator, source)
                    _eo_comparison_timed_solve!(solution_eo, eo_operator, source)

                    times_full, times_eo = zeros(repeats), zeros(repeats)
                    residuals_full, residuals_eo = zeros(repeats), zeros(repeats)
                    differences = zeros(repeats)
                    for sample in 1:repeats
                        # Alternate the order to reduce systematic timing bias.
                        order = isodd(sample) ? (false, true) : (true, false)
                        for use_eo in order
                            GC.gc()
                            if use_eo
                                times_eo[sample] = _eo_comparison_timed_solve!(
                                    solution_eo, eo_operator, source)
                            else
                                times_full[sample] = _eo_comparison_timed_solve!(
                                    solution_full, full_operator, source)
                            end
                        end

                        # Use the SAME full equation for both residuals, outside
                        # timing; reduced/normal residuals alone do not suffice.
                        residuals_full[sample] = _eo_comparison_residual!(
                            work, full_operator, solution_full, source)
                        residuals_eo[sample] = _eo_comparison_residual!(
                            work, full_operator, solution_eo, source)
                        differences[sample] = _eo_comparison_relative_difference!(
                            work, solution_eo, solution_full)
                        @test residuals_full[sample] < tolerance
                        @test residuals_eo[sample] < tolerance
                        @test differences[sample] < tolerance
                    end

                    # Also compare CG, BiCG, and BiCGStab against each other.
                    if method == "cg"
                        substitute_fermion!(reference, solution_full)
                    end
                    @test _eo_comparison_relative_difference!(work, solution_full, reference) < tolerance
                    @test _eo_comparison_relative_difference!(work, solution_eo, reference) < tolerance

                    full_ms = 1000 * _eo_comparison_median(times_full)
                    eo_ms = 1000 * _eo_comparison_median(times_eo)
                    speedup = full_ms / eo_ms
                    @test isfinite(speedup) && speedup > 0
                    # Above 1 means EO is faster; below 1 means EO is slower.
                    # The speed ratio is informational, not a pass criterion.
                    @info "EO/non-EO solve timing (median; speedup = non-EO / EO)" method direction non_eo_ms=full_ms eo_ms speedup
                    @info "EO/non-EO solution agreement" method direction non_eo_residual=maximum(residuals_full) eo_residual=maximum(residuals_eo) relative_solution_difference=maximum(differences)

                    # Exercise the new non-EO CG public API beyond timing:
                    # in-place inputs, diagnostics, and the PV-composed wrapper.
                    if method == "cg"
                        substitute_fermion!(solution_full, source)
                        diagnostics = solve_DinvX!(solution_full, full_operator, solution_full)
                        @test diagnostics isa SolverDiagnostics
                        @test diagnostics.method === :cg
                        @test _eo_comparison_relative_difference!(work, solution_full, reference) < tolerance

                        full_wrapped = adj ? full_outer' : full_outer
                        eo_wrapped = adj ? eo_outer' : eo_outer
                        diagnostics = solve_DinvX!(solution_full, full_wrapped, source)
                        solve_DinvX!(solution_eo, eo_wrapped, source)
                        @test diagnostics isa SolverDiagnostics
                        @test _eo_comparison_residual!(work, full_wrapped, solution_full, source) < tolerance
                        @test _eo_comparison_relative_difference!(work, solution_eo, solution_full) < tolerance
                    end
                end
            end

            if method == "cg"
                # A zero RHS and a failed normal solve must release all leases.
                clear_fermion!(work)
                clear_fermion!(solution_full)
                @test solve_DinvX!(solution_full, D_full, work).iterations == 0
                @test real(dot(solution_full, solution_full)) == 0
                for adj in (false, true)
                    failing = Dirac_operator(U, source,
                        merge(full_parameters, Dict("MaxCGstep" => 0))).D5DW
                    @test_throws ErrorException solve_DinvX!(
                        solution_full, adj ? failing' : failing, source)
                    @test !any(failing._temporary_fermi._data._flagusing)
                    @test !any(failing._temporary_fermion_forCG._data._flagusing)
                end
            end
        end
    end
end
