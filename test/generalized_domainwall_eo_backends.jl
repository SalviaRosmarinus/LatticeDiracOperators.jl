# Run with the active project's JACC backend; no vendor-specific imports.
# LDO_TEST_MPI=true enables MPI, LDO_TEST_REQUIRE_GPU=true rejects CPU fallback.
# See test/eo/README.md for serial, MPI, GPU, and MPI+GPU commands.
import JACC
JACC.@init_backend

using Gaugefields, LatticeDiracOperators, LatticeMatrices, LinearAlgebra, Test
include(joinpath(@__DIR__, "test_communicator.jl"))

const EO_REQUIRE_GPU = lowercase(get(ENV, "LDO_TEST_REQUIRE_GPU", "false")) == "true"
EO_REQUIRE_GPU && JACC.backend == "threads" && error(
    "GPU test requested, but JACC selected threads; configure JACC in the active project first")
const EO_EXPECTED_RANKS = parse(Int, get(ENV, "LDO_TEST_EXPECT_RANKS", string(ldo_test_comm_size())))
ldo_test_comm_size() == EO_EXPECTED_RANKS || error(
    "Expected $EO_EXPECTED_RANKS ranks, got $(ldo_test_comm_size()); check LDO_TEST_MPI and the launcher")

# Device selection belongs to the v1 backend layer and uses node-local rank.
# It runs before allocation and also supports scheduler-isolated devices.
const EO_DEVICE_SELECTION = select_device_by_mpi_rank!(LDO_TEST_COMM)
@info "EO backend test" backend=JACC.backend rank=ldo_test_comm_rank() ranks=ldo_test_comm_size() selection=EO_DEVICE_SELECTION

function _eo_backend_serial_reference(links, source, phases, halo)
    comm = SerialCommunicator()
    grid = (1, 1, 1, 1)
    gsize = size(links[1])[3:end]
    U = gauge_configuration(gsize; colors=3, halo, start=:cold,
        process_grid=grid, comm, verbose=0)
    for mu in 1:4
        field = LatticeMatrix(links[mu], 4, grid;
            nw=halo, comm0=comm, device_mapping=:current)
        LatticeMatrices.substitute!(U[mu].U, field)
    end
    set_wing_U!(U)
    L5 = size(source, 7)
    x = DomainwallFermion_5D_MPILattice(U[1], L5;
        operator_name="GeneralizedDomainwall", boundarycondition=phases)
    field = LatticeMatrix(source, 5, (grid..., 1);
        nw=halo, phases, comm0=comm, device_mapping=:current)
    LatticeMatrices.substitute!(x.f, field)
    set_wing_fermion!(x)
    parameters = merge(_eo_parameters(L5), Dict("method_CG" => "bicgstab"))
    A = Dirac_operator(U, x, parameters).D5DW
    full = Dirac_operator(U, x, merge(parameters, Dict("evenodd" => false))).D5DW
    S = EO_Dop.D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)
    return (; x, image=similar(x), A, S, full)
end

function _eo_backend_decomposition_test(axis, halo)
    nprocs = ldo_test_comm_size()
    rank = ldo_test_comm_rank()
    # On two ranks, local extent 3 makes the parity origin differ on rank 1.
    # The global periodic graph stays even. Both spatial and time cuts are used.
    local_extent = iseven(nprocs) ? 3 : 4
    gsize = ntuple(d -> d == axis ? local_extent * nprocs : 2, 4)
    grid = ntuple(d -> d == axis ? nprocs : 1, 4)
    phases = [cis(0.31), 1, 1, -1, 1]
    L5 = 3
    U = gauge_configuration(gsize; colors=3, halo, start=:hot, seed=UInt64(91),
        process_grid=grid, comm=LDO_TEST_COMM, verbose=0)
    x = DomainwallFermion_5D_MPILattice(U[1], L5;
        operator_name="GeneralizedDomainwall", boundarycondition=phases)
    gauss_distribution_fermion!(x; seed=92)
    saved, image, solution = similar(x), similar(x), similar(x)
    substitute_fermion!(saved, x)
    A = Dirac_operator(U, x,
        merge(_eo_parameters(L5), Dict("method_CG" => "bicgstab"))).D5DW
    S = EO_Dop.D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)

    @test x.f.A isa JACC.array_type()
    @test all(link -> link.U.A isa JACC.array_type(), U)
    @test A.eo_cache.lower isa JACC.array_type()
    @test A.eo_cache.upper_adjoint isa JACC.array_type()
    if EO_REQUIRE_GPU
        @test !(x.f.A isa Array)
        @test !(A.eo_cache.lower isa Array)
    end
    @test x.f.PN[axis] == local_extent
    @test x.f.dims == (grid..., 1)
    @info "EO lattice layout" rank axis halo global_size=gsize local_size=x.f.PN storage=typeof(x.f.A) transport=mpi_transport_info(x.f)

    # Every rank participates in gathers; only rank 0 constructs the serial
    # oracle, so another rank's selected GPU is never changed to device zero.
    links = [gather_matrix(U[mu].U) for mu in 1:4]
    source = gather_matrix(x.f)
    reference = rank == 0 ? _eo_backend_serial_reference(links, source, phases, halo) : nothing
    for which in (:full, :schur), adj in (false, true)
        op = which === :full ? A : S
        # Restore just the core after poisoning all ghosts. Communication must
        # repair remote and periodic faces before a stencil reads them.
        fill!(x.f.A, ComplexF64(NaN, NaN))
        LatticeMatrices.substitute!(x.f, saved.f)
        @test halo_is_dirty(x.f)
        mul!(image, adj ? op' : op, x)
        @test !halo_is_dirty(image.f)
        result = gather_matrix(image.f)
        if rank == 0
            refop = which === :full ? reference.A : reference.S
            mul!(reference.image, adj ? refop' : refop, reference.x)
            expected = gather_matrix(reference.image.f)
            @test result ≈ expected rtol=3e-11 atol=3e-11
        end
    end

    for adj in (false, true)
        op = adj ? A' : A
        clear_fermion!(solution)
        solve_DinvX!(solution, op, saved)
        @test !halo_is_dirty(solution.f)
        mul!(image, op, solution)
        @test _eo_relative_difference(image, saved) < 1e-8
        result = gather_matrix(solution.f)
        if rank == 0
            clear_fermion!(reference.image)
            refop = adj ? reference.full' : reference.full
            solve_DinvX!(reference.image, refop, reference.x)
            expected = gather_matrix(reference.image.f)
            @test norm(result - expected) / norm(expected) < 1e-8
        end
    end
    @test _eo_pool_free(A._temporary_fermion_forCG)
    return nothing
end

@testset "Generalized domain-wall EO on JACC backend" begin
    # Reuse the numerical/force and all-three-solver checks on the actual
    # backend and communicator instead of maintaining GPU-only copies.
    include("generalized_domainwall_evenodd.jl")
    @testset "decomposition axis=$axis halo=$halo" for axis in (1, 4), halo in (1, 2)
        _eo_backend_decomposition_test(axis, halo)
    end
    include("generalized_domainwall_eo_solve_comparison.jl")
end
