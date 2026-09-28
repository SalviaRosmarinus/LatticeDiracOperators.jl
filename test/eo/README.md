# Generalized domain-wall EO: JACC and MPI tests

Run these commands from the LatticeDiracOperators repository root. The test
uses the active project's JACC backend; it does not import CUDA, AMDGPU, or
another vendor package directly. LatticeMatrices v1 supplies device arrays,
portable kernel launches, rank-to-device selection, and halo transport.

## Environment

Create an isolated test environment with this checkout as the LDO dependency:

```sh
julia --project=test/eo -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

To test local Gaugefields/LatticeMatrices changes as well, `Pkg.develop` their
checkouts in the same environment. Configure JACC there, then start a fresh
Julia process. For example, the following selects the CPU backend:

```sh
backend=threads
julia --project=test/eo -e 'import JACC; JACC.set_backend(ARGS[1])' "$backend"
```

Set `backend` to the JACC backend installed on the machine to use a GPU. JACC
loads its backend with `JACC.@init_backend`. The tests use `ComplexF64`, so the
device/backend must support double precision. Selecting a backend alone does
not establish that these EO kernels have been validated on that hardware.

## One process

```sh
julia --startup-file=no --project=test/eo test/domainwall_eo.jl
```

For a GPU run, add `LDO_TEST_REQUIRE_GPU=true`. This rejects the Threads backend
and verifies that fields and the EO diagonal-inverse cache use JACC device
arrays rather than CPU arrays:

```sh
LDO_TEST_REQUIRE_GPU=true julia --startup-file=no --project=test/eo test/domainwall_eo.jl
```

## MPI, with the same selected JACC backend

The launcher below uses the MPI binary selected by MPI.jl, avoiding a mismatch
between a system `mpiexec` and the MPI library loaded in Julia:

```sh
LDO_TEST_MPI=true LDO_TEST_EXPECT_RANKS=2 \
julia --startup-file=no --project=test/eo -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=test/eo test/domainwall_eo.jl`)'
```

Add `LDO_TEST_REQUIRE_GPU=true` to that command for MPI+GPU. Device assignment
is handled by `LatticeMatrices.select_device_by_mpi_rank!`, which uses the
node-local rank and respects schedulers exposing one device per process.
Available devices, double precision, and MPI transport support remain backend
requirements. In the current v1 implementation, automatic multi-rank Metal
device mapping is unsupported. The runner prints the array type, process
layout, and the resolved halo transport on every rank; it does not assume
device-direct MPI or that distinct physical GPUs were assigned by a scheduler.

The existing `test/runtests_mpi.jl` includes this runner in its two-rank CI
tests. It selects one timing sample by default to limit CI time.

## Checks and timings

The single test file `domainwall_eo.jl` contains block inverses,
forward/adjoint Schur operators, solver cleanup, pseudofermion refresh, and
finite-difference force checks. Reference construction and numerical/halo
assertions may transfer arrays to the host; production solves, actions, and
forces use the selected backend.

It adds comparisons against an undistributed reference using exactly the same
gauge links and source. The reference runs on rank zero with an explicit serial
communicator. Spatial and temporal splits, complex/antiperiodic boundary phases,
halo widths 1 and 2, poisoned ghosts, and odd local extents are exercised. The
two-rank case has local extent 3 and global extent 6 in the split direction.

The same file checks CG, BiCG, and BiCGStab, with and without EO, for both
forward and adjoint equations. Its settings are:

| Variable | Default | Meaning |
|---|---:|---|
| `LDO_EO_BENCH_N` | 4 | Global extent in each of the four directions |
| `LDO_EO_BENCH_L5` | 4 | Local fifth-direction extent |
| `LDO_EO_BENCH_REPEATS` | 5 | Samples per solver and direction |
| `LDO_EO_BENCH_AXIS` | 1 | Physical direction split across ranks, 1–4 |

The global extent must be even and divisible by the number of ranks. Each
timing sample starts after backend synchronization and an MPI barrier, includes
the complete solve and final device synchronization, and reports the maximum
rank time. The median of those samples is printed on rank zero. Full-system
residuals and solution agreement are pass criteria; speedups are informational.

## Validation of this change

With Julia 1.11.8, Gaugefields 1.1.9, LatticeMatrices 1.2.8, JACC 1.3.1
(`threads`), and MPI.jl 0.20.26 (MPICH 5.0.1), the runner passed with one Julia
thread per process and `LDO_EO_BENCH_REPEATS=1`:

- Serial: 526 assertions passed.
- Two MPI ranks: 481 assertions on rank zero and 457 on rank one passed.
  Rank zero also checks the gathered results against the serial reference.
- Requesting a GPU while the Threads backend was selected failed as intended.

GPU and MPI+GPU execution have not been validated on accelerator hardware.
These assertion counts belong to this runner and configuration; repetitions
and rank-specific checks change the totals.
