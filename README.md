# LatticeDiracOperators


[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://akio-tomiya.github.io/LatticeDiracOperators.jl/dev)
[![CI](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/LatticeDiracOperators.jl/actions/workflows/CI.yml)

# Abstract

This is a package for lattice QCD codes.
Treating pseudo-femrion fields with various lattice Dirac operators, fermion actions with MPI.

<img src="LQCDjl_block.png" width=300> 

This package is used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl)
and a code in a project [JuliaQCD](https://github.com/JuliaQCD/).


## Q&A/Issues
If you have questions and comments. Please use the issues section of this repository or use [Discussions in JuliaQCD](https://github.com/orgs/JuliaQCD/discussions/5).

[In Japanese] 質問やコメントを日本語でしたい方は[JuliaQCDのディスカッションボード](https://github.com/orgs/JuliaQCD/discussions/6)に書き込みをしてください。

# What this package can do:
- Constructing actions and its derivative for Staggered Fermion with 1-8 tastes (with the use of the rational HMC technique)
- Constructing actions and its derivative for Wilson Fermion
- (EXPERIMENTAL) Constructing actions and its derivative for Standard Domainwall Fermion 
- Hybrid Monte Carlo method with fermions.

With the use of the Gaugefields.jl, we can also do the HMC with STOUT smearing. 

This package will be used in LatticeQCD.jl. 
This package uses [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl). 
This package can be regarded as the additional package of the Gaugefields.jl to treat with Lattice fermions (pseudo- fermions). 

# Install

In the package mode, Julia REPL,
```
add LatticeDiracOperators
```

# How to use

## Definition of the pseudo-fermion fields

The pseudo-fermin field is defined as 

```julia
using Gaugefields
using LatticeDiracOperators

NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4
NC = 3

U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
x = Initialize_pseudofermion_fields(U[1],"Wilson")
```

Now, x is a pseudo fermion fields for Wilson Dirac operator. 
The element of x is ```x[ic,ix,iy,iz,it,ialpha]```. ic is an index of the color. ialpha is the internal degree of the gamma matrix. 

Then, the Wilson Dirac operator can be defined as 

```julia
params = Dict()
params["Dirac_operator"] = "Wilson"
params["κ"] = 0.141139
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2

D = Dirac_operator(U,x,params)
```

If you want to get the Gaussian distributed pseudo-fermions, just do

```julia
gauss_distribution_fermion!(x)
```

Then, you can apply the Dirac operator to the pseudo-fermion fields. 

```julia
using LinearAlgebra
y = similar(x)
mul!(y,D,x)
```

And you can solve the equation $D x = b$ like

```julia
solve_DinvX!(y,D,x)
println(y[1,1,1,1,1,1])
```
If you want to see the convergence of the CG method, you can change the "verbose_level" in the Dirac operator. 

```julia
params["verbose_level"] = 3
D = Dirac_operator(U,x,params)
gauss_distribution_fermion!(x)
solve_DinvX!(y,D,x)
println(y[1,1,1,1,1,1])
```

The output is like 

```
bicg method
1-th eps: 1742.5253056262081
2-th eps: 758.2899742222573
3-th eps: 378.7020470573924
4-th eps: 210.17029515182503
5-th eps: 118.00493128655506
6-th eps: 63.31719669150997
7-th eps: 36.18603541453448
8-th eps: 21.593691953496077
9-th eps: 16.02895509383768
10-th eps: 12.920647360667004
11-th eps: 9.532250164198402
12-th eps: 5.708202470516758
13-th eps: 3.1711913019834337
14-th eps: 0.9672090407947617
15-th eps: 0.14579004932559966
16-th eps: 0.02467506197970277
17-th eps: 0.005588563782732157
18-th eps: 0.002285284357387675
19-th eps: 5.147142014626153e-5
20-th eps: 3.5632092739322066e-10
Converged at 20-th step. eps: 3.5632092739322066e-10
```

## Other operators
You can use the adjoint of the Dirac operator 

```julia
gauss_distribution_fermion!(x)
solve_DinvX!(y,D',x)
println(y[1,1,1,1,1,1])
```

You can define the ```D^{\dagger} D``` operator. 

```julia
DdagD = DdagD_operator(U,x,params)
gauss_distribution_fermion!(x)
solve_DinvX!(y,DdagD,x) 
println(y[1,1,1,1,1,1])
```

# Staggared Fermions
The Dirac operator of the staggered fermions is defined as 

```julia
x = Initialize_pseudofermion_fields(U[1],"staggered")
gauss_distribution_fermion!(x)
params = Dict()
params["Dirac_operator"] = "staggered"
params["mass"] = 0.1
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2
D = Dirac_operator(U,x,params)

y = similar(x)
mul!(y,D,x)
println(y[1,1,1,1,1,1])

solve_DinvX!(y,D,x)
println(y[1,1,1,1,1,1])
```

The "tastes" of the Staggered Fermion is defined in the action. 

# (EXPERIMENTAL) Domainwall Fermions
This package supports standard domainwall fermions. 
The Dirac operator of the domainwall fermion is defined as 

```julia
L5 = 4
x = Initialize_pseudofermion_fields(U[1],"Domainwall",L5=L5)
println("x ", x.w[1][1,1,1,1,1,1])
gauss_distribution_fermion!(x)

params = Dict()
params["Dirac_operator"] = "Domainwall"
params["eps_CG"] = 1.0e-16
params["MaxCGstep"] = 3000
params["verbose_level"] = 3
params["mass"] = 0.1
params["L5"] = L5
D = Dirac_operator(U,x,params)

println("x ", x[1,1,1,1,1,1,1])
y = similar(x)
solve_DinvX!(y,D,x)
println("y ", y[1,1,1,1,1,1,1])

z = similar(x)
mul!(z,D,y)
println("z ", z[1,1,1,1,1,1,1])

```

The domainwall fermion is defined in 5D space. The element of x is ```x[ic,ix,iy,iz,it,ialpha,iL]```, where iL is an index on the five dimensional axis. 

# Fermion Action

## Wilson Fermion

The action for pseudo-fermion is defined as 

```julia

NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4
NC = 3

U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
x = Initialize_pseudofermion_fields(U[1],"Wilson")
gauss_distribution_fermion!(x)

params = Dict()
params["Dirac_operator"] = "Wilson"
params["κ"] = 0.141139
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2

D = Dirac_operator(U,x,params)

parameters_action = Dict()
fermi_action = FermiAction(D,parameters_action)


```

The fermion action with given pseudo-fermion fields is evaluated as 

```julia
Sfnew = evaluate_FermiAction(fermi_action,U,x)
println(Sfnew)
```

The derivative of the fermion action dSf/dU can be calculated as 

```julia
UdSfdUμ = calc_UdSfdU(fermi_action,U,x)
```
The function calc_UdSfdU calculates the ```U dSf/dU```,
You can also use ```calc_UdSfdU!(UdSfdUμ,fermi_action,U,x)```


## Staggered Fermion
In the case of the Staggered fermion, we can choose "taste". 
The action is defined as 

```julia
x = Initialize_pseudofermion_fields(U[1],"staggered")
gauss_distribution_fermion!(x)
params = Dict()
params["Dirac_operator"] = "staggered"
params["mass"] = 0.1
params["eps_CG"] = 1.0e-8
params["verbose_level"] = 2
D = Dirac_operator(U,x,params)

Nf = 2

println("Nf = $Nf")
parameters_action = Dict()
parameters_action["Nf"] = Nf
fermi_action = FermiAction(D,parameters_action)

Sfnew = evaluate_FermiAction(fermi_action,U,x)
println(Sfnew)

UdSfdUμ = calc_UdSfdU(fermi_action,U,x)
```

This package uses the RHMC techniques. 

## (EXPERIMENTAL) Domainwall Fermions
In the case of the domainwall fermion, the action is defined as 

```julia
 L5 = 4
x = Initialize_pseudofermion_fields(U[1],"Domainwall",L5 = L5)
gauss_distribution_fermion!(x)

params = Dict()
params["Dirac_operator"] = "Domainwall"
params["mass"] = 0.1
params["L5"] = L5
params["eps_CG"] = 1.0e-19
params["verbose_level"] = 2
params["method_CG"] = "bicg"
D = Dirac_operator(U,x,params)

parameters_action = Dict()
fermi_action = FermiAction(D,parameters_action)

Sfnew = evaluate_FermiAction(fermi_action,U,x)
println(Sfnew)

UdSfdUμ = calc_UdSfdU(fermi_action,U,x)
```


# Hybrid Monte Carlo with fermions

## Wilson Fermion
We show the HMC code with this package. 

```julia
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using InteractiveUtils
using Random

function MDtest!(gauge_action,U,Dim,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    Random.seed!(123)

    numtrj = 10
    for itrj = 1:numtrj
        @time accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
    Δτ = 1/MDsteps
    NC,_,NN... = size(U[1])
    
    gauss_distribution!(p)
    
    substitute_U!(Uold,U)
    gauss_sampling_in_action!(ξ,U,fermi_action)
    sample_pseudofermions!(η,U,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Sfnew = evaluate_FermiAction(fermi_action,U,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")

    accept = exp(Sold - Snew) >= rand()

    #ratio = min(1,exp(Snew-Sold))
    if accept != true #rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end

function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    calc_UdSfdU!(UdSfdUμ,fermi_action,U,η)

    for μ=1:Dim
        Traceless_antihermitian_add!(p[μ],factor,UdSfdUμ[μ])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end
end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    x = Initialize_pseudofermion_fields(U[1],"Wilson")


    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)


    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)

    y = similar(x)

    
    MDtest!(gauge_action,U,Dim,fermi_action,x,y)

end


test1()
```

## Staggered Fermion
if you want to use the Staggered fermions in HMC, the code is like: 

```julia
function test2()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    x = Initialize_pseudofermion_fields(U[1],"staggered")
    gauss_distribution_fermion!(x)
    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = 0.1
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)
    
    Nf = 2
    
    println("Nf = $Nf")
    parameters_action = Dict()
    parameters_action["Nf"] = Nf
    fermi_action = FermiAction(D,parameters_action)

    y = similar(x)

    
    MDtest!(gauge_action,U,Dim,fermi_action,x,y)

end

```

# HMC with fermions with stout smearing
We show the code of HMC with Wilson fermions with stout smearing. 

```julia
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using LatticeDiracOperators

function MDtest!(gauge_action,U,Dim,nn,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)
    
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU,fermi_action,η,ξ)
    

    Δτ = 1/MDsteps
    gauss_distribution!(p)

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Sold = calc_action(gauge_action,Uout,p)

    substitute_U!(Uold,U)

    gauss_sampling_in_action!(ξ,Uout,fermi_action)
    sample_pseudofermions!(η,Uout,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)


    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    #Snew = calc_action(gauge_action,Uout,p)

    Sfnew = evaluate_FermiAction(fermi_action,Uout,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end


function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,dSdU,nn,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    Uout,Uout_multi,_ = calc_smearedU(U,nn)

    for μ=1:Dim
        calc_UdSfdU!(UdSfdUμ,fermi_action,Uout,η)
        mul!(dSdU[μ],Uout[μ]',UdSfdUμ[μ])
    end

    dSdUbare = back_prop(dSdU,nn,Uout_multi,U) 
    

    for μ=1:Dim
        mul!(temps[1],U[μ],dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end
end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)

    show(gauge_action)

    L = [NX,NY,NZ,NT]
    nn = CovNeuralnet(U)
    ρ = [0.1]
    layername = ["plaquette"]
    #st = STOUT_Layer(layername,ρ,L)
    st = STOUT_Layer(layername, ρ, U)
    push!(nn,st)
    #push!(nn,st)

    x = Initialize_pseudofermion_fields(U[1],"Wilson")


    params = Dict()
    params["Dirac_operator"] = "Wilson"
    params["κ"] = 0.141139
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U,x,params)


    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)

    y = similar(x)
    

    MDtest!(gauge_action,U,Dim,nn,fermi_action,x,y)

end


test1()
```

# SLHMC

```julia
using LinearAlgebra
using Optimisers
using Wilsonloop
using Gaugefields
using LatticeDiracOperators
import Gaugefields.Abstractsmearing_module: get_parameters, zero_grad!, set_parameters!

# For debug
import InteractiveUtils
versionstring = """
$(InteractiveUtils.versioninfo())
LatticeDiracOperators $(pkgversion(LatticeDiracOperators))
Gaugefields $(pkgversion(Gaugefields))
Wilsonloop $(pkgversion(Wilsonloop))
"""
println(versionstring)
#---

const NX = 4
const NY = 4
const NZ = 4
const NT = 4

const Ncin = 2
const β0 = 2.45
const mass = 0.3
const mass_eff = 0.4

const Nf = 4
const filename = "mass005eff01_stout_2_1109.txt"
const paramfilename = "param_" * filename
const actionfilename = "diff_" * filename

const MDsteps = 20
const numtrj = 100
const numbatch = 1
const eta = 1e-4 #parameter for ADAM 
const optimiser = Optimisers.Adam(eta)
const trainable = true

function MDtest!(gauge_action, U, Dim, nn, fermi_action, η, ξ, fermi_action_eff, numtrj, isbare)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)

    substitute_U!(Uold, U)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0
    fp = open(filename, "w")
    fp2 = open(paramfilename, "w")
    fp3 = open(actionfilename, "w")

    θ = get_parameters(nn)
    state = Optimisers.setup(optimiser, θ)
    dLdθ = zero(θ)


    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, fermi_action, η, ξ, fermi_action_eff, state, θ, dLdθ, fp3, isbare)

        if itrj % numbatch == 0 && !isbare
            println("θ_before = $θ        ")
            if trainable
                println("dLdθ ")
                display(dLdθ)
                Optimisers.update!(state, θ, dLdθ / numbatch)
            end
            println("θ = $θ        ")
            set_parameters!(nn, θ)
            dLdθ .= 0
        end


        for p in θ
            print(fp2, p, "\t")
        end
        println(fp2, "\t")
        flush(fp2)
        flush(fp3)
        numaccepted += ifelse(accepted, 1, 0)

        plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
        println("$itrj plaq_t = $plaq_t")
        #println(fp,"$itrj $plaq_t")
        println("acceptance ratio ", numaccepted / itrj)
        println(fp, "$itrj $plaq_t $(numaccepted / itrj) #itrj plaq_t acceptanceratio")
        flush(fp)

    end
    close(fp)
    close(fp2)
    close(fp3)
end

function calc_action(gauge_action, U, p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action, U) / NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p * p / 2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, nn, dSdU, fermi_action, η, ξ,
    fermi_action_eff, state, θ, dLdθ, fp3, isbare)

    Δτ = 1 / MDsteps
    gauss_distribution!(p)


    #Uout, Uout_multi, _ = calc_smearedU(U, nn)
    substitute_U!(Uold, U)

    gauss_sampling_in_action!(ξ, U, fermi_action)
    sample_pseudofermions!(η, U, fermi_action, ξ)
    Sfold = real(dot(ξ, ξ))
    println("Sfold = $Sfold")

    Sgold = calc_action(gauge_action, U, p)
    println("Sgold = $Sgold")
    Sold = Sgold + Sfold
    println("Sold = ", Sold)


    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action)
        #P_update_fermion!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn, fermi_action, η)
        P_update_fermion!(U, p, 1.0, Δτ, Dim, gauge_action, dSdU, nn, fermi_action_eff, η, isbare)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action)
    end

    Sfnew = evaluate_FermiAction(fermi_action, U, η)
    println("Sfnew = $Sfnew")
    if !isbare
        Uout, Uout_multi, _ = calc_smearedU(U, nn)
        Sfnew_eff = evaluate_FermiAction(fermi_action_eff, Uout, η)
        println(fp3, "$Sfnew \t $(Sfnew_eff) #Sf, Sf_eff")
        zero_grad!(nn)
    end




    if !isbare
        temps = get_temporary_gaugefields(gauge_action)
        UdSfdUμ = temps[1:Dim]
        for μ = 1:Dim
            calc_UdSfdU!(UdSfdUμ, fermi_action_eff, Uout, η)
            mul!(dSdU[μ], Uout[μ]', UdSfdUμ[μ])
        end

        dSdUbare = back_prop(dSdU, nn, Uout_multi, U)
        dSdw = deepcopy(get_parameter_derivatives(nn) * -1)

        loss = (Sfnew - Sfnew_eff)^2
        dLdw = (-2) * dSdw * (Sfnew - Sfnew_eff)


        dLdθ .+= dLdw
        println(dSdw)
        println(dLdw)
        println("loss = $loss")
    end


    Sgnew = calc_action(gauge_action, U, p)
    Snew = Sgnew + Sfnew
    println("Sgnew = $Sgnew")
    println("Sg: Sgnew Sgold Sgnew-Sgold: $Sgnew $Sgold $(Sgnew-Sgold)")
    println("Sf: Sfnew Sfold Sfnew-Sfold: $Sfnew $Sfold $(Sfnew-Sfold)")

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-(Snew - Sold)))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end


function U_update!(U, p, ϵ, Δτ, Dim, gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]
    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)

    end
end

function P_update!(U, p, ϵ, Δτ, Dim, gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor = -ϵ * Δτ / (NC)

    for μ = 1:Dim
        calc_dSdUμ!(dSdUμ, gauge_action, μ, U)
        mul!(temps[1], U[μ], dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ], factor, temps[1])
    end
end


function P_update_fermion!(U, p, ϵ, Δτ, Dim, gauge_action, dSdU,
    nn, fermi_action, η, isbare)  # p -> p +factor*U*dSdUμ
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor = -ϵ * Δτ

    if !isbare
        Uout, Uout_multi, _ = calc_smearedU(U, nn)
        for μ = 1:Dim
            calc_UdSfdU!(UdSfdUμ, fermi_action, Uout, η)
            mul!(dSdU[μ], Uout[μ]', UdSfdUμ[μ])
        end
        dSdUbare = back_prop(dSdU, nn, Uout_multi, U)

        for μ = 1:Dim
            mul!(temps[1], U[μ], dSdUbare[μ]) # U*dSdUμ
            Traceless_antihermitian_add!(p[μ], factor, temps[1])
        end

    else
        for μ = 1:Dim
            calc_UdSfdU!(UdSfdUμ, fermi_action, U, η)
            mul!(dSdU[μ], U[μ]', UdSfdUμ[μ])
        end

        for μ = 1:Dim
            mul!(temps[1], U[μ], dSdU[μ]) # U*dSdUμ
            Traceless_antihermitian_add!(p[μ], factor, temps[1])
        end
    end


end

function test1()
    Nwing = 0
    Dim = 4
    NC = Ncin

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop, plaqloop')
    β = β0 / 2
    push!(gauge_action, β, plaqloop)

    show(gauge_action)

    L = [NX, NY, NZ, NT]

    nn = CovNeuralnet(U)
    layername = ["plaquette", "polyakov_x", "polyakov_y", "polyakov_z", "polyakov_t"]
    ρ = (2 * rand(length(layername)) .- 1) * 1e-3
    st = STOUT_Layer(layername, ρ, U)
    push!(nn, st)
    ρ = (2 * rand(length(layername)) .- 1) * 1e-3
    st2 = STOUT_Layer(layername, ρ, U)
    push!(nn, st2)

    x = Initialize_pseudofermion_fields(U[1], "staggered")
    gauss_distribution_fermion!(x)
    params = Dict()
    params["Dirac_operator"] = "staggered"
    params["mass"] = mass
    params["eps_CG"] = 1.0e-8
    params["verbose_level"] = 2
    D = Dirac_operator(U, x, params)

    parameters_action = Dict()
    parameters_action["Nf"] = Nf
    fermi_action = FermiAction(D, parameters_action)

    y = similar(x)

    isbare = true
    MDtest!(gauge_action, U, Dim, nn, fermi_action, x, y, fermi_action, 10, isbare)

    x_eff = Initialize_pseudofermion_fields(U[1], "staggered")
    params_eff = Dict()
    params_eff["Dirac_operator"] = "staggered"
    params_eff["mass"] = mass_eff

    params_eff["eps_CG"] = 1.0e-8
    params_eff["verbose_level"] = 2
    D_eff = Dirac_operator(U, x_eff, params_eff)
    parameters_action_eff = Dict()
    parameters_action_eff["Nf"] = Nf
    fermi_action_eff = FermiAction(D_eff, parameters_action_eff)

    isbare = false
    MDtest!(gauge_action, U, Dim, nn, fermi_action, x, y, fermi_action_eff, numtrj, isbare)

end

test1()
```

# Acknowledgment
If you write a paper using this package, please refer this code.

BibTeX citation is following
```
@article{Nagai:2024yaf,
    author = "Nagai, Yuki and Tomiya, Akio",
    title = "{JuliaQCD: Portable lattice QCD package in Julia language}",
    eprint = "2409.03030",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "9",
    year = "2024"
}
```
and the paper is [arXiv:2409.03030](https://arxiv.org/abs/2409.03030).
