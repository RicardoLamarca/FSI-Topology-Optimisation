using Pkg
# Pkg.activate(".") # Activates the current directory as the project
# Pkg.develop(path="MultiTopOpt.jl-main")
using Pkg
Pkg.activate()
import Pkg
# Add here your .msh file with all the corresponding labels, file must be created with Gmsh or GridapGmsh.jl
mesh_file_path = "Model File here" 
# Setup Environment, specially if used for the first time 
function setup_environment()
    Pkg.activate(".")
    if Sys.iswindows()
        try
            installed = haskey(Pkg.project().dependencies, "P4est_jll")
            if !installed
                Pkg.add(name="P4est_jll", version="2.8.1") #Further versions might cause issues in windows 
                Pkg.pin("P4est_jll")
            else
                println("P4est_jll is already installed.")
            end
        catch e
            @warn "Failed to pin P4est_jll. This might cause GridapTopOpt precompilation errors on Windows." exception=e
        end
    end
    required_packages = [
        "LineSearches",
        "WriteVTK",
        "Gridap",
        "PartitionedArrays",
        "GridapDistributed",
        "Metis",
        "GridapGmsh",
        "NLopt",
        "GridapTopOpt",
        "ChainRulesCore",
        "GridapSolvers",
        "Distributions"
    ]
    for pkg in required_packages
        if Base.find_package(pkg) === nothing
            println("  -> Installing missing package: $pkg...")
            Pkg.add(pkg)
        end
    end
    Pkg.instantiate()
    println("Environment ready.")
end

setup_environment()
Pkg.develop(path="MultiTopOpt.jl-main")
using MultiTopOpt
using Gridap
using GridapGmsh 
using Distributions 
using Statistics 
using GridapSolvers.NonlinearSolvers 
using NLopt
using Printf
using PartitionedArrays
using GridapDistributed
import MultiTopOpt: history

function check_convergence_mkts(history::Vector{Float64}, t::Int, params::NamedTuple)

    # 1. Check if we have enough iterations
    if t < params.minIter
        return false
    end

    # 2. Define window size W
    W = min(params.winSize, t) # Use up to t elements if t < winSize
    if W < 3
        return false # Not enough data to check
    end

    # 3. Get the data window
    x = history[t-W+1 : t]


    # Calculate S statistic
    S = 0.0
    for i in 1:W-1
        for j in i+1:W
        S += sign(x[j] - x[i])
        end
    end

    # Calculate Var(S) with tie correction
    tie_correction = 0.0
    ux = unique(x)
    if length(ux) < W
        for val in ux
            tp = count(==(val), x) # tp is the size of the p-th group of identical values
            if tp > 1
                tie_correction += tp * (tp - 1) * (2 * tp + 5)
            end
        end
    end
    varS = (W * (W - 1) * (2 * W + 5) - tie_correction) / 18.0

    # Calculate standardized statistic Z_MK
    z_mk = 0.0
    if varS > 0
        if S > 0
            z_mk = (S - 1) / sqrt(varS)
            elseif S < 0
             z_mk = (S + 1) / sqrt(varS)
        end
    end 

    # Calculate bilateral p-value
    p_mk = 2 * (1 - cdf(Normal(0, 1), abs(z_mk)))

    # MK Citerion: no trend if p_mk > α
    mk_passed = p_mk > params.α

    # Calculate all pairwise slopes
    slopes = Float64[]
    sizehint!(slopes, W * (W - 1) ÷ 2)
    for i in 1:W-1
        for j in i+1:W
            # t_j - t_i is just j - i
            push!(slopes, (x[j] - x[i]) / (j - i))
        end
    end

    # TS slope is the median
    beta_ts = isempty(slopes) ? 0.0 : median(slopes)

    # TS Criterion: small slope, checking for safeguard relaxation
    current_tol_factor = if t >= params.safeguardAt
        params.relaxFactor * params.slopeTolRel # Relaxed rule
    else
        params.slopeTolRel # Base rule
    end

    # Using relative tolerance as defined on page 1 of the document
    x_bar = mean(x) 
    # Handle case where x_bar is 0 or very small
    ts_criterion_val = abs(current_tol_factor * x_bar)
    # Use a small absolute tolerance if relative is near zero
    ts_criterion = ts_criterion_val < 1e-12 ? 1e-12 : ts_criterion_val
    
    ts_passed = abs(beta_ts) <= ts_criterion

    # Stop if MK detects no trend AND TS slope is small
    if mk_passed && ts_passed
        # Printing is handled in the callback
        return true
    end

    return false
end

function compute_navier_stokes_load(Ω, Ω_f, Ω_s; k=2, μ=0.05, ρf=0.5, U=10.0, verbose=true)
    
    # 1. Setup 3D Parameters
    D = 3 
    Re = ρf * U / μ
    if verbose; println(">>> Reynolds number: ", Re); end
    
    # 2. Boundary Conditions (3D)
    uf_in(x) = VectorValue( U, 0.0, 0.0 ) # Inlet along X
    uf_0(x)  = VectorValue(0.0, 0.0, 0.0)
    f(x)     = VectorValue(0.0, 0.0, 0.0) 
    s(x)     = VectorValue(0.0, 0.0, 0.0) 

    # 3. FE Spaces
    reffe_u = ReferenceFE(lagrangian, VectorValue{D,Float64}, k)
    reffe_p = ReferenceFE(lagrangian, Float64, k - 1)

    # Note: Ensure your mesh tags match these strings
    Vf = TestFESpace(Ω_f, reffe_u, conformity=:H1, dirichlet_tags=["inlet", "NoSlip", "interface"])
    Qf = TestFESpace(Ω_f, reffe_p, conformity=:C0)
    Y = MultiFieldFESpace([Vf, Qf])

    Uf = TrialFESpace(Vf, [uf_in, uf_0, uf_0])
    Pf = TrialFESpace(Qf)
    X = MultiFieldFESpace([Uf, Pf])

    # 4. Measures & Interfaces
    degree = 2 * k
    dΩ_f = Measure(Ω_f, degree)
    
    # Interface Fluid-Solid (Normal points Fluid -> Solid)
    Γ_fs = InterfaceTriangulation(Ω_f, Ω_s)
    n_Γfs = get_normal_vector(Γ_fs)
    dΓ_fs = Measure(Γ_fs, degree)

    # 5. Weak Form (Navier-Stokes)
    σ_dev_f(ε) = 2*μ*ε
    a_stokes((u, p), (v, q)) = ∫(ε(v) ⊙ (σ_dev_f ∘ ε(u)) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ_f
    
    conv(u, ∇u) = ρf * (∇u') ⋅ u
    dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)
    l_f((v,q)) = ∫(v⋅f)dΩ_f

    # Residual & Jacobian
    res_f((u, p), (v, q)) = a_stokes((u, p), (v, q)) + ∫(v ⋅ (conv ∘ (u, ∇(u))))dΩ_f - l_f((v,q))
    jac_f((u, p), (du, dp), (v, q)) = a_stokes((du, dp), (v, q)) + ∫(v ⋅ (dconv ∘ (du, ∇(du), u, ∇(u))))dΩ_f

    # 6. Solve
    op_f = FEOperator(res_f, jac_f, X, Y)
    
    if verbose; println(">>> Solving 3D Navier-Stokes..."); end
    nls = NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-7, verbose=verbose)
    xh = solve(nls, op_f)
    uhf, ph = xh
    if verbose; println(">>> Fluid solution converged."); end

    # 7. Compute Interaction Load (Traction)
    # The load acting on the solid is the traction T = n . (sigma_f)
    # T = n . (2μ ε(u) - pI)
    l_s_vol(v,ρ,dΩ,dΓ) = ∫(v ⋅ s)dΩ
    
    # Neumann Load term on the interface
    l_s_f(v,ρ,dΩ,dΓ) = ∫( v.⁻ ⋅ (n_Γfs.⁺ ⋅ (σ_dev_f∘ε(uhf.⁺)) - ph.⁺*n_Γfs.⁺ ) )dΓ_fs
    
    # Total Linear Form for Solid
    lₛ(v,ρ,dΩ,dΓ) = l_s_vol(v,ρ,dΩ,dΓ) + l_s_f(v,ρ,dΩ,dΓ)

    # Save fluid results for visualization
    writevtk(Ω_f,"Results/Fluid_Solution_3D",cellfields=["ph"=>ph,"uhf"=>uhf])

    return lₛ, dΓ_fs
end


function NavierImmersedBodyMain3D(ranks, np, solvers;
    mesh_file::String="",
    action=:optimize_w_mma,
    fe_order::Int=1,
    filter_radius_variation=:MeshSizeProportional,
    filter_radius_parameter::Float64=1.5,
    E::Float64=70e9,
    ν::Float64=0.33,
    volume_fraction=0.3,
    path="Results",
    title="WingOpt3D",
    names=["uh","ρfh"],
    output_freq=1,
    verbose=false,
    max_iter::Int=120,
    U_inlet=10.0,
    kwargs...)

    solid_solver = solvers[1]
    filter_solver = solvers[2]
    
    # 1. Output Directory
    if !isdir(path)
        mkpath(path)
        i_am_main(ranks) && println("Created output directory: $path")
    end

    # 2. Geometry
    i_am_main(ranks) && println(">>> Loading Model: $mesh_file")
    model = GmshDiscreteModel(ranks, mesh_file)
    writevtk(model, joinpath(path, "ModelGeometry"))

    # Labels (Ensure these match your .msh file)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "design", "Wing") 
    add_tag_from_tags!(labels, "nodesign", "Fluid")

    Ω   = Interior(model)
    Ω_f = Interior(model, tags="Fluid")
    Ω_s = Interior(model, tags="Wing")
    dΩ_s = Measure(Ω_s, 2*fe_order)

    # 3. Calculate Fluid Load
    # We pass the domain info to the helper function we wrote above
    lₛ, dΓ_fs = compute_navier_stokes_load(Ω, Ω_f, Ω_s; U=U_inlet, verbose=verbose) 
    
    # 4. Topology Setup
    χ = DensityBasedTopology(Ω_s, type=:uniform, value=volume_fraction)
    
    # Helmholtz Filter
    R = HelmholtzFilter(Ω_s, dΩ_s, χ; solver=filter_solver, 
                        radius_variation=filter_radius_variation, 
                        radius_parameter=filter_radius_parameter)

    # 5. Physics (3D Elasticity)
    # Using the SIMP_ElasticityBEAM you provided (it works for 3D if ε is 3D)
    aₛ, daₛ = SIMP_Elasticity(E, ν; p=3) # Assuming this is available in scope

    # Heaviside Projection
    β = 0.01; η = 0.5
    T(ρ) = ((tanh(β * η) + tanh(β * (ρ - η))) / (tanh(β * η) + tanh(β * (1.0 - η))))
    ∂T∂ρ(ρ) = β * (1.0 - tanh(β * (ρ - η))^2) / (tanh(β * η) + tanh(β * (1.0 - η)))

    aₚ(u, v, ρ, dΩ, dΓ) = aₛ(u, v, T ∘ ρ, dΩ)
    daₚ(u, v, ρ, δρ, dΩ, dΓ) = daₛ(u, v, T ∘ ρ, ((∂T∂ρ ∘ ρ) * δρ), dΩ)

    # State Operator (Solves K(ρ)u = F_fluid)
    # tags="fixed" must match your .msh support tag
    S = StateOperator(Val(:SelfAdjointAffine), Val(:Vector), Ω_s, lₛ, aₚ, χ, R, dΩ_s, dΓ_fs; 
                      tags="fixed", solver=solid_solver)

    # 6. Functionals
    J1(u, ρ, dΩ, dΓ) = aₚ(u, u, ρ, dΩ, dΓ)
    J0 = NormalizationFactor(J1, χ, S, R, dΩ_s, dΓ_fs)
    J(u, ρ, dΩ, dΓ) = J0 * J1(u, ρ, dΩ, dΓ)
    dJ(δρ, u, ρ, dΩ, dΓ) = J0 * daₚ(u, u, ρ, δρ, dΩ, dΓ)
    
    Jc = PDEsConstrainedFunctional(J, S, R; differential=dJ)
    V0 = sum(∫(1)dΩ_s)
    V(ϱ,ρ,dΩ) = (1.0/V0)*∫((T∘ϱ)-volume_fraction)dΩ
    dV(δρ,ϱ,ρ,dΩ) = (1.0/V0)*∫(∂T∂ρ∘(ϱ)*δρ)dΩ
    #V0 = sum(∫(1)dΩ_s)
    #function V(ϱ, ρ, dΩ)
    #    val = sum( ∫((T ∘ ϱ) - volume_fraction)dΩ ) 
    #    return val / V0
    #end
    #function dV(δρ, ϱ, ρ, dΩ)
    #    val = sum( ∫( (∂T∂ρ ∘ ϱ) * δρ )dΩ )
    #    return val / V0
    #end
    Vc = PDEConstrainedFunctional(V, R; differential=dV)

    # 7. Optimizer Setup
    mkts_params = (α = 0.02, slopeTolRel = 2e-3, winSize = 12, minIter = 25, relaxFactor = 3.0, safeguardAt = 2000)
    
    h = history(2, verbose, max_iter, 1, joinpath(path,title), output_freq, names)

    if action == :optimize_w_mma
        
        # Optimizer variables
        ρ_prev_ref = Ref(Float64[])
        mkts_converged_flag = Ref(false)
        DENSITY_DIFF_TOL = 0.01

        P = MultiTopOpt.get_trial_space(χ)
        n_dofs = num_free_dofs(P)
        i_am_main(ranks) && println(">>> Starting MMA Optimizer ($n_dofs DOFs)...")

        nlopt_obj = Opt(:LD_MMA, n_dofs)
        nlopt_obj.lower_bounds = 0.0
        nlopt_obj.upper_bounds = 1.0
        nlopt_obj.maxeval = max_iter
        nlopt_obj.ftol_rel = 1e-5

        # --- CUSTOM OBJECTIVE CALLBACK ---
        function objective_callback(ρ::Vector{Float64}, ∂F∂ρ::Vector{Float64})
            
            # 1. Evaluate
            f_val = MultiTopOpt.serialized!(ρ, ∂F∂ρ, P, Jc, h, 1) 
            t_after = h.iter[1] 

            # 2. Stats
            change = 0.0
            if length(ρ_prev_ref[]) == length(ρ)
                 change = norm(ρ - ρ_prev_ref[], Inf)
            end
            vol_frac_curr = mean(ρ)

            # 3. Print (The format you requested)
            if i_am_main(ranks)
                @printf("It.:%3i Obj.:%8.4f Vol.:%6.3f ch.:%6.3f\n", t_after, f_val, vol_frac_curr, change)
            end

            # 4. Convergence Check (MK-TS)
            if t_after >= mkts_params.minIter 
                mkts_passed = check_convergence_mkts(h.vals[1:t_after, 1], t_after, mkts_params)
                density_passed = change < DENSITY_DIFF_TOL

                if mkts_passed && density_passed
                    i_am_main(ranks) && Printf.@printf(">>> CONVERGED (MK-TS + Density). Stopping.\n")
                    NLopt.force_stop!(nlopt_obj)
                end
            end 
            
            ρ_prev_ref[] = copy(ρ) 
            return f_val
        end

        function constraint_callback(ρ::Vector{Float64}, ∂F∂ρ::Vector{Float64})
            return MultiTopOpt.serialized!(ρ, ∂F∂ρ, P, Vc, h, 2)
        end

        nlopt_obj.min_objective = objective_callback
        inequality_constraint!(nlopt_obj, constraint_callback, 1e-6)

        # Run
        opt = MultiTopOpt.NLopt_MMA(h, nlopt_obj)
        optimize!(opt, χ)
    end
    
    io = open(joinpath(path,"history.txt"),"w"); write(io,"$h"); close(io)
    return nothing
end

function NavierImmersedBody3D(;
    solid_solver=:julia,
    filter_solver=:julia, kwargs...)
    solvers = [solid_solver, filter_solver] # Filter solver defaults to julia
    MultiTopOpt.run_with_backend(NavierImmersedBodyMain3D, solvers; kwargs...)
end

# EXECUTION

println("Starting 3D FSI Topology Optimization...")


NavierImmersedBody3D(   
    backend=:sequential, 
    np=(1,1,1),
    mesh_file=mesh_file_path, # 
    action=:optimize_w_mma,
    solid_solver=:julia,
    path="Resultsfine2",        
    title="Wing3Dfine",
    verbose=true         
)
