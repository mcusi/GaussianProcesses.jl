#Constant mean function

"""
    MeanVec <: Mean

Mean function for a 2D GP, 
with constant mean along one dimension
```math
m(x) = [β_1, ..., β_n]
```
with constants ``β_1``, ..., ``β_n``.
"""
mutable struct MeanVec <: Mean
    "Constant vector"
    β::Array{Float64,1}
    "x points corresponding to `β`"
    x0::Array{Float64,1}
    "Priors for mean parameters"
    priors::Array
    """
        MeanVec(β::Array{Float64,1})

    Create `MeanVec` with constant vector `β`.
    """
    MeanVec(β::Array{Float64,1}, x0::Array{Float64,1}) = new(β, x0, [])
end

mean(mVec::MeanVec, x::AbstractVector) = mVec.β[ [findfirst(isequal(_x[2]),mVec.x0) for _x in x] ]
function mean(mVec::MeanVec, X::AbstractMatrix)
    println("meanfunc")
    println(size(X,2))
    mean_vector = zeros(size(X,2))
    for i in 1:size(X,2)
        println(mVec.β[findfirst(isequal(X[2,i]),mVec.x0)])
        mean_vector[i] = mVec.β[findfirst(isequal(X[2,i]),mVec.x0)]
    end
    println("mean_vector: ", mean_vector)
    return mean_vector
end

get_params(mVec::MeanVec) = mVec.β
get_param_names(::MeanVec) = [:β]
num_params(mVec::MeanVec) = length(mVec.β)
function set_params!(mVec::MeanVec, hyp::AbstractVector)
    println("set_params")
    println(hyp)
    println(mVec.β)
    length(hyp) == length(mVec.β) || throw(ArgumentError("ConstVec mean function only has $(mVec.dim) parameters"))
    copyto!(mVec.β, hyp)
    copyto!(mVec.x0, mVec.x0)
end
function grad_mean(mVec::MeanVec, x::AbstractVector)
    println("Grad_mean x: ", x)
    println("Grad_mean mVec: ", mVec)
    dM_theta = ones(length(x))
    return dM_theta
end
