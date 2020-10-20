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
    mean_vector = zeros(size(X,2))
    for i in 1:size(X,2)
        mean_vector[i] = mVec.β[findfirst(isequal(X[2,i]),mVec.x0)]
    end
    return mean_vector
end

get_params(mVec::MeanVec) = mVec.β
get_param_names(::MeanVec) = [:β]
num_params(mVec::MeanVec) = length(mVec.β)
function set_params!(mVec::MeanVec, hyp::AbstractVector)
    length(hyp) == length(mVec.β) || throw(ArgumentError("ConstVec mean function only has $(mVec.dim) parameters"))
    copyto!(mVec.β, hyp)
end
function grad_mean(mVec::MeanVec, x::AbstractVector)
    dM_theta = ones(length(x))
    return dM_theta
end
