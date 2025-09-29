abstract type AbstractReferencePointModel <: DiscreteMultivariateDistribution end

"""
    ReferencePointModel{T<:Real} <: AbstractReferencePointModel

A model object for the Quantum Prisoner's Dilemma Model. The ReferencePointModel has four basis states:
    
1. win first gamble, accept second gamble 
2. win first gamble, decline second gamble 
3. lose first gamble, accept second gamble 
4. lose first gamble, decline second gamble 

The bases are orthonormal and in standard form. The model returns the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. 

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble

# Fields 

- `α::T`: utility curvature parameter where α < 1 is risk averse and α > 1 is risk seeking
- `λ::T`: loss aversion parameter 
- `w₁:T`: decision weight for the first outcome
- `p_rep::T`: the probability remembering and repeating the first response
- `γ::T`: entanglement parameter for beliefs and actions 

# Constructors 

    ReferencePointModel(; α, λ, w₁, p_rep, γ)

    ReferencePointModel(α, λ, w₁, p_rep, γ)

# Example 

```julia
using QuantumDynamicInconsistencyModels
model = ReferencePointModel(; α = .9, λ = 2, w₁ = .5, p_rep, = .6, γ = -1.74)
```

# References 

Busemeyer, J. R., Wang, Z., & Shiffrin, R. M. (2015). Bayesian model comparison favors quantum over standard decision theory account of dynamic inconsistency. Decision, 2(1), 1.
"""
struct ReferencePointModel{T <: Real} <: AbstractReferencePointModel
    α::T
    λ::T
    w₁::T
    p_rep::T
    θ::T
end

ReferencePointModel(; α = 0.65, λ = 1.6, w₁ = 0.50, p_rep = 0.30, θ = 1.0) =
    ReferencePointModel(α, λ, w₁, p_rep, θ)

function ReferencePointModel(α, λ, w₁, p_rep, θ)
    return ReferencePointModel(promote(α, λ, w₁, p_rep, θ)...)
end

Base.broadcastable(dist::AbstractReferencePointModel) = Ref(dist)
