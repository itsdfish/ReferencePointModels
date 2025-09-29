"""
    predict(
        model::AbstractReferencePointModel,
        outcomes1::Vector{<:Real},
        outcomes2::Vector{<:Real},
        won_first::Bool
    )

Returns the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. 

# Arguments

- `model::AbstractReferencePointModel`: a subtype of `AbstractReferencePointModel``
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble
- `won_first`: set true if first gamble was won, in which case evaluation of final decision
    is conditioned on winning first gamble 

# Returns 

Returns a vector respresenting the joint distribution over planned and final choices, where elements correspond to 

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble

# Example 

```julia 
using ReferencePointModels
using ReferencePointModels: predict
model = ReferencePointModel(; α = .9, λ = 2, w₁ = .5, θ = 1, p_rep = .3)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
won_first = true
preds = predict(model, outcomes1, outcomes2, won_first)
```
"""
function predict(
    model::AbstractReferencePointModel,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool;
)
    outcome_stage_1 = won_first ? outcomes1[1] : outcomes1[2]
    p_plan = predict(model, outcomes2, 0)
    p_final = predict(model, outcomes2, outcome_stage_1)
    return predict_joint_probs(model, p_plan, p_final)
end

adjust(x) = x == 1 ? 1 - eps() : x == 0 ? eps() : x

"""
    predict(
        model::AbstractReferencePointModel,
        outcomes2::Vector{<:Real},
        outcome1::Real
    )

Returns the probability of choosing the risky option. Set outcome1 = 0 for planned decision condition.

# Arguments

- `model::AbstractReferencePointModel`: a subtype of `AbstractReferencePointModel``
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble
- `outcomes1::Real`: outcome from the first gamble. For the planned condition, the value is set to zero.

# Example 

```julia 
using ReferencePointModels
using ReferencePointModels: predict

model = ReferencePointModel(; α = .9, λ = 2, w₁ = .5, θ = 1, p_rep = .3)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
won_first = true
preds = predict(model, outcomes1, outcomes2, won_first)
```
"""
function predict(
    model::AbstractReferencePointModel,
    outcomes2::Vector{<:Real},
    outcome1::Real
)
    (; θ) = model
    diff = compute_utility_diff(model, outcomes2, outcome1)
    prob = pdf(BernoulliLogit(θ * diff), 1)
    return adjust(prob)
end

function compute_utility_diff(
    model::AbstractReferencePointModel,
    outcomes2::Vector{<:Real},
    outcome1::Real
)
    return get_expected_utility(model, outcomes2 .+ outcome1) - get_utility(model, outcome1)
end

""" 
    predict_joint_probs(
        model::AbstractReferencePointModel,
        p_plan::Real,
        p_final::Real
    )

Returns the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. The joint probability distribution includes 
the probability of repeating of remembering and repeating the choice, denoted  by parameter `p_rep`, which 
allows dependencies in the joint probability distribution 

# Arguments

- `model::AbstractReferencePointModel`: a subtype of `AbstractReferencePointModel``
- `p_plan::Real`: probability of planning to accept gamble 
- `p_final::Real`: probability of accepting gamble in final decision 

# Returns 

Returns a vector respresenting the joint distribution over planned and final choices, where elements correspond to 

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble
"""
function predict_joint_probs(
    model::AbstractReferencePointModel,
    p_plan::Real,
    p_final::Real
)
    (; p_rep) = model
    p_rep = adjust(p_rep)
    # probability of planning to accept second gamble and accepting second gamble
    p_aa = p_plan * (p_rep + (1 - p_rep) * p_final)
    # probability of planning to accept second gamble and rejecting second gamble
    p_ar = p_plan * (1 - p_rep) * (1 - p_final)
    # probability of planning to reject second gamble and accepting second gamble
    p_ra = (1 - p_plan) * (1 - p_rep) * p_final
    # probability of planning to reject second gamble and rejecting second gamble
    p_rr = (1 - p_plan) * (p_rep + (1 - p_rep) * (1 - p_final))
    return [p_aa, p_ar, p_ra, p_rr]
end

"""
    rand(
        model::AbstractReferencePointModel, 
        outcomes1::Vector{<:Real}, 
        outcomes2::Vector{<:Real}, 
        n::Int
    )

Returns samples from the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. 

# Arguments

- `model::AbstractReferencePointModel`: a subtype of `AbstractReferencePointModel``
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble 
- `won_first::Bool`: true if first gamble won
- `n`: the number of trials per condition 

# Returns 

Returns a vector respresenting the samples from the joint distribution over planned and final choices, where elements correspond to 

1. frequency of planning to accept second gamble and accepting second gamble
2. frequency of planning to accept second gamble and rejecting second gamble
3. frequency of planning to reject second gamble and accepting second gamble
4. frequency of planning to reject second gamble and rejecting second gamble

# Example 

```julia 
using ReferencePointModels
model = ReferencePointModel(; α = .9, λ = 2, w₁ = .5, θ = 1, p_rep = .3)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
won_first = true
n_trials = 10
data = rand(model, outcomes1, outcomes2, won_first, n_trials)
```
"""
function rand(
    model::AbstractReferencePointModel,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool,
    n::Int;
)
    Θ = predict(model, outcomes1, outcomes2, won_first)
    return rand(Multinomial(n, Θ))
end

function rand(
    model::AbstractReferencePointModel,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool
)
    return rand(model, outcomes1, outcomes2, won_first, 1)
end

"""
    logpdf(
        model::AbstractReferencePointModel, 
        outcomes1::Vector{<:Real}, 
        outcomes2::Vector{<:Real},
        data::Vector{<:Int}, 
    )

Returns the multinomial log loglikelihood for the joint planned decision and final decision of the 
second gamble conditioned on outcome of first gamble. The joint distribution is as follows:

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble

# Arguments

- `model::AbstractReferencePointModel`: a subtype of `AbstractReferencePointModel`
- `outcomes1::Vector{<:Real}`: the win and loss outcomes for stage 1 
- `outcomes2::Vector{<:Real}`: the win and loss outcomes for stage 2
- `won_first::Bool`: indicates true if the larger of the two outcomes is won
- ` data::Vector{<:Int}`: a vector of response frequencies joint planned decision and final decision of the 
    second gamble conditioned on outcome of first gamble. The frequencies correspond to the joint distribution above.

# Example 

```julia 
model = ReferencePointModel(; α = .9, λ = 2, w₁ = .5, θ = 1, p_rep = .3)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
won_first = true
n_trials = 10
data = rand(model, outcomes1, outcomes2, won_first, n_trials)
logpdf(model, outcomes1, outcomes2, won_first, data)
```
"""
function logpdf(
    model::AbstractReferencePointModel,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool,
    data::Vector{<:Int};
)
    Θ = predict(model, outcomes1, outcomes2, won_first)
    n = sum(data)
    return logpdf(Multinomial(n, Θ), data)
end

loglikelihood(d::AbstractReferencePointModel, data::Tuple) = sum(logpdf.(d, data...))

logpdf(model::AbstractReferencePointModel, x::Tuple) = logpdf(model, x...)

function get_expected_utility(model::AbstractReferencePointModel, vals::Vector{<:Real})
    (; λ, α, w₁) = model
    utils = get_utility.(vals, α, λ)
    w = [w₁, 1 - w₁]
    return utils' * w
end

get_utility(v, α, λ) = v < 0 ? -λ * abs(v)^α : v^α

function get_utility(model::AbstractReferencePointModel, v)
    (; α, λ) = model
    return get_utility(v, α, λ)
end
