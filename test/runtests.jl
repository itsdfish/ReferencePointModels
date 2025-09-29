using SafeTestsets

@safetestset "rand" begin
    @safetestset "rand 1" begin
        using ReferencePointModels
        using Test
        using Random

        Random.seed!(7878)

        n = 100_000

        parms = (α = 0.9, λ = 1, p_rep = 0.30, w₁ = 0.5, θ = 0.80)

        model = ReferencePointModel(; parms...)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        won = true

        data = rand(model, outcomes1, outcomes2, won, n)
        preds = predict(model, outcomes1, outcomes2, won)
        probs = data ./ n
        @test probs ≈ preds atol = 1e-2
    end

    @safetestset "rand 2" begin
        using ReferencePointModels
        using Test
        using Random

        Random.seed!(958)

        n = 100_000

        parms = (α = 0.9, λ = 2, w₁ = 0.5, p_rep = 0.30, θ = 1.2)

        model = ReferencePointModel(; parms...)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        won = false

        data = rand(model, outcomes1, outcomes2, won, n)
        preds = predict(model, outcomes1, outcomes2, won)
        probs = data ./ n
        @test probs ≈ preds atol = 1e-2
    end
end

@safetestset "logpdf" begin
    using ReferencePointModels
    using Test
    using Random

    Random.seed!(514)

    parms = (α = 0.9, λ = 2, p_rep = 0.3, w₁ = 0.5, θ = 1)

    outcomes1 = [[2, -1], [5, -3], [0.5, -0.25], [2, -2], [5, -5], [0.5, -0.50]]
    outcomes2 = [[2, -1], [5, -3], [0.5, -0.25], [2, -2], [5, -5], [0.5, -0.50]]
    won = [true, false, true, true, false, false]

    ns = fill(20_000, 6)
    model = ReferencePointModel(; parms...)
    data = rand.(model, outcomes1, outcomes2, won, ns)

    θs = range(0.8 * parms.θ, 1.2 * parms.θ, length = 100)
    LLs =
        map(
            θ -> sum(logpdf.(
                ReferencePointModel(; parms..., θ),
                outcomes1,
                outcomes2,
                won,
                data
            )),
            θs
        )
    _, mxi = findmax(LLs)
    @test θs[mxi] ≈ parms.θ rtol = 1e-2

    αs = range(0.8 * parms.α, 1.2 * parms.α, length = 100)
    LLs =
        map(
            α -> sum(logpdf.(
                ReferencePointModel(; parms..., α),
                outcomes1,
                outcomes2,
                won,
                data
            )),
            αs
        )
    _, mxi = findmax(LLs)
    @test αs[mxi] ≈ parms.α rtol = 1e-2

    λs = range(0.8 * parms.λ, 1.2 * parms.λ, length = 100)
    LLs =
        map(
            λ -> sum(logpdf.(
                ReferencePointModel(; parms..., λ),
                outcomes1,
                outcomes2,
                won,
                data
            )),
            λs
        )
    _, mxi = findmax(LLs)
    @test λs[mxi] ≈ parms.λ rtol = 1e-1

    w₁s = range(0.8 * parms.w₁, 1.2 * parms.w₁, length = 100)
    LLs = map(
        w₁ -> sum(logpdf.(
            ReferencePointModel(; parms..., w₁),
            outcomes1,
            outcomes2,
            won,
            data
        )),
        w₁s
    )
    _, mxi = findmax(LLs)
    @test w₁s[mxi] ≈ parms.w₁ rtol = 1e-2
end

@safetestset "predict" begin
    @safetestset "loss1" begin
        using ReferencePointModels
        using ReferencePointModels: predict
        using Test

        model = ReferencePointModel(; α = 0.70, λ = 2.0, w₁ = 0.50, p_rep = 0.50, θ = 1)
        outcomes1 = [2, -1]
        outcomes2 = 0
        p_plan = predict(model, outcomes1, outcomes2)
        p_plan_true = 0.4532

        @test p_plan ≈ p_plan_true atol = 1e-4

        outcomes2 = -1
        p_final = predict(model, outcomes1, outcomes2)
        p_final_true = 0.7059

        @test p_final ≈ p_final_true atol = 1e-4
    end

    @safetestset "loss2" begin
        using ReferencePointModels
        using ReferencePointModels: predict
        using Test

        model = ReferencePointModel(; α = 0.80, λ = 1.5, w₁ = 0.50, p_rep = 0.30, θ = 1.5)
        outcomes1 = [20, -22]
        outcomes2 = 0
        p_plan = predict(model, outcomes1, outcomes2)
        p_plan_true = 6.0671e-03

        @test p_plan ≈ p_plan_true atol = 1e-4

        outcomes2 = -22
        p_final = predict(model, outcomes1, outcomes2)
        p_final_true = 0.8167

        @test p_final ≈ p_final_true atol = 1e-4
    end

    @safetestset "win1" begin
        using ReferencePointModels
        using ReferencePointModels: predict
        using Test

        model = ReferencePointModel(; α = 0.70, λ = 2.0, w₁ = 0.50, p_rep = 0.50, θ = 1)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        preds = predict(model, outcomes1, outcomes2, true)
        preds_true = [0.3509, 0.1023, 0.1500, 0.3968]

        @test preds ≈ preds_true atol = 1e-4
    end
end

@safetestset "predict joint" begin
    @safetestset "loss1" begin
        using ReferencePointModels
        using ReferencePointModels: predict
        using Test

        model = ReferencePointModel(; α = 0.70, λ = 2.0, w₁ = 0.50, p_rep = 0.50, θ = 1)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        preds = predict(model, outcomes1, outcomes2, false)
        preds_true = [0.386555, 0.066646, 0.192990, 0.353810]

        @test preds ≈ preds_true atol = 1e-4
    end
    @safetestset "win1" begin
        using ReferencePointModels
        using ReferencePointModels: predict
        using Test

        model = ReferencePointModel(; α = 0.70, λ = 2.0, w₁ = 0.50, p_rep = 0.50, θ = 1)
        outcomes1 = [2, -1]
        outcomes2 = 0
        p_plan = predict(model, outcomes1, outcomes2)
        p_plan_true = 0.4532

        @test p_plan ≈ p_plan_true atol = 1e-4

        outcomes2 = 2
        p_final = predict(model, outcomes1, outcomes2)
        p_final_true = 0.5486

        @test p_final ≈ p_final_true atol = 1e-4
    end

    @safetestset "win2" begin
        using ReferencePointModels
        using ReferencePointModels: predict
        using Test

        model = ReferencePointModel(; α = 0.80, λ = 1.5, w₁ = 0.50, p_rep = 0.30, θ = 1.5)
        outcomes1 = [20, -22]
        outcomes2 = 0
        p_plan = predict(model, outcomes1, outcomes2)
        p_plan_true = 6.0671e-03

        @test p_plan ≈ p_plan_true atol = 1e-4

        outcomes2 = 20
        p_final = predict(model, outcomes1, outcomes2)
        p_final_true = 0.016434

        @test p_final ≈ p_final_true atol = 1e-4
    end
end

@safetestset "get_utility" begin
    using ReferencePointModels
    using ReferencePointModels: get_utility
    using Test

    utility = get_utility(4, 0.5, 2)
    @test utility ≈ 2

    utility = get_utility(-4, 0.5, 2)
    @test utility ≈ -4
end

@safetestset "utility examples" begin
    @safetestset "utility example 1" begin
        using ReferencePointModels
        using ReferencePointModels: compute_utility_diff
        using Test

        # example based on Busemeyer et. al (2015) page 5
        model = ReferencePointModel(; α = 0.65, λ = 1.6, w₁ = 0.50, p_rep = 0.30, θ = 1)
        outcomes2 = [2, -1]
        outcome1 = 0
        x = compute_utility_diff(model, outcomes2, outcome1)
        @test x ≈ -0.0154 atol = 0.001
    end

    @safetestset "utility example 2" begin
        using ReferencePointModels
        using ReferencePointModels: compute_utility_diff
        using Test

        # example based on Busemeyer et. al (2015) page 5
        model = ReferencePointModel(; α = 0.65, λ = 1.6, w₁ = 0.50, p_rep = 0.30, θ = 1)
        outcomes2 = [2, -1]
        outcome1 = 2
        x = compute_utility_diff(model, outcomes2, outcome1)
        @test x ≈ 0.162 atol = 0.001
    end

    @safetestset "utility example 3" begin
        using ReferencePointModels
        using ReferencePointModels: compute_utility_diff
        using Test

        # example based on Busemeyer et. al (2015) page 5
        model = ReferencePointModel(; α = 0.65, λ = 1.6, w₁ = 0.50, p_rep = 0.30, θ = 1)
        outcomes2 = [2, -1]
        outcome1 = -1
        x = compute_utility_diff(model, outcomes2, outcome1)
        @test x ≈ 0.8447 atol = 0.001
    end
end

@safetestset "predict choice" begin
    using ReferencePointModels
    using ReferencePointModels: compute_utility_diff
    using ReferencePointModels: predict
    using Test

    θ = 2
    model = ReferencePointModel(; α = 0.65, λ = 1.6, w₁ = 0.50, p_rep = 0.30, θ)
    outcomes2 = [2, -1]
    outcome1 = -1
    x = compute_utility_diff(model, outcomes2, outcome1)
    prob = predict(model, outcomes2, outcome1)
    @test prob ≈ 1 / (1 + exp(-θ * x))
    @test prob > 0.50
end
