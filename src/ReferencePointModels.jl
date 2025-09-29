module ReferencePointModels

using Distributions: BernoulliLogit
using Distributions: Multinomial
using Distributions: DiscreteMultivariateDistribution
using PrettyTables

import Distributions: logpdf
import Distributions: pdf
import Distributions: rand
import Distributions: loglikelihood

export get_expected_utility
export predict
export loglikelihood
export logpdf
export pdf
export ReferencePointModel
export rand

include("structs.jl")
include("functions.jl")
include("utilities.jl")

end
