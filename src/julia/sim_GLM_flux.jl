__precompile__
module simGLM
    using StaticArrays
    using LinearAlgebra: dot
    using Flux
    using Flux.Tracker
    using Flux.Tracker: update!

    mutable struct GLM
        M::Int
        N::Int
        dh::Int
        dt::Float64
        ds::Int
        θ_b::AbstractArray
        θ_h::AbstractArray
        θ_w::AbstractArray
        θ_k::AbstractArray
        n::Int
    end

    function GLM(w::AbstractArray, h::AbstractArray, b::AbstractArray, k::AbstractArray, params::AbstractDict)
        M = params["numSamples"]
        N = params["numNeurons"]
        dh = params["hist_dim"]
        dt = params["dt"]
        ds = params["stim_dim"]

        @assert size(w) == (N, N)
        @assert size(h) == (N, dh)
        @assert size(b) == (N,)
        @assert size(k) == (N, ds)

        θ_b, θ_h, θ_w, θ_k = param(b), param(reverse(h, dims=2)), param(w), param(k)

        GLM(M, N, dh, dt, ds, θ_b, θ_h, θ_w, θ_k, 0)
    end

    function ll(o::GLM, data::AbstractArray, stim::AbstractArray)
        o.N, o.M = size(data)
        check_array!(o)

        @assert o.N <= size(o.θ_w)[1]

        expo = zeros(Tracker.TrackedReal{Float64}, o.N, o.M)

        for j in 1:o.M
            if j <= o.dh  # Only stim
                expo[:,j] = runModelStep(o, zeros(o.N, o.dh), stim[:, j])
            else
                expo[:,j] = runModelStep(o, data[:, j-o.dh+1:j], stim[:, j])
            end
        end

        r̂ = o.dt .* exp.(expo)
        (sum(r̂) - sum(data .* log.(r̂ .+ eps(Float64)))) / (o.M * o.N)  # Log-likelihood
    end

    function ll_wrapper(o::GLM)
        function wrap(d, s)
            ll(o, d, s)
        end
    end

    function runModelStep(o::GLM, data_j, stim_j)
        tⱼ = size(data_j)[2]
        expoⱼ = zeros(eltype(o.θ_h), o.N)

        for i in 1:o.N
            cal_h = sum(o.θ_h[i, :] .* data_j[i,:])
            cal_w = dot(o.θ_w[i, 1:o.N], data_j[:, tⱼ-1])
            cal_stim = dot(o.θ_k[i, :], stim_j)

            expoⱼ[i] = o.θ_b[i] + cal_h + cal_w + cal_stim
        end

        expoⱼ
    end

    function ll_grad(o::GLM, data, stim; opt=Descent(1e-3))#, rate=x -> 1/x)
        o.n += 1
        grads = Tracker.gradient(() -> ll(o, data, stim), Params([o.θ_b, o.θ_h, o.θ_w, o.θ_k]))
        grads[o.θ_b]
        # for p in (o.θ_b, o.θ_h, o.θ_w)
        #     update!(opt, p, grads[p])
        # end
    end

    function ll_grad_wrapper(o::GLM; kwargs...)
        function wrap(d, s)
            ll_grad(o, d, s, kwargs...)
        end
    end

    Float64(x::Tracker.TrackedReal{Float64}) = x.data  # Handle conversion back to Python

    function check_array!(o::GLM)
        if o.N > size(o.θ_w)[1]  # Use as proxy for old N
            println("Add!")
            new_N = 2 * o.N

            # Weights
            temp = zeros(new_N, new_N)
            temp[1:size(o.θ_w)[1], 1:size(o.θ_w)[2]] = o.θ_w
            o.θ_w = param(temp)

            # History
            temp = zeros(new_N, o.dh)
            temp[1:size(o.θ_h)[1], 1:size(o.θ_h)[2]] = o.θ_h
            o.θ_h = param(temp)

            # Bias
            temp = zeros(new_N)
            temp[1:size(o.θ_b)[1]] = o.θ_b
            o.θ_b = param(temp)

            # K
            temp = zeros(new_N, o.ds)
            temp[1:size(o.θ_k)[1], 1:size(o.θ_k)[2]] = o.θ_k
            o.θ_k = param(temp)
        end
    end
end

N = 2
dh = 2

p = Dict("numSamples" => 8,
         "numNeurons" => 2,
         "hist_dim" => 2,
         "dt" => 0.1,
         "stim_dim" => 8)

weights = zeros(N,N)
hist = zeros(N,dh)
base = zeros(N)
k = zeros(N,8)
data = ones(2, 10)

theta = [weights[:]; hist[:]; base[:]; k[:]]

g = simGLM.GLM(weights, hist, base, k, p)
println(simGLM.ll(g, data, ones(8,10)))
println(simGLM.ll_grad(g, data, ones(8,10)))
