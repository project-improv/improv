module simGLM
    using Flux, Zygote

    mutable struct GLM{T<:Array{Float64, 2}}
        θ_b::T
        θ_h::T
        θ_w::T
        θ_k::T
    end

    mutable struct Parameters
        M::Int
        N::Int
        dh::Int
        dt::Float64
        ds::Int
        n::Int
    end

    function Parameters(params::AbstractDict)
        M = params["numSamples"]
        N = params["numNeurons"]
        dh = params["hist_dim"]
        dt = params["dt"]
        ds = params["stim_dim"]
        Parameters(M, N, dh, dt, ds, 0)
    end

    function GLM(p::Parameters, w::T, h::T, b::T, k::T) where T<:AbstractArray{Float64, 2}
        @assert size(w) == (p.N, p.N)
        @assert size(h) == (p.N, p.dh)
        @assert size(b) == (p.N, 1)
        @assert size(k) == (p.N, p.ds)

        θ_b, θ_h, θ_w, θ_k = b, reverse(h, dims=2), w, k

        GLM(θ_b, θ_h, θ_w, θ_k)
    end

    function GLM(p::Parameters)
        N = p.N
        θ_h = zeros(Float64, N, p.dh)
        θ_k = zeros(Float64, N, p.ds)
        θ_w = zeros(Float64, N, N)
        θ_b = zeros(Float64, N, 1)

        GLM(θ_b, θ_h, θ_w, θ_k)
    end

    function ll(o::GLM, p::Parameters, data::T, stim::T) where T<:AbstractArray{Float64, 2}
        p.N, p.M = size(data)
        N = p.N

        check_θ_size!(o, p)
        @assert size(stim) == (p.ds, p.M)

        @views begin
            cal_weight = o.θ_w[1:N, 1:N] * data
            cal_weight = hcat(zeros(N, p.dh), cal_weight[:, p.dh:p.M-1])

            cal_hist = convolve(data, o.θ_h[1:N, :], p)
            cal_stim = o.θ_k[1:N, :] * stim
            total = @. o.θ_b[1:N] + cal_stim + cal_weight + cal_hist
        end

        r̂ = p.dt .* exp.(total)
        (sum(r̂) - sum(data .* log.(r̂ .+ eps(Float64)))) / length(data)
    end

    function convolve(data, th, p::Parameters)
        """ Sliding window convolution. """
        @assert size(data)[1] == size(th)[1]
        N, M = size(data)
        @views out = th[:, 1] .* data[:, 1:M-p.dh]
        for i in 2:p.dh
          @views out += th[:, i] .* data[:, i:M-(p.dh-i+1)]
        end
        hcat(zeros(N, p.dh), out)

    end

    function check_θ_size!(o::GLM, p::Parameters)
        """ Resize weights array to accommodate new neurons. """

        Δ = p.N - size(o.θ_w)[1]
        if Δ > 0
            println("Add ", Δ)
            old = size(o.θ_w)[1]

            o.θ_w = hcat(o.θ_w, zeros(old, Δ))
            o.θ_w = vcat(o.θ_w, zeros(Δ, old+Δ))

            o.θ_h = vcat(o.θ_h, zeros(Δ, p.dh))
            o.θ_b = vcat(o.θ_b, zeros(Δ))
            o.θ_k = vcat(o.θ_k, zeros(Δ, p.ds))
        end
    end

    function ll_wrapper(o::GLM, p::Parameters)
        """ To enable calls from Python without referencing GLM and Parameters. """

        function wrap(data, stim)
            ll(o, p, data, stim)
        end
    end

    function ll_grad_wrapper(o::GLM, p::Parameters)
        function wrap(data, stim)
            # Need to move array resizing here to prevent interference with Zygote.
            p.N, p.M = size(data)
            check_θ_size!(o, p)

            grads = gradient(() -> ll(o, p, data, stim), Params([o.θ_b, o.θ_h, o.θ_w, o.θ_k]))

            # fun = x -> ll(x, p, data, stim)
            # gradient(fun, o)

            if p.n > 0  # First call is for JIT, do not modify gradient.
                opt = Descent(1e-5 * 1/sqrt(p.n))
                for p in (o.θ_b, o.θ_h, o.θ_w, o.θ_k)
                    Flux.Optimise.update!(opt, p, grads[p])
                end
            end

            p.n += 1
        end
    end
end


N = 50
M = 100
dh = 3
ds = 8

p = Dict("numSamples" => M,
         "numNeurons" => N,
         "hist_dim" => dh,
         "dt" => 0.1,
         "stim_dim" => ds)

weights = zeros(N, N)
#weights[:,1] = collect(0.:9.)
hist = ones(N, dh)
#hist[1,:] = collect(0.:2.)

k = zeros(N, ds)
base = zeros(N, 1)

para = simGLM.Parameters(p)
g = simGLM.GLM(para, weights, hist, base, k)

data = ones(N, M);
#data[1,:] = collect(0.:9.)
s  = zeros(ds, M);
# println(simGLM.ll(g, para, data, s))
ll_grad = simGLM.ll_grad_wrapper(g, para)


# function test(g, para, data, s)
#     fun = x -> simGLM.ll(x, para, data, s)
#     gradient(fun, g)
# end
#
# gw = simGLM.ll_grad_wrapper(g, para)
# t = gw(data,s)

∇