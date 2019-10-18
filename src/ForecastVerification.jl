module ForecastVerification
using Statistics
using FFTW
using PyPlot

include("power_spectrum_2d.jl")

"""
    rms(x,y)

Root mean square difference between `x` and `y`
"""
rms(x,y) = sqrt(mean((x - y).^2))
export rms

"""
    crms(x,y)

Centred root mean square difference between `x` and `y`
"""
crms(x,y) = rms(x .- mean(x),y .- mean(y))
export crms

bias(x,y) = mean(x) - mean(y)
export bias

rms(x,y,sel) = rms(x[sel],y[sel])
crms(x,y,sel) = crms(x[sel],y[sel])
bias(x,y,sel) = bias(x[sel],y[sel])

function summary(x,obs)
    s = (rms = rms(x,obs),
         crms = crms(x,obs),
         bias = bias(x,obs),
         cor = cor(x,obs),
         std_x_obs = std(x)/std(obs),
         count = length(x))

    println("RMS         ",s.rms)
    println("CRMS        ",s.crms)
    println("bias        ",s.bias)
    println("std_x_obs   ",s.std_x_obs)
    println("correlation ",s.cor)
    return s
end


function taylorstat(x::Vector,obs::Vector)
    σ_x = std(x,corrected=false)
    σ_obs = std(obs,corrected=false)

    c = cor(x,obs)
    CRMS = crms(x,obs)

    return CRMS,c,σ_x,σ_obs
end

function taylorstat(x::Matrix,obs::Vector)
    n = size(x,2)
    CRMS = zeros(eltype(x),n)
    c = zeros(eltype(x),n)
    σ_x = zeros(eltype(x),n)
    σ_obs = zero(eltype(x))

    for i = 1:n
        CRMS[i],c[i],σ_x[i],σ_obs = taylorstat(x[:,i],obs)
    end
    return CRMS,c,σ_x,σ_obs
end
export taylorstat

function taylorplot(x::Matrix,obs::Vector; labels = fill(nothing,size(x,2)))
    n = size(x,2)
    CRMS,c,σ_x,σ_obs = taylorstat(x,obs)
    taylorplot(CRMS,c,σ_x,σ_obs; labels = labels)
end

function taylorplot(CRMS,c,σ_x,σ_obs;
                    labels = fill(nothing,length(CRMS)),
                    obslabel = "observation")
    n = length(CRMS)
    ϕ = acos.(c)

    plot(σ_obs,0,"x",label = obslabel)
    for i = 1:n
        plot(cos(ϕ[i]) * σ_x[i],sin(ϕ[i]) * σ_x[i],"o",label = labels[i])
    end

    lim = maximum([xlim()...,ylim()...])

    xlim(0,lim)
    ylim(0,lim)
    loc,labels = xticks()

    ax = gca()
    for radius = loc
        ax.add_patch(plt.Circle((0,0),radius = radius,fill = false, linewidth = 1, edgecolor = "gray"))
    end

    for radius = loc
        ax.add_patch(plt.Circle((σ_obs,0),radius = radius,fill = false, linewidth = 1, edgecolor = "gray", linestyle = ":"))
    end
    #axis("scaled")
    gca().set_aspect(1)

    legend()
end
export taylorplot

"""
    BS = brierscore(y_true,y_prob::AbstractVector)

Compute the Brier Score for a series of binary events y_true,
forecasted with a probability `y_prob`

``BS = \\frac{1}{N}\\sum_{t=1}^{N}(f_t - o_t)^2 ``

"""
function brierscore(y_true,y_prob::AbstractVector)
    bs = zero(eltype(y_prob))

    for i = 1:length(y_prob)
        bs += (y_true[i] - y_prob[i])^2
    end
    return bs/length(y_prob)
end


"""
    BS,REL,RES,UNC = brierscoredecom(y_true,y_prob::AbstractVector)

The Brier score can be decomposed into 3 additive components: Uncertainty, Reliability, and Resolution. (Murphy 1973)>


``BS = REL - RES + UNC``


``REL=\\frac{1}{N} \\sum_{k=1}^{K}{n_{k}(\\mathbf{f_{k}} - \\mathbf{\\bar{o}}_{\\mathbf{k}})}^{2}``

``RES = \\frac{1}{N}\\sum\\limits _{k=1}^{K}{n_{k}(\\mathbf{\\bar{o}_{k}}-\\bar{\\mathbf{o}})}^{2}``

``UNC = \\mathbf{\\bar{o}}\\left({1-\\mathbf{\\bar{o}}}\\right)``

With ``N`` being the total number of forecasts issued, ``K`` the number of unique forecasts issued, ``\\mathbf{\\bar{o}}={\\sum_{t=1}^{N}}\\mathbf{{o_t}}/N`` the observed climatological base rate for the event to occur, `` n_{k}`` the number of forecasts with the same probability category and ``\\mathbf{\\overline{o}}_{\\mathbf{k}}`` the observed frequency, given forecasts of probability ``\\mathbf{f_{k}}``.

"""
function brierscoredecom(y_true,y_prob::AbstractVector)
    N = length(y_prob)

    yy = unique(y_prob);

    n = zeros(Int,length(yy));

    for k = 1:length(yy);
        n[k] = sum(yy[k] .== y_prob);
    end

    ob = zeros(length(yy));
    for k = 1:length(yy);
        ob[k] = mean(y_true[yy[k] .== y_prob]);
    end

    obb = mean(y_true)

    REL = 1/N * sum(n .* (yy-ob).^2)
    RES = 1/N * sum(n .* (ob .- obb).^2);
    UNC = obb*(1-obb)

    BS = REL - RES + UNC

    return BS,REL,RES,UNC
end



"""
    crps = CRPS(X::AbstractMatrix,xobs::AbstractVector)

The Continuous Rank Probability Score (CRPS) from an ensemble X (cases,members)
using the observation xobs (cases). The CRPS is averaged over all cases.
"""
function CRPS(X::AbstractMatrix,xobs::AbstractVector)
    crps =
        mean([mean(abs.(X[icase,:] .- xobs[icase])) - 0.5 * mean(abs.(X[icase,:] .- X[icase,:]')) for icase = 1:length(xobs)])

    return crps
end

end
