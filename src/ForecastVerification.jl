module ForecastVerification
using Statistics


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
