module ForecastVerificationPyPlotExt

import ForecastVerification: taylorplot

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


end
