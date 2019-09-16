
function power_spectrum(Δx,Δy,f::Array{T,2}) where T
    Nx,Ny = size(f)
    Lx = Nx*Δx
    Ly = Ny*Δy

    Δk = 2*pi/Lx
    Δl = 2*pi/Ly


    k = (-ceil(Int,(Nx-1)/2) : floor(Int,(Nx-1)/2)) * Δk
    l = (-ceil(Int,(Ny-1)/2) : floor(Int,(Ny-1)/2)) * Δl

    spec = fftshift(abs.(fft(f))) :: Array{T,2}

    return k,l,spec
end


function rad_power_spectrum(Δx,Δy,f::Array{T,2}) where T
    Nbins=80
    Nx,Ny = size(f)

    k,l,spec = power_spectrum(Δx,Δy,f)

    Δkrad = min(maximum(k),maximum(l)) / Nbins
    radspec = zeros(Nbins)
    radspec_count = zeros(Int,Nbins)

    for j in 1:Ny
        for i in 1:Nx
            kmag = @inbounds sqrt(k[i]^2 + l[j]^2)

            ii = floor(Int,kmag / Δkrad) + 1
            if ii <= Nbins
                @inbounds radspec[ii] += spec[i,j]
                @inbounds radspec_count[ii] += 1
            end
        end
    end

    radspec ./= radspec_count

    krad = (1:Nbins)*Δkrad .- 0.5*Δkrad
    return krad,radspec
end



function rad_power_spectrum(Δx,Δy,f::Array{T,3}) where T
    krad,radspec = rad_power_spectrum(Δx,Δy,f[:,:,1])

    for n = 2:size(f,3)
        radspec .+= rad_power_spectrum(Δx,Δy,f[:,:,n])[2]
    end

    radspec ./= size(f,3)

    return krad,radspec
end
