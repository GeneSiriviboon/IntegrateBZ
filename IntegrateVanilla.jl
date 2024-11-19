using MultiQuad

function calculate_dos_quad(ω, ε_k, η = 1e-5, atol = 1e-6)
    integral, error = dblquad((kx, ky) -> η/(η^2 + (ω -ε_k(kx, ky))^2), 0., 1., 0., 1., atol=atol)
    return integral * 1/π
end

function green_test1()
    function ε_k(kx, ky)
        return  cos.(2* π *  kx) + cos.(2 * π * ky)
    end

    t0  = time()
    Ns = []

    ωs = range(-3., 3.,30)

    for ω in ωs
        dos = calculate_dos_quad(ω, ε_k)
        push!(Ns, real)
        println("time: ", time() - t0, " dos: ", dos)
    end

    println("time: ", time() - t0)
    Plots.plot(ωs, Ns, seriestype=:scatter)
    Plots.ylabel!("G(ω)")
end

green_test1()