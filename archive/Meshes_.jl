
using Plots
# using PlotlyJS

"""
Data Structure 

Denote grid with tree

Mesh: List of mesh and unit mesh
UnitMesh: triangular mesh
"""
abstract type Mesh end
# abstract type UnitMesh2D          <: Mesh2D end

struct Triangle 
    v1::Int
    v2::Int 
    v3::Int
end

struct Vertex2D
    x::Float64
    y::Float64
end

struct Edge2D
    p1::Vertex2D
    p2::Vertex2D
end


struct Mesh2D <: Mesh
    _vertices::Vector{Vertex2D}
    _triangles::Vector{Triangle}
    
    function Mesh2D()
        new([], [])
    end 
end

struct Surface2D
    _mesh::Mesh2D 
    _values::Vector{Float64}

    function Surface2D(mesh::Mesh2D, f::Any)
        values = []
        for pos in mesh._vertices
            Base.push!(values, f(pos.x, pos.y))
        end
        new(mesh, values)
    end
end

struct SurfacePQ
    _mesh::Mesh2D 
    _values_P::Vector{Float64}
    _values_Q::Vector{Float64}
    _P_fn::Any
    _Q_fn::Any

    function SurfacePQ(mesh::Mesh2D, f::Any, g::Any)
        values_P = []
        values_Q = []
        for pos in mesh._vertices
            Base.push!(values_P, f(pos.x, pos.y))
            Base.push!(values_Q, g(pos.x, pos.y))
        end
        new(mesh, values_P, values_Q, f, g)
    end
end



function get_square_grid(X, Y, n)::Mesh2D
    dx = X/n
    dy = Y/n
    mesh = Mesh2D()

    #(i, j) vertices locate at (i -1) * (n+1) + j 
    for i = 1:n 
        for j = 1:n
            Base.push!(mesh._vertices, Vertex2D(i*dx, j*dy))
            if i < n  && j < n
                v1 = (i - 1) * (n) + j
                v2 = (i) * (n) + j
                v3 = (i - 1) * (n) + (j+1)
                v4 = (i) * (n) + (j+1)
                Base.push!(mesh._triangles,Triangle(v1, v2, v3))
                Base.push!(mesh._triangles,Triangle(v3, v2, v4))
            end
        end
    end
    return mesh
end


function plot(mesh::Mesh2D)
    p = Plots.plot()
    for m in mesh._triangles
        # println(m.v1, " ",m.v2," ", m.v3)
        v1 = mesh._vertices[m.v1]
        v2 = mesh._vertices[m.v2]
        v3 = mesh._vertices[m.v3]
        Plots.plot!([v1.x, v2.x, v3.x, v1.x], 
        [v1.y, v2.y, v3.y, v1.y],
        color = "black", legend = false,
        grid=false)
    end
    return p
end

function plot(surf::Surface2D)
    p = Plots.plot()
    mesh = surf._mesh
    for m in mesh._triangles
        # println(m.v1, " ",m.v2," ", m.v3)
        v1 = mesh._vertices[m.v1]
        v2 = mesh._vertices[m.v2]
        v3 = mesh._vertices[m.v3]
        xs = [v1.x, v2.x, v3.x, v1.x]
        ys = [v1.y, v2.y, v3.y, v1.y]

        f1 = surf._values[m.v1]
        f2 = surf._values[m.v2]
        f3 = surf._values[m.v3]
        z = (f1 + f2 + f3)/3 
        

        Plots.plot!(Shape(xs, ys),
        fill_z = z,
        linecolor = RGBA(255, 255, 255, 0.2),
        c = :redsblues,
        clims = (-0.1, 0.1),
        legend = false,
        grid=false)
    end
    return p
end

function plot(surf::SurfacePQ)
    p = Plots.plot()
    mesh = surf._mesh
    for m in mesh._triangles
        # println(m.v1, " ",m.v2," ", m.v3)
        v1 = mesh._vertices[m.v1]
        v2 = mesh._vertices[m.v2]
        v3 = mesh._vertices[m.v3]
        xs = [v1.x, v2.x, v3.x, v1.x]
        ys = [v1.y, v2.y, v3.y, v1.y]

        q1 = surf._values_Q[m.v1]
        q2 = surf._values_Q[m.v2]
        q3 = surf._values_Q[m.v3]
        z = (q1 + q2 + q3)/3
        
        # is_sing = Int(z < threshold) 

        Plots.plot!(Shape(xs, ys),
        fill_z = z,
        linecolor = RGBA(255, 255, 255, 0.2),
        c = :redsblues,
        clims = (-10, 10),
        legend = false,
        grid=false)
    end
    return p
end

function subsample!(mesh::Mesh2D, i::Int)
    t = mesh._triangles[i]
    v1 = mesh._vertices[t.v1]
    v2 = mesh._vertices[t.v2]
    v3 = mesh._vertices[t.v3]
    
    u1 = Vertex2D((v2.x + v3.x) /2, 
                  (v2.y + v3.y) /2)
    u2 = Vertex2D((v3.x + v1.x) /2, 
                  (v3.y + v1.y) /2)
    u3 = Vertex2D((v1.x + v2.x) /2, 
                  (v1.y + v2.y) /2)

    idx1 = length(mesh._vertices) + 1
    idx2 = length(mesh._vertices) + 2
    idx3 = length(mesh._vertices) + 3

    Base.push!(mesh._vertices, u1)
    Base.push!(mesh._vertices, u2)
    Base.push!(mesh._vertices, u3)

    t0 = Triangle(idx1, idx2, idx3)
    t1 = Triangle(t.v1, idx3, idx2)
    t2 = Triangle(t.v2, idx1, idx3)
    t3 = Triangle(t.v3, idx2, idx1)
    # print(t0)
    
    Base.push!(mesh._triangles, t1)
    Base.push!(mesh._triangles, t2)
    Base.push!(mesh._triangles, t3)
    mesh._triangles[i] = t0
end

function subsample!(surf::SurfacePQ, i::Int)
    L = length(surf._mesh._vertices)

    subsample!(surf._mesh, i)
    
    u1, u2, u3 = surf._mesh._vertices[L+1], surf._mesh._vertices[L+2], surf._mesh._vertices[L + 3]
    
    Base.push!(surf._values_P, surf._P_fn(u1.x, u1.y))
    Base.push!(surf._values_P, surf._P_fn(u2.x, u2.y))
    Base.push!(surf._values_P, surf._P_fn(u3.x, u3.y))

    Base.push!(surf._values_Q, surf._Q_fn(u1.x, u1.y))
    Base.push!(surf._values_Q, surf._Q_fn(u2.x, u2.y))
    Base.push!(surf._values_Q, surf._Q_fn(u3.x, u3.y))
end


function subsample_help!(surf::SurfacePQ, abs_err::Float64, idxs::Vector{Int})
    idx_new = []
    for i in idxs
        m = surf._mesh._triangles[i]
        # println(m.v1, " ",m.v2," ", m.v3)
        v1 = mesh._vertices[m.v1]
        v2 = mesh._vertices[m.v2]
        v3 = mesh._vertices[m.v3]

        p1 = surf._values_P[m.v1]
        p2 = surf._values_P[m.v2]
        p3 = surf._values_P[m.v3]

        q1 = surf._values_Q[m.v1]
        q2 = surf._values_Q[m.v2]
        q3 = surf._values_Q[m.v3]

        integrand1 = p1/q1 
        integrand2 = p2/q2
        integrand3 = p3/q3

        x_center = (v1.x + v2.x + v3.x)/3
        y_center = (v1.x + v2.x + v3.x)/3

        p4 = surf._P_fn(x_center, y_center)
        q4 = surf._Q_fn(x_center, y_center)

        integrand4 = p4/q4        

        err = abs(integrand4 - (integrand1 + integrand2 + integrand3)/3)
        
        if err > abs_err
            subsample!(surf, i)
            push!(idx_new, i)
            L = length(surf._values_P)
            push!(idx_new, L+1)
            push!(idx_new, L+2)
            push!(idx_new, L+3)
        end

    end

    # print("new: ", length(idx_new))

    return idx_new
end

function subsample!(surf::SurfacePQ, abs_err::Float64, depth::Int)
    idxs::Vector{Int} = 1:length(surf._mesh._triangles)
    for n = 1:depth
        if length(idxs) > 0
            println(n)
            # println(length(idxs))
            idxs = subsample_help!(surf, abs_err, idxs)
        end
    end
end

function levelset2D_helper(
    l_set::Vector{Edge2D},
    surf::Surface2D, 
    i::Int,
    z::Float64)
    t = surf._mesh._triangles[i]
    f1 = surf._values[t.v1]
    f2 = surf._values[t.v2]
    f3 = surf._values[t.v3]

    p1 = surf._mesh._vertices[t.v1]
    p2 = surf._mesh._vertices[t.v2]
    p3 = surf._mesh._vertices[t.v3]

    pts = []
    
    if z > min(f1, f2) && z < max(f1, f2)
        p = (z - f2)/(f1 - f2)
        Base.push!(pts, Vertex2D(p * p1.x + (1 - p) * p2.x,
                                 p * p1.y + (1 - p) * p2.y))
    end      
    
    if z > min(f1, f3) && z < max(f1, f3)
        p = (z - f3)/(f1 - f3)
        Base.push!(pts, Vertex2D(p * p1.x + (1 - p) * p3.x,
                                 p * p1.y + (1 - p) * p3.y))
    end    

    if z > min(f2, f3) && z < max(f2, f3)
        p = (z - f3)/(f2 - f3)
        Base.push!(pts, Vertex2D(p * p2.x + (1 - p) * p3.x,
                                 p * p2.y + (1 - p) * p3.y))
    end    

    if length(pts) == 2
        Base.push!(l_set, Edge2D(pts[1], pts[2]))
    elseif length(pts) == 3
        Base.push!(l_set, Edge2D(pts[1], pts[2]))
        Base.push!(l_set, Edge2D(pts[2], pts[3]))
        Base.push!(l_set, Edge2D(pts[3], pts[1]))
    end
end

function levelset2D(
    surf::Surface2D, 
    z::Float64)::Vector{Edge2D}
    l_set::Vector{Edge2D} = []

    for i=1:length(surf._mesh._triangles)
        levelset2D_helper(l_set, surf, i, z)
    end

    return l_set
end

function plot_level_set!(l_set::Vector{Edge2D})
    for edge in l_set 
        Plots.plot!([edge.p1.x, edge.p2.x], 
                    [edge.p1.y, edge.p2.y],
                    color = "white")
    end
end

function integrate_non_sing()
end

function integrate_sing()
end


mesh = get_square_grid(1., 1., 10)

function p_fn(x, y)
    return 1.
end

function q_fn(x, y)
    r = ((x - 0.50)^2 + (y - 0.50)^2)^2
    return r
end

surf = SurfacePQ(mesh, p_fn, q_fn)
subsample!(surf, 1.0, 2)
p = plot(surf)
    
Plots.display(p)




