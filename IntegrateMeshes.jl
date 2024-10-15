
using Plots;
using Meshes;
using StatsBase;
# using MeshViz;
import CairoMakie as Mke;
# using PlotlyJS

"""
Data Structure 

Denote grid with tree

Mesh: List of mesh and unit mesh
UnitMesh: triangular mesh
"""
abstract type Mesh end
abstract type Vertex end
abstract type Surface end
# abstract type UnitMesh2D          <: Mesh2D end

struct Triangle 
    v1::Int
    v2::Int 
    v3::Int
end

struct Vertex2D <: Vertex
    x::Float64
    y::Float64
end

Base.:+(v1::Vertex2D, v2::Vertex2D) = Vertex2D(v1.x + v2.x, v1.y + v2.y)
Base.:-(v1::Vertex2D, v2::Vertex2D) = Vertex2D(v1.x - v2.x, v1.y - v2.y)
Base.:/(v::Vertex2D, c::Float64) = Vertex2D(v.x /c, v.y /c)

struct Edge
    p1::Vertex
    p2::Vertex
end

struct Mesh2D <: Mesh
    _vertices::Vector{Vertex2D}
    _triangles::Vector{Triangle}
    
    function Mesh2D()
        new([], [])
    end 
end

struct Surface2D<:Surface
    _mesh::Mesh2D 
    _values::Vector{Float64}
    _F_fn::Any

    function Surface2D(mesh::Mesh2D, f::Any)
        values = []
        for pos in mesh._vertices
            Base.push!(values, f(pos.x, pos.y))
        end
        new(mesh, values, f)
    end
end

function get_area(mesh::Mesh2D, i::Int)::Float64
    v1 = mesh._triangles[i].v1 
    v2 = mesh._triangles[i].v2 
    v3 = mesh._triangles[i].v3
    
    u1 = mesh._vertices[v2] - mesh._vertices[v1]
    u2 = mesh._vertices[v3] - mesh._vertices[v1]

    area = (u1.x * u2.y - u1.y * u2.x)/2
    return area
end

struct Surface2DPQ <: Surface
    _mesh::Mesh2D 
    _P_values::Vector{Float64} #vertices
    _Q_values::Vector{Float64} #vertices
    _areas::Vector{Float64}    #triangle
    _P_fn::Any
    _Q_fn::Any

    function Surface2DPQ(mesh::Mesh2D, p::Any, q::Any)
        P_values = []
        Q_values = []
        for pos in mesh._vertices
            Base.push!(P_values, p(pos.x, pos.y))
            Base.push!(Q_values, q(pos.x, pos.y))
        end
        areas = []
        for i = 1:length(mesh._triangles)
            Base.push!(areas, get_area(mesh, i))
        end
        new(mesh, P_values, Q_values, areas, p_fn, q_fn)
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
    points = [(p.x, p.y) for p in mesh._vertices]
    connec = [connect((t.v1 , t.v2, t.v3)) for t in mesh._triangles]
    simple_mesh   = SimpleMesh(points, connec)
    Meshes.viz(simple_mesh, showsegments = true)
end

function plot(surf::Surface2D)
    points = [(p.x, p.y) for p in surf._mesh._vertices]
    connec = [connect((t.v1 , t.v2, t.v3)) for t in surf._mesh._triangles]
    colors = [surf._values[i] for i = 1:length(surf._mesh._vertices)]
    simple_mesh   = SimpleMesh(points, connec)
    # println(simple_mesh)
    vmax = StatsBase.percentile(abs.(colors), 95)
    Meshes.viz(simple_mesh, 
                color = clamp.(colors, -vmax, vmax), 
                showsegments = true, 
                segmentcolor = RGBA(0,0,0,0.1),
                colormap = :balance)
end

function plotQ(surf::Surface2DPQ)
    points = [(p.x, p.y) for p in surf._mesh._vertices]
    connec = [connect((t.v1 , t.v2, t.v3)) for t in surf._mesh._triangles]
    colors = [surf._Q_values[i] for i = 1:length(surf._mesh._vertices)]
    simple_mesh   = SimpleMesh(points, connec)
    # println(simple_mesh)
    vmax = StatsBase.percentile(abs.(colors), 95)
    Meshes.viz(simple_mesh, 
                color = clamp.(colors, -vmax, vmax), 
                showsegments = true, 
                segmentcolor = RGBA(0,0,0,0.1),
                colormap = :balance)
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

function subsample!(surf::Surface2D, i::Int)
    L = length(surf._mesh._vertices)

    subsample!(surf._mesh, i)
    
    u1, u2, u3 = surf._mesh._vertices[L+1], surf._mesh._vertices[L+2], surf._mesh._vertices[L + 3]
    
    Base.push!(surf._values, surf._F_fn(u1.x, u1.y))
    Base.push!(surf._values, surf._F_fn(u2.x, u2.y))
    Base.push!(surf._values, surf._F_fn(u3.x, u3.y))
end

function subsample!(surf::Surface2DPQ, i::Int)
    L = length(surf._mesh._vertices)

    subsample!(surf._mesh, i)
    area =  surf._areas[i]/4
    surf._areas[i] = area
    Base.push!(surf._areas,  area)
    Base.push!(surf._areas,  area)
    Base.push!(surf._areas,  area)
    
    u1, u2, u3 = surf._mesh._vertices[L+1], surf._mesh._vertices[L+2], surf._mesh._vertices[L + 3]
    
    Base.push!(surf._P_values, surf._P_fn(u1.x, u1.y))
    Base.push!(surf._P_values, surf._P_fn(u2.x, u2.y))
    Base.push!(surf._P_values, surf._P_fn(u3.x, u3.y))

    Base.push!(surf._Q_values, surf._Q_fn(u1.x, u1.y))
    Base.push!(surf._Q_values, surf._Q_fn(u2.x, u2.y))
    Base.push!(surf._Q_values, surf._Q_fn(u3.x, u3.y))

end

function subsample_check(surf::Surface2D, abs_err::Float64, i::Int)::Bool
    m = surf._mesh._triangles[i]
    # println(m.v1, " ",m.v2," ", m.v3)
    v1, v2, v3 = mesh._vertices[m.v1], mesh._vertices[m.v2], mesh._vertices[m.v3]

    integrand1, integrand2, integrand3 = surf._values[m.v1], surf._values[m.v2], surf._values[m.v3]

    x_center = (v1.x + v2.x + v3.x)/3
    y_center = (v1.x + v2.x + v3.x)/3

    integrand4 = surf._F_fn(x_center, y_center)      
    
    l1_sq = (v2.x -v1.x)^2 + (v2.y -v1.y)^2
    l2_sq = (v3.x -v1.x)^2 + (v3.y -v1.y)^2

    area = (l2_sq * l1_sq)^0.5

    err = area * abs(integrand4 - (integrand1 + integrand2 + integrand3)/3)

    success = err < abs_err

    return success
end

function subsample!(surf::Surface2D, abs_err::Float64, depth::Int)
    idxs::Vector{Int} = 1:length(surf._mesh._triangles)
    for n = 1:depth
        idx_new::Vector{Int} = []
        if length(idxs) > 0
            println(n)
            for i in idxs
                success = subsample_check(surf, abs_err, i)
                if !success
                    subsample!(surf, i)
                    L = length(surf._mesh._triangles)
                    push!(idx_new, i)
                    push!(idx_new, L-1)
                    push!(idx_new, L-2)
                    push!(idx_new, L-3)
                end
            end
        end
        idxs = idx_new
    end
end


function interp(u::Vertex2D, v::Vertex2D, p::Float64)
    return Vertex2D(p * u.x + (1-p) * v.x, p * u.y + (1-p) * v.y)
end


function level_set_helper(surf::Surface2D, c::Float64, i::Int)::Union{Edge, Nothing}
    t = surf._mesh._triangles[i]
    f1, f2, f3 = surf._values[t.v1], surf._values[t.v2], surf._values[t.v3]
    p1, p2, p3 = surf._mesh._vertices[t.v1], surf._mesh._vertices[t.v2], surf._mesh._vertices[t.v3]
    val1 = f1 - c; val2 = f2 - c; val3 = f3 - c
    pts = []

    if val1 == 0
        push!(pts, p1)
    end

    if val2 == 0
        push!(pts, p2)
    end

    if val3 == 0
        push!(pts, p3)
    end

    
    if val1 * val2 < 0
        p = val2/(val2 - val1)
        push!(pts, interp(p1, p2, p))
    end

    if val2 * val3 < 0
        p = val3/(val3 - val2)
        push!(pts, interp(p2, p3, p))
    end

    if val3 * val1 < 0
        p = val3/(val3 - val1)
        push!(pts, interp(p1, p3, p))
    end

    
    if length(pts) == 2
        return Edge(pts[1], pts[2])
    else
        return nothing 
    end
end

norm(v::Vertex2D) = sqrt(v.x^2 + v.y^2)

function level_set_helper(surf::Surface2DPQ, 
                            c::Float64, 
                            i::Int)::Union{Nothing, Tuple{Edge, Vector{Float64}}}
    t = surf._mesh._triangles[i]
    f1, f2, f3 = surf._Q_values[t.v1], surf._Q_values[t.v2], surf._Q_values[t.v3]
    p1, p2, p3 = surf._mesh._vertices[t.v1], surf._mesh._vertices[t.v2], surf._mesh._vertices[t.v3]
    val1 = f1 - c; val2 = f2 - c; val3 = f3 - c
    
    pts = []
    df = []
    dl = []

    dl12 = p2 - p1
    df12 = f2 - f1 
    
    dl23 = p3 - p2
    df23 = f3 - f2 
    
    dl31 = p1 - p3
    df31 = f1 - f3 


    if val1 == 0
        push!(pts, p1)
        push!(df, df12)
        push!(dl, dl12)
    end

    if val2 == 0
        push!(pts, p2)
        push!(df, df23)
        push!(dl, dl23)
    end

    if val3 == 0
        push!(pts, p3)
        push!(df, df31)
        push!(dl, dl31)
    end
    
    if val1 * val2 < 0
        p = val2/(val2 - val1)
        push!(pts, interp(p1, p2, p))
        push!(df, df12)
        push!(dl, dl12)
    end

    if val2 * val3 < 0
        p = val3/(val3 - val2)
        push!(pts, interp(p2, p3, p))
        push!(df, df23)
        push!(dl, dl23)
    end

    if val3 * val1 < 0
        p = val3/(val3 - val1)
        push!(pts, interp(p1, p3, p))
        push!(df, df31)
        push!(dl, dl31)
    end

    
    if length(pts) == 2
        edge =  Edge(pts[1], pts[2])
        du::Vertex2D = pts[2] - pts[1]
        du = du/norm(du)
        grad1 = abs(df[1]/(dl[1].x * du.y - dl[1].y * du.x))
        grad2 = abs(df[2]/(dl[2].x * du.y - dl[2].y * du.x))
        grad = [grad1, grad2]
        return edge, grad
    else
        return nothing 
    end


end


function level_set_subsample!(surf::Surface, c::Float64, depth::Int)::Vector{Int}
    idxs = 1:length(surf._mesh._triangles) 
    for n =1:depth - 1
        idx_new = []
        for i in idxs
            edge = level_set_helper(surf, c, i)
            if !isnothing(edge)
                subsample!(surf, i)
                L = length(surf._mesh._triangles)
                push!(idx_new, i)
                push!(idx_new, L-3)
                push!(idx_new, L-2)
                push!(idx_new, L-1)
            end
        end
        idxs = idx_new
    end
    return idxs
end

function level_set(surf::Surface2D, c::Float64, idxs::Vector{Int})::Vector{Edge}
    level_set_edges::Vector{Edge} = []
        
    for i in idxs
        edge::Union{Edge, Nothing} = level_set_helper(surf, c, i)
        if !isnothing(edge)
            push!(level_set_edges, edge)
        end
    end

    return level_set_edges
end

function level_set(surf::Surface2DPQ, c::Float64, idxs::Vector{Int})::Tuple{Vector{Edge}, Vector{Vector{Float64}}}
    level_set_edges = []
    gradients = []
        
    for i in idxs
        e::Union{Nothing, Tuple{Edge, Vector{Float64}}} = level_set_helper(surf, c, i)
        if !isnothing(e)
            edge, grad = e
            push!(level_set_edges, edge)
            push!(gradients, grad)
        end
    end

    return level_set_edges, gradients
end


function level_set!(surf::Surface, c::Float64, depth::Int)
    
    idxs = level_set_subsample!(surf, c, depth)

    level_set_edges = level_set(surf, c, idxs)

    return level_set_edges
end


function plot_level_set!(l_set::Vector{Edge})
    
    for edge in l_set 
        segment = Meshes.Segment((edge.p1.x, edge.p1.y), 
                                (edge.p2.x, edge.p2.y))
        viz!(segment, color = "black")
    end
end

function integrate1D(e::Edge, grad::Vector{Float64}, f::Any)
    ul, ur = e.p1, e.p2
    dl = norm(ur - ul)
    fl, fr = f(ul.x, ul.y), f(ur.x, ur.y)
    gl = grad[1]
    gr = grad[2]
    uc = interp(ul, ur, 0.5)
    gc = (gl + gr)/2
    intl, intr = fl/gl, fr/gr
    intc = f(uc.x, uc.y)/gc
   
    return (intl + 4 * intc + intr)/6 * dl
end

function integrate1D(l_set::Vector{Edge}, grad::Vector{Vector{Float64}}, p::Any)
    acc = 0
    for i = 1:length(l_set)
        acc += integrate1D(l_set[i], grad[i], p)
    end
    return acc
end

mesh = get_square_grid(1., 1., 30)

function p_fn(x, y)
    return 1.0
end

function q_fn(x, y) 
    return 0.1 - cos(2π * x) * cos(2π * y)
end


function f_fn(x, y)
    return q_fn(x, y)
end

t0  = time()
surf = Surface2DPQ(mesh, p_fn, f_fn)
println("time: ", time() - t0, " create mesh")
level_set_edges, gradients = level_set!(surf, 0., 2)
println("time: ", time() - t0, " level set")
res = integrate1D(level_set_edges, gradients, p_fn)
println("integral: ", res)


# p = plotQ(surf)
# plot_level_set!(level_set_edges)
    
# Plots.display(p)
# println("grads: ",gradients)

println("done: ", time() - t0)




