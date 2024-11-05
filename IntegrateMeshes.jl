
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
Base.:+(c::Float64, v2::Vertex2D) = Vertex2D(c + v2.x, c + v2.y)
Base.:-(v1::Vertex2D, v2::Vertex2D) = Vertex2D(v1.x - v2.x, v1.y - v2.y)
Base.:/(v::Vertex2D, c::Float64) = Vertex2D(v.x /c, v.y /c)
Base.:*(c::Float64, v::Vertex2D) = Vertex2D(v.x *c, v.y*c)

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
    _Q_fn::Any

    function Surface2D(mesh::Mesh2D, f::Any)
        values = []
        for pos in mesh._vertices
            Base.push!(values, f(pos))
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

function get_square_grid(X, Y, n, rand_pos = 0.0)::Mesh2D
    dx = X/n
    dy = Y/n
    mesh = Mesh2D()

    #(i, j) vertices locate at (i - 1) * (n+1) + j 
    for i = 1:n+1 
        for j = 1:n+1
            x = (i-1)*dx
            y = (j-1)*dy
            
            if i > 1 && i < n+1
                x = clamp(x * (1 - rand_pos/2 + rand_pos*rand()), 0, X)
            end
            
            if j > 1 && j < n+1
                y = clamp(y * (1 - rand_pos/2 + rand_pos*rand()), 0, Y)
            end

            Base.push!(mesh._vertices, 
                        Vertex2D(x, y))

            if i < n+1  && j < n+1
                v1 = (i - 1) * (n+1) + j
                v2 = (i) * (n+1) + j
                v3 = (i - 1) * (n+1) + (j+1)
                v4 = (i) * (n+1) + (j+1)
                v5 = (n+1)*(n+1) + (i - 1) * n + j

                Base.push!(mesh._triangles,Triangle(v1, v2, v5))  
                Base.push!(mesh._triangles,Triangle(v1, v5, v3))
                Base.push!(mesh._triangles,Triangle(v2, v4, v5))
                Base.push!(mesh._triangles,Triangle(v4, v3, v5))
            end
        end
    end

    for i=1:n
        for j=1:n
            x = (i-0.5)*dx* (1 - rand_pos/2 + rand_pos*rand())
            y = (j-0.5)*dy* (1 - rand_pos/2 + rand_pos*rand())
            Base.push!(mesh._vertices, 
                        Vertex2D(x, y))
        end
    end

    println("n: ", n, " len ", length(mesh._vertices))
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
                segmentcolor = RGBA(0,0,0,0.05),
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
    
    Base.push!(surf._values, surf._Q_fn(u1))
    Base.push!(surf._values, surf._Q_fn(u2))
    Base.push!(surf._values, surf._Q_fn(u3))
end


function subsample_check(surf::Surface2D, abs_err::Float64, i::Int)::Bool
    m = surf._mesh._triangles[i]
    # println(m.v1, " ",m.v2," ", m.v3)
    v1, v2, v3 = mesh._vertices[m.v1], mesh._vertices[m.v2], mesh._vertices[m.v3]

    integrand1, integrand2, integrand3 = surf._values[m.v1], surf._values[m.v2], surf._values[m.v3]

    x_center = (v1.x + v2.x + v3.x)/3
    y_center = (v1.x + v2.x + v3.x)/3

    integrand4 = surf._Q_fn(x_center, y_center)      
    
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


norm(v::Vertex2D) = sqrt(v.x^2 + v.y^2)


function find_zeros(p1::Vertex, p2::Vertex, 
                    f1::Float64, f2::Float64,
                    f::Any, steps = 2)::Tuple{Vertex2D, Float64, Vertex2D}
    fl, fr = f1, f2
    pl, pr = p1, p2
    dl = pr - pl
    
    if f1 > f2
        fl, fr  = f2, f1
        pl, pr = p2, p1
    end
    
    p = fr/(fr - fl)
    pc =  interp(pl, pr, p)
    # println("pc: ", pc)
    fc = f(pc)

    for i=1:steps
        if fc == 0
            return pc, (fr - fl), dl
        elseif fc < 0
            fl = fc
            pl = pc
            dl = p * dl
        else
            fr = fc
            pr = pc
            dl = (1 - p) * dl
        end
        p = fr/(fr - fl)
        pc =  interp(pl, pr, p)
        fc = f(pc)
    end

    return pc, (fr - fl), dl
end

#TODO: has to deals with the edge cases
function level_set_helper(surf::Surface2D, 
    c::Float64, 
    i::Int,
    tol::Float64)::Tuple{Bool, Union{Nothing, Tuple{Edge, Vector{Float64}}}}
    
    t = surf._mesh._triangles[i]

    fs  = [surf._values[t.v1], surf._values[t.v2], surf._values[t.v3]]
    ps  = [surf._mesh._vertices[t.v1], surf._mesh._vertices[t.v2], surf._mesh._vertices[t.v3]]
    vals = [fs[1] - c,  fs[2] - c,  fs[3] - c]

    e = nothing 
    is_subsample = false

    num_zeros = Int(vals[1] == 0) + Int(vals[2] == 0) + Int(vals[3] == 0)

    if  num_zeros == 3
        is_subsample = true
        edge = Edge(ps[1], ps[2])
        grad = 0.0
        e = edge, grad
    elseif  num_zeros == 2
        for i = 0:2
            i1 = i+1
            i2 = (i+1)%3+1
            i3 = (i+2)%3+1
            if vals[i1] == 0 && vals[i2] == 0
                edge = Edge(ps[i1], ps[i2])
                df = fs[i1] - fs[i3]
                dl = ps[i2] - ps[i3]
                du = ps[i2] - ps[i2]
                grad1 = abs(df/(dl.x * du.y - dl.y * du.x))
                
                grad = [grad1, grad1]
                
                e = (edge, grad)
                is_subsample = level_set_is_subsample(edge, surf._Q_fn, tol)
            end
        end
    else 
        pts_ = []
        dfs_ = []
        dls_ = []

        for i = 0:2
            i1 = i+1
            i2 = (i+1)%3+1

            if vals[i1] * vals[i2] < 0
                pc, dfc, dlc = find_zeros(ps[i1], ps[i2], 
                                          fs[i1], fs[i2], 
                                          surf._Q_fn)
                push!(pts_, pc)
                push!(dfs_, dfc)
                push!(dls_, dlc)
            elseif vals[i1] == 0 
                push!(pts_, ps[i1])
                push!(dfs_, fs[i2] - fs[i1])
                push!(dls_, ps[i2] - ps[i1])
            end
        end

        if length(pts_) == 2
            edge =  Edge(pts_[1], pts_[2])
            is_subsample = level_set_is_subsample(edge, surf._Q_fn, tol)
            if pts_[2] == pts_[1]
                e = nothing
            else
                du = pts_[2] - pts_[1]
                du = du/norm(du)
                grad1 = abs(dfs_[1]/(dls_[1].x * du.y - dls_[1].y * du.x))
                grad2 = abs(dfs_[2]/(dls_[2].x * du.y - dls_[2].y * du.x))
                grad = [grad1, grad2]
                e = (edge, grad)
            end
        end
    end

    return is_subsample, e
end

function level_set_is_subsample(edge::Edge, f::Any, err::Float64)
    p1, p2 = edge.p1, edge.p2 
    p3 = (edge.p1 + edge.p2)/2.

    dl = norm(edge.p1 - edge.p2)

    return abs(f(p3) - (f(p1) + f(p2))/2) > err * dl
end

function level_set_subsample!(surf::Surface, 
                                c::Float64, 
                                depth::Int, 
                                tol::Float64)::Vector{Int}
    idxs = 1:length(surf._mesh._triangles) 
    idx_keep = []
    idx_new = []
    for n =1:depth - 1
        idx_new = []
        for i in idxs
            is_subsample, val = level_set_helper(surf, c, i, tol)

            if is_subsample
                push!(idx_new, i)
                subsample!(surf, i)
                L = length(surf._mesh._triangles)
                push!(idx_new, L-3)
                push!(idx_new, L-2)
                push!(idx_new, L-1)
            elseif !isnothing(val)
                push!(idx_keep, i)
            end


        end
        idxs = idx_new
    end
    idxs =Base.vcat(idx_keep, idx_new)
    return idxs
end

#TODO: remove redundancy since edge can be reuse
#TODO: Perhaps could redefine contour?
function level_set(surf::Surface, c::Float64, idxs::Vector{Int}, tol::Float64)::Tuple{Vector{Edge}, Vector{Vector{Float64}}}
    level_set_edges = []
    gradients = []
        
    for i in idxs
        is_subsmaple, e = level_set_helper(surf, c, i, tol)
        if !isnothing(e)
            edge, grad = e
            push!(level_set_edges, edge)
            push!(gradients, grad)
        end
    end

    return level_set_edges, gradients
end

function level_set!(surf::Surface, c::Float64, depth::Int, tol::Float64)
    
    idxs = level_set_subsample!(surf, c, depth, tol)

    level_set_edges = level_set(surf, c, idxs, tol)

    return level_set_edges
end

function plot_level_set!(l_set::Vector{Edge})
    
    for edge in l_set 
        segment = Meshes.Segment((edge.p1.x, edge.p1.y), 
                                (edge.p2.x, edge.p2.y))
        viz!(segment, color = "black")
    end
end

function integrate1D_helper(f::Any, a::Float64, b::Float64)
    fl = f(a)
    fr = f(b)
    fc = f((a + b)/2)
    L = b-a
    err = abs(fc - (fl + fr)/2.)

    res = (fl + 4 * fc + fr)/6 * L

    return res, err/L
end

function integrate1D(e::Edge, grad::Vector{Float64}, f::Any, depth::Int, tol::Float64)
    vl, vr = e.p1, e.p2
    L = norm(vr - vl)

    x2p(p) = interp(vl, vr, 1 - p)

    f_grad(x) = f(x2p(x))/(grad[1] + x/L * grad[2])

    xl = [0.]
    xr = [L]
    acc = 0.

    for n=1:depth 
        xl_ = []
        xr_ = []
        for i = 1:length(xl)
            res, err = integrate1D_helper(f_grad, xl[i], xr[i])

            
            if (err < tol || n == depth) && !isnan(res)
                acc += res
            else
                push!(xl_, xl[i])
                push!(xr_, (xl[i] + xr[i])/2.)
                
                push!(xl_, (xl[i] + xr[i])/2.)
                push!(xr_, xr[i])
            end
        end

        xl = xl_
        xr = xr_
    end

    return acc
end

function integrate1D(l_set::Vector{Edge}, grad::Vector{Vector{Float64}}, p::Any, depth::Int, tol::Float64)
    acc = 0

    for i = 1:length(l_set)
        acc += integrate1D(l_set[i], grad[i], p, depth,  tol)
    end
    return acc
end


mesh = get_square_grid(1., 1., 13)

function p_fn(v)
    return 1.0
end

function q_fn(v) 
    return 0.2 - (v.x-0.5)^2 - (v.y-0.5)^2
end

function f_fn(v)
    return q_fn(v)
end

t0  = time()
surf = Surface2D(mesh, q_fn)
# println(level_set_helper(surf, 0., 1, ))
println("time: ", time() - t0, " create mesh")
level_set_edges, gradients = level_set!(surf, 0., 4, 1e-6)
println("time: ", time() - t0, " level set: ", length(level_set_edges))
res = integrate1D(level_set_edges, gradients, p_fn, 5,  1e-9)
println("time: ", time() - t0," integral: ", res)

p = plot(surf)
plot_level_set!(level_set_edges)
    
Plots.display(p)

println("done: ", time() - t0)




