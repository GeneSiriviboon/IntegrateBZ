


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

function plot(surf::SurfacePQ)
    points = [(p.x, p.y) for p in surf._mesh._vertices]
    connec = [connect((t.v1 , t.v2, t.v3)) for t in surf._mesh._triangles]
    colors = [surf._values_P[i]/surf._values_Q[i] for i = 1:length(surf._mesh._vertices)]
    simple_mesh   = SimpleMesh(points, connec)
    # println(simple_mesh)
    vmax = StatsBase.percentile(abs.(colors), 95)
    Meshes.viz(simple_mesh, 
                color = clamp.(colors, -vmax, vmax), 
                showsegments = true, 
                segmentcolor = RGBA(255,255,255,0.05),
                colormap = :balance)
   
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
        v1, v2, v3 = mesh._vertices[m.v1], mesh._vertices[m.v2], mesh._vertices[m.v3]

        p1, p2, p3 = surf._values_P[m.v1], surf._values_P[m.v2], surf._values_P[m.v3]

        q1, q2, q3 =  surf._values_Q[m.v1], surf._values_Q[m.v2], surf._values_Q[m.v3]

        integrand1, integrand2, integrand3 = p1/q1,p2/q2, p3/q3

        x_center = (v1.x + v2.x + v3.x)/3
        y_center = (v1.x + v2.x + v3.x)/3

        p4 = surf._P_fn(x_center, y_center)
        q4 = surf._Q_fn(x_center, y_center)

        integrand4 = p4/q4       
        
        l1_sq = (v2.x -v1.x)^2 + (v2.y -v1.y)^2
        l2_sq = (v3.x -v1.x)^2 + (v3.y -v1.y)^2

        area = (l2_sq * l1_sq)^0.5

        err = area * abs(integrand4 - (integrand1 + integrand2 + integrand3)/3)
        q_min = min(abs(q1), abs(q2), abs(q3), abs(q4))
        if err > abs_err  || q_min < abs_err
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