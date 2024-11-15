import numpy as np
import mcubes
import plotly.graph_objects as go
import plotly.offline as py
from ipywidgets import interactive, HBox, VBox
import plotly.express as px
from IPython.display import display

import plotly.io as pio
pio.renderers.default = "vscode"



def get_level_set(X, Y, Z, f, h):
    vertices, triangles = mcubes.marching_cubes(f, h) 
    x = vertices[:, 0] * (X.max() - X.min())/(X.shape[0]) +  X.min()
    y = vertices[:, 1] * (Y.max() - Y.min())/(Y.shape[1]) +  Y.min()
    z = vertices[:, 2] * (Z.max() - Z.min())/(Z.shape[2]) +  Z.min()
    return np.stack([x, y, z], axis = 1), triangles

#TODO: might need adaptive or quadruture for better result
def surface_integral(vertices, triangles, integrand):
    f = integrand(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    r1 = vertices[triangles[:, 1]] \
        - vertices[triangles[:, 0]]
    r2 = vertices[triangles[:, 2]] \
        - vertices[triangles[:, 0]]
    area = np.linalg.norm(np.cross(r1, r2), axis = -1)/2
    integral = (f[triangles[:, 0]] + \
                f[triangles[:, 1]] + \
                f[triangles[:, 2]]) * area/3
    return np.sum(integral)

def volume_integral(X, Y, Z, f, hs, integrand, absJ):
    dh = hs[1] - hs[0]
    areas = []
    integrand_helper = lambda x, y, z: integrand(x, y, z)/absJ(x, y, z)
    for h in hs:   
        vertices, triangles = get_level_set(X, Y, Z, f, h)
        area = surface_integral(vertices, triangles, integrand_helper)
        areas.append(area)
        print(h, area)


    integrand_ = np.array(areas)/hs
    integral = dh/2 * (integrand_[:-1] + integrand_[1:])
    return np.sum(integral)

def plot_level_set(X, Y, Z, f, hs):
    h = hs[0]

    vertices, triangles = get_level_set(X, Y, Z, f, h)
    fig = go.FigureWidget(data=[
    go.Mesh3d(
        x=vertices[:,0],
        y=vertices[:,1],
        z=vertices[:,2],
        # i, j and k give the vertices of triangles
        # here we represent the 4 triangles of the tetrahedron surface
        i=triangles[:,0],
        j=triangles[:,1],
        k=triangles[:,2],
        showscale=True
    )])


    def update(h):
        vertices, triangles = get_level_set(X, Y, Z, f, h)
        fig.data[0].x = vertices[:,0]
        fig.data[0].y = vertices[:,1]
        fig.data[0].z = vertices[:,2]
        fig.data[0].i = triangles[:,0]
        fig.data[0].j = triangles[:,1]
        fig.data[0].k = triangles[:,2]



    h_slider = interactive(update, h=hs)

    vb = VBox((fig, h_slider))
    vb.layout.align_items = 'center'

    return vb









