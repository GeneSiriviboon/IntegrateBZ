import numpy as np
import mcubes
import plotly.graph_objects as go
import plotly.offline as py
from ipywidgets import interactive, HBox, VBox
from IPython.display import display

import plotly.io as pio
pio.renderers.default = "vscode"



def get_level_set(X, Y, Z, f, h):
    vertices, triangles = mcubes.marching_cubes(f(X, Y, Z), h)
    return vertices, triangles

def integrate_triangle(vertex, triangle, I):
    pass

def integrate_level():
    pass


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


if __name__ == "__main__":
    l = np.linspace(-np.pi, np.pi, num = 100)

    hs = (-3, 3, 0.01)

    X, Y, Z = np.meshgrid(l,l,l)

    f = lambda X, Y, Z: (np.cos(X) + np.cos(Y) + np.cos(Z))

    vb = plot_level_set(X, Y, Z, f, hs)
    
    display(vb)


#(X - 50)**2 + (Y-50)**2 + (Z-50)**2
# print(u)
# u = (X-50)**2 +  (Y - 50)**2 - (Z - 50)**2 
# R = 20

# Create figure



