from levelSetIntegrate import *


"""
Fermi surface plot test
"""
def fermi_plot_test():
    l = np.linspace(-np.pi, np.pi, num = 100)

    

    X, Y, Z = np.meshgrid(l,l,l)

    f = (np.cos(X) + np.cos(Y) + np.cos(Z))

    hs = (f.min(), f.max(), 0.01)

    vb = plot_level_set(X, Y, Z, f, hs)
    
    display(vb)

def optical_plot_test():
    l = np.linspace(-np.pi, np.pi, num = 100)
    # hs = (-3, 3, 0.01)

    X, Y, Z = np.meshgrid(l,l,l)

    q = 1

    f = (np.cos(X) + np.cos(Y) + np.cos(Z)) - \
        (np.cos(X + q) + np.cos(Y + q) + np.cos(Z))

    hs = (f.min(), f.max(), 0.01)

    vb = plot_level_set(X, Y, Z, f, hs)
    
    display(vb)

"""
Integral test (Surface)
"""

def surface_int_test():
    l = np.linspace(-1.3, 1.3, num = 300)
    X, Y, Z = np.meshgrid(l,l,l)
    f = X**2 + Y**2 + Z**2
    hs = np.linspace(0, 1, num = 10)
    integrand = lambda x, y, z: np.ones(x.shape)
    areas = []
    for h in hs:
        vertices, triangles = get_level_set(X, Y, Z, f, h)
        area = surface_integral(vertices, triangles, integrand)
        areas.append(area)
        print(area, 4*np.pi*h)
    fig = go.Figure(data=[go.Scatter(x=hs, y=areas, name = 'calculated'),
                          go.Scatter(x=hs, y=4*np.pi*hs, name = 'expected')])
    display(fig)

"""
volume integral
"""
def volume_int_test():
    l = np.linspace(-1.3, 1.3, num = 300)
    X, Y, Z = np.meshgrid(l,l,l)
    Q = X**2 + Y**2 + Z**2
    hs = np.linspace(1e-5, 1, num = 30)
    integrand = lambda x, y, z: x**2 + y**2 + z**2
    absJ = lambda x, y, z: 2 * np.sqrt(x**2 + y**2 + z**2)
    res = volume_integral(X, Y, Z, Q, hs, integrand, absJ)
    print(res, 4/3 * np.pi)

"""
2-pt correlation test
"""
def χ2_test():
    l = np.linspace(-np.pi, np.pi, num = 300)
    kx, ky, kz = np.meshgrid(l,l,l)
    ω = 0
    q = 0.5
    Q = ω - (np.cos(kx) + np.cos(ky) + np.cos(kz)) \
        - (np.cos(kx + q) + np.cos(ky + q) + np.cos(kz + q))
    hs = np.linspace(-3, 3, num = 30)
    P = lambda kx, ky, kz: \
        np.where( kx**2 + ky**2 + kz**2 < 1, 1., 0.) \
        - np.where((kx-q)**2 + (ky-q)**2 + (kz-q)**2 < 1, 1., 0.)
    absJ = lambda kx, ky, kz: 2 * q
    res = volume_integral(kx, ky, kz, Q, hs, P, absJ)
    print(res)

if __name__ == '__main__':
    # optical_plot_test()
    # volume_int_test()
    χ2_test()




