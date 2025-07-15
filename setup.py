"""
setup.py
This module provides setup functions for defining boundary conditions, source terms, and coefficient functions for various PDE test problems using FEniCSx/dolfinx. 
It includes utilities for homogeneous Dirichlet and Robin boundaries, as well as spatially varying coefficients such as channels, squares, and random fields. 
"""

import numpy as np
from dolfinx.fem import Function



def getSetupSourceDirichlet(xL, yL, xR, yR, V, msh):
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC
    # u_D.interpolate(lambda x : np.cos(direction[0] * k * x[0] + direction[1] * k * x[1]) + imag_unit * np.sin(direction[0] * k * x[0] + direction[1] * k * x[1]))
    # u_D.interpolate(lambda x: x[0] * x[1])

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.7, 0.9
    f.interpolate(lambda x: np.exp(-((x[0] - x0)**2 + (x[1] - y0)**2) ))

    # Coeff in PDE
    coeff_V_function = lambda x : 1.0 + 0 * x[0]

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_V_function


def getSetupIID(xL, yL, xR, yR, V, contrast):
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    f.interpolate(lambda x: np.full(x.shape[1], 1.0))

    # Coeff in PDE
    coeff_A_function = getIIDCoefficient(contrast)

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function

def getIIDCoefficient(contrast = 10):
    Nx = 64 # mesh elements for the iid coefficient

    if contrast < 1:
        raise ValueError("Contrast must be greater than 1.")

    # set seed
    np.random.seed(0)
    # Generate an NxN matrix with entries uniformly distributed between 0 and 1
    coeff = np.random.uniform(low=0, high=1, size=(Nx, Nx))

    coeff = coeff  - np.min(coeff)  

    coeff_max = np.max(coeff)

    coeff = 1 + (contrast - 1) * coeff/coeff_max

    coeff_function = lambda x : coeff[np.floor(Nx * x[0]).astype(int), np.floor(Nx * x[1]).astype(int)]

    return coeff_function
    
def getSetupChannel(xL, yL, xR, yR, V, ny, Ny, contrast):
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.5, 0.9
    f.interpolate(lambda x: np.full(x.shape[1],  1.0)) 

    # Coeff in PDE
    coeff_A_function = lambda x : 1.0 + (contrast - 1) * channel(x[0], x[1], ny, Ny)

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function


def vertical_channel(x_midpoint, y_midpoint, length, width, x, y):
    return np.logical_and(np.abs(x - x_midpoint) < width/2 - 1e-6, np.abs(y - y_midpoint) < length/2 - 1e-6)
    
def horizontal_channel(x_midpoint, y_midpoint, length, width, x, y):
    return np.logical_and(np.abs(y - y_midpoint) < width/2 - 1e-6, np.abs(x - x_midpoint) < length/2 - 1e-6)

def on_box(x_midpoint, y_midpoint, sidelength, h, layers, x, y):
    return np.logical_and(
        np.abs(x-x_midpoint) < (sidelength - h * layers)/2, 
        np.abs(y-y_midpoint) < (sidelength - h * layers)/2
    )

def channel(x,y, ny, N):
    # Take the unit channel from below and repeat it in the four quadrants
    # N = 4
    on_channel = np.full(x.shape, False)
    length = 1
    width = 0.05
    # layers = 4 # Layers aways from dirichlet boundary
    layers = 8
    h = 1/ ny

    checkerboard_bool = True

    for i in range(N):
        for j in range(N):
            on_channel = np.logical_or(on_channel, np.logical_or(horizontal_channel(1/(2*N) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y), np.logical_or(vertical_channel(1/(2*N) -1 /(2*N *5) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y), 
                np.logical_or(vertical_channel(1/(2*N) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y), vertical_channel(1/(2*N) +1 /(2*N *5) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y)))))
            # only connect the channels on each second (checkerboard) domain
            if checkerboard_bool:
                sidelength = 1/ N
                on_channel = np.logical_and(on_channel, np.logical_not(on_box(1/(2*N) + i/N, 1/(2*N) + j/N, sidelength, h, 2 * layers, x, y )))
                # on unconnected regions, connect the three vertical channels by horizontal channels
                # connect upper part:
                on_channel = np.logical_or(on_channel, horizontal_channel(1/(2*N) + i/N,  (j+1)/N - layers * h , length/(3 * N), width/N, x, y))
                # connect lower part:
                on_channel = np.logical_or(on_channel, horizontal_channel(1/(2*N) + i/N,  j/N + layers * h , length/(3 * N), width/N, x, y))


                checkerboard_bool = False
            else:
                checkerboard_bool = True
        checkerboard_bool = not checkerboard_bool

    # Make that channel does not touch the boundary
    layers = 4
    on_channel = np.logical_and(on_channel, x > layers * h -1e-9)
    on_channel = np.logical_and(on_channel, y > layers * h -1e-9)
    on_channel = np.logical_and(on_channel, x < 1 - layers * h + 1e-9)
    on_channel = np.logical_and(on_channel, y < 1 - layers * h + 1e-9)


    return on_channel
    
def unit_channel(x,y):
    # Check whether the point (x,y) is on one of the following 
    # three vertical channels, which have midpoints at  x = 0.4, 0.5, 0.6, y = 0.5, 0.5, 0.5
    # length 0.5; and width 0.05.
    return np.logical_or(horizontal_channel(0.5, 0.5, 0.5, 0.05, x, y), np.logical_or(vertical_channel(0.4, 0.5, 0.5, 0.05, x, y), np.logical_or(vertical_channel(0.5, 0.5, 0.5, 0.05, x, y), vertical_channel(0.6, 0.5, 0.5, 0.05, x, y))))


def getSetupSquareMiddle(xL, yL, xR, yR, V, contrast):
    # Only consider homogeneous dirichlet BC
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.5, 0.5
    f.interpolate(lambda x: np.full(x.shape[1], 1.0))

    # Coeff in PDE
    coeff_A_function = lambda x : 1.0 + (contrast - 1) * square_middle(x[0], x[1])

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function

def square_middle(x,y):
    # Check whether the point (x,y) is in the square with midpoint (0.5, 0.5), length 0.1, and width 0.1.
    return np.logical_and(np.abs(x - 0.5) < 0.05 +1e-6, np.abs(y - 0.5) < 0.05 +1e-6)

def getSetupSkyscraper(xL, yL, xR, yR, V, contrast):
    # Only consider homogeneous dirichlet BC
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.5, 0.5
    f.interpolate(lambda x: np.full(x.shape[1], 1.0))

    # Coeff in PDE
    coeff_A_function = lambda x : np.array([skyscraper(x[0,i], x[1,i]) for i in range(x.shape[1])])

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function

def skyscraper(x,y):
    # Compute Nx and Ny
    Nx = int(np.floor(x * 8))
    Ny = int(np.floor(y * 8))
    
    # Compute Nx_c and Ny_c
    Nx_c = 8 * x - Nx
    Ny_c = 8 * y - Ny
    
    # Define the r_number array
    r_number = 1.0e5 * np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    ])
    
    # Initial value
    value = 1.0
    
    # Check conditions for Nx_c and Ny_c
    if 0.2 < Nx_c < 0.8 and 0.2 < Ny_c < 0.8:
        value = r_number[Ny, Nx]
    
    # Additional conditions based on rotations
    theta3 = np.pi / 3
    xc, yc = x - 0.825, y - 0.7
    x3 = np.cos(theta3) * xc + np.sin(theta3) * yc + 0.825
    y3 = -np.sin(theta3) * xc + np.cos(theta3) * yc + 0.675
    if 0.8 < x3 < 0.85 and 0.4 < y3 < 0.95:
        value = 1.5e5
    
    theta4 = 3 * np.pi / 4
    xc, yc = x - 0.875, y - 0.25
    x4 = np.cos(theta4) * xc + np.sin(theta4) * yc + 0.875
    y4 = -np.sin(theta4) * xc + np.cos(theta4) * yc + 0.25
    if 0.85 < x4 < 0.9 and 0 < y4 < 0.5:
        value = 1.5e5
    
    theta2 = 3.25 * np.pi / 10
    xc, yc = x - 0.15, y - 0.55
    x2 = np.cos(theta2) * xc + np.sin(theta2) * yc + 0.15
    y2 = -np.sin(theta2) * xc + np.cos(theta2) * yc + 0.55
    if 0.12 < x2 < 0.18 and -0.15 < y2 < 0.6:
        value = 2.0e6
    
    theta1 = np.pi / 10
    xc, yc = x - 0.15, y - 0.55
    x1 = np.cos(theta1) * xc + np.sin(theta1) * yc + 0.15
    y1 = -np.sin(theta1) * xc + np.cos(theta1) * yc + 0.55
    if 0.15 < x1 < 0.67 and 0.56 < y1 < 0.61:
        value = 2.0e6
    
    return value

