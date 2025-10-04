import numpy as np
import math
## functional imports 
from Solver import *
from Simulator import * 
from Params import * 
import numpy as np




def atmosphere(altitude, delta_ISA=0.0):
    """
    Calculates temperature, pressure, density, and speed of sound 
    at a given altitude with ISA temperature offset.
    
    Inputs:
        altitude   : Altitude [m]
        delta_ISA  : ISA temperature offset [K] (default = 0)
        
    Returns:
        dict with T [K], p [Pa], rho [kg/m³], a [m/s]
    """
    
    # Constants
    T0 = 288.15      # Sea level standard temp [K]
    p0 = 101325.0    # Sea level standard pressure [Pa]
    rho0 = 1.225     # Sea level density [kg/m³]
    a0 = 340.294     # Sea level speed of sound [m/s]
    g = 9.80665      # Gravity [m/s²]
    R = 287.058      # Gas constant for air [J/kg-K]
    L = -0.0065      # Temperature lapse rate [K/m] up to 11 km
    
    # Troposphere only (0–11 km)
    if altitude <= 11000:
        T = T0 + L * altitude + delta_ISA
        p = p0 * (T / (T0 + delta_ISA)) ** (-g / (L * R))
    else:
        # Very simplified: isothermal above 11 km
        T = 216.65 + delta_ISA
        p = p0 * (T / (T0 + delta_ISA)) ** (-g / (L * R)) \
            * math.exp(-g * (altitude - 11000) / (R * T))
    
    rho = p / (R * T)
    a = math.sqrt(1.4 * R * T)
    
    return {"T": T, "p": p, "rho": rho, "a": a}




def airfoil_lift(Cl0, Cl_alpha, alpha0, alpha,alpha_stall=15.0):
    """
    Calculates lift coefficient for a given airfoil at a specified angle of attack in degrees.
    """
    if alpha > alpha_stall:
        return None
    
    alpha_rad = math.radians(alpha)
    alpha0_rad = math.radians(alpha0)
    Cl = Cl0 + Cl_alpha * (alpha_rad - alpha0_rad)
    return Cl

def airfoil_drag(Cd0, Cl, e, AR):
    """
    Calculates drag coefficient using parabolic drag polar.
    """
    Cd = Cd0 + (Cl ** 2) / (math.pi * AR * e)
    return Cd

def airfoil_moment(Cm0, Cm_alpha,alpha0, alpha):
    """
    Calculates pitching moment coefficient for a given airfoil.
    """
    alpha_rad = math.radians(alpha)
    alpha0_rad = math.radians(alpha0)
    Cm = Cm0 + Cm_alpha * (alpha_rad - alpha0_rad)
    return Cm


def chord_r(rotor, r):
    """
    Calculates the chord length at a given radius assuming linear variation from root to tip.
    """
    Rr= rotor["Rr"]
    Rt = rotor["Rt"]
    tip_chord = rotor["chord_tip"]
    root_chord = rotor["chord_root"]
    return root_chord + (tip_chord - root_chord) * (r - Rr) / (Rt - Rr)


def pitch_x(rotor,r):
    """
    Calculates the pitch angle along the blade span assuming linear variation from root to tip in degrees
    """
    Rr= rotor["Rr"]
    Rt = rotor["Rt"]
    theta_root = rotor["theta_root"] 
    theta_tip = rotor["theta_tip"] 
    collective = rotor["collective"]

    return  collective + theta_root + (theta_tip - theta_root) * (r - Rr) / (Rt - Rr)


def pitch_x_forward(rotor,r,sigh):
    """
    Calculates the pitch angle along the blade span assuming linear variation from root to tip in degrees
    """
    Rr= rotor["Rr"]
    Rt = rotor["Rt"]
    theta_root = rotor["theta_root"] 
    theta_tip = rotor["theta_tip"] 
    collective = rotor["collective"]
    theta_1c = rotor["cyclic_c"]
    theta_1s = rotor["cyclic_s"]

    return  collective + theta_root + (theta_tip - theta_root) * (r - Rr) / (Rt - Rr) + theta_1c* np.cos(sigh) + theta_1s * np.sin(sigh)




def solidity(b: int, c_root: float, c_tip: float, R: float, R_root: float = 0.0) -> float:
    """
    Rotor solidity for linearly tapered chord with a root cut-out.

    Formula:
    σ = (Nb * (R - R_root) * (c_root + c_tip) / 2) / (π * R^2)

    Nb:     number of blades [-]
    c_root: chord at R_root [m]
    c_tip:  chord at R [m]
    R:      rotor tip radius [m]
    R_root: root cut-out radius [m]
    """
    if not (0 <= R_root < R):
        raise ValueError("Require 0 <= R_root < R")
    span = R - R_root
    avg_chord = 0.5 * (c_root + c_tip)
    return (b * avg_chord * span) / (math.pi * R**2)



## inflow calculator 
def Lambda(sigma, a, theta, r, R, V, Omega, R_root=0.0):
    """
    Computes inflow ratio (lambda) at a blade section, accounting for root cut-out.

    Parameters:
    sigma   : float   # Rotor solidity
    a       : float   # Lift curve slope (per rad)
    theta   : float   # Pitch angle (deg)
    r       : float   # Radial location (m)
    R       : float   # Rotor tip radius (m)
    V       : float   # Freestream velocity (m/s)
    Omega   : float   # Rotor angular speed (rad/s)
    R_root  : float   # Root cut-out radius (m), default 0

    Returns:
    lambda_val : float
    """

    # Effective span and normalized radius
    span = R - R_root
    r_bar = (r - R_root) / span  # normalized radius from root cut-out

    theta_r = math.radians(theta)
    lambda_c = V / (Omega * R)

    term1 = (sigma * a / 16) - (lambda_c / 2)
    term2 = (sigma * a / 8) * theta_r * r_bar

    lambda_val = math.sqrt(term1**2 + term2) - term1

    return lambda_val



def induced_velocity(Lambda,Omega,R,V):
    """
    Computes induced velocity at the rotor disk for a given inflow ratio.

    Parameters:
    lambda : float  # Inflow ratio
    Omega  : float  # Rotor angular speed (rad/s)
    R      : float  # Rotor radius (m)

    Returns:
    vi : float
    """
    vi = Lambda * Omega * R - V
    return vi
    
   

   
def LambdaP_hover(sigma, a, F, theta, r, R_tip, R_root=0.0):
    """
    Computes inflow ratio (lambda) for hover with Prandtl's tip loss, accounting for root cut-out.

    Parameters:
    sigma   : float   # Rotor solidity
    a       : float   # Lift curve slope (per rad)
    F       : float   # Tip loss factor
    theta   : float   # Pitch angle (deg)
    r       : float   # Radial location (m)
    R_root  : float   # Root cut-out radius (m), default 0

    Returns:
    lam     : float   # Inflow ratio
    """
    if F == 0:
        return 0
    theta_rad = math.radians(theta)
    term = 1 + (32 * F / (sigma * a)) * theta_rad * r/R_tip
    lam = (sigma * a / (16 * F)) * (math.sqrt(term) - 1)
    return lam


def LambdaP_climb(sigma, a, F, theta, r, R_tip, lambda_c, R_root=0.0):
    """
    Computes inflow ratio (lambda) for climb with Prandtl's tip loss, accounting for root cut-out.

    Parameters:
    sigma     : float   # Rotor solidity
    a         : float   # Lift curve slope (per rad)
    F         : float   # Tip loss factor
    theta     : float   # Pitch angle (deg)
    r         : float   # Radial location (m)
    R         : float   # Rotor tip radius (m)
    lambda_c  : float   # Advance ratio (V / Omega R)
    R_root    : float   # Root cut-out radius (m), default 0

    Returns:
    lam       : float   # Inflow ratio
    """
    if F <= 0:
        return 0
    theta_rad = math.radians(theta)
    term1 = (sigma * a) / (16 * F) - (lambda_c / 2)
    term2 = (sigma * a / (8 * F)) * theta_rad * r/R_tip
    lam = math.sqrt(term1**2 + term2) - term1
    return lam


def LambdaP(flight_condition, sigma, a, F, theta, r, lambda_c, R_tip, R_root=0.0):
    """ 
    Wrapper to compute inflow ratio (lambda) for hover or climb with Prandtl's tip loss, accounting for root cut-out.
    Decides between hover and climb based on vertical velocity component.
    Parameters:     
    flight_condition : dict with key "velocity" (3-element list or array)
    sigma           : float   # Rotor solidity  
    a               : float   # Lift curve slope (per rad)
    F               : float   # Tip loss factor
    theta           : float   # Pitch angle (deg)
    r               : float   # Radial location (m)
    lambda_c       : float   # Advance ratio (V / Omega R)
    R_tip          : float   # Rotor tip radius (m)
    R_root         : float   # Root cut-out radius (m), default 0
    Returns:    
    lam             : float   # Inflow ratio
    """


    if flight_condition["velocity"][2] == 0:
        return LambdaP_hover(sigma, a, F, theta, r, R_tip, R_root)
    else:
        return LambdaP_climb(sigma, a, F, theta, r, R_tip, lambda_c, R_root)
    



def tip_loss_factor(b, r, R_tip, lam, R_root=0.0):
    """
    Computes Prandtl's tip loss factor, accounting for root cut-out.

    Parameters:
    b      : int    # Number of blades
    r      : float  # Radial location (m)
    R      : float  # Rotor tip radius (m)
    lam    : float  # Local inflow ratio
    R_root : float  # Root cut-out radius (m), default 0

    Returns:
    F      : float  # Tip loss factor
    """
    if lam == 0:
        return 0.0
    f = (b / 2) * (1 - r/R_tip) / lam
    F = (2 / math.pi) * math.acos(math.exp(-f))
    return F


def compute_phi(V, v, Omega, r, R_root=0.0):
    """
    Computes the inflow angle (phi) at a blade section and returns it in degrees.

    Parameters:
    V      : float  # Freestream velocity (m/s)
    v      : float  # Induced velocity (m/s)
    Omega  : float  # Rotor angular speed (rad/s)
    r      : float  # Radial location (m), should be >= R_root
    R_root : float  # Root cut-out radius (m), default 0

    Returns:
    phi_deg : float  # Inflow angle in degrees
    """
    Ut = Omega * r
    Up = V + v
    phi_rad = math.atan2(Up, Ut)
    phi_deg = math.degrees(phi_rad)
    
    return phi_deg



def solve_lambda_tiploss(flight_condition, sigma, a, b, theta_deg, r,lambda_c, R_tip, R_root=0.0,tol=1e-3, max_iter=5):
    """
    Iteratively solves for lambda and Prandtl's tip loss factor F at a blade station.
    Starts from F=1, updates lambda and F until convergence.
    Returns: (lambda, F)
    """
    F = 1.0
    for _ in range(max_iter):
        lam_new = LambdaP(flight_condition, sigma, a, F, theta_deg, r, lambda_c, R_tip, R_root)
        F_new = tip_loss_factor(b, r, R_tip, lam_new, R_root)
        #print("F_new",F_new)
        if abs(F_new - F) < tol:
            return lam_new, F_new
        F = F_new
    return (lam_new, F_new)  # Return last values if not converged



def density_ratio(h):
    """
    Computes density ratio (rho / rho0) as a function of altitude h (meters).
    Equation: (1 - 0.00198*h / 288.16) ^ 4.2553
    """
    return (1 - (0.00198 * h) / 288.16) ** 4.2553
''' 
--------------------------------------------------------------------------------------------------------------------------------
forward Flight functions
--------------------------------------------------------------------------------------------------------------------------------
'''
def induced_velocity_forward(Lambda,Omega,R,V,alpha_TPP):
    """
    Computes induced velocity at the rotor disk for a given inflow ratio.

    Parameters:
    lambda : float  # Inflow ratio
    Omega  : float  # Rotor angular speed (rad/s)
    R      : float  # Rotor radius (m)

    Returns:
    vi : float
    """
    vi = Lambda * Omega * R 
    return vi


def compute_phi_forward(Vinf, v, Omega,alpha_TPP,sigh, r, B,Bdot,R_root=0.0):
    """
    Computes the inflow angle (phi) at a blade section and returns it in degrees.

    Parameters:
    V      : float  # Freestream velocity (m/s)
    v      : float  # Induced velocity (m/s)
    Omega  : float  # Rotor angular speed (rad/s)
    r      : float  # Radial location (m), should be >= R_root
    R_root : float  # Root cut-out radius (m), default 0

    Returns:
    phi_deg : float  # Inflow angle in degrees
    """
    Ut = Omega * r + Vinf*np.cos(alpha_TPP)*np.sin(sigh)  # tangential velocity component
    Up = Vinf*np.sin(alpha_TPP) + v + r*Bdot + Vinf*np.sin(B)*np.cos(sigh)  # axial velocity component
    phi_rad = np.arctan2(Up, Ut)

    return phi_rad



def fuselage_drag(fuselage, rho, V):
    """
    Computes fuselage drag force.
    Inputs:
        fuselage : dict with keys Cdx, Cdy, Cdz, X_flat_area, Y_flat_area, Z_flat_area
        rho      : air density [kg/m³]
        V        : velocity vector [m/s] (3-element list or array)
    Returns:
        D_fuse   : drag force [N]
    """
    Vx, Vy, Vz = V
    Cdx = fuselage["Cdx"]
    Cdy = fuselage["Cdy"]
    Cdz = fuselage["Cdz"]
    Ax = fuselage["X_flat_area"]
    Ay = fuselage["Y_flat_area"]
    Az = fuselage["Z_flat_area"]
    
    D_x = 0.5 * rho * Cdx * Ax * Vx * abs(Vx)
    D_y = 0.5 * rho * Cdy * Ay * Vy * abs(Vy)
    D_z = 0.5 * rho * Cdz * Az * Vz * abs(Vz)
    
    D_fuse = {"Dx":D_x,"Dy" : D_y,"Dz":D_z}
    return D_fuse

def alphaTPP(fuselage_drag,TOGW):
    """
    Computes the angle of attack of the total parasite drag force.
    Inputs:
        fuselage_drag : dict with keys Dx, Dy, Dz
        W             : weight [N]
    Returns:
        alpha_tpp     : angle of attack [deg]
    """
    D_x = fuselage_drag["Dx"]
    D_y = fuselage_drag["Dy"]
    D_z = fuselage_drag["Dz"]
    
    alpha_tpp = np.tanh(D_x/(TOGW*9.8)) #in radians
    return alpha_tpp

def advance_ratio(V,alpha_tpp, Omega, R):
    """
    Computes the advance ratio (u).
    Inputs:
        V : freestream velocity [m/s]
        alpha_tpp : angle of attack of the total parasite drag force [deg]
        Omega : rotor angular speed [rad/s]
        R : rotor radius [m]
    Returns:
        advance ration u 
    """
    return V*np.cos(alpha_tpp) / (Omega * R)

def Ct(Thrust, rho, Omega, R):
    """
    Computes the thrust coefficient (Ct).
    Inputs:
        Thrust : thrust [N]
        rho : air density [kg/m³]
        Omega : rotor angular speed [rad/s]
        R : rotor radius [m]
    Returns:
        Ct : thrust coefficient
    """
    A = math.pi * R**2
    return Thrust / (rho * A * (Omega * R)**2)

def Cpi(Ct, lambda_i,k):
    """
    Computes the induced power coefficient (Cpi).
    Inputs:
        Ct : thrust coefficient
        lambda_i : induced inflow ratio
        k : induced power factor
    Returns:
        Cpi : induced power coefficient
    """
    return k * Ct * lambda_i
    

def Cpp(f,mu,R):
    """
    Computes the profile power coefficient (Cpp).
    Inputs:
        f : profile drag coefficient
        u : advance ratio
        R : rotor radius [m]
    Returns:
        Cpp : profile power coefficient
    """
    return f * mu**3 / 2*np.pi*R**2

def Cpo(sigma,Cd0,mu):
    """
    Computes the profile power coefficient (Cpo).
    Inputs:
        sigma : rotor solidity
        Cd0 : zero-lift drag coefficient
        mu : advance ratio
    Returns:
        Cpo : profile power coefficient
    """
    return sigma * Cd0 / 8 * (1 + 4.6 * mu**2)
    


def LambdaIG(Ct,mu,alpha_TPP):
    """
    Solves for induced inflow ratio (lambda_i) using Glauret's method.
    Inputs:
        Ct : thrust coefficient
        mu : advance ratio
        alpha_TPP : angle of attack of the total parasite drag force [deg]
    Returns:
        lambda_i : induced inflow ratio
    """

    t = np.tan(alpha_TPP)  

    # polynomial coefficients for 4 x^4 + 8 mu t x^3 + 4 mu^2(1+t^2) x^2 - CT^2 = 0
    coeffs = [4.0, 8.0*mu*t, 4.0*mu*mu*(1.0+t*t), 0.0, -Ct*Ct]

    roots = np.roots(coeffs)
    # filter real, positive roots (allow tiny imag numerical noise)
    real_pos = [r.real for r in roots if abs(r.imag) < 1e-2 and r.real > 0]
    #print("candidate lambda_i_g roots:", real_pos)
    if not real_pos:
        raise ValueError("No positive real root found for lambda_i")
    #print(min(real_pos))
    return min(real_pos)  
        

def LambdaG(V,lambdaig,Omega,R):
    """
    Computes induced velocity using Glauret's method.
    Inputs:
        V : freestream velocity [m/s]
        lambdaig : induced inflow ratio from Glauret's method
        Omega : rotor angular speed [rad/s]
        R : rotor radius [m]
    Returns:
        vi_g : induced velocity [m/s]
    """
    return (lambdaig  + V/(Omega*R))


def lambda_i_forward(mu, r, R, sigh,alpha_TPP,Omega,rho,Vinf):
    """
    Compute inflow ratio lambda_i.

    Parameters
    ----------
    mu : float
        Advance ratio
    lambda_G : float
        Reference inflow parameter (λ_G)
    lambda_Glauert : float
        Glauert inflow λ_i,Glauert
    r : float
        Radial location (spanwise position)
    R : float
        Rotor radius
    psi : float
        Azimuth angle [radians]

    Returns
    -------
    float
        λ_i at given r, ψ
    """
    TOGW = fuselage["Empty_Weight"] + payload["weight"] + flight_condition["fuel_weight"]
    Thrust = TOGW/ np.cos(alpha_TPP)  # Thrust required to balance weight in forward flight
    lambdaig = LambdaIG(Ct(Thrust,rho,Omega,R),mu,alpha_TPP)
    lambda_G = LambdaG(Vinf,lambdaig,Omega,R)
    frac = ( (4/3) * (mu / lambda_G) ) / (1.2 + (mu / lambda_G))
    correction = frac * (r / R) * np.cos(sigh)
    return lambdaig * (1 + correction)


from scipy.integrate import quad

def beta0(lock_number,alpha_effective):
    """
    Compute β0 or Coning Angle numerically from the given parameters.
    
    Parameters
    ----------
    rho : float
        Air density
    a : float
        Lift curve slope (per radian)
    c : float
        Chord length
    R : float
        Rotor radius
    I : float
        Blade moment of inertia about flapping hinge
    
    Returns
    -------
    float
        β0 value
    """
    
    # Define the integrand f(r/R)
    integrand = lambda r_over_R: (r_over_R**3) * alpha_effective(r_over_R)
    
    # Perform numerical integration from 0 to 1
    integral, _ = quad(integrand, 0, 1)
    
    # Apply scaling factor
    beta0_val = (lock_number) * integral
    return beta0_val

def Beta(Sigh):
    """             
    Compute blade flapping angle β at azimuth angle Sigh (radians).    
    Parameters
    """

    return beta0(rotor["lock_number"]) + rotor["cyclic_c"] * np.cos(Sigh) + rotor["cyclic_s"] * np.sin(Sigh)



