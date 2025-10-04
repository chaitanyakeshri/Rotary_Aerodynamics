from Params import *
from Solver import *
from helper_functions import *
import numpy as np
from typing import Callable, Dict
from scipy.optimize import brentq


def make_Cl(cyclic_c, cyclic_s, phi_fn):
    Cl_fn = lambda r, sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha=pitch_x_forward(rotor, r, sigh, cyclic_c, cyclic_s) - phi_fn(r, sigh),
        alpha_stall=rotor_aero["alpha_stall"]
    )
    return Cl_fn

def find_cyclic_bounds(
    Mp_init, Mr_init,
    theta_c_init,theta_s_init,
    b: int,
    rho: float,
    Ut_fn: Callable[[float, float], float],
    Up_fn: Callable[[float, float], float],
    c_fn: Callable[[np.ndarray], np.ndarray],
    phi_fn: Callable[[float, float], float],
    R_root: float,
    R_tip: float,
    ):
    print("Mp_init:",Mp_init,"Mr_init:",Mr_init)
    cyclic_c=cyclic_s=0

    # Upper/Lower bound for pitch
    cyclic_c = 5+theta_c_init if Mp_init > 0 else -5-theta_c_init
    cyclic_s=theta_s_init
    
    Cl_fn = make_Cl(cyclic_c, cyclic_s,phi_fn)
    out=iterative_solver_cyclic(
        b=b, rho=rho,
        Ut_fn=Ut_fn, Up_fn=Up_fn, c_fn=c_fn, Cl_fn=Cl_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter= 4, tol=1e-3
    )
    print("out Mp:",out["Mp"])
    if out["Mp"]*Mp_init>0:
        cyclic_c+=(np.sign(Mp_init)*5)

    # Upper/Lower bound for Roll
    cyclic_s_2 = -10-theta_s_init if Mr_init >= 0 else 10+theta_s_init
    cyclic_c_2=theta_c_init
    Cl_fn = make_Cl(cyclic_c_2, cyclic_s_2,phi_fn)
    out=iterative_solver_cyclic(
        b=b, rho=rho,
        Ut_fn=Ut_fn, Up_fn=Up_fn, c_fn=c_fn, Cl_fn=Cl_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter= 4, tol=1e-3
    )
    print("out Mr:",out["Mr"])
    if out["Mr"]*Mr_init>0:
        cyclic_s_2-=(np.sign(Mr_init)*5)

    return cyclic_c,cyclic_s_2


def find_cyclic(
    bound_c, bound_s,
    theta_c_init, theta_s_init,
    Mp_init, Mr_init, tol_mode,
    b: int,
    rho: float,
    Ut_fn: Callable[[float, float], float],
    Up_fn: Callable[[float, float], float],
    c_fn: Callable[[np.ndarray], np.ndarray],            # c_fn(R) -> 1D array of chord vs R
    phi_fn: Callable[[float, float], float],
    R_root: float,
    R_tip: float,
    t_horizon_s=60
    ):
    """
    Efficient 2D trim using Nelder-Mead and caching.

    simulate_forward_flight: function(theta1c, theta1s) -> {"Mr":..., "Mp":...}
    theta_init: starting guess (rad, rad)
    I_roll, I_pitch: inertias (kg*m^2)
    theta_max_deg: allowed attitude drift (deg)
    t_horizon_s: horizon in seconds
    """

    I_values=compute_MoI(fuselage,payload)
    I_roll=I_values["I_x"]
    I_pitch=I_values["I_y"]
    theta_max_deg=5.0
    omega_max=1

    if Mp_init is None or Mr_init is None:
        raise ValueError("Provide both initial pitch and roll values.")

    # compute allowed moments
    if tol_mode == "angle":
        theta_max = np.deg2rad(theta_max_deg)
        M_roll_allow  = 2 * I_roll  * theta_max / (t_horizon_s ** 2)
        M_pitch_allow = 2 * I_pitch * theta_max / (t_horizon_s ** 2)
        M_roll_allow = float(M_roll_allow)
        M_pitch_allow = float(M_pitch_allow)
    elif tol_mode == "rate":
        omega_max = np.deg2rad(omega_max)
        M_roll_allow  = I_roll  * omega_max
        M_pitch_allow = I_pitch * omega_max
    else:
        raise ValueError("tol_mode must be 'angle' or 'rate'")
    

    # Step 1: Solve cyclic_c for Mp = 0
    if Mp_init < 0:
        lower_c, upper_c = bound_c, theta_c_init
    else:
        lower_c, upper_c = theta_c_init, bound_c

    def Mp_func(c):
        Cl_fn = make_Cl(c, theta_s_init, phi_fn)  # cyclic_s fixed at initial
        out = iterative_solver_cyclic(
            b=b, rho=rho, Ut_fn=Ut_fn, Up_fn=Up_fn, c_fn=c_fn, Cl_fn=Cl_fn,
            R_root=R_root, R_tip=R_tip, N0=25, max_iter=4, tol=1e-2
        )
        Mp = out["Mp"]
        print("Mp->",Mp)

        # Treat anything within tolerance as zero
        if abs(Mp) <= M_pitch_allow:
            return 0.0
        return Mp

    cyclic_c_trim = brentq(Mp_func, lower_c, upper_c)
    
    # Step 2: Solve cyclic_s for Mr = 0
    if Mr_init < 0:
        lower_s, upper_s = theta_s_init, bound_s
    else:
        lower_s, upper_s = bound_s, theta_s_init

    def Mr_func(s):
        Cl_fn = make_Cl(cyclic_c_trim, s, phi_fn)  # cyclic_s fixed at initial
        out = iterative_solver_cyclic(
            b=b, rho=rho, Ut_fn=Ut_fn, Up_fn=Up_fn, c_fn=c_fn, Cl_fn=Cl_fn,
            R_root=R_root, R_tip=R_tip, N0=25, max_iter=4, tol=1e-2
        )
        Mr = out["Mr"]
        print("Mr->",Mr)
    
        # Treat anything within tolerance as zero
        if abs(Mr) <= M_roll_allow:
            return 0.0
        return Mr

    cyclic_s_trim = brentq(Mr_func, lower_s, upper_s)

    return cyclic_c_trim, cyclic_s_trim

    
def trim_cyclic(
        Mp_init,Mr_init,tol_mode,
        theta_c_init,theta_s_init,
        alpha_tpp,B1c,V_inf,B0_final,
        b, rho, t_horizon_s, max_iter
    ):

    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    Omega = engine["omega"]
    B_dot=B1c*Omega
    mu =  V_inf*np.cos(alpha_tpp)/ (Omega * R_tip)  # advance ratio


    B_fn = lambda sigh: B0_final + B1c*np.cos(sigh)

    Lambda_induced_forward = lambda r,sigh: lambda_i_forward(mu,r,R_tip,sigh,alpha_tpp,Omega,rho,V_inf)
    v_fn = lambda r,sigh: induced_velocity_forward(Lambda_induced_forward(r,sigh),Omega, R_tip, V_inf, alpha_tpp)

    phi_fn = lambda r,sigh: compute_phi_forward(V_inf,v_fn(r,sigh),Omega,alpha_tpp,
                                                            sigh,r,B_fn(sigh),B_dot,R_root)
    c_fn = lambda r: chord_r(rotor,r)
    
    AR = (R_tip - R_root) / ((rotor["chord_root"] + rotor["chord_tip"]) / 2)
    

    bound_c,bound_s=find_cyclic_bounds(
        Mp_init, Mr_init, 
        theta_c_init, theta_s_init,
        b=b, rho=rho,
        c_fn=c_fn, phi_fn=phi_fn,
        Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
        Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
        R_root=R_root, R_tip=R_tip,
    )
    print("bound_c:",bound_c,"bound_s:",bound_s)

    cyclic_c_trim,cyclic_s_trim=find_cyclic(
            bound_c,bound_s,
            theta_c_init,theta_s_init,
            Mp_init, Mr_init, tol_mode,
            b=b, rho=rho,
            c_fn=c_fn, phi_fn=phi_fn,
            Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
            Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
            R_root=R_root, R_tip=R_tip, 
            t_horizon_s=t_horizon_s
    )

    Cl_fn=make_Cl(cyclic_c_trim, cyclic_s_trim,phi_fn)
    # Drag coefficient as a function of r
    Cd_fn = lambda r,sigh: airfoil_drag(
        Cd0=rotor_aero["Cd0"],
        Cl=Cl_fn(r,sigh),
        e=rotor_aero["e"],
        AR=AR
    )
    res=iterative_solver_forward(
        b=b, rho=rho,
        Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
        Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
        c_fn=c_fn, Cl_fn=Cl_fn, 
        phi_fn=phi_fn, Cd_fn=Cd_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter = 4, tol=1e-3
    )

    return cyclic_c_trim,cyclic_s_trim,res["Q"]

