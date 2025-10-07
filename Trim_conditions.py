from Params import rotor_aero
from Solver import *
from helper_functions import *
from calc_functions import *
import numpy as np
from typing import Callable, Dict
from scipy.optimize import brentq

def stall_check(R_root,R_tip,theta_fn,phi_fn):
    r_all = np.linspace(R_root, R_tip, 80)       # dense discretization in radius
    sigh_all = np.linspace(0, 2*np.pi, 90)       # dense discretization in azimuth

    for r in r_all:
        for sigh_val in sigh_all:
            alpha = theta_fn(r,sigh_val) - phi_fn(r, sigh_val)
            if alpha > rotor_aero["alpha_stall"]:
                print(f"Stall detected for main rotor at r = {r:.3f} m, sigh = {sigh_val:.3f} rad, alpha = {alpha:.3f} deg")
                return {"stall_status": 1, "r": r, "sigh": sigh_val, "alpha": alpha, "out_of_power": False}
    
    return {"stall_status": 0}


def find_moments(rotor, cyclic_c, cyclic_s, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn):

    theta_fn = lambda r,sigh: pitch_x_forward(rotor, r,sigh, cyclic_c, cyclic_s, coll) 
    Cl_fn = lambda r,sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha = theta_fn(r,sigh) - phi_fn(r,sigh),
        alpha_stall=rotor_aero["alpha_stall"]
    )

    stall=stall_check(R_root,R_tip,theta_fn,phi_fn)
    if stall["stall_status"]==1:
        return stall
    B_fn,B0_final=beta_fn(rho, alpha_tpp, Omega, V_inf, v_fn, c_fn, Cl_fn, R_root, R_tip, I)

    res = iterative_solver_cyclic(
        b=b, rho=rho,  
        Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
        Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
        c_fn=c_fn, Cl_fn=Cl_fn, 
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter = 4, tol=1e-2
    )
    return res


def find_cyclic_bounds(
    Mp_init, Mr_init,
    theta_c_init, theta_s_init, coll,
    alpha_tpp, I, Omega, V_inf, B_dot,
    b, rho, R_root, R_tip,
    v_fn, c_fn, phi_fn
    ):
    print("Mp_init:",Mp_init,"Mr_init:",Mr_init)
    print(theta_c_init,theta_s_init)
    # Upper/Lower bound for pitch
    cyclic_c = 5+theta_c_init if Mp_init > 0 else -5+theta_c_init
    cyclic_s=theta_s_init
    
    out=find_moments(rotor, cyclic_c, cyclic_s, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
    print("out Mp:",out["Mp"])
    if out["Mp"]*Mp_init>0:
        cyclic_c+=(np.sign(Mp_init)*5)


    # Upper/Lower bound for Roll
    if theta_s_init<-4:
        print("here")
        cyclic_s_2=-2.5+theta_s_init if Mr_init>=0 else 2.5+theta_s_init
    else:
        cyclic_s_2 = -5+theta_s_init if Mr_init >= 0 else 5+theta_s_init
    cyclic_c_2=theta_c_init

    out=find_moments(rotor, cyclic_c_2, cyclic_s_2, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
    print("out Mr:",out["Mr"])
    if out["Mr"]*Mr_init>0:
        cyclic_s_2-=(np.sign(Mr_init)*5)

    return cyclic_c,cyclic_s_2


def find_cyclic(
    bound_c, bound_s, skipMp, skipMr,
    theta_c_init, theta_s_init, coll,
    Mp_init, Mr_init, Mp_allow, Mr_allow,
    alpha_tpp, I, Omega, V_inf, B_dot,
    b, rho, R_root, R_tip,
    v_fn, c_fn, phi_fn
    ):
    """
    Efficient 2D trim using Nelder-Mead and caching.

    simulate_forward_flight: function(theta1c, theta1s) -> {"Mr":..., "Mp":...}
    theta_init: starting guess (rad, rad)
    I_roll, I_pitch: inertias (kg*m^2)
    theta_max_deg: allowed attitude drift (deg)
    t_horizon_s: horizon in seconds
    """

    cyclic_c_trim=theta_c_init
    if skipMp==False:
        # Step 1: Solve cyclic_c for Mp = 0
        if Mp_init < 0:
            lower_c, upper_c = bound_c, theta_c_init
        else:
            lower_c, upper_c = theta_c_init, bound_c

        def Mp_func(c):
            out = find_moments(rotor, c, theta_s_init, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
            Mp = out["Mp"]
            print("Mp->",Mp,", c->",c)

            # Treat anything within tolerance as zero
            if abs(Mp) <= Mp_allow:
                return 0.0
            return Mp

        cyclic_c_trim = brentq(Mp_func, lower_c, upper_c)
    
    cyclic_s_trim=theta_s_init
    if skipMr==False:
        # Step 2: Solve cyclic_s for Mr = 0
        if Mr_init < 0:
            lower_s, upper_s = theta_s_init, bound_s
        else:
            lower_s, upper_s = bound_s, theta_s_init

        def Mr_func(s):
            out = find_moments(rotor, cyclic_c_trim, s, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
            Mr = out["Mr"]
            print("Mr->",Mr,", s->",s)
        
            # Treat anything within tolerance as zero
            if abs(Mr) <= Mr_allow:
                return 0.0
            return Mr

        cyclic_s_trim = brentq(Mr_func, lower_s, upper_s)

    return cyclic_c_trim, cyclic_s_trim

    
def trim_cyclic(rotor, tol_mode,
        theta_c_init, theta_s_init, coll,
        alpha_tpp, V_inf, b, rho, t_horizon_s
    ):

    I_values=compute_MoI(fuselage,payload)
    I_roll=I_values["I_x"]
    I_pitch=I_values["I_y"]
    theta_max_deg=5.0
    omega_max=1

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
    print("Mp_allow:",M_pitch_allow,"Mr_allow:",M_roll_allow)

    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    C_tip = rotor["chord_tip"]
    C_root = rotor["chord_root"]

    Omega = engine["omega"]
    a = rotor_aero["Cl_alpha"]

    B1c=alpha_tpp
    B_dot=B1c*Omega
    B0_old=0

    mu =  V_inf*np.cos(alpha_tpp)/ (Omega * R_tip)  # advance ratio
    I = rho*a*(C_tip + C_root)*R_root**4/(2*rotor["lock_number"] ) # Lock number

    Lambda_induced_forward = lambda r,sigh: lambda_i_forward(mu,r,R_tip,sigh,alpha_tpp,Omega,rho,V_inf)
    B_fn = lambda sigh: B0_old + B1c*np.cos(sigh)

    v_fn = lambda r,sigh: induced_velocity_forward(Lambda_induced_forward(r,sigh),Omega, R_tip, V_inf, alpha_tpp)
    phi_fn = lambda r,sigh: compute_phi_forward(V_inf,v_fn(r,sigh),Omega,alpha_tpp,sigh,r,B_fn(sigh),B_dot,R_root)

    c_fn = lambda r: chord_r(rotor,r)
    AR = (R_tip - R_root) / ((rotor["chord_root"] + rotor["chord_tip"]) / 2)
    
    out = find_moments(rotor, theta_c_init, theta_s_init, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
    Mp_init,Mr_init=out["Mp"],out["Mr"]

    skipMp=False
    skipMr=False
    if abs(Mp_init)<=abs(M_pitch_allow):
        skipMp=True
    if abs(Mr_init)<=abs(M_roll_allow):
        skipMr=True

    end_run=True
    cyclic_c_trim,cyclic_s_trim=theta_c_init,theta_s_init

    if skipMp==False or skipMr==False:
        end_run=False
        bound_c,bound_s=find_cyclic_bounds(
            Mp_init, Mr_init,
            theta_c_init, theta_s_init, coll,
            alpha_tpp, I, Omega, V_inf, B_dot,
            b, rho, R_root, R_tip,
            v_fn, c_fn, phi_fn
        )
        print("bound_c:",bound_c,"bound_s:",bound_s)

        cyclic_c_trim,cyclic_s_trim=find_cyclic(
                bound_c, bound_s, skipMp, skipMr,
                theta_c_init, theta_s_init, coll,
                Mp_init, Mr_init, M_pitch_allow, M_roll_allow,
                alpha_tpp, I, Omega, V_inf, B_dot,
                b, rho, R_root, R_tip,
                v_fn, c_fn, phi_fn
        )

    Cl_fn = lambda r, sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha=pitch_x_forward(rotor, r, sigh, cyclic_c_trim, cyclic_s_trim, coll) - phi_fn(r, sigh),
        alpha_stall=rotor_aero["alpha_stall"]
    )
    B_fn,B0_final=beta_fn(rho, alpha_tpp, Omega, V_inf, v_fn, c_fn, Cl_fn, R_root, R_tip, I)
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
        c_fn=c_fn, Cl_fn=Cl_fn, v_fn=v_fn,
        phi_fn=phi_fn, Cd_fn=Cd_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter = 4, tol=1e-3
    )

    res["cyclic_c"]   = cyclic_c_trim
    res["cyclic_s"]   = cyclic_s_trim
    res["collective"] = coll
    res["end_run"]    = end_run
    res["end_T"]    = False
    res["B0_final"]   = B0_final
    print("Converged B0 (coning angle):", B0_final)
    return res

