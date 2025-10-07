import numpy as np
import pandas as pd
from typing import Callable, Dict
from Solver import *
from helper_functions import *
from Params import tail_rotor_aero,vertical_stabilizers
from scipy.optimize import brentq

import math

def yaw_moment(delE, rho, V_inf, S, l, Cd=0.01):

    Cl_alpha=vertical_stabilizers["Cl_alpha"]
    Cl0=vertical_stabilizers["Cl0"]
    S = vertical_stabilizers["verti_area"]
    
    x_arm = vertical_stabilizers["x_arm"]
    z_arm = vertical_stabilizers["z_arm"]

    Cl = Cl0 + Cl_alpha * delE
    L = 0.5 * rho * V_inf**2 * S * Cl
    yaw = L * x_arm
    roll = L * z_arm

    return yaw,roll

delE = math.radians(10)
rho = 1.225
V = 50
S = 2.0
l = 10.0

moment = yaw_moment(delE, rho, V, S, l)
print(f"Yaw moment: {moment:.2f} N·m")

def lambda_i_tail_forward(mu, r, R, sigh,alpha_TPP,Omega,rho,Vinf,Thrust):

    lambdaig = LambdaIG(Ct(Thrust,rho,Omega,R),mu,alpha_TPP)
    lambda_G = LambdaG(Vinf,lambdaig,Omega,R)

    frac = ( (4/3) * (mu / lambda_G) ) / (1.2 + (mu / lambda_G))
    correction = frac * (r / R) * np.cos(sigh)

    return lambdaig * (1 + correction)


def stall_check(R_root,R_tip,theta_fn,phi_fn):
    r_all = np.linspace(R_root, R_tip, 80)       # dense discretization in radius
    sigh_all = np.linspace(0, 2*np.pi, 90)       # dense discretization in azimuth

    for r in r_all:
        for sigh_val in sigh_all:
            alpha = theta_fn(r) - phi_fn(r, sigh_val)
            if alpha > tail_rotor_aero["alpha_stall"]:
                print(f"Stall detected for tail rotor at r = {r:.3f} m, sigh = {sigh_val:.3f} rad, alpha = {alpha:.3f} deg")
                return {"stall_status": 1, "r": r, "sigh": sigh_val, "alpha": alpha}
    
    return {"stall_status": 0}


def find_thrust(tail_rotor, tail_coll, R_root, R_tip, C_root, C_tip, alpha_tpp,
                 Omega, b, rho, V_inf, phi_fn, v_fn, c_fn):
    
    theta_fn = lambda r: tail_pitch_x_forward(tail_rotor, r, tail_coll)
    Cl_fn = lambda r,sigh: airfoil_lift(
            Cl0=tail_rotor_aero["Cl0"],
            Cl_alpha=tail_rotor_aero["Cl_alpha"],
            alpha0=tail_rotor_aero["alpha0"],
            alpha = theta_fn(r) - phi_fn(r,sigh),
            alpha_stall=tail_rotor_aero["alpha_stall"]
    )

    AR = (R_tip - R_root) / ((C_root + C_tip) / 2)
    Cd_fn = lambda r,sigh: airfoil_drag(
        Cd0=tail_rotor_aero["Cd0"],
        Cl=Cl_fn(r,sigh),
        e=tail_rotor_aero["e"],
        AR=AR
    )
    stall=stall_check(R_root,R_tip,theta_fn,phi_fn)
    if stall["stall_status"]==1:
        return stall
    
    res = iterative_solver_forward(
            b=b, rho=rho,
            Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
            Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh),
            c_fn=c_fn, Cl_fn=Cl_fn, v_fn=v_fn, phi_fn=phi_fn, Cd_fn=Cd_fn,
            R_root=R_root, R_tip=R_tip,
            N0 = 25, max_iter= 4, tol=1e-2
    )
    return res
    
    
def Simulate_tail_rotor(tail_rotor, engine, flight_condition, coll_init, Q_rotor=78732, tol2=1e-3):
    
    b = tail_rotor["b"]
    altitude = flight_condition["altitude"]
    rho = atmosphere(altitude, delta_ISA=flight_condition["delta_ISA"])["rho"]
    R_tip  = tail_rotor["Rt"]
    R_root = tail_rotor["Rr"]
    C_tip = tail_rotor["chord_tip"]
    C_root = tail_rotor["chord_root"]
    Omega = engine["omega"]


    V_inf = flight_condition["velocity"][0] # horizontal velocity component (w)
    arm_len=tail_rotor['arm_length']
    Thrust = Q_rotor/arm_len
    
    alpha_tpp = 0    #in radians
    B1c = 0 
    B_dot = 0

    mu =  V_inf*np.cos(alpha_tpp)/ (Omega * R_tip)  # advance ratio


    # Inflow ratio lambda as a function of r, sigh
    Lambda_induced_forward = lambda r, sigh: lambda_i_tail_forward(mu, r, R_tip, sigh, alpha_tpp, Omega, rho, V_inf,Thrust)
    B_fn = 0

    # Induced velocity as a function of r,sigh
    v_fn = lambda r,sigh: induced_velocity_forward(Lambda_induced_forward(r,sigh), Omega, R_tip, V_inf,alpha_tpp)
 
    # Inflow angle phi as a function of r,sigh
    phi_fn = lambda r,sigh: compute_phi_forward(V_inf, v_fn(r,sigh), Omega, alpha_tpp,sigh,r,B_fn,B_dot,R_root)

    # Chord as a function of r
    c_fn = lambda r: chord_r(rotor,r)
    
    res=find_thrust(tail_rotor, coll_init, R_root, R_tip, C_root, C_tip, alpha_tpp,
                 Omega, b, rho, V_inf, phi_fn, v_fn, c_fn)
    if res["stall_status"]==1:
        return res
    
    print("Thrust needed by tail:",Thrust)
    T_init=res["T"]
    print(T_init)
    coll_bound=coll_init

    if T_init-Thrust < 0:
        iter=1
        max_iter=200
        while T_init-Thrust < 0:
            if iter>max_iter:
                coll_bound+=4.5
                break

            coll_bound+=5
            res=find_thrust(tail_rotor, coll_bound, R_root, R_tip, C_root, C_tip, alpha_tpp,
                            Omega, b, rho, V_inf, phi_fn, v_fn, c_fn)
            if res["stall_status"]==1:
                coll_bound-=5.005
            else:
                T_init=res["T"]
            iter+=1
        coll_lower,coll_upper=coll_init,coll_bound
    else:
        while T_init-Thrust > 0:
            coll_bound-=2.5
            res=find_thrust(tail_rotor, coll_bound, R_root, R_tip, C_root, C_tip, alpha_tpp,
                            Omega, b, rho, V_inf, phi_fn, v_fn, c_fn)
            T_init=res["T"]
        coll_lower,coll_upper=coll_bound,coll_init


    def T_func(coll):
        res = find_thrust(tail_rotor, coll, R_root, R_tip, C_root, C_tip, alpha_tpp,
                            Omega, b, rho, V_inf, phi_fn, v_fn, c_fn)
        T = res["T"]
        print("T->",T,", coll->",coll)
    
        # Treat anything within tolerance as zero
        if abs(T-Thrust)/Thrust <= tol2:
            return 0.0
        return T-Thrust
    
    try:
        collective = brentq(T_func, coll_lower, coll_upper)
    except:
        return {"stall_status": 1}
        
    
    res = find_thrust(tail_rotor, collective, R_root, R_tip, C_root, C_tip, alpha_tpp,
                Omega, b, rho, V_inf, phi_fn, v_fn, c_fn)
    res["tail_collective"]=collective
    
    if res["stall_status"]==1:
        return res

    # --- Print converged scalar outputs (including new items) ---
    print("\nConverged results for tail rotor:")
    print(f"T (Thrust)        = {res['T']:.4f} N")
    print(f"L (Lift)          = {res['T']* np.cos(alpha_tpp)} N")
    print(f"D (Drag)          = {res['D']:.4f} N")
    print(f"Q (Torque)        = {res['Q']:.4f} Nm")
    print(f"P (Power)         = {res['P']:.4f} W")
    print(f"Rolling moment    = {res['Mr']:.4f} Nm")
    print(f"Pitching moment   = {res['Mp']:.4f} Nm")

    # --- Warning if not converged ---
    if "warning" in res:
        print("\nWarning:", res["warning"])


    return res


res=Simulate_tail_rotor(tail_rotor, engine, flight_condition, 0, Q_rotor=182693.8676, tol2=5e-3)
print(res["tail_collective"])