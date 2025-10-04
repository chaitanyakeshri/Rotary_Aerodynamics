import numpy as np
import pandas as pd
from typing import Callable, Dict
from Solver import *
from helper_functions import *
from Params import *


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

    
def find_collective(collective_vals,R_root,R_tip,Thrust,
                    phi_fn,v_fn,c_fn,Cd_fn,
                    b,rho,Omega,V_inf,alpha_tpp,increase_coll):
    
    for coll in collective_vals:
        print("Current coll:",coll)
        theta_fn = lambda r: tail_pitch_x_forward(tail_rotor, r,coll)

        stall = stall_check(R_root,R_tip,theta_fn,phi_fn)
        if stall["stall_status"]==1:
            stall["tail_coll"]=coll
            return stall

        Cl_fn = lambda r,sigh: airfoil_lift(
            Cl0=tail_rotor_aero["Cl0"],
            Cl_alpha=tail_rotor_aero["Cl_alpha"],
            alpha0=tail_rotor_aero["alpha0"],
            alpha = theta_fn(r) - phi_fn(r,sigh),
            alpha_stall=tail_rotor_aero["alpha_stall"]
        )

        res = iterative_solver_forward(
            b=b, rho=rho,
            Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
            Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh),
            c_fn=c_fn, Cl_fn=Cl_fn, phi_fn=phi_fn, Cd_fn=Cd_fn,
            R_root=R_root, R_tip=R_tip,
            N0 = 25, max_iter= 3, tol=1e-3
        )

        if res["T"]>Thrust and increase_coll==True:
            return coll,res
        if res["T"]<Thrust and increase_coll==False:
            return coll,res
    
    
def Simulate_tail_rotor(tail_rotor, tail_rotor_aero, engine, flight_condition, coll_init=tail_rotor["collective"], Q_rotor=78732):
    
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
    
    # Pitch angle as a function of r
    theta_fn = lambda r: tail_pitch_x_forward(tail_rotor, r, coll_init)

    # Lift coefficient as a function of r
    Cl_fn = lambda r,sigh: airfoil_lift(
        Cl0=tail_rotor_aero["Cl0"],
        Cl_alpha=tail_rotor_aero["Cl_alpha"],
        alpha0=tail_rotor_aero["alpha0"],
        alpha = theta_fn(r) - phi_fn(r,sigh),
        alpha_stall=tail_rotor_aero["alpha_stall"]
    )

    # Aspect ratio estimate for drag calculation?
    AR = (R_tip - R_root) / ((C_root + C_tip) / 2)

    # Drag coefficient as a function of r
    Cd_fn = lambda r,sigh: airfoil_drag(
        Cd0=tail_rotor_aero["Cd0"],
        Cl=Cl_fn(r,sigh),
        e=tail_rotor_aero["e"],
        AR=AR
    )

    # Check stall across all radii (not just samples)
    # Check stall across all radii and all azimuth angles (sigh)
    stall = stall_check(R_root, R_tip, theta_fn,phi_fn)
    if stall["stall_status"]==1:
        stall["tail_coll"]=coll_init
        return stall
        
    # Now run the first iterative thrust calculation
    res = iterative_solver_forward(
            b=b, rho=rho,
            Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
            Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh),
            c_fn=c_fn, Cl_fn=Cl_fn, phi_fn=phi_fn, Cd_fn=Cd_fn,
            R_root=R_root, R_tip=R_tip,
            N0 = 25, max_iter= 4, tol=1e-3
        )
    
    increase_coll=True
    print("Thrust needed by tail:",Thrust)

    # Get tail collective
    if res["T"]<Thrust:
        if coll_init==tail_rotor["collective"]:
            collective_vals=np.arange(coll_init,20,0.5)
            collective,res=find_collective(collective_vals,R_root,R_tip,Thrust,
                    phi_fn,v_fn,c_fn,Cd_fn,
                    b,rho,Omega,V_inf,alpha_tpp,increase_coll)
        
            if res["stall_status"]==1:
                return res
            
            coll_vals=np.arange(collective-0.5,collective+0.55,0.05)
            collective,res=find_collective(coll_vals,R_root,R_tip,Thrust,
                        phi_fn,v_fn,c_fn,Cd_fn,
                        b,rho,Omega,V_inf,alpha_tpp,increase_coll)
        else:
            collective_vals=np.arange(coll_init+0.05,coll_init+5.05,0.05)

            collective,res=find_collective(collective_vals,R_root,R_tip,Thrust,
                    phi_fn,v_fn,c_fn,Cd_fn,
                    b,rho,Omega,V_inf,alpha_tpp,increase_coll)
    else:
        increase_coll=False
        collective_vals=np.arange(coll_init-0.05,coll_init-5.05,-0.05)

        collective,res=find_collective(collective_vals,R_root,R_tip,Thrust,
                    phi_fn,v_fn,c_fn,Cd_fn,
                    b,rho,Omega,V_inf,alpha_tpp,increase_coll)
    
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

    res["tail_coll"]=collective
    return res


# Simulate_tail_rotor(tail_rotor,tail_rotor_aero,engine, flight_condition, )