import numpy as np
from helper_functions import *
from Params import rotor_aero,engine
from calc_functions import beta_fn
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


def find_thrust(rotor, cyclic_c, cyclic_s, coll, R_root, R_tip, alpha_tpp,
                I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn 
                ):
    theta_fn = lambda r,sigh: pitch_x_forward(rotor, r, sigh, cyclic_c, cyclic_s, coll) 
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
    res["B0_final"]=B0_final
    return res


def balance_thrust(rotor, flight_condition, Omega, alpha_tpp, cyclic_c, cyclic_s, coll_init, TOGW, tol1=5e-3):
    b = rotor["b"]
    altitude = flight_condition["altitude"]
    rho = atmosphere(altitude, delta_ISA=flight_condition["delta_ISA"])["rho"]
    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    C_tip = rotor["chord_tip"]
    C_root = rotor["chord_root"]
    V_inf = flight_condition["velocity"][0]

    Omega = engine["omega"]
    a = rotor_aero["Cl_alpha"]

    B1c = alpha_tpp
    B_dot = B1c*Omega 
    B0_old=0

    mu =  V_inf*np.cos(alpha_tpp)/ (Omega * R_tip)  # advance ratio
    I = rho*a*(C_tip + C_root)*R_root**4/(2*rotor["lock_number"] ) # Lock number

    Lambda_induced_forward = lambda r, sigh: lambda_i_forward(mu, r, R_tip, sigh, alpha_tpp, Omega, rho, V_inf)
    B_fn = lambda sigh: B0_old + B1c*np.cos(sigh)

    v_fn = lambda r,sigh: induced_velocity_forward(Lambda_induced_forward(r,sigh), Omega, R_tip, V_inf,alpha_tpp)
    phi_fn = lambda r,sigh: compute_phi_forward(V_inf, v_fn(r,sigh), Omega, alpha_tpp,sigh,r,B_fn(sigh),B_dot,R_root)

    c_fn = lambda r: chord_r(rotor,r)

    res=find_thrust(rotor, cyclic_c, cyclic_s, coll_init, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
    if res["stall_status"]==1:
        return res

    T_init=res["T"]*np.cos(alpha_tpp)/9.8
    print(T_init)
    if abs(T_init-TOGW)/TOGW <= tol1:
        res["cyclic_c"]   = cyclic_c
        res["cyclic_s"]   = cyclic_s
        res["collective"] = coll_init
        res["end_run"]    = False
        res["end_T"]      = True
        return res
    
    coll_bound=coll_init
    if T_init-TOGW < 0:
        max_iter=200
        iter=1
        while T_init-TOGW < 0:
            if iter>max_iter:
                coll_bound+=5
                break
            print("here")
            coll_bound+=5
            res=find_thrust(rotor, cyclic_c, cyclic_s, coll_bound, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
            if res["stall_status"]==1:
                coll_bound-=5.005
            else:
                T_init=res["T"]*np.cos(alpha_tpp)/9.8
            iter+=1
        coll_lower,coll_upper=coll_init,coll_bound
    else:
        while T_init-TOGW > 0:
            print("Here")
            coll_bound-=2.5
            res=find_thrust(rotor, cyclic_c, cyclic_s, coll_bound, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn)
            T_init=res["T"]*np.cos(alpha_tpp)/9.8
        coll_lower,coll_upper=coll_bound,coll_init

    def T_func(coll):
        res = find_thrust(rotor, cyclic_c, cyclic_s, coll, R_root, R_tip, alpha_tpp, I, Omega, b, rho, V_inf, B_dot, phi_fn, v_fn, c_fn,)
        T = res["T"]*np.cos(alpha_tpp)/9.8
        print("T->",T,", coll->",coll)
    
        # Treat anything within tol1erance as zero
        if abs(T-TOGW)/TOGW <= tol1:
            return 0.0
        return T-TOGW
    
    try:
        collective = brentq(T_func, coll_lower, coll_upper)
    except:
        return {"stall_status": 1}
        
    Cl_fn = lambda r, sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha=pitch_x_forward(rotor, r, sigh, cyclic_c, cyclic_s, collective) - phi_fn(r, sigh),
        alpha_stall=rotor_aero["alpha_stall"]
    )
    B_fn,B0_final=beta_fn(rho, alpha_tpp, Omega, V_inf, v_fn, c_fn, Cl_fn, R_root, R_tip, I)
    Cd_fn = lambda r,sigh: airfoil_drag(
        Cd0=rotor_aero["Cd0"],
        Cl=Cl_fn(r,sigh),
        e=rotor_aero["e"],
        AR = (R_tip - R_root) / ((rotor["chord_root"] + rotor["chord_tip"]) / 2)
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
    res["cyclic_c"]   = cyclic_c
    res["cyclic_s"]   = cyclic_s
    res["collective"] =collective
    res["end_run"]    = False
    res["end_T"]      = True
    res["B0_final"]   = B0_final

    print("Converged B0 (coning angle):", B0_final)
    return res
