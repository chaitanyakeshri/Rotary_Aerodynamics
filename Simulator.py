import numpy as np
import pandas as pd
from Solver import *
from helper_functions import *
from Params import *
from Balance_main_and_tail import balance_main_and_tail


def Sim_Start_Hover_Climb(rotor, rotor_aero, engine, flight_condition):
    
    b = rotor["b"]
    altitude = flight_condition["altitude"]
    rho = atmosphere(altitude, delta_ISA=flight_condition["delta_ISA"])["rho"]
    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    C_tip = rotor["chord_tip"]
    C_root = rotor["chord_root"]
    Omega = engine["omega"]


    sigma = solidity(b, C_root,C_tip, R_tip)
    a = rotor_aero["Cl_alpha"]

    V_val = flight_condition["velocity"][2] # vertical velocity component (w)
    lambda_c =  V_val/ (Omega * R_tip)  # advance ratio

    # Pitch angle as a function of r
    theta_fn = lambda r: pitch_x(rotor, r)

    # Inflow ratio lambda as a function of r, with tip loss factor
    Lambda_tiploss = lambda r: solve_lambda_tiploss(flight_condition, sigma, a, b, theta_fn(r), r, lambda_c, R_tip, R_root)
    lembda_fn = lambda r: Lambda_tiploss(r)[0]
    F_fn = lambda r: Lambda_tiploss(r)[1]
    
    #[print(Lambda_tiploss(r)) for r in np.linspace(R_root, R_tip, 5)] # debug print
    
    # Induced velocity as a function of r
    v_fn = lambda r: induced_velocity(lembda_fn(r), Omega, R_tip, V_val)
 
    # Inflow angle phi as a function of r
    phi_fn = lambda r: compute_phi(V_val, v_fn(r), Omega, r, R_root)

    # Chord as a function of r
    c_fn = lambda r: chord_r(rotor,r)

    

    # Lift coefficient as a function of r
    Cl_fn = lambda r: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha = theta_fn(r) - phi_fn(r),
        alpha_stall=rotor_aero["alpha_stall"]

    )

    # Aspect ratio estimate for drag calculation?
    AR = (R_tip - R_root) / ((C_root + C_tip) / 2)

    # Drag coefficient as a function of r
    Cd_fn = lambda r: airfoil_drag(
        Cd0=rotor_aero["Cd0"],
        Cl=Cl_fn(r),
        e=rotor_aero["e"],
        AR=AR
    )


    # Check stall across all radii (not just samples)
    r_all = np.linspace(R_root, R_tip, 200)   # dense discretization
    for r in r_all:
        alpha = theta_fn(r) - phi_fn(r)
        if alpha > rotor_aero["alpha_stall"]:
            print(f"Stall detected at r = {r:.3f} m, alpha = {alpha:.3f} deg")
            return ({"stall_status": 1, "r": r, "alpha": alpha},)
        
        
    # Now run the iterative thrust calculation
    res = iterative_solver_hover_climb(
        b=b, rho=rho,
        Ut_fn=lambda r: Omega * r,
        Up_fn=lambda r: V_val + v_fn(r),
        c_fn=c_fn, Cl_fn=Cl_fn, phi_fn=phi_fn, Cd_fn=Cd_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 200, max_iter= 100, tol=1e-3
    )

    # Print scalar variables
    print("b =", b)
    print("rho =", rho)
    print("altitude =", flight_condition["altitude"])
    print("R_tip =", R_tip)
    print("R_root =", R_root)
    print("Omega =", Omega)
    print("sigma =", sigma)
    print("a =", a)
    print("V_val =", V_val)
    print("AR =", AR)

    # Tabulate sample values at selected radial positions
    sample_r = np.linspace(R_root, R_tip, 10)
    table = []

    for r in sample_r:
        table.append({
            "r (m)": round(r, 3),
            "theta (deg)": round(theta_fn(r), 3),
            "F (tip loss)": round(F_fn(r), 4),
            "lambda": round(lembda_fn(r), 5),
            "v_induced (m/s)": round(v_fn(r), 4),
            "phi (deg)": round(phi_fn(r), 3),
            "chord (m)": round(c_fn(r), 4),
            "Cl": round(Cl_fn(r), 4),
            "Cd": round(Cd_fn(r), 5)
        })


    df = pd.DataFrame(table)
    print("\nSample values at selected r:")
    print(df.to_string(index=False))

    # # Tabulate arrays from iterative_thrust result (first 5 values for brevity)
    # tab_array = pd.DataFrame({
    #     "r (m)": res["r"],
    #     "dT_dr_per_blade": res["dT_dr_per_blade"],
    #     "dD_dr_per_blade": res["dD_dr_per_blade"],
    #     "dQ_dr_per_blade": res["dQ_dr_per_blade"],
    #     "dP_dr_per_blade": res["dP_dr_per_blade"],
    #     "q": res["q"]
    # })
    # #print("\nElemental values along blade (first 5 rows):")
    # #print(tab_array.head().to_string(index=False))
    # # Print histories and converged values

    print("\nConverged T (Thrust) =", res["T"], "N")
    print("Converged D (Drag)   =", res["D"], "N")
    print("Converged Q (Torque) =", res["Q"], "Nm")
    print("Converged P (Power)  =", res["P"], "W")

    # print("Grid points used  =", res["N"])
    # if "warning" in res:
    #     print("Warning:", res["warning"])
        
        
    return (res,lembda_fn, v_fn, phi_fn, c_fn, Cl_fn, Cd_fn)



"""
-------------Forward Simulation Start----------------
"""


def Sim_Start_Forward(rotor, rotor_aero, engine, flight_condition, t_horizon_s):

    b = rotor["b"]
    #sigh = np.linspace(0, 2*np.pi, 360)  # azimuth angles for flapping
    altitude = flight_condition["altitude"]
    rho = atmosphere(altitude, delta_ISA=flight_condition["delta_ISA"])["rho"]
    R_tip  = rotor["Rt"]
    R_root = rotor["Rr"]
    C_tip = rotor["chord_tip"]
    C_root = rotor["chord_root"]
    Omega = engine["omega"]
    TOGW = fuselage["Empty_Weight"] + payload["weight"] + flight_condition["fuel_weight"]

    
    

    sigma = solidity(b, C_root,C_tip, R_tip)
    a = rotor_aero["Cl_alpha"]

    V_inf = flight_condition["velocity"][0] # horizontal velocity component (w)

    drag = fuselage_drag(fuselage, rho, flight_condition["velocity"])

    alpha_tpp = alphaTPP(drag,TOGW)    #in radians
    B1c = alpha_tpp 
    B_dot = B1c*Omega  
  
    
    B0_old = 0. ## Assuming no coning for now

    mu =  V_inf*np.cos(alpha_tpp)/ (Omega * R_tip)  # advance ratio

    I = rho*a*(C_tip + C_root)*R_root**4/(2*rotor["lock_number"] ) # Lock number

    # Pitch angle as a function of r
    theta_fn = lambda r,sigh: pitch_x_forward(rotor, r,sigh) 

    # Inflow ratio lambda as a function of r, with tip loss factor
    Lambda_induced_forward = lambda r, sigh: lambda_i_forward(mu, r, R_tip, sigh, alpha_tpp, Omega, rho, V_inf)

    
    #[print(Lambda_tiploss(r)) for r in np.linspace(R_root, R_tip, 5)] # debug print
    
    # Induced velocity as a function of r
    v_fn = lambda r,sigh: induced_velocity_forward(Lambda_induced_forward(r,sigh), Omega, R_tip, V_inf,alpha_tpp)
    B_fn = lambda sigh: B0_old + B1c*np.cos(sigh)
 
    # Inflow angle phi as a function of r
    phi_fn = lambda r,sigh: compute_phi_forward(V_inf, v_fn(r,sigh), Omega, alpha_tpp,sigh,r,B_fn(sigh),B_dot,R_root)

    # Chord as a function of r
    c_fn = lambda r: chord_r(rotor,r)

    #alpha_eff = theta_fn(r,sigh) - phi_fn(r,sigh)

    # Lift coefficient as a function of r
    Cl_fn = lambda r,sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha = theta_fn(r,sigh) - phi_fn(r,sigh),
        alpha_stall=rotor_aero["alpha_stall"]

    )

    # Aspect ratio estimate for drag calculation?
    AR = (R_tip - R_root) / ((C_root + C_tip) / 2)

    # Drag coefficient as a function of r
    Cd_fn = lambda r,sigh: airfoil_drag(
        Cd0=rotor_aero["Cd0"],
        Cl=Cl_fn(r,sigh),
        e=rotor_aero["e"],
        AR=AR
    )
    
    # Check stall across all radii (not just samples)
    # Check stall across all radii and all azimuth angles (sigh)
    r_all = np.linspace(R_root, R_tip, 200)       # dense discretization in radius
    sigh_all = np.linspace(0, 2*np.pi, 360)       # dense discretization in azimuth

    for r in r_all:
        for sigh_val in sigh_all:
            alpha = theta_fn(r, sigh_val) - phi_fn(r, sigh_val)
            if alpha > rotor_aero["alpha_stall"]:
                print(f"Stall detected in tail rotor at r = {r:.3f} m, sigh = {sigh_val:.3f} rad, alpha = {alpha:.3f} deg")
                return {"stall_status": 1, "r": r, "sigh": sigh_val, "alpha": alpha}

    # # --- Coning angle iteration ---
    tolerance = 1e-2
    max_iter = 7
    iter_count = 0

    while True:
        # Define coning angle function with current guess
        B_fn = lambda sigh: B0_old + B1c * np.cos(sigh)

        # Solve for new coning angle
        B0_new = Coning_Angle_Solver(
            rho=rho,
            Ut_fn=lambda r, sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
            Up_fn=lambda r, sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot
                                + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
            c_fn=c_fn, Cl_fn=Cl_fn,
            R_root=R_root, R_tip=R_tip, I=I, Omega=Omega,
            N0=50, max_iter=2, tol=1e-3
        )
        print(B0_new)

        # Check convergence
        if abs(B0_new - B0_old) < tolerance:
            break

        # Update and continue
        B0_old = B0_new
        iter_count += 1
        if iter_count >= max_iter:
            print("Warning: B0 iteration did not converge")
            break

    # Final converged value
    B0_final = B0_new
    print("Converged B0 (coning angle):", B0_final)
    B_fn = lambda sigh: B0_final + B1c*np.cos(sigh)

        
    # Now run the first iterative thrust calculation
    res = iterative_solver_forward(
        b=b, rho=rho,
        Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
        Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
        c_fn=c_fn, Cl_fn=Cl_fn, 
        phi_fn=phi_fn, Cd_fn=Cd_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter = 4, tol=1e-3
    )

    out=balance_main_and_tail(
        res["Mp"], res["Mr"],
        alpha_tpp, B1c, V_inf, B0_final,
        b, rho, t_horizon_s, max_iter = 3
    )
    if out["stall_status"]==1:
        print("Mission not possible due to tail rotor stall.")
        return
    
    theta_fn = lambda r,sigh: pitch_x_forward(rotor, r,sigh, out["cyclic_c"], out["cyclic_s"]) 
    Cl_fn = lambda r,sigh: airfoil_lift(
        Cl0=rotor_aero["Cl0"],
        Cl_alpha=rotor_aero["Cl_alpha"],
        alpha0=rotor_aero["alpha0"],
        alpha = theta_fn(r,sigh) - phi_fn(r,sigh),
        alpha_stall=rotor_aero["alpha_stall"]

    )
    res = iterative_solver_forward(
        b=b, rho=rho,
        Ut_fn=lambda r,sigh: Omega * r + V_inf*np.cos(alpha_tpp)*np.sin(sigh),
        Up_fn=lambda r,sigh: V_inf*np.sin(alpha_tpp) + v_fn(r,sigh) + r*B_dot + V_inf*np.sin(B_fn(sigh))*np.cos(sigh),
        c_fn=c_fn, Cl_fn=Cl_fn, 
        phi_fn=phi_fn, Cd_fn=Cd_fn,
        R_root=R_root, R_tip=R_tip,
        N0 = 25, max_iter = 5, tol=5e-4
    )

    # Print scalar parameters
    print("Scalar parameters:")
    for name, val in [("b", b), ("rho", rho), ("altitude", flight_condition["altitude"]),
                    ("R_tip", R_tip), ("R_root", R_root), ("Omega", Omega),
                    ("sigma", sigma), ("a", a), ("V_inf", V_inf), ("AR", AR),("TOGW", TOGW),('alpha_tpp',alpha_tpp),('B_dot',B_dot),('B1c',B1c),('mu',mu)]:
        print(f"{name} = {val}")

    # Tabulate sample values at selected radial and azimuth positions
    import itertools
    sample_r = np.linspace(R_root, R_tip, 10)
    sample_sigh = np.linspace(0, 2*np.pi, 10)
    table = []

    for r, sigh in itertools.product(sample_r, sample_sigh):
        table.append({
            "r (m)": round(r, 3),
            "theta (deg)": round(theta_fn(r, sigh), 3),
            "lambda": round(Lambda_induced_forward(r, sigh), 5),
            "v_induced (m/s)": round(v_fn(r, sigh), 4),
            "phi (deg)": round(phi_fn(r, sigh), 3),
            "chord (m)": round(c_fn(r), 4),
            "Cl": round(Cl_fn(r, sigh), 4),
            "Cd": round(Cd_fn(r, sigh), 5)
        })

    df = pd.DataFrame(table)
    print("\nSample values at selected r and azimuth positions:")
    print(df.to_string(index=False))

    # --- Print induced velocity at r = 6 m over full azimuth ---
    r_fixed = 6.0
    phi_vals = np.linspace(0, 2*np.pi, 20)  # adjust number of points for resolution
    vi_table = []

    for phi in phi_vals:
        vi_table.append({
            "phi (deg)": round(np.degrees(phi), 2),
            "v_induced (m/s)": round(v_fn(r_fixed, phi), 5)
        })

    vi_df = pd.DataFrame(vi_table)
    print(f"\nInduced velocity at r = {r_fixed} m for phi = 0 to 360 deg:")
    print(vi_df.to_string(index=False))
    # --- End of induced velocity table ---


    # Tabulate elemental values along the blade
    # Use mean over azimuth for 2D arrays
    # --- Elemental distributions (mean over azimuth for q) ---
    q_1d = np.mean(res["q"], axis=1)

    tab_array = pd.DataFrame({
        "R (m)": res["R"],
        "dT_dr_per_blade": res["dT_dr_per_blade"],
        "dD_dr_per_blade": res["dD_dr_per_blade"],
        "dQ_dr_per_blade": res["dQ_dr_per_blade"],
        "dP_dr_per_blade": res["dP_dr_per_blade"],
        "q (mean over azimuth)": q_1d
    })

    # print("\nElemental values along the blade (first 5 radial points):")
    # print(tab_array.head().to_string(index=False))

    # --- Print converged scalar outputs (including new items) ---
    print("\nConverged results:")
    print(f"T (Thrust)        = {res['T']:.4f} N")
    print(f"L (Lift)          = {res['T']* np.cos(alpha_tpp)} N")
    print(f"D (Fuselgae)      = {res['T']* np.sin(alpha_tpp)}) N")
    print(f"D (Drag)          = {res['D']:.4f} N")
    print(f"Q (Torque)        = {res['Q']:.4f} Nm")
    print(f"P (Power)         = {res['P']:.4f} W")
    print(f"Rolling moment    = {res['Mr']:.4f} Nm")
    print(f"Pitching moment   = {res['Mp']:.4f} Nm")

    print(f"cyclic_c:          = {out['cyclic_c']:.2f} deg")
    print(f"cyclic_s:          = {out['cyclic_s']:.2f} deg")
    print(f"Tail collective    = {out['tail_coll']:.2f} deg")

    # # --- Optional: print convergence histories ---
    # print("\nHistory of convergence:")
    # print("Thrust (T):", res["history_T"])
    # print("Drag (D):", res["history_D"])
    # print("Torque (Q):", res["history_Q"])
    # print("Power (P):", res["history_P"])
    # print("Ind. Drag (Di):", res["history_Di"])
    # print("Ind. Torque (Qi):", res["history_Qi"])
    # print("Ind. Power (Pi):", res["history_Pi"])
    # print("Rolling moment:", res["history_rolling_moment"])
    # print("Pitching moment:", res["history_pitching_moment"])

    # --- Warning if not converged ---
    if "warning" in res:
        print("\nWarning:", res["warning"])
