import numpy as np
from Solver import *

def beta_fn(rho, alpha_tpp, Omega, V_inf, v_fn, c_fn, Cl_fn, R_root, R_tip, I):
    # # --- Coning angle iteration ---
    B0_old=0
    B1c=alpha_tpp
    B_dot=B1c*Omega

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
            N0=50, max_iter=2, tol=1e-2
        )

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
    # print("Converged B0 (coning angle):", B0_final)
    B_fn = lambda sigh: B0_final + B1c*np.cos(sigh)

    return B_fn, B0_final