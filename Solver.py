import numpy as np
from typing import Callable, Dict
import pandas as pd

def iterative_solver_hover_climb(
    b: int,
    rho: float,
    Ut_fn: Callable[[float], float],
    Up_fn: Callable[[float], float],
    c_fn: Callable[[float], float],
    Cl_fn: Callable[[float], float],
    phi_fn: Callable[[float], float],
    Cd_fn: Callable[[float], float],
    R_root: float,
    R_tip: float,
    N0: int = 200,       
    max_iter: int = 10,
    tol: float = 1e-3   
) -> Dict:
    
    def compute_T_D_Q_P_on_grid(N: int):
        r = np.linspace(R_root, R_tip, N)
        Ut = np.array([Ut_fn(ri) for ri in r])
        Up = np.array([Up_fn(ri) for ri in r])
        c  = np.array([c_fn(ri)  for ri in r])
        Cl = np.array([Cl_fn(ri) for ri in r], dtype=float)
        phi = np.array([phi_fn(ri) for ri in r])
        Cd  = np.array([Cd_fn(ri) for ri in r])

        V2 = Ut**2 + Up**2
        q = 0.5 * rho * V2
        phi_r = np.radians(phi) 

        dT_dr_1b = q * c * (Cl * np.cos(phi_r) - Cd * np.sin(phi_r))
        dD_dr_1b = q * c * (Cl * np.sin(phi_r) + Cd * np.cos(phi_r))
        dQ_dr_1b = r * q * c * (Cd * np.cos(phi_r) + Cl * np.sin(phi_r))
        dP_dr_1b = (Ut / r) * dQ_dr_1b

        T = b * np.trapezoid(dT_dr_1b, r)
        D = b * np.trapezoid(dD_dr_1b, r)
        Q = b * np.trapezoid(dQ_dr_1b, r)
        P = b * np.trapezoid(dP_dr_1b, r)

        return T, D, Q, P, r, dT_dr_1b, dD_dr_1b, dQ_dr_1b, dP_dr_1b, q

    N = N0
    history_T, history_D, history_Q, history_P = [], [], [], []
    T_prev = D_prev = Q_prev = P_prev = None

    for it in range(max_iter):
        T, D, Q, P, r, dT_dr_1b, dD_dr_1b, dQ_dr_1b, dP_dr_1b, q = compute_T_D_Q_P_on_grid(N)

        history_T.append(T)
        history_D.append(D)
        history_Q.append(Q)
        history_P.append(P)

        if T_prev is not None:
            rel_err_T = abs((T - T_prev) / T_prev)
            rel_err_D = abs((D - D_prev) / D_prev)
            rel_err_Q = abs((Q - Q_prev) / Q_prev)
            rel_err_P = abs((P - P_prev) / P_prev)
            # check relative errors
            if max(rel_err_T, rel_err_Q, rel_err_P, rel_err_D) < tol:
                return {
                    "T": float(T), "D": float(D), "Q": float(Q), "P": float(P),
                    "N": N,
                    "r": r,
                    "dT_dr_per_blade": dT_dr_1b,
                    "dD_dr_per_blade": dD_dr_1b,
                    "dQ_dr_per_blade": dQ_dr_1b,
                    "dP_dr_per_blade": dP_dr_1b,
                    "q": q,
                    "history_T": history_T,
                    "history_D": history_D,
                    "history_Q": history_Q,
                    "history_P": history_P,
                    "stall_status": 0
                }

        T_prev, D_prev, Q_prev, P_prev = T, D, Q, P

    return {
        "T": float(T), "D": float(D), "Q": float(Q), "P": float(P),
        "N": N,
        "r": r,
        "dT_dr_per_blade": dT_dr_1b,
        "dD_dr_per_blade": dD_dr_1b,
        "dQ_dr_per_blade": dQ_dr_1b,
        "dP_dr_per_blade": dP_dr_1b,
        "q": q,
        "history_T": history_T,
        "history_D": history_D,
        "history_Q": history_Q,
        "history_P": history_P,
        "stall_status": 0,
        "warning": "Did not converge"
    }

def iterative_solver_forward(
    b: int,
    rho: float,
    Ut_fn: Callable[[float, float], float],
    Up_fn: Callable[[float, float], float],
    c_fn: Callable[[np.ndarray], np.ndarray],            # c_fn(R) -> 1D array of chord vs R
    Cl_fn: Callable[[float, float], float],
    phi_fn: Callable[[float, float], float],
    Cd_fn: Callable[[float, float], float],
    R_root: float,
    R_tip: float,
    N0: int = 25,
    max_iter: int = 5,
    tol: float = 1e-3
) -> Dict:

    def compute_T_D_Q_P_on_grid(N: int):
        # grids
        R = np.linspace(R_root, R_tip, N)
        SIGH = np.linspace(0, 2*np.pi, 90)
        R_grid, SIGH_grid = np.meshgrid(R, SIGH, indexing='ij')  # shape (N, 360)


        # allocate arrays for results
        Ut  = np.empty_like(R_grid)
        Up  = np.empty_like(R_grid)
        Cl  = np.empty_like(R_grid)
        phi = np.empty_like(R_grid)
        Cd  = np.empty_like(R_grid)

        # loop element-wise
        for i in range(R_grid.shape[0]):      # over radius
            for j in range(R_grid.shape[1]):  # over azimuth
                R_val = R_grid[i, j]
                sigh_val = SIGH_grid[i, j]

                Ut[i, j]  = Ut_fn(R_val, sigh_val)
                Up[i, j]  = Up_fn(R_val, sigh_val)
                phi[i, j] = phi_fn(R_val, sigh_val)
                Cl[i, j]  = Cl_fn(R_val, sigh_val)
                Cd[i, j]  = Cd_fn(R_val, sigh_val)

            # --- Print the 2D matrices ---
       
        # print("\nUt matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(Ut).round(4).to_string(index=False, header=True))

        # print("\nUp matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(Up).round(4).to_string(index=False, header=False))

        # print("\nCl matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(Cl).round(4).to_string(index=False, header=False))

        # print("\nphi matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(phi).round(4).to_string(index=False, header=False))

        # print("\nCd matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(Cd).round(5).to_string(index=False, header=False))



        # print("\nUt matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(Ut).round(4).to_string(index=False, header=False))

        # chord only varies with R -> produce 2D chord array
        c_1d = c_fn(R)                    # shape (N,)
        c = np.tile(c_1d[:, np.newaxis], (1, len(SIGH)))  # shape (N, 360)

        # --- Replace negative Ut with 0 ---
    
        # dynamic pressure with sign logic
        V2 = np.where(Ut < 0, -(Ut**2 + Up**2), (Ut**2 + Up**2))
        q = 0.5 * rho * V2


        # prevent division by zero for Ut and radius
        Ut_safe = np.where(np.abs(Ut) < 1e-12, 1e-12, Ut)
        R_safe_grid = np.where(np.abs(R_grid) < 1e-12, 1e-12, R_grid)

        # differential loads per blade element (per blade, 2D arrays)
        # differential loads per blade element (per blade, 2D arrays)
        dT_dr_dsigh_1b = q * c * Cl
        dD_total_dr_dsigh_1b = q * c * (Cd + Cl * Up / Ut_safe)  # D + Di
        dQ_total_dr_dsigh_1b = R_grid * dD_total_dr_dsigh_1b    # Q + Qi
        dP_total_dr_dsigh_1b = Ut * q * c * Cl * (Up / Ut_safe) + (Ut / R_safe_grid) * R_grid * q * c * Cd  # P + Pi

        # Moments remain the same
        rolling_moment_dr_dsigh = dT_dr_dsigh_1b * R_grid * np.sin(SIGH_grid)
        pitching_moment_dr_dsigh = -dT_dr_dsigh_1b * R_grid * np.cos(SIGH_grid)

        # ---- DOUBLE INTEGRATION ----
        dT_dr_1b = np.trapz(dT_dr_dsigh_1b, x=SIGH, axis=1)
        dD_total_dr_1b = np.trapz(dD_total_dr_dsigh_1b, x=SIGH, axis=1)
        dQ_total_dr_1b = np.trapz(dQ_total_dr_dsigh_1b, x=SIGH, axis=1)
        dP_total_dr_1b = np.trapz(dP_total_dr_dsigh_1b, x=SIGH, axis=1)

        rolling_moment_dr = np.trapz(rolling_moment_dr_dsigh, x=SIGH, axis=1)
        pitching_moment_dr = np.trapz(pitching_moment_dr_dsigh, x=SIGH, axis=1)

        # integrate over radius
        T = b * np.trapz(dT_dr_1b, x=R) / (2 * np.pi)
        D_total = b * np.trapz(dD_total_dr_1b, x=R) / (2 * np.pi)
        Q_total = b * np.trapz(dQ_total_dr_1b, x=R) / (2 * np.pi)
        P_total = b * np.trapz(dP_total_dr_1b, x=R) / (2 * np.pi)
        rolling_moment = b * np.trapz(rolling_moment_dr, x=R) / (2 * np.pi)
        pitching_moment = b * np.trapz(pitching_moment_dr, x=R) / (2 * np.pi)


        return T, D_total, Q_total, P_total, R, SIGH,rolling_moment,pitching_moment,dT_dr_1b, dD_total_dr_1b, dQ_total_dr_1b, dP_total_dr_1b, q

    N = N0
    history_T, history_D, history_Q, history_P, history_rolling_moment,history_pitching_moment= [], [], [], [], [], []
    T_prev = D_prev = Q_prev = P_prev = rolling_moment_prev=pitching_moment_prev= None

    for it in range(max_iter):
        T, D, Q, P, R, SIGH,rolling_moment,pitching_moment,dT_dr_1b, dD_total_dr_1b, dQ_total_dr_1b, dP_total_dr_1b, q= compute_T_D_Q_P_on_grid(N)

        # --- history initialization (ensure counts match) ---
        history_T = []
        history_D = []
        history_Q = []
        history_P = []
        history_rolling_moment = []
        history_pitching_moment = []


        if T_prev is not None:
            rel_err_T = abs((T - T_prev) / T_prev) if T_prev != 0 else np.inf
            rel_err_D = abs((D - D_prev) / D_prev) if D_prev != 0 else np.inf
            rel_err_Q = abs((Q - Q_prev) / Q_prev) if Q_prev != 0 else np.inf
            rel_err_P = abs((P - P_prev) / P_prev) if P_prev != 0 else np.inf
            rel_err_rolling_moment = abs((rolling_moment - rolling_moment_prev) / rolling_moment_prev) if rolling_moment_prev != 0 else np.inf
            rel_err_pitching_moment = abs((pitching_moment - pitching_moment_prev) / pitching_moment_prev) if pitching_moment_prev != 0 else np.inf
            
            # check relative errors
            # print(rel_err_pitching_moment,rel_err_rolling_moment,rel_err_T)
            if max(rel_err_T,rel_err_pitching_moment,rel_err_rolling_moment) < tol:
                return {
                    "T": float(T), "D": float(D), "Q": float(Q), "P": float(P),
                    "N": N,
                    "R": R,                     # 1D radial coordinates
                    "SIGH": SIGH, # 1D azimuth coordinates                 # Coning Angle
                    "Mr": float(rolling_moment),        
                    "Mp": float(pitching_moment),
                    "dT_dr_per_blade": dT_dr_1b,
                    "dD_dr_per_blade": dD_total_dr_1b,
                    "dQ_dr_per_blade": dQ_total_dr_1b,
                    "dP_dr_per_blade": dP_total_dr_1b,
                    "q": q,
                    "history_T": history_T,
                    "history_D": history_D,
                    "history_Q": history_Q,
                    "history_P": history_P,      
                    "history_rolling_moment": history_rolling_moment,
                    "history_pitching_moment": history_pitching_moment,
                    "stall_status": 0
                }

        T_prev, D_prev, Q_prev, P_prev,rolling_moment_prev,pitching_moment_prev = T, D, Q, P,rolling_moment,pitching_moment
        N=N*2

    return {
        "T": float(T), "D": float(D), "Q": float(Q), "P": float(P),
        "N": N,
        "R": R,                     # 1D radial coordinates
        "SIGH": SIGH, # 1D azimuth coordinates                
        "Mr": float(rolling_moment),        
        "Mp": float(pitching_moment),
        "dT_dr_per_blade": dT_dr_1b,
        "dD_dr_per_blade": dD_total_dr_1b,
        "dQ_dr_per_blade": dQ_total_dr_1b,
        "dP_dr_per_blade": dP_total_dr_1b,
        "q": q,
        "history_T": history_T,
        "history_D": history_D,
        "history_Q": history_Q,
        "history_P": history_P,
        "history_rolling_moment": history_rolling_moment,
        "history_pitching_moment": history_pitching_moment,
        "stall_status": 0,
        "warning": "Did not converge"
    }


def Coning_Angle_Solver(
    rho: float,
    Ut_fn: Callable[[float, float], float],
    Up_fn: Callable[[float, float], float],
    c_fn: Callable[[np.ndarray], np.ndarray],            # c_fn(R) -> 1D array of chord vs R
    Cl_fn: Callable[[float, float], float],
    R_root: float,
    R_tip: float,
    I:float,
    Omega:float,
    N0: int = 25,
    max_iter: int = 4,
    tol: float = 1e-3
) -> Dict:

    def compute_B0_on_meshgrid(N: int):
        # grids
        R = np.linspace(R_root, R_tip, N)
        SIGH = np.linspace(0, 2*np.pi, 90)
        R_grid, SIGH_grid = np.meshgrid(R, SIGH, indexing='ij')  # shape (N, 360)
        # allocate arrays for results
        Ut  = np.empty_like(R_grid)
        Up  = np.empty_like(R_grid)
        Cl  = np.empty_like(R_grid)

        # loop element-wise
        for i in range(R_grid.shape[0]):      # over radius
            for j in range(R_grid.shape[1]):  # over azimuth
                R_val = R_grid[i, j]
                sigh_val = SIGH_grid[i, j]

                Ut[i, j]  = Ut_fn(R_val, sigh_val)
                Up[i, j]  = Up_fn(R_val, sigh_val)
                Cl[i, j]  = Cl_fn(R_val, sigh_val)


            # --- Print the 2D matrices ---
       
        # print("\nUt matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(Ut).round(4).to_string(index=False, header=False))

        # --- Replace negative Ut with 0 ---
        Ut = np.maximum(Ut, 0.0)
        # print("\nUt matrix (rows=r, cols=phi):")
        # print(pd.DataFrame(Ut).round(4).to_string(index=False, header=False))
        # chord only varies with R -> produce 2D chord array
        c_1d = c_fn(R)                    # shape (N,)
        c = np.tile(c_1d[:, np.newaxis], (1, len(SIGH)))  # shape (N, 360)
        # dynamic pressure
        V2 = Ut**2 + Up**2
        q = 0.5 * rho * V2
        dB0_dr_dsigh = q*c*Cl*R_grid/(2*np.pi*I*Omega**2)  # Coning Angle
        # ---- DOUBLE INTEGRATION ----
        dB0_dr = np.trapz(dB0_dr_dsigh, x=SIGH, axis=1)
        # integrate over radius
        B0 = np.trapz(dB0_dr, x=R)
        return  B0

    N = N0
    history_B0 = []
    B0_prev = None
    for it in range(max_iter):
        B0 = compute_B0_on_meshgrid(N)
        history_B0.append(B0)
        
        if B0_prev is not None:
            rel_err_B0 = abs((B0 - B0_prev) / B0_prev) if B0_prev != 0 else np.inf
            if rel_err_B0 < tol:
                return B0

        B0_prev =  B0
        N=N*2
    return B0

def iterative_solver_cyclic(
    b: int,
    rho: float,
    Ut_fn: Callable[[float, float], float],
    Up_fn: Callable[[float, float], float],
    c_fn: Callable[[np.ndarray], np.ndarray],            # c_fn(R) -> 1D array of chord vs R
    Cl_fn: Callable[[float, float], float],
    R_root: float,
    R_tip: float,
    N0: int = 50,
    max_iter: int = 10,
    tol: float = 1e-3
) -> Dict:
    
    def compute_T_D_Q_P_on_grid(N: int):
        # grids
        R = np.linspace(R_root, R_tip, N)
        SIGH = np.linspace(0, 2*np.pi, 90)
        R_grid, SIGH_grid = np.meshgrid(R, SIGH, indexing='ij')  # shape (N, SIGH_points)

        # ---- compute Ut, Up, Cl ----
        # allocate arrays for results
        Ut  = np.empty_like(R_grid)
        Up  = np.empty_like(R_grid)
        Cl  = np.empty_like(R_grid)

        # loop element-wise
        for i in range(R_grid.shape[0]):      # over radius
            for j in range(R_grid.shape[1]):  # over azimuth
                R_val = R_grid[i, j]
                sigh_val = SIGH_grid[i, j]

                Ut[i, j]  = Ut_fn(R_val, sigh_val)
                Up[i, j]  = Up_fn(R_val, sigh_val)
                Cl[i, j]  = Cl_fn(R_val, sigh_val)

        # chord only varies with radius
        c_1d = c_fn(R)                    # shape (N,)
        c = np.tile(c_1d[:, np.newaxis], (1, len(SIGH)))  # shape (N, 360)

        # ---- dynamic pressure ----
        V2 = np.where(Ut < 0, -(Ut**2 + Up**2), (Ut**2 + Up**2))
        q = 0.5 * rho * V2

        # differential loads per blade element
        dT_dr_dsigh_1b = q * c * Cl
        rolling_moment_dr_dsigh = dT_dr_dsigh_1b * R_grid * np.sin(SIGH_grid)
        pitching_moment_dr_dsigh = -dT_dr_dsigh_1b * R_grid * np.cos(SIGH_grid)

        # ---- integrate over azimuth ----
        dT_dr_1b = np.trapz(dT_dr_dsigh_1b, x=SIGH, axis=1)
        rolling_moment_dr = np.trapz(rolling_moment_dr_dsigh, x=SIGH, axis=1)
        pitching_moment_dr = np.trapz(pitching_moment_dr_dsigh, x=SIGH, axis=1)

        # ---- integrate over radius ----
        T = b * np.trapz(dT_dr_1b, x=R) / (2 * np.pi)
        rolling_moment = b * np.trapz(rolling_moment_dr, x=R) / (2 * np.pi)
        pitching_moment = b * np.trapz(pitching_moment_dr, x=R) / (2 * np.pi)


        return T, rolling_moment, pitching_moment
    
    N = N0
    T_prev = rolling_moment_prev=pitching_moment_prev= None

    for it in range(max_iter):
        T, rolling_moment, pitching_moment= compute_T_D_Q_P_on_grid(N)

        if T_prev is not None:
            rel_err_T = abs((T - T_prev) / T_prev) if T_prev != 0 else np.inf
            rel_err_rolling_moment = abs((rolling_moment - rolling_moment_prev) / rolling_moment_prev) if rolling_moment_prev != 0 else np.inf
            rel_err_pitching_moment = abs((pitching_moment - pitching_moment_prev) / pitching_moment_prev) if pitching_moment_prev != 0 else np.inf
            # check relative errors
            # print(rel_err_pitching_moment,rel_err_rolling_moment,rel_err_T)

            if max(rel_err_T,rel_err_pitching_moment,rel_err_rolling_moment) < tol:
                return {
                    "T": float(T),
                    "Mr": float(rolling_moment),        
                    "Mp": float(pitching_moment),
                    "stall_status": 0
                }

        T_prev, rolling_moment_prev, pitching_moment_prev = T, rolling_moment, pitching_moment
        N=N*2
    
    return {
        "T": float(T),            
        "Mr": float(rolling_moment),        
        "Mp": float(pitching_moment),
        "stall_status": 0,
        "warning": "Did not converge"
    }