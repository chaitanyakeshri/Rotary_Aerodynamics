from Tail_rotor import Simulate_tail_rotor
from Trim_conditions import trim_cyclic
from Params import *

def balance_main_and_tail(Mp_init, Mr_init, alpha_tpp, B1c, V_inf, B0_final, b, rho, t_horizon, max_iter):
    
    theta_c_init=rotor["cyclic_c"]
    theta_s_init=rotor["cyclic_s"]
    tail_coll_init=tail_rotor["collective"]

    Q_prev=None
    iter=1
    while True:
        if iter>max_iter:
            return{
                "tail_coll": res["tail_coll"],
                "cyclic_c": cyclic_c,
                "cyclic_s": cyclic_s,
                "stall_status": 0
            }
        # Find cyclic values to trim
        cyclic_c,cyclic_s,Q_rotor=trim_cyclic(
            Mp_init,Mr_init,"angle",
            theta_c_init,theta_s_init,
            alpha_tpp,B1c,V_inf,B0_final,
            b, rho, t_horizon, max_iter
        )
        print("Cyclic_c",cyclic_c,"Cyclic_c:",cyclic_s)

        # Find tail collective to balance torque
        res=Simulate_tail_rotor(tail_rotor,tail_rotor_aero,engine,flight_condition,tail_coll_init,Q_rotor)
        if res["stall_status"]==1:
            return res
        
        # To converge 
        if Q_prev is not None:
            rel_err_Q = abs((res["Q"] - Q_prev) / Q_prev) if Q_prev != 0 else np.inf
            if rel_err_Q < 1e-3:
                return {
                    "tail_coll": res["tail_coll"],
                    "cyclic_c": cyclic_c,
                    "cyclic_s": cyclic_s,
                    "stall_status": 0
                }

        Q_prev=res["Q"]
        theta_c_init, theta_s_init = cyclic_c, cyclic_s
        Mp_init, Mr_init=res["Q"], res["Mr"]
        tail_coll_init=res["tail_coll"]
        iter+=1
    