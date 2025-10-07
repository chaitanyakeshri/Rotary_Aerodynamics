import numpy as np
from Tail_rotor import Simulate_tail_rotor
from Trim_conditions import trim_cyclic
from Balance_thrust import balance_thrust
from Params import engine


def balance_main_and_tail(coll, theta_1c, theta_1s,
                        TOGW, Omega, alpha_tpp, rotor, tail_rotor, flight_condition, 
                        b, rho, t_horizon_s, tol1, tol2, max_iter):
    
    theta_c_init=theta_1c
    theta_s_init=theta_1s
    coll_init=coll
    V_inf=flight_condition["velocity"][0]

    iter=1
    while True:
        if iter>max_iter:
            print(res["stall_status"],"------------------------")
            res["stall_status"] = 0
            res["collective"]=coll_init
            break
        
        # Find cyclic values to trim
        res=trim_cyclic(rotor,"angle",
            theta_c_init, theta_s_init, coll_init,
            alpha_tpp, V_inf, b, rho, t_horizon_s
        )
        print("Cyclic_c",res["cyclic_c"],"Cyclic_s:",res["cyclic_s"])

        if res["end_run"]==False:
            res = balance_thrust(rotor, flight_condition, Omega, alpha_tpp, 
                                 res["cyclic_c"], res["cyclic_s"], coll_init, TOGW, tol1)

            if res["stall_status"]==1:
                return res,None
            # if res["out_of_power"]:
            #     print("Power insufficient for this mission.")
            #     return res,None
        else:
            res["collective"]   = coll_init
            res["stall_status"] = 0
            break

        cyclic_s_prev=None
        if cyclic_s_prev is not None:
            err_s = abs((res["cyclic_s"]-cyclic_s_prev)) if cyclic_s_prev != 0 else np.inf
            if err_s < 0.02 and res["end_T"]==True:
                res["collective"]   = coll_init
                res["stall_status"] = 0
                break

        theta_c_init, theta_s_init = res["cyclic_c"], res["cyclic_s"]
        cyclic_s_prev=res["cyclic_s"]
        coll_init=res["collective"]
        iter+=1
    
    # Find tail collective to balance torque
    # tail_coll_init = tail_rotor["collective"]
    # res_tail = Simulate_tail_rotor(tail_rotor,engine,flight_condition,tail_coll_init,res["Q"],tol2)
    res_tail={"tail_collective":5.90, "stall_status": 0}

    print("Converged B0 (coning angle):", res["B0_final"])
    return res,res_tail
        
    
    # Mr_prev=None

    # if Mr_prev is not None:
    #         rel_err_Mr = abs((res["Mr"] - Mr_prev) / Mr_prev) if Mr_prev != 0 else np.inf
    #         if rel_err_Mr < 1e-3:
    #             return {
    #                 "tail_coll": res["tail_coll"],
    #                 "cyclic_c": out["cyclic_c"],
    #                 "cyclic_s": out["cyclic_s"],
    #                 "stall_status": 0
    #             }

    #     Mr_prev=res["Mr"]
    #     theta_c_init, theta_s_init = out["cyclic_c"], out["cyclic_s"]
    #     Mp_init, Mr_init=res["Mr"], res["Mr"]
    #     tail_coll_init=res["tail_coll"]
    #     iter+=1
    