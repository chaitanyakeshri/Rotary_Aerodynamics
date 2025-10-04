import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We only import the low-level solver and the high-level hover simulator
from Solver import iterative_solver_forward
from Simulator import Sim_Start_Hover_Climb
from helper_functions import *
from Params import rotor, rotor_aero, engine, flight_condition, fuselage, payload, tail_rotor, mission_profile, trim_settings

def Mission_Planner(initial_fuel_weight):
   
    if initial_fuel_weight > fuselage["max_fuel_weight"]:
        print(f"Mission Failed: Fuel weight exceeds max of {fuselage['max_fuel_weight']} kg")
        return

    # --- INITIALIZATION ---
    current_payload = payload["weight"]
    current_fuel = initial_fuel_weight
    times, weights, fuels, distances = [0], [fuselage["Empty_Weight"] + current_payload + current_fuel], [current_fuel], [0]
    t, total_distance_km = 0, 0
    print(f"Initial TOGW: {weights[0]:.2f} kg")

    # --- SIMULATION LOOP ---
    for i, segment in enumerate(mission_profile):
        print(f"\n--- Starting Segment {i+1}: {segment['type'].upper()} ---")
        current_altitude = segment.get("altitude_m", flight_condition["altitude"])
        rho = atmosphere(current_altitude)["rho"]
        corrected_engine_power = engine["max_power_avail"] * density_ratio(current_altitude)
        
        # --- EVENT SEGMENT ---
        if segment['type'] == 'event':
            payload_change = segment.get("payload_change_kg", 0)
            current_payload += payload_change
            print(f"Payload changed by {payload_change} kg. New payload: {current_payload} kg.")
            weights[-1] = fuselage["Empty_Weight"] + current_payload + fuels[-1]
            continue

        # --- HOVER/CLIMB SEGMENTS (This logic is stable) ---
        elif segment['type'] in ['hover', 'climb']:
            if segment['type'] == 'hover':
                duration_minutes = segment['duration_minutes']
                vertical_velocity = 0
            else: # climb
                target_alt = segment['target_altitude_m']
                climb_rate = segment['climb_rate_mps']
                duration_minutes = (target_alt - current_altitude) / (climb_rate * 60)
                vertical_velocity = climb_rate
            
            for minute in range(int(round(duration_minutes))):
                current_weight = fuselage["Empty_Weight"] + current_payload + current_fuel
                thrust_req = current_weight * 9.81
                flight_condition_now = flight_condition.copy()
                flight_condition_now["velocity"] = [0, 0, vertical_velocity]
                
                found_collective, res = None, None
                for col in np.linspace(0, rotor_aero["alpha_stall"] + 1, 100):
                    rotor["collective"] = col
                    res_tuple = Sim_Start_Hover_Climb(rotor, rotor_aero, engine, flight_condition_now)
                    if res_tuple[0]["stall_status"] == 1: continue
                    if res_tuple[0]["T"] >= thrust_req:
                        found_collective, res = col, res_tuple[0]
                        break
                
                if found_collective is None:
                    print(f"Mission Failed: Cannot sustain {segment['type']} at t={t} min")
                    return

                phase_power = res["P"] / (1 - tail_rotor["power_fraction"]) / (1 - engine["engines_loss"])
                if phase_power > corrected_engine_power:
                    print(f"Mission Failed: Insufficient power for {segment['type']} at t={t} min")
                    return
                
                fuel_rate_kg_min = (engine["bsfc"] * (phase_power / 1000)) / 60
                current_fuel -= fuel_rate_kg_min
                if current_fuel <= 0:
                    print(f"Fuel exhausted at t={t+1} min. Mission Failed.")
                    return
                
                t += 1
                times.append(t); fuels.append(current_fuel); distances.append(total_distance_km)
                weights.append(fuselage["Empty_Weight"] + current_payload + current_fuel)
                print(f"t={t} min | {segment['type'].capitalize()}ing | Fuel: {current_fuel:.2f} kg | Weight: {weights[-1]:.2f} kg")

        # --- SIMPLIFIED FORWARD FLIGHT LOGIC (CRUISE & LOITER) ---
        elif segment['type'] in ['cruise', 'loiter']:
            air_speed_kph = segment.get("speed_kph", 0)
            
            # Get pre-calculated trim controls from Params.py file
            if air_speed_kph not in trim_settings:
                print(f"Mission Failed: No trim settings found in Params.py for {air_speed_kph} kph.")
                return
            trim = trim_settings[air_speed_kph]
            
            # --- Setup for the low-level solver ---
            V_inf = air_speed_kph * 1000 / 3600
            alpha_tpp, mu = 0, V_inf / (engine["omega"] * rotor["Rt"])
            
            # Define all necessary functions for the solver
            Lambda_induced_forward = lambda r, sigh: lambda_i_forward(mu, r, rotor["Rt"], sigh, alpha_tpp, engine["omega"], rho, V_inf)
            v_fn = lambda r, sigh: induced_velocity_forward(Lambda_induced_forward(r, sigh), engine["omega"], rotor["Rt"], V_inf, alpha_tpp)
            phi_fn = lambda r, sigh: compute_phi_forward(V_inf, v_fn(r, sigh), engine["omega"], alpha_tpp, sigh, r, 0, 0, rotor["Rr"])
            c_fn = lambda r: chord_r(rotor, r)
            
            # Create Cl and Cd functions using the known trim values from Params.py
            Cl_fn = lambda r, sigh: airfoil_lift(
                Cl0=rotor_aero["Cl0"], Cl_alpha=rotor_aero["Cl_alpha"], alpha0=rotor_aero["alpha0"],
                alpha=pitch_x_forward(rotor, r, sigh, trim["cyclic_c"], trim["cyclic_s"], trim["collective"]) - np.degrees(phi_fn(r, sigh)),
                alpha_stall=rotor_aero["alpha_stall"]
            )
            AR = (rotor["Rt"] - rotor["Rr"]) / ((rotor["chord_root"] + rotor["chord_tip"]) / 2)
            Cd_fn = lambda r, sigh: airfoil_drag(Cd0=rotor_aero["Cd0"], Cl=Cl_fn(r, sigh), e=rotor_aero["e"], AR=AR)
            
            
            res = iterative_solver_forward(
                b=rotor["b"], rho=rho, Ut_fn=lambda r, sigh: engine["omega"] * r + V_inf,
                Up_fn=lambda r, sigh: v_fn(r, sigh),
                c_fn=c_fn, Cl_fn=Cl_fn, phi_fn=phi_fn, Cd_fn=Cd_fn,
                R_root=rotor["Rr"], R_tip=rotor["Rt"]
            )

            # --- Fuel burn and logging logic ---
            phase_power = res["P"] / (1 - tail_rotor["power_fraction"]) / (1 - engine["engines_loss"])
            if phase_power > corrected_engine_power:
                print(f"Mission Failed: Insufficient power for {segment['type']}.")
                return

            fuel_rate_kgh = engine["bsfc"] * (phase_power / 1000)
            wind_kph = segment.get("wind_kph", 0)
            ground_speed_kph = air_speed_kph + wind_kph

            if segment['type'] == 'cruise':
                distance_to_cover_km = segment.get("distance_km", 0)
                if ground_speed_kph <= 0:
                    print("Mission Failed: Ground speed non-positive.")
                    return
                duration_hours = distance_to_cover_km / ground_speed_kph
            else: # loiter
                duration_hours = segment.get("duration_minutes", 0) / 60
                distance_to_cover_km = ground_speed_kph * duration_hours
            
            fuel_needed = fuel_rate_kgh * duration_hours
            if fuel_needed > current_fuel:
                print(f"Mission Failed: Insufficient fuel for {segment['type']}.")
                return

            current_fuel -= fuel_needed
            t += duration_hours * 60
            total_distance_km += distance_to_cover_km

            times.append(t); fuels.append(current_fuel); distances.append(total_distance_km)
            weights.append(fuselage["Empty_Weight"] + current_payload + current_fuel)
            print(f"Leg End | Power: {phase_power/1e3:.1f} kW | Fuel: {current_fuel:.2f} kg | Weight: {weights[-1]:.2f} kg")

    print("\nMission Feasible ✅")
    # PLOTTING
    
