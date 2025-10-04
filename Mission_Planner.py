import numpy as np
import matplotlib.pyplot as plt


from Solver import *
from helper_functions import *
from Simulator import *
from Simulator import Sim_Start_Forward
from Params import rotor, rotor_aero, engine, flight_condition, fuselage, payload, tail_rotor, mission_profile,trim_settings

def Mission_Planner(initial_fuel_weight):

    if initial_fuel_weight > fuselage["max_fuel_weight"]:
        print(f"Mission Failed: Fuel weight exceeds max of {fuselage['max_fuel_weight']} kg")
        return

    corrected_engine_power = engine["max_power_avail"] * density_ratio(flight_condition["altitude"])

     # Track payload separately to handle in-mission changes
    current_payload = payload["weight"]
    current_fuel = initial_fuel_weight
    TOGW = fuselage["Empty_Weight"] + current_payload + current_fuel
    
    # Initialize arrays with the starting point (t=0)
    times = [0]
    weights = [TOGW]
    fuels = [current_fuel]
    distances = [0] # Add distance tracking

    t = 0  # Represents total elapsed time in minutes
    total_distance_km = 0
    print(f"Initial TOGW: {TOGW:.2f} kg")
    #---------------------Simulation Setup---------------------
    for i, segment in enumerate(mission_profile):
        print(f"\n--- Starting Segment {i+1}: {segment['type'].upper()} ---")

        # Set altitude for the segment
        flight_condition["altitude"] = segment.get("altitude_m", flight_condition["altitude"])
        corrected_engine_power = engine["max_power_avail"] * density_ratio(flight_condition["altitude"])
        
        # --- HANDLE EVENT SEGMENT (e.g., payload drop) ---
        if segment['type'] == 'event':
            payload_change = segment.get("payload_change_kg", 0)
            current_payload += payload_change
            print(f"Payload changed by {payload_change} kg. New payload: {current_payload} kg.")
            # Update the last recorded weight to reflect the change instantly
            weights[-1] = fuselage["Empty_Weight"] + current_payload + fuels[-1]
            continue # Move to the next segment

        # --- HANDLE HOVER SEGMENT ---
        elif segment['type'] == 'hover':
            # This block will contain our existing hover logic, but adapted
            # to run for the duration specified in the segment dictionary.
            duration_minutes = segment['duration_minutes']
            for minute in range(int(duration_minutes)):
                current_weight = fuselage["Empty_Weight"] + current_payload + current_fuel
                thrust_req = current_weight * 9.81
                flight_condition_now = flight_condition.copy()
                flight_condition_now["velocity"] = [0, 0, 0]   
                found_collective = None 
                for col in np.linspace(0, rotor_aero["alpha_stall"]+1, 100):
                    rotor["collective"] = col
                    res = Sim_Start_Hover_Climb(rotor=rotor, rotor_aero=rotor_aero,
                                    engine=engine, flight_condition=flight_condition_now)
                    if res[0]["stall_status"] == 1:
                        continue
                    if res[0]["T"] >= thrust_req:
                        found_collective = col
                        break
                if found_collective is None:
                    print(f"Mission Failed: Cannot sustain hover at t={t} min")
                    return
                phase_power = res[0]["P"] / (1 - tail_rotor["power_fraction"]) / (1 - engine["engines_loss"])
                if phase_power > corrected_engine_power:
                    print(f"Mission Failed: Insufficient hover power {phase_power} > {corrected_engine_power} at t={t} min for {TOGW} kg")
                    return
                # Fuel burn this minute
                bsfc = engine["bsfc"]  # kg/kWh
                hover_fuel_rate = (bsfc * (phase_power/1000)) / 60  # kg/min
                current_fuel -= hover_fuel_rate
                current_fuel = max(current_fuel, 0)
                if current_fuel <= 0:
                    print(f"Fuel exhausted at t={t+1} min during hover. Mission Failed.")
                    return
                           
                 # 3. Update and append logs for this time step
                t += 1
                # Distance does not change in hover, so total_distance_km is constant
                
                times.append(t)
                fuels.append(current_fuel)
                weights.append(fuselage["Empty_Weight"] + current_payload + current_fuel)
                distances.append(total_distance_km)

                print(f"t={t} min | Hovering | Fuel: {current_fuel:.2f} kg | Weight: {weights[-1]:.2f} kg")
                continue  # Move to the next segment after hover
        # --- HANDLE F0RWARD FLIGHT  ---
        elif segment['type'] in ['cruise','loiter']:
            current_weight = fuselage["Empty_Weight"] + current_payload + current_fuel
            air_speed_kph = segment.get("speed_kph", 0)
            wind_kph = segment.get("wind_kph", 0)
            ground_speed_kph = air_speed_kph + wind_kph
            if ground_speed_kph <= 0:
                print(f"Mission Failed: Non-positive ground speed {ground_speed_kph} kph in {segment['type']} segment.")
                return
            #Simulator with airspeed and trim conditions
            flight_condition_now = flight_condition.copy()
            flight_condition_now["velocity"] = [air_speed_kph * 1000 / 3600, 0, 0]  # Convert kph to m/s
            trim = trim_settings[air_speed_kph]
            rotor["collective"] = trim["collective"]
            rotor["cyclic_s"] = trim["cyclic_s"]
            rotor["cyclic_c"] = trim["cyclic_c"]
            res = Sim_Start_Forward(rotor=rotor, rotor_aero=rotor_aero,
                                           engine=engine, flight_condition=flight_condition_now,t_horizon_s=30)
            phase_power = res[0]["P"] / (1 - tail_rotor["power_fraction"]) / (1 - engine["engines_loss"])
            if phase_power > corrected_engine_power:
                print(f"Mission Failed: Insufficient {segment['type']} power {phase_power} > {corrected_engine_power} at t={t} min for {TOGW} kg")
                return
            fuel_rate_kgh = engine["bsfc"] * (phase_power / 1000)  # kg/h
            #Calculate Fuel burn and distance based on segment type
            if segment['type'] == 'cruise':
                distance_km = segment.get("distance_km", 0)
                duration_hours = distance_km / ground_speed_kph
                duration_minutes = duration_hours * 60
                cruise_fuel_rate = fuel_rate_kgh / 60  # kg/min
                total_fuel_burn = cruise_fuel_rate * duration_minutes
                if total_fuel_burn > current_fuel:
                    print(f"Mission Failed: Insufficient fuel for cruise segment at t={t} min.")
                    return
                current_fuel -= total_fuel_burn
                t += int(duration_minutes)
                total_distance_km += distance_km
            elif segment['type'] == 'loiter':
                duration_minutes = segment.get("duration_minutes", 0)
                loiter_fuel_rate = fuel_rate_kgh / 60  # kg/min
                total_fuel_burn = loiter_fuel_rate * duration_minutes
                if total_fuel_burn > current_fuel:
                    print(f"Mission Failed: Insufficient fuel for loiter segment at t={t} min.")
                    return
                current_fuel -= total_fuel_burn
                t += int(duration_minutes)
                # Distance does not change in loiter, so total_distance_km is constant
                # Update logs for this segment
            times.append(t)
            fuels.append(current_fuel)
            weights.append(fuselage["Empty_Weight"] + current_payload + current_fuel)
            distances.append(total_distance_km)
            print(f"t={t} min | {segment['type'].capitalize()} | Fuel: {current_fuel:.2f} kg | Weight: {weights[-1]:.2f} kg | Distance: {total_distance_km:.2f} km")
            continue  
        #-----------------Plotting and Summary-----------------
     
    print("\nMission Feasible ✅")
    print(f"Total Fuel Used: {initial_fuel_weight - current_fuel:.2f} kg")
    print(f"Total Mission Time: {t:.1f} minutes")
    print(f"Total Distance Covered: {total_distance_km:.1f} km")
    
   