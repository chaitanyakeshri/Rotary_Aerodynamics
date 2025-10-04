import numpy as np
import matplotlib.pyplot as plt


from Solver import *
from helper_functions import *
from Simulator import *
from Params import *

def Mission_Planner(fuel_weight, hover_endurance, climb_endurance, climb_velocity):

    if fuel_weight > fuselage["max_fuel_weight"]:
        print(f"Mission Failed: Fuel weight exceeds max of {fuselage['max_fuel_weight']} kg")
        return

    corrected_engine_power = engine["max_power_avail"] * density_ratio(flight_condition["altitude"])

    # -------------------- INITIAL TOGW --------------------
    TOGW = fuel_weight + fuselage["Empty_Weight"] + payload["weight"]
    print("TAKe offf gross weight",TOGW)
    
    times, weights, fuels, hover_SEs, climb_SEs = [], [], [], [], []
    fuel_rates = []   # store phase_fuel_rate values
    phases = []       # keep track of phase (hover/climb)
    pahse_power_arr = []

    current_fuel = fuel_weight
    current_weight = TOGW
    t = 0

    # -------------------- SIMULATION LOOP --------------------
    for phase, duration in [("hover", hover_endurance), ("climb", climb_endurance)]:
        for i in range(int(duration)):

            thrust_req = current_weight * 9.81

            if phase == "hover":
                flight_condition_now = flight_condition.copy()
                flight_condition_now["velocity"] = [0, 0, 0]
            else:
                flight_condition_now = flight_condition.copy()
                flight_condition_now["velocity"] = [0, 0, climb_velocity]

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
                print(f"Mission Failed: Cannot sustain {phase} at t={t} min")
                return

            phase_power = res[0]["P"] / (1 - tail_rotor["power_fraction"]) / (1 - engine["engines_loss"])
            pahse_power_arr.append(phase_power)
            if phase_power > corrected_engine_power:
                print(f"Mission Failed: Insufficient {phase} power {phase_power} > {corrected_engine_power} at t={t} min for {TOGW} kg")
                return

            # Fuel burn this minute
            bsfc = engine["bsfc"]  # kg/kWh
            phase_fuel_rate = (bsfc * (phase_power/1000)) / 60  # kg/min
            current_fuel -= phase_fuel_rate
            current_fuel = max(current_fuel, 0)

            # Update weights
            current_weight = fuselage["Empty_Weight"] + payload["weight"] + current_fuel

            # Store results
            times.append(t)
            weights.append(current_weight)
            fuels.append(current_fuel)
            fuel_rates.append(phase_fuel_rate)
            phases.append(phase)

            # SE storage
            SE = 1/phase_fuel_rate if phase_fuel_rate > 0 else np.inf
            if phase == "hover":
                hover_SEs.append(SE)
                climb_SEs.append(None)
            else:
                hover_SEs.append(None)
                climb_SEs.append(SE)

            # Print status
            print(f"\n--- t = {t} min | Phase: {phase.upper()} ---")
            print(f"Gross Weight: {current_weight:.2f} kg | Fuel Remaining: {current_fuel:.2f} kg")
            print(f"Fuel Rate: {phase_fuel_rate:.4f} kg/min")

            t += 1
            if current_fuel <= 0:
                print(f"Fuel exhausted at t={t} min")
                return 
            

    print("\nMission Feasible ✅")
    print(f"Initial TOGW: {TOGW:.1f} kg")
    print(f"Total Fuel Used: {fuel_weight - current_fuel:.2f} kg")
    print(f"Final TOGW: {current_weight:.2f} kg")   
    print("collective angle",found_collective)
    print(f"Total Mission Time: {t} minutes")
    print("Avialable Power (W)",corrected_engine_power)
    print("Power Required(W)", pahse_power_arr[-1])
    

    # -------------------- PLOTTING --------------------
    # 1. Vehicle Weight vs Time
    plt.figure(figsize=(10, 5))
    plt.plot(times, weights, label="Vehicle Weight (kg)", linewidth=2)
    plt.title("Vehicle Weight vs Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Weight (kg)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()

    # 2. Fuel Remaining vs Time
    plt.figure(figsize=(10, 5))
    plt.plot(times, fuels, label="Fuel Remaining (kg)", linewidth=2, color="orange")
    plt.title("Fuel Remaining vs Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Fuel (kg)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()

    # 3. Phase Fuel Rate vs TOGW (kgf instead of kg)
    plt.figure(figsize=(8, 6))
    for phase_name, color in zip(["hover", "climb"], ["green", "red"]):
        x = [f for f, p in zip(fuel_rates, phases) if p == phase_name]
        y = [w * 9.81 for w, p in zip(weights, phases) if p == phase_name]  # convert to kgf
        if x and y:
            plt.plot(x, y, label=phase_name.capitalize(), linewidth=2, color=color)

    plt.title("Phase Fuel Rate vs TOGW")
    plt.xlabel("Fuel Burn Rate (kg/min)")
    plt.ylabel("Gross Weight (kgf)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()

