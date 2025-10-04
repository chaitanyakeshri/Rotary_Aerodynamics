import numpy as np

rotor= {
    "b": 4,
    "Rr": 1,
    "Rt": 8,
    "chord_root": 0.52,
    "chord_tip": 0.52,
    "theta_root": 0,
    "theta_tip": 0,
    "collective": 0,
    "cyclic_c":  0, # default trim values
    "cyclic_s": 0, 
    "lock_number": 0.002  # @AMSL
}

rotor_aero = {
    "Cl_alpha": 5.73,
    "Cl0": 0.2,
    "Cd0": 0.01,
    "alpha_stall": 13.0,
    "e": 0.8,
    "alpha0": -1,
    "Cm0": -0.02,
    "Cm_alpha": -0.5
}

tail_rotor = {
    "b": 5,
    "Rr": 0.1,
    "Rt": 2,
    "chord_root": 0.1,
    "chord_tip": 0.1,
    "theta_root": 4.0,
    "theta_tip": 0.0,
    "collective": 0,
    "arm_length": 10, # length of driveshaft
    "power_fraction": 0.2
}

tail_rotor_aero= {
    "Cl_alpha": 5.73,
    "Cl0": 0.0,
    "Cd0": 0.08,
    "alpha_stall": 15,
    "e": 0.8,
    "alpha0": 0,
    "Cm0": -0.020,
    "Cm_alpha": 0
}

engine = {
    "max_power_avail": 3000000,
    "omega": 27,
    "bsfc": 0.3,
    "engines_loss": 0.15
}

fuselage = {
    "Cdx": 1,
    "Cdy": 1,
    "Cdz": 1,
    "X_flat_area": 2.0,
    "Y_flat_area": 4.0,
    "Z_flat_area": 7.0,
    "max_fuel_weight": 1000,
    "Empty_Weight": 5000,
    # Fuselage design: Two cylinders each with specified length and radii
    "length_front": 8.0,
    "length_tail": 10.0,
    "radius_front": 1.5,
    "radius_tail": 0.5,
    "fuel_pos": (0.0,0.0,0.0) # at centroid
}

horizontal_stabilizers = {
    "horiz_area": 4.18,
    "horiz_Cd": 0.05,
    "horiz_arm": 5.0
}

vertical_stabilizers = {
    "verti_area": 1.92,
    "verti_Cd": 0.05,
    "verti_arm": 5.0
}

flight_condition = {
    "altitude": 2000,
    "velocity": [100, 0,0],
    "wind": [0, 0, 0],
    "delta_ISA": 5,
    "fuel_weight": 800,
}

payload = {
    "weight": 700,
    "payload_pos": (1.0,0.5,0.0), # relative to centroid
}


mission_profile = [
    {"type": "hover", "duration_minutes": 5, "altitude_m": 2000, "payload_change_kg": 0},
    {"type": "climb", "target_altitude_m": 2500, "climb_rate_mps": 5},
    {"type": "cruise", "distance_km": 50, "speed_kph": 200, "altitude_m": 2500, "wind_kph": -30},  # 30 kmph headwind
    {"type": "loiter", "duration_minutes": 10, "speed_kph": 150, "altitude_m": 2500, "wind_kph": -30},
    {"type": "event", "payload_change_kg": -700}, # Unloading payload
    {"type": "cruise", "distance_km": 50, "speed_kph": 200, "altitude_m": 2500, "wind_kph": 15},   # 15 kmph tailwind
    # will add descent and landing segments later
]
trim_settings = {}  # Fill this with your trim data

