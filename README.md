````markdown
# Test Run Instructions

## Running the Simulation
Run the `test_run.ipynb` file.

1. Run **Cell 2** for hover/climb.  
2. Run **Cell 3** for forward flight.

---

### Adjusting Flight Parameters

**Forward Flight Velocity:**
```python
flight_condition["velocity"] = [input_velocity, 0, 0]
````

**Main Rotor Parameters:**

```python
rotor["theta_root"] = 5  # example
```

**Tail Rotor Parameters:**

```python
tail_rotor["theta_root"] = 5  # example
```

**Collective Adjustment (Thrust Control):**

```python
rotor["collective"] = input_collective
```

> Changeable parameters for both main and tail rotors can be found in `Params.py`.

---

## Generating Plots

Run the `Generate_plots.ipynb` file.

1. Run **Cells 1–2** to run the simulation and generate data.
2. Run **Cell 3** to generate standard plots.
3. Run **Cell 4** to generate normalized plots.

---

## Mission Planner

Run the `Mission_planner.ipynb` file.

### Adjust Mission Parameters

1. **Hover/Climb/Cruise Durations:**

   ```python
   hover_endurance = time_required
   climb_endurance = time_required
   cruise_endurance = time_required
   ```
2. **Cruise/Climb Velocity:**

   ```python
   cruise_speed = input_value
   climb_velocity = input_value
   ```
3. **Main Rotor Collective:**

   ```python
   main_collective = "input"
   ```
4. **Fuel Weight:**

   ```python
   fuel_weight = input_value
   ```

---

## Running the Mission

1. Specify climb and cruise velocities.
2. Set desired durations for each phase.
3. Define the fuel weight to carry.
4. Adjust rotor collective to achieve liftoff.

> **Note:** Tail collective is automatically calculated by the code.

```
```
