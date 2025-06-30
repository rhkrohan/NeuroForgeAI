#!/usr/bin/env python3
"""
Refactored Obaid simulation script using optimized parameters from emodel-runner.

This script loads optimized parameters from BluePyOpt optimization results
and runs the original Obaid simulation with those parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import neuron
from neuron import h
import sys
import json
import yaml


def load_optimized_parameters(params_file):
    """
    Load optimized parameters from JSON file.
    
    Args:
        params_file (Path): Path to best parameters JSON file
        
    Returns:
        dict: Optimized parameters
    """
    if not params_file.exists():
        print(f"Warning: Optimized parameters not found at {params_file}")
        return None
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    return params


def instantiate_emodel(project_dir):
    """
    Load optimized e-model from optimization results.
    
    Args:
        project_dir (Path): Project root directory
        
    Returns:
        neuron.h.Section: Instantiated cell with optimized parameters
    """
    # Define paths
    config_file = project_dir / "configs" / "emodel.yaml"
    params_file = project_dir / "optimisation" / "best_parameters.json"
    mechanisms_dir = project_dir / "mechanisms"
    
    # Load configuration
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load mechanisms
    if mechanisms_dir.exists():
        neuron.load_mechanisms(str(mechanisms_dir))
    
    # Load morphology
    morph_path = project_dir / config["morphology"]["path"].replace("../", "")
    
    # Initialize NEURON
    h.load_file("stdrun.hoc")
    
    # Create cell
    if morph_path.exists():
        # Load SWC morphology
        cell = h.Section(name='soma')
        morph = h.Import3d_SWC_read()
        morph.input(str(morph_path))
        imprt = h.Import3d_GUI(morph, 0)
        imprt.instantiate(cell)
        
        # Find soma section
        soma = None
        for sec in h.allsec():
            if "soma" in sec.name():
                soma = sec
                break
        
        if soma is None:
            soma = cell
    else:
        # Create basic soma
        soma = h.Section(name='soma')
        soma.L = 20
        soma.diam = 20
    
    # Insert passive mechanism
    soma.insert('pas')
    
    # Load and apply optimized parameters
    best_params = load_optimized_parameters(params_file)
    
    if best_params:
        print(f"Applying {len(best_params)} optimized parameters...")
        
        for param_name, param_value in best_params.items():
            try:
                if param_name == "g_pas":
                    soma.g_pas = param_value
                elif param_name == "Ra":
                    soma.Ra = param_value
                elif param_name == "cm":
                    soma.cm = param_value
                elif param_name.startswith("gbar_"):
                    # Extract mechanism name
                    mech_name = param_name.replace("gbar_", "")
                    
                    # Insert mechanism if available
                    try:
                        soma.insert(mech_name)
                        setattr(soma, param_name, param_value)
                    except:
                        print(f"Warning: Could not insert mechanism {mech_name}")
                        
            except Exception as e:
                print(f"Warning: Could not set parameter {param_name}: {e}")
    else:
        print("Using default parameters (optimization not run yet)")
        soma.g_pas = 0.0001
        soma.Ra = 100
        soma.cm = 1.0
    
    # Set reversal potential
    if not hasattr(soma, 'e_pas'):
        soma.e_pas = -65
    
    return soma


def setup_stimulation(cell, amp_start=0.1, amp_end=0.4, num_steps=7, 
                     delay=500, duration=1000, total_time=2000):
    """
    Setup current clamp stimulation protocol.
    
    Args:
        cell: NEURON cell object
        amp_start (float): Starting current amplitude (nA)
        amp_end (float): Ending current amplitude (nA)
        num_steps (int): Number of current steps
        delay (float): Stimulation delay (ms)
        duration (float): Stimulation duration (ms)
        total_time (float): Total simulation time (ms)
        
    Returns:
        tuple: (stimuli, recordings, current_amplitudes)
    """
    # Create current amplitudes
    current_amps = np.linspace(amp_start, amp_end, num_steps)
    
    stimuli = []
    recordings = []
    
    for amp in current_amps:
        # Create current clamp
        stim = h.IClamp(cell(0.5))
        stim.delay = delay
        stim.dur = duration
        stim.amp = amp
        stimuli.append(stim)
        
        # Create voltage recording
        v_rec = h.Vector()
        v_rec.record(cell(0.5)._ref_v)
        recordings.append(v_rec)
    
    return stimuli, recordings, current_amps


def run_simulation(cell, total_time=2000, dt=0.025):
    """
    Run NEURON simulation.
    
    Args:
        cell: NEURON cell object
        total_time (float): Total simulation time (ms)
        dt (float): Integration time step (ms)
    """
    # Setup simulation
    h.dt = dt
    h.tstop = total_time
    h.v_init = -65
    
    # Create time vector
    t_vec = h.Vector()
    t_vec.record(h._ref_t)
    
    # Initialize and run
    h.finitialize(h.v_init)
    h.run()
    
    return np.array(t_vec)


def plot_results(time, voltage_traces, current_amps, save_path=None):
    """
    Plot voltage traces for different current injections.
    
    Args:
        time (np.array): Time vector
        voltage_traces (list): List of voltage vectors
        current_amps (np.array): Current amplitudes
        save_path (Path, optional): Path to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, (v_trace, i_amp) in enumerate(zip(voltage_traces, current_amps)):
        voltage = np.array(v_trace)
        ax.plot(time, voltage, label=f'{i_amp:.2f} nA', linewidth=1.5)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title('Voltage Response to Current Steps (Optimized Parameters)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main simulation function."""
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    print("Starting Obaid simulation with optimized parameters...")
    
    # Load optimized e-model
    print("Loading e-model...")
    try:
        cell = instantiate_emodel(project_dir)
        print("E-model loaded successfully")
    except Exception as e:
        print(f"Failed to load e-model: {e}")
        print("Creating fallback basic cell model...")
        
        # Fallback: create basic cell (for testing without optimization)
        cell = h.Section(name='soma')
        cell.L = 20
        cell.diam = 20
        cell.insert('pas')
        cell.g_pas = 0.0001
        cell.e_pas = -65
        cell.Ra = 100
        cell.cm = 1
    
    # Setup stimulation protocol
    stimuli, recordings, current_amps = setup_stimulation(cell)
    
    # Run simulations for each current step
    print(f"Running {len(current_amps)} current step simulations...")
    voltage_traces = []
    
    for i, (stim, v_rec) in enumerate(zip(stimuli, recordings)):
        print(f"Running step {i+1}/{len(stimuli)}: {current_amps[i]:.2f} nA")
        
        # Run simulation
        time = run_simulation(cell)
        voltage_traces.append(list(v_rec))
        
        # Clear stimulus for next run
        stim.amp = 0
    
    print("Simulations completed")
    
    # Plot results
    output_path = project_dir / "obaid_simulation_results.png"
    plot_results(time, voltage_traces, current_amps, output_path)
    
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()