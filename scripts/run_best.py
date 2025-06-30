#!/usr/bin/env python3
"""
Run simulation with best optimized parameters and plot results.

Loads the best parameters from optimization and runs a simulation
to validate the optimized model.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import neuron
from neuron import h
import yaml


def load_best_parameters(params_file):
    """
    Load best parameters from optimization results.
    
    Args:
        params_file (Path): Path to best parameters JSON file
        
    Returns:
        dict: Best parameters dictionary
    """
    if not params_file.exists():
        raise FileNotFoundError(f"Best parameters file not found: {params_file}")
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    return params


def create_optimized_cell(morph_path, best_params, mechanisms_dir):
    """
    Create cell with optimized parameters.
    
    Args:
        morph_path (Path): Path to morphology file
        best_params (dict): Optimized parameters
        mechanisms_dir (Path): Directory containing mechanisms
        
    Returns:
        neuron.h.Section: NEURON cell with optimized parameters
    """
    # Load mechanisms
    if mechanisms_dir.exists():
        neuron.load_mechanisms(str(mechanisms_dir))
    
    # Initialize NEURON
    h.load_file("stdrun.hoc")
    
    # Load morphology
    cell = h.Section(name='soma')
    
    if morph_path.exists():
        # Load SWC morphology
        morph = h.Import3d_SWC_read()
        morph.input(str(morph_path))
        imprt = h.Import3d_GUI(morph, 0)
        imprt.instantiate(cell)
        
        # Get soma section
        soma = None
        for sec in h.allsec():
            if "soma" in sec.name():
                soma = sec
                break
        
        if soma is None:
            soma = cell
    else:
        # Create simple soma
        soma = cell
        soma.L = 20
        soma.diam = 20
    
    # Insert passive mechanism
    soma.insert('pas')
    
    # Apply optimized parameters
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
                
                # Insert mechanism if not already present
                if not hasattr(soma, mech_name):
                    try:
                        soma.insert(mech_name)
                    except:
                        print(f"Warning: Could not insert mechanism {mech_name}")
                        continue
                
                # Set conductance
                setattr(soma, param_name, param_value)
            
            print(f"Set {param_name} = {param_value}")
            
        except Exception as e:
            print(f"Warning: Could not set parameter {param_name}: {e}")
    
    # Set passive parameters if not already set
    if not hasattr(soma, 'e_pas'):
        soma.e_pas = -65
    
    return soma


def run_validation_simulation(cell, current_amplitudes, delay=500, duration=1000, total_time=2000):
    """
    Run validation simulation with multiple current steps.
    
    Args:
        cell: NEURON cell object
        current_amplitudes (list): List of current amplitudes to test
        delay (float): Stimulus delay in ms
        duration (float): Stimulus duration in ms
        total_time (float): Total simulation time in ms
        
    Returns:
        tuple: (time_array, voltage_traces, current_amplitudes)
    """
    voltage_traces = []
    
    # Setup simulation parameters
    h.dt = 0.025
    h.tstop = total_time
    h.v_init = -65
    
    # Create time vector
    t_vec = h.Vector()
    t_vec.record(h._ref_t)
    
    for i, amp in enumerate(current_amplitudes):
        print(f"Running simulation {i+1}/{len(current_amplitudes)}: {amp:.3f} nA")
        
        # Create current clamp
        stim = h.IClamp(cell(0.5))
        stim.delay = delay
        stim.dur = duration
        stim.amp = amp
        
        # Create voltage recording
        v_vec = h.Vector()
        v_vec.record(cell(0.5)._ref_v)
        
        # Run simulation
        h.finitialize(h.v_init)
        h.run()
        
        # Store results
        voltage_traces.append(np.array(v_vec))
        
        # Clean up stimulus
        stim = None
    
    time_array = np.array(t_vec)
    
    return time_array, voltage_traces, current_amplitudes


def plot_validation_results(time, voltage_traces, current_amps, best_params, save_path=None):
    """
    Plot validation simulation results.
    
    Args:
        time (np.array): Time array
        voltage_traces (list): List of voltage arrays
        current_amps (list): Current amplitudes
        best_params (dict): Best parameters for title
        save_path (Path, optional): Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot voltage traces
    for i, (v_trace, i_amp) in enumerate(zip(voltage_traces, current_amps)):
        ax1.plot(time, v_trace, label=f'{i_amp:.3f} nA', linewidth=1.5)
    
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Optimized E-Model Validation - Voltage Responses')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot I-V relationship
    steady_state_voltages = []
    for v_trace in voltage_traces:
        # Get steady state voltage (last 100ms average)
        steady_state = np.mean(v_trace[-int(100/0.025):])
        steady_state_voltages.append(steady_state)
    
    ax2.plot(current_amps, steady_state_voltages, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Current Amplitude (nA)')
    ax2.set_ylabel('Steady State Voltage (mV)')
    ax2.set_title('Current-Voltage Relationship')
    ax2.grid(True, alpha=0.3)
    
    # Add parameter summary
    param_text = "Optimized Parameters:\n"
    for param, value in list(best_params.items())[:6]:  # Show first 6 parameters
        param_text += f"{param}: {value:.6f}\n"
    if len(best_params) > 6:
        param_text += f"... and {len(best_params)-6} more"
    
    fig.text(0.02, 0.02, param_text, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Validation plot saved to: {save_path}")
    
    plt.show()


def analyze_model_properties(voltage_traces, current_amps, time):
    """
    Analyze basic electrophysiological properties of the optimized model.
    
    Args:
        voltage_traces (list): Voltage traces
        current_amps (list): Current amplitudes
        time (np.array): Time array
    """
    print("\n=== Model Analysis ===")
    
    # Find rheobase (minimum current that elicits spikes)
    rheobase = None
    spike_threshold = -20  # mV
    
    for i, (v_trace, i_amp) in enumerate(zip(voltage_traces, current_amps)):
        if np.max(v_trace) > spike_threshold:
            rheobase = i_amp
            break
    
    if rheobase is not None:
        print(f"Rheobase: {rheobase:.3f} nA")
    else:
        print("Rheobase: Not reached in tested range")
    
    # Analyze input resistance (subthreshold responses)
    subthreshold_responses = []
    subthreshold_currents = []
    
    for v_trace, i_amp in zip(voltage_traces, current_amps):
        if np.max(v_trace) < spike_threshold and i_amp != 0:
            # Calculate voltage deflection
            baseline = np.mean(v_trace[:int(500/0.025)])  # Before stimulus
            steady_state = np.mean(v_trace[-int(100/0.025):])  # End of trace
            deflection = steady_state - baseline
            
            if abs(deflection) > 1:  # Significant deflection
                subthreshold_responses.append(deflection)
                subthreshold_currents.append(i_amp)
    
    if len(subthreshold_responses) >= 2:
        # Calculate input resistance from linear fit
        resistance = np.polyfit(subthreshold_currents, subthreshold_responses, 1)[0]
        print(f"Input Resistance: {resistance:.1f} MΩ")
    
    # Count spikes for suprathreshold responses
    for i, (v_trace, i_amp) in enumerate(zip(voltage_traces, current_amps)):
        if np.max(v_trace) > spike_threshold:
            # Simple spike counting
            spikes = 0
            above_threshold = v_trace > spike_threshold
            for j in range(1, len(above_threshold)):
                if above_threshold[j] and not above_threshold[j-1]:
                    spikes += 1
            print(f"Current {i_amp:.3f} nA: {spikes} spikes")


def main():
    """Main validation function."""
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    config_file = project_dir / "configs" / "emodel.yaml"
    params_file = project_dir / "optimisation" / "best_parameters.json"
    mechanisms_dir = project_dir / "mechanisms"
    
    # Load configuration
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get morphology path
    morph_path = project_dir / config["morphology"]["path"].replace("../", "")
    
    # Load best parameters
    try:
        best_params = load_best_parameters(params_file)
        print(f"Loaded {len(best_params)} optimized parameters")
    except FileNotFoundError:
        print(f"Warning: Best parameters not found at {params_file}")
        print("Using default parameters for validation")
        best_params = {
            "g_pas": 0.0001,
            "Ra": 100,
            "cm": 1.0
        }
    
    # Create optimized cell
    print("Creating optimized cell model...")
    cell = create_optimized_cell(morph_path, best_params, mechanisms_dir)
    
    # Define test currents
    current_amplitudes = [-0.2, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    
    # Run validation simulation
    print("Running validation simulations...")
    time, voltage_traces, current_amps = run_validation_simulation(cell, current_amplitudes)
    
    # Plot results
    output_path = project_dir / "validation_results.png"
    plot_validation_results(time, voltage_traces, current_amps, best_params, output_path)
    
    # Analyze model properties
    analyze_model_properties(voltage_traces, current_amps, time)
    
    print("\nValidation completed successfully!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()