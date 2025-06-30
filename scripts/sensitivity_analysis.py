#!/usr/bin/env python3
"""
Sensitivity analysis script for optimized e-model.

Performs parameter sensitivity analysis to understand which parameters
most influence model behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import yaml
from copy import deepcopy
import neuron
from neuron import h


def load_base_parameters(params_file):
    """
    Load base optimized parameters.
    
    Args:
        params_file (Path): Path to best parameters JSON file
        
    Returns:
        dict: Base parameters
    """
    if not params_file.exists():
        print(f"Warning: Base parameters not found at {params_file}")
        return {
            "g_pas": 0.0001,
            "Ra": 100,
            "cm": 1.0
        }
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    return params


def create_test_cell(morph_path, parameters, mechanisms_dir):
    """
    Create NEURON cell with given parameters.
    
    Args:
        morph_path (Path): Path to morphology file
        parameters (dict): Cell parameters
        mechanisms_dir (Path): Directory containing mechanisms
        
    Returns:
        neuron.h.Section: NEURON cell
    """
    # Load mechanisms
    if mechanisms_dir.exists():
        neuron.load_mechanisms(str(mechanisms_dir))
    
    # Initialize NEURON
    h.load_file("stdrun.hoc")
    
    # Create basic soma (simplified for sensitivity analysis)
    soma = h.Section(name='soma')
    soma.L = 20
    soma.diam = 20
    
    # Insert passive mechanism
    soma.insert('pas')
    
    # Apply parameters
    for param_name, param_value in parameters.items():
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
                    pass  # Skip if mechanism not available
                        
        except Exception as e:
            pass  # Skip problematic parameters
    
    # Set reversal potential
    if not hasattr(soma, 'e_pas'):
        soma.e_pas = -65
    
    return soma


def run_test_simulation(cell, current_amp=0.2, delay=100, duration=500, total_time=800):
    """
    Run a single test simulation.
    
    Args:
        cell: NEURON cell object
        current_amp (float): Current amplitude (nA)
        delay (float): Stimulus delay (ms)
        duration (float): Stimulus duration (ms)
        total_time (float): Total simulation time (ms)
        
    Returns:
        dict: Simulation results
    """
    # Setup simulation parameters
    h.dt = 0.025
    h.tstop = total_time
    h.v_init = -65
    
    # Create recordings
    t_vec = h.Vector()
    v_vec = h.Vector()
    t_vec.record(h._ref_t)
    v_vec.record(cell(0.5)._ref_v)
    
    # Create stimulus
    stim = h.IClamp(cell(0.5))
    stim.delay = delay
    stim.dur = duration
    stim.amp = current_amp
    
    # Run simulation
    h.finitialize(h.v_init)
    h.run()
    
    # Extract results
    time = np.array(t_vec)
    voltage = np.array(v_vec)
    
    # Calculate features
    results = calculate_features(time, voltage, delay, duration)
    
    return results


def calculate_features(time, voltage, delay, duration):
    """
    Calculate basic electrophysiological features.
    
    Args:
        time (np.array): Time array
        voltage (np.array): Voltage array
        delay (float): Stimulus delay
        duration (float): Stimulus duration
        
    Returns:
        dict: Calculated features
    """
    features = {}
    
    # Baseline voltage
    baseline_idx = time < delay
    if np.any(baseline_idx):
        features['baseline_voltage'] = np.mean(voltage[baseline_idx])
    else:
        features['baseline_voltage'] = voltage[0]
    
    # Stimulus period
    stim_idx = (time >= delay) & (time <= delay + duration)
    
    if np.any(stim_idx):
        stim_voltage = voltage[stim_idx]
        
        # Maximum voltage during stimulus
        features['max_voltage'] = np.max(stim_voltage)
        
        # Minimum voltage during stimulus
        features['min_voltage'] = np.min(stim_voltage)
        
        # Voltage deflection
        features['voltage_deflection'] = features['max_voltage'] - features['baseline_voltage']
        
        # Count spikes (simple threshold crossing)
        spike_threshold = -20  # mV
        spikes = 0
        above_threshold = stim_voltage > spike_threshold
        for i in range(1, len(above_threshold)):
            if above_threshold[i] and not above_threshold[i-1]:
                spikes += 1
        features['spike_count'] = spikes
    else:
        features['max_voltage'] = features['baseline_voltage']
        features['min_voltage'] = features['baseline_voltage']
        features['voltage_deflection'] = 0
        features['spike_count'] = 0
    
    # Post-stimulus features
    post_idx = time > delay + duration
    if np.any(post_idx):
        post_voltage = voltage[post_idx]
        features['steady_state_voltage'] = np.mean(post_voltage[-int(50/0.025):])  # Last 50ms
    else:
        features['steady_state_voltage'] = features['baseline_voltage']
    
    return features


def parameter_sensitivity_analysis(base_params, morph_path, mechanisms_dir, 
                                 variation_percent=20, n_points=5):
    """
    Perform parameter sensitivity analysis.
    
    Args:
        base_params (dict): Base optimized parameters
        morph_path (Path): Path to morphology file
        mechanisms_dir (Path): Directory containing mechanisms
        variation_percent (float): Percentage variation around base values
        n_points (int): Number of points to test for each parameter
        
    Returns:
        pd.DataFrame: Sensitivity analysis results
    """
    results = []
    
    # Test each parameter
    for param_name, base_value in base_params.items():
        print(f"Testing sensitivity of {param_name}...")
        
        # Create parameter variations
        if base_value == 0:
            # For zero values, test small positive values
            test_values = np.linspace(0, base_value + 0.01, n_points)
        else:
            # Create variations around base value
            variation = abs(base_value) * variation_percent / 100
            test_values = np.linspace(base_value - variation, 
                                    base_value + variation, n_points)
            # Ensure no negative values for conductances
            if param_name.startswith('gbar_') or param_name == 'g_pas':
                test_values = np.maximum(test_values, 0)
        
        # Test each variation
        for test_value in test_values:
            # Create modified parameters
            test_params = deepcopy(base_params)
            test_params[param_name] = test_value
            
            try:
                # Create test cell
                cell = create_test_cell(morph_path, test_params, mechanisms_dir)
                
                # Run simulation
                sim_results = run_test_simulation(cell)
                
                # Store results
                result_row = {
                    'parameter': param_name,
                    'base_value': base_value,
                    'test_value': test_value,
                    'percent_change': ((test_value - base_value) / base_value * 100) if base_value != 0 else 0
                }
                result_row.update(sim_results)
                results.append(result_row)
                
            except Exception as e:
                print(f"Error testing {param_name} = {test_value}: {e}")
                continue
    
    return pd.DataFrame(results)


def calculate_sensitivity_indices(df):
    """
    Calculate sensitivity indices for each parameter.
    
    Args:
        df (pd.DataFrame): Sensitivity analysis results
        
    Returns:
        pd.DataFrame: Sensitivity indices
    """
    sensitivity_indices = []
    
    features = ['baseline_voltage', 'max_voltage', 'voltage_deflection', 'spike_count']
    
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param].copy()
        
        if len(param_data) < 2:
            continue
        
        param_indices = {'parameter': param}
        
        for feature in features:
            if feature in param_data.columns:
                # Calculate sensitivity as normalized standard deviation
                feature_values = param_data[feature].values
                param_changes = param_data['percent_change'].values
                
                if len(feature_values) > 1 and np.std(param_changes) > 0:
                    # Linear regression to find sensitivity slope
                    sensitivity = np.polyfit(param_changes, feature_values, 1)[0]
                    param_indices[f'{feature}_sensitivity'] = abs(sensitivity)
                else:
                    param_indices[f'{feature}_sensitivity'] = 0
        
        sensitivity_indices.append(param_indices)
    
    return pd.DataFrame(sensitivity_indices)


def plot_sensitivity_results(df, sensitivity_df, output_dir):
    """
    Plot sensitivity analysis results.
    
    Args:
        df (pd.DataFrame): Raw sensitivity data
        sensitivity_df (pd.DataFrame): Sensitivity indices
        output_dir (Path): Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Parameter variations vs features
    features = ['voltage_deflection', 'spike_count', 'max_voltage']
    parameters = df['parameter'].unique()
    
    n_params = len(parameters)
    n_features = len(features)
    
    fig, axes = plt.subplots(n_features, n_params, figsize=(4*n_params, 3*n_features))
    if n_params == 1:
        axes = axes.reshape(-1, 1)
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(features):
        for j, param in enumerate(parameters):
            param_data = df[df['parameter'] == param]
            
            if len(param_data) > 0 and feature in param_data.columns:
                axes[i, j].plot(param_data['percent_change'], param_data[feature], 'o-')
                axes[i, j].set_xlabel(f'{param} change (%)')
                axes[i, j].set_ylabel(feature.replace('_', ' ').title())
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_title(f'{param}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Sensitivity indices heatmap
    if not sensitivity_df.empty:
        feature_cols = [col for col in sensitivity_df.columns if col.endswith('_sensitivity')]
        
        if feature_cols:
            sens_matrix = sensitivity_df[['parameter'] + feature_cols].set_index('parameter')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(sens_matrix.values.T, aspect='auto', cmap='viridis')
            
            ax.set_xticks(range(len(sens_matrix.index)))
            ax.set_xticklabels(sens_matrix.index, rotation=45, ha='right')
            ax.set_yticks(range(len(feature_cols)))
            ax.set_yticklabels([col.replace('_sensitivity', '').replace('_', ' ').title() 
                               for col in feature_cols])
            
            ax.set_title('Parameter Sensitivity Matrix')
            plt.colorbar(im, ax=ax, label='Sensitivity Index')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Sensitivity plots saved to: {output_dir}")


def main():
    """Main sensitivity analysis function."""
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    config_file = project_dir / "configs" / "emodel.yaml"
    params_file = project_dir / "optimisation" / "best_parameters.json"
    mechanisms_dir = project_dir / "mechanisms"
    output_dir = project_dir / "sensitivity_analysis"
    
    # Load configuration
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get morphology path
    morph_path = project_dir / config["morphology"]["path"].replace("../", "")
    
    # Load base parameters
    base_params = load_base_parameters(params_file)
    print(f"Loaded {len(base_params)} base parameters")
    
    # Run sensitivity analysis
    print("Running parameter sensitivity analysis...")
    df = parameter_sensitivity_analysis(
        base_params, morph_path, mechanisms_dir,
        variation_percent=20, n_points=5
    )
    
    if df.empty:
        print("No successful sensitivity tests completed")
        return
    
    print(f"Completed {len(df)} sensitivity tests")
    
    # Calculate sensitivity indices
    sensitivity_df = calculate_sensitivity_indices(df)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "sensitivity_data.csv", index=False)
    sensitivity_df.to_csv(output_dir / "sensitivity_indices.csv", index=False)
    
    # Plot results
    plot_sensitivity_results(df, sensitivity_df, output_dir)
    
    # Print summary
    print("\n=== Sensitivity Analysis Summary ===")
    if not sensitivity_df.empty:
        feature_cols = [col for col in sensitivity_df.columns if col.endswith('_sensitivity')]
        for feature_col in feature_cols:
            if feature_col in sensitivity_df.columns:
                most_sensitive = sensitivity_df.loc[sensitivity_df[feature_col].idxmax()]
                print(f"{feature_col.replace('_sensitivity', '').title()}: "
                      f"Most sensitive to {most_sensitive['parameter']} "
                      f"(index: {most_sensitive[feature_col]:.4f})")
    
    print(f"\nResults saved to: {output_dir}")
    print("Sensitivity analysis completed!")


if __name__ == "__main__":
    main()