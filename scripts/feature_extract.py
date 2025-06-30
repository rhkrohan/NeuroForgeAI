#!/usr/bin/env python3
"""
Feature extraction script using BluePyEfe.

Extracts electrophysiological features from voltage recordings
for use in e-model optimization.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
import bluepyefe as bpefe
from bluepyefe.reader import BluepyEfeReader
from bluepyefe.cell import Cell
from bluepyefe.protocol import Protocol
from bluepyefe.feature import Feature


def create_step_protocol(amplitudes, delay=500, duration=1000, total_time=2000):
    """
    Create step current protocol configuration.
    
    Args:
        amplitudes (list): List of current amplitudes in nA
        delay (float): Stimulus delay in ms
        duration (float): Stimulus duration in ms
        total_time (float): Total recording time in ms
        
    Returns:
        dict: Protocol configuration
    """
    protocol = {
        "name": "step_current",
        "type": "current_clamp",
        "parameters": {
            "delay": delay,
            "duration": duration,
            "total_time": total_time,
            "amplitudes": amplitudes
        },
        "stimuli": []
    }
    
    for i, amp in enumerate(amplitudes):
        stimulus = {
            "stimulus_name": f"step_{i:02d}",
            "amplitude": amp,
            "delay": delay,
            "duration": duration,
            "totduration": total_time
        }
        protocol["stimuli"].append(stimulus)
    
    return protocol


def create_feature_configuration():
    """
    Create feature extraction configuration.
    
    Returns:
        dict: Feature configuration
    """
    features = {
        "voltage_features": [
            # Spike count features
            {
                "feature_name": "Spikecount",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            
            # Spike timing features
            {
                "feature_name": "ISI_mean",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            {
                "feature_name": "ISI_CV",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            
            # Spike shape features
            {
                "feature_name": "AP_amplitude",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            {
                "feature_name": "AP_width",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            {
                "feature_name": "AHP_depth",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            {
                "feature_name": "AP_threshold",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            
            # Passive features
            {
                "feature_name": "voltage_base",
                "threshold": -20.0,
                "strict_stiminterval": False
            },
            {
                "feature_name": "steady_state_voltage",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            {
                "feature_name": "sag_amplitude",
                "threshold": -20.0,
                "strict_stiminterval": True
            },
            {
                "feature_name": "sag_ratio1",
                "threshold": -20.0,
                "strict_stiminterval": True
            }
        ]
    }
    
    return features


def load_recordings(recordings_dir, file_pattern="*.csv"):
    """
    Load voltage recordings from CSV files.
    
    Args:
        recordings_dir (Path): Directory containing recordings
        file_pattern (str): File pattern to match
        
    Returns:
        dict: Dictionary of loaded recordings
    """
    recordings = {}
    
    for file_path in recordings_dir.glob(file_pattern):
        try:
            # Load CSV file
            data = pd.read_csv(file_path)
            
            # Assume columns are: time, voltage, current
            if len(data.columns) >= 2:
                recordings[file_path.stem] = {
                    'time': data.iloc[:, 0].values,
                    'voltage': data.iloc[:, 1].values,
                    'current': data.iloc[:, 2].values if len(data.columns) > 2 else None
                }
                print(f"Loaded recording: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return recordings


def extract_features(recordings, protocol_config, feature_config, output_dir):
    """
    Extract features from recordings using BluePyEfe.
    
    Args:
        recordings (dict): Dictionary of recordings
        protocol_config (dict): Protocol configuration
        feature_config (dict): Feature configuration
        output_dir (Path): Output directory for results
    """
    # Create BluePyEFE reader
    reader = BluepyEfeReader()
    
    # Create cell object
    cell = Cell(name="obaid_cell")
    
    # Add recordings to cell
    for rec_name, rec_data in recordings.items():
        # Create protocol
        protocol = Protocol(
            name=protocol_config["name"],
            stimulus=protocol_config["parameters"]
        )
        
        # Add recording
        cell.add_recording(
            protocol_name=protocol_config["name"],
            recording_name=rec_name,
            time=rec_data['time'],
            voltage=rec_data['voltage'],
            current=rec_data.get('current')
        )
    
    # Extract features
    print("Extracting features...")
    
    try:
        # Run feature extraction
        features_df = bpefe.extract_features(
            cells=[cell],
            protocols=[protocol_config],
            feature_list=feature_config["voltage_features"]
        )
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature DataFrame
        features_csv = output_dir / "extracted_features.csv"
        features_df.to_csv(features_csv, index=False)
        print(f"Features saved to: {features_csv}")
        
        # Save summary statistics
        summary = features_df.groupby('feature_name').agg({
            'feature_value': ['mean', 'std', 'count']
        }).round(4)
        
        summary_csv = output_dir / "feature_summary.csv"
        summary.to_csv(summary_csv)
        print(f"Feature summary saved to: {summary_csv}")
        
        return features_df
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise


def main():
    """Main function to run feature extraction."""
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    recordings_dir = project_dir / "recordings"
    protocols_dir = project_dir / "protocols"
    output_dir = project_dir / "features"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Define current amplitudes (example values)
    current_amplitudes = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    
    # Create protocol configuration
    protocol_config = create_step_protocol(current_amplitudes)
    
    # Save protocol configuration
    protocols_dir.mkdir(exist_ok=True)
    protocol_file = protocols_dir / "step_proto.json"
    with open(protocol_file, 'w') as f:
        json.dump(protocol_config, f, indent=2)
    print(f"Protocol configuration saved to: {protocol_file}")
    
    # Create feature configuration
    feature_config = create_feature_configuration()
    
    # Save feature configuration
    feature_file = protocols_dir / "step_features.json"
    with open(feature_file, 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"Feature configuration saved to: {feature_file}")
    
    # Check if recordings exist
    if not recordings_dir.exists() or not any(recordings_dir.glob("*.csv")):
        print(f"Warning: No recordings found in {recordings_dir}")
        print("Please add voltage recording files (CSV format) to the recordings/ directory")
        print("Expected format: time (ms), voltage (mV), current (nA)")
        return
    
    # Load recordings
    recordings = load_recordings(recordings_dir)
    
    if not recordings:
        print("No valid recordings found")
        return
    
    print(f"Found {len(recordings)} recordings")
    
    # Extract features
    features_df = extract_features(recordings, protocol_config, feature_config, output_dir)
    
    print("Feature extraction completed successfully!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()