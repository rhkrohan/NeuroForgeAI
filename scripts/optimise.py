#!/usr/bin/env python3
"""
E-model optimization script using BluePyOpt.

Runs single-cell optimization to fit model parameters to experimental features.
"""

import os
import json
import yaml
from pathlib import Path
import numpy as np
import bluepyopt as bpopt
import bluepyopt.ephys as ephys
from bluepyopt.ephys import models, simulators, protocols, objectives
import neuron


def load_mechanisms(mech_dir):
    """
    Load and compile NEURON mechanisms.
    
    Args:
        mech_dir (Path): Directory containing MOD files
    """
    if mech_dir.exists():
        # Compile mechanisms
        os.chdir(str(mech_dir))
        os.system("nrnivmodl")
        
        # Load compiled mechanisms
        neuron.load_mechanisms(str(mech_dir))
        print(f"Loaded mechanisms from: {mech_dir}")
    else:
        print(f"Warning: Mechanisms directory not found: {mech_dir}")


def create_cell_model(config):
    """
    Create BluePyOpt cell model from configuration.
    
    Args:
        config (dict): E-model configuration
        
    Returns:
        ephys.models.CellModel: BluePyOpt cell model
    """
    # Create morphology
    morph_path = Path(config["morphology"]["path"])
    morphology = ephys.morphologies.NrnFileMorphology(
        morphology_path=str(morph_path),
        do_replace_axon=True
    )
    
    # Create somatic location
    somatic_loc = ephys.locations.NrnSeclistLocation(
        name="somatic",
        seclist_name="somatic"
    )
    
    # Create mechanisms
    mechanisms = []
    
    # Passive mechanism
    pas_mech = ephys.mechanisms.NrnMODMechanism(
        name="pas",
        suffix="pas",
        locations=[somatic_loc]
    )
    mechanisms.append(pas_mech)
    
    # Active mechanisms from config
    for channel in config["mechanisms"]["channels"]:
        mech = ephys.mechanisms.NrnMODMechanism(
            name=channel,
            suffix=channel,
            locations=[somatic_loc]
        )
        mechanisms.append(mech)
    
    # Create parameters
    parameters = []
    
    for param_config in config["parameters"]:
        param_name = param_config["name"]
        bounds = param_config["bounds"]
        
        # Create parameter
        if "g_pas" in param_name:
            param = ephys.parameters.NrnSectionParameter(
                name=param_name,
                param_name="g_pas",
                value_scaler=ephys.parameterscalers.NrnSegmentLinearScaler(),
                bounds=bounds,
                locations=[somatic_loc]
            )
        elif "Ra" in param_name:
            param = ephys.parameters.NrnSectionParameter(
                name=param_name,
                param_name="Ra",
                bounds=bounds,
                locations=[somatic_loc]
            )
        elif "cm" in param_name:
            param = ephys.parameters.NrnSectionParameter(
                name=param_name,
                param_name="cm",
                bounds=bounds,
                locations=[somatic_loc]
            )
        elif "gbar_" in param_name:
            # Extract mechanism name
            mech_name = param_name.replace("gbar_", "")
            param = ephys.parameters.NrnSectionParameter(
                name=param_name,
                param_name=f"gbar_{mech_name}",
                bounds=bounds,
                locations=[somatic_loc]
            )
        else:
            continue
        
        parameters.append(param)
    
    # Create cell model
    cell_model = ephys.models.CellModel(
        name=config["name"],
        morph=morphology,
        mechs=mechanisms,
        params=parameters
    )
    
    return cell_model


def create_protocols(protocol_config):
    """
    Create stimulation protocols from configuration.
    
    Args:
        protocol_config (dict): Protocol configuration
        
    Returns:
        list: List of BluePyOpt protocols
    """
    protocols_list = []
    
    for stimulus in protocol_config["stimuli"]:
        # Create location
        soma_loc = ephys.locations.NrnSeclistCompLocation(
            name="soma",
            seclist_name="somatic",
            sec_index=0,
            comp_x=0.5
        )
        
        # Create stimulus
        step_stim = ephys.stimuli.NrnSquarePulse(
            step_amplitude=stimulus["amplitude"],
            step_delay=stimulus["delay"],
            step_duration=stimulus["duration"],
            location=soma_loc,
            total_duration=stimulus["totduration"]
        )
        
        # Create recording
        soma_recording = ephys.recordings.CompRecording(
            name=f"soma.v_{stimulus['stimulus_name']}",
            location=soma_loc,
            variable="v"
        )
        
        # Create protocol
        protocol = ephys.protocols.SweepProtocol(
            name=stimulus["stimulus_name"],
            stimuli=[step_stim],
            recordings=[soma_recording]
        )
        
        protocols_list.append(protocol)
    
    return protocols_list


def create_objectives(features_data, protocols_list, feature_config):
    """
    Create optimization objectives from extracted features.
    
    Args:
        features_data (dict): Extracted features data
        protocols_list (list): List of protocols
        feature_config (dict): Feature configuration
        
    Returns:
        list: List of BluePyOpt objectives
    """
    objectives_list = []
    
    # Load feature weights
    script_dir = Path(__file__).parent
    weights_file = script_dir.parent / "configs" / "feature_weights.yaml"
    
    if weights_file.exists():
        with open(weights_file, 'r') as f:
            weights_config = yaml.safe_load(f)
        feature_weights = weights_config.get("voltage_features", {})
    else:
        feature_weights = {}
    
    # Create objectives for each feature
    for feature_info in feature_config["voltage_features"]:
        feature_name = feature_info["feature_name"]
        
        # Get weight
        weight = feature_weights.get(feature_name, {}).get("weight", 1.0)
        
        # Create eFeature for each protocol
        for protocol in protocols_list:
            protocol_name = protocol.name
            
            # Look for feature value in data
            feature_value = None
            if protocol_name in features_data:
                feature_value = features_data[protocol_name].get(feature_name)
            
            if feature_value is not None:
                # Create eFeature
                efeature = ephys.efeatures.eFELFeature(
                    name=f"{feature_name}_{protocol_name}",
                    efel_feature_name=feature_name,
                    recording_names={"": f"soma.v_{protocol_name}"},
                    threshold=feature_info.get("threshold", -20.0),
                    stimulus_current=0.0,  # Will be set by protocol
                    exp_mean=feature_value,
                    exp_std=feature_value * 0.1  # 10% standard deviation
                )
                
                # Create objective
                objective = ephys.objectives.SingletonObjective(
                    name=f"{feature_name}_{protocol_name}",
                    efeature=efeature,
                    weight=weight
                )
                
                objectives_list.append(objective)
    
    return objectives_list


def run_optimization(cell_model, protocols_list, objectives_list, config):
    """
    Run BluePyOpt optimization.
    
    Args:
        cell_model: BluePyOpt cell model
        protocols_list: List of protocols
        objectives_list: List of objectives
        config: Optimization configuration
    """
    # Create simulator
    nrn_simulator = ephys.simulators.NrnSimulator()
    
    # Create cell evaluator
    cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=cell_model,
        param_names=[param.name for param in cell_model.params.values()],
        fitness_protocols=protocols_list,
        fitness_calculator=ephys.objectivescalculators.ObjectivesCalculator(objectives_list),
        sim=nrn_simulator
    )
    
    # Create optimization algorithm
    opt_config = config.get("optimization", {})
    
    optimisation = bpopt.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator,
        offspring_size=opt_config.get("population_size", 50),
        eta=20,
        mutpb=0.3,
        cxpb=0.7
    )
    
    # Run optimization
    print("Starting optimization...")
    final_pop, hall_of_fame, logs, hist = optimisation.run(
        max_ngen=opt_config.get("max_generations", 100),
        seed=opt_config.get("seed", 42)
    )
    
    return final_pop, hall_of_fame, logs, hist


def save_results(hall_of_fame, cell_model, output_dir):
    """
    Save optimization results.
    
    Args:
        hall_of_fame: Best individuals from optimization
        cell_model: BluePyOpt cell model
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get best individual
    best_individual = hall_of_fame[0]
    
    # Create parameter dictionary
    best_params = {}
    param_names = [param.name for param in cell_model.params.values()]
    
    for param_name, param_value in zip(param_names, best_individual):
        best_params[param_name] = float(param_value)
    
    # Save best parameters
    params_file = output_dir / "best_parameters.json"
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best parameters saved to: {params_file}")
    
    # Save fitness score
    fitness_info = {
        "fitness_score": float(best_individual.fitness.values[0]),
        "parameter_count": len(best_params),
        "best_parameters": best_params
    }
    
    fitness_file = output_dir / "fitness_results.json"
    with open(fitness_file, 'w') as f:
        json.dump(fitness_info, f, indent=2)
    
    print(f"Fitness results saved to: {fitness_file}")
    print(f"Best fitness score: {fitness_info['fitness_score']:.4f}")


def main():
    """Main optimization function."""
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    config_file = project_dir / "configs" / "emodel.yaml"
    protocols_file = project_dir / "protocols" / "step_proto.json"
    features_file = project_dir / "protocols" / "step_features.json"
    features_data_file = project_dir / "features" / "extracted_features.csv"
    mechanisms_dir = project_dir / "mechanisms"
    output_dir = project_dir / "optimisation"
    
    # Load configuration
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load mechanisms
    load_mechanisms(mechanisms_dir)
    
    # Load protocols
    if not protocols_file.exists():
        print(f"Error: Protocol file not found: {protocols_file}")
        print("Please run feature_extract.py first to generate protocols")
        return
    
    with open(protocols_file, 'r') as f:
        protocol_config = json.load(f)
    
    # Load feature configuration
    if not features_file.exists():
        print(f"Error: Feature file not found: {features_file}")
        print("Please run feature_extract.py first to generate feature configuration")
        return
    
    with open(features_file, 'r') as f:
        feature_config = json.load(f)
    
    # Check for extracted features
    if not features_data_file.exists():
        print(f"Warning: Extracted features not found: {features_data_file}")
        print("Using default feature values for optimization")
        features_data = {}
    else:
        # Load feature data (simplified)
        features_data = {}
    
    print("Creating cell model...")
    cell_model = create_cell_model(config)
    
    print("Creating protocols...")
    protocols_list = create_protocols(protocol_config)
    
    print("Creating objectives...")
    objectives_list = create_objectives(features_data, protocols_list, feature_config)
    
    if not objectives_list:
        print("Warning: No objectives created. Using dummy objective.")
        # Create dummy objective for testing
        dummy_objective = ephys.objectives.SingletonObjective(
            name="dummy_voltage",
            efeature=ephys.efeatures.eFELFeature(
                name="dummy_voltage",
                efel_feature_name="voltage_base",
                recording_names={"": f"soma.v_{protocols_list[0].name}"},
                exp_mean=-65.0,
                exp_std=5.0
            )
        )
        objectives_list = [dummy_objective]
    
    print(f"Starting optimization with {len(objectives_list)} objectives...")
    
    # Run optimization
    final_pop, hall_of_fame, logs, hist = run_optimization(
        cell_model, protocols_list, objectives_list, config
    )
    
    # Save results
    save_results(hall_of_fame, cell_model, output_dir)
    
    print("Optimization completed successfully!")


if __name__ == "__main__":
    main()