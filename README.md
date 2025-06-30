# Neuron E-Model Optimization Pipeline

A comprehensive, reproducible Git-based pipeline for single-cell electrophysiological model optimization using advanced computational neuroscience tools. This project combines morphologically realistic neuron models with biophysically detailed ion channel mechanisms to create optimized electrical models (e-models) that match experimental patch-clamp data.

## Scientific Background & Methodology

### Computational Neuroscience Approach

This pipeline implements a **data-driven model optimization workflow** that bridges experimental electrophysiology with computational modeling:

1. **Morphological Reconstruction**: Uses digitally reconstructed neuron morphologies from the Allen Brain Atlas
2. **Biophysical Modeling**: Implements detailed Hodgkin-Huxley type ion channel mechanisms 
3. **Feature-Based Optimization**: Extracts quantitative features from experimental recordings
4. **Multi-objective Optimization**: Uses evolutionary algorithms to fit model parameters
5. **Model Validation**: Validates optimized models against held-out experimental data

### Key Methodologies

#### 1. **Feature Extraction Methodology**
- **eFEL (Electrophysiology Feature Extraction Library)**: Extracts >170 electrophysiological features
- **BluePyEFE**: Automated feature extraction from experimental voltage traces
- **Feature Categories**:
  - **Spike features**: Action potential amplitude, width, threshold, afterhyperpolarization
  - **Firing patterns**: Spike count, inter-spike intervals, adaptation indices
  - **Subthreshold**: Input resistance, membrane time constant, sag amplitude
  - **Frequency-current**: F-I curve characteristics, rheobase current

#### 2. **Multi-Objective Optimization Methodology**
- **BluePyOpt Framework**: Uses DEAP (Distributed Evolutionary Algorithms in Python)
- **CMA-ES Algorithm**: Covariance Matrix Adaptation Evolution Strategy
- **Objective Function**: Weighted sum of feature errors between model and experiment
- **Parameter Space**: Explores physiologically realistic parameter bounds
- **Convergence Criteria**: Monitors fitness evolution and parameter stability

#### 3. **Model Validation Methodology**
- **Cross-validation**: Tests optimized models on independent stimulus protocols
- **Sensitivity Analysis**: Identifies critical parameters using local perturbation analysis
- **Robustness Testing**: Evaluates model behavior across parameter uncertainty ranges

## Overview

This pipeline:
1. **Loads realistic neuron morphology** (.swc format) into NEURON simulator
2. **Inserts custom ion-channel mechanisms** (.mod files) for biophysical realism
3. **Extracts electrophysiological features** from patch-clamp recordings using BluePyEFE
4. **Optimizes passive and active parameters** using BluePyOpt evolutionary algorithms
5. **Validates optimized models** and provides comprehensive sensitivity analysis

## Repository Structure & File Descriptions

```
neuron-emodel-optimisation/
├── environment.yml              # Conda environment definition
├── .gitignore                  # Git ignore patterns
├── README.md                   # This comprehensive documentation
├── morphologies/               # Neuron morphology files
│   └── 720575940622093546_obaid.swc
├── mechanisms/                 # NEURON mechanism files (.mod)
│   ├── README.md
│   ├── NaTa_t.mod             # Transient sodium channel
│   ├── Kv3_1.mod              # Fast delayed rectifier K+
│   ├── Ca_HVA.mod             # High voltage-activated Ca2+
│   └── Ih.mod                 # HCN channel
├── recordings/                 # Experimental recordings (CSV/ABF)
│   └── example_recording.csv   # Example data format
├── protocols/                  # Stimulus protocols and feature definitions (auto-generated)
├── configs/                    # Configuration files
│   ├── emodel.yaml            # E-model optimization config
│   └── feature_weights.yaml   # Feature importance weights
├── scripts/                    # Analysis scripts
│   ├── Obaid_simulation_file.py  # Original simulation (legacy)
│   ├── load_morph_check.py      # Morphology visualization
│   ├── feature_extract.py       # Feature extraction
│   ├── optimise.py              # Parameter optimization
│   ├── run_best.py              # Run with best parameters
│   ├── run_obaid.py             # Refactored Obaid simulation
│   └── sensitivity_analysis.py  # Parameter sensitivity
├── optimisation/               # Optimization outputs (ignored by git)
├── features/                   # Extracted features (auto-generated)
├── sensitivity_analysis/       # Sensitivity analysis results (auto-generated)
└── .github/workflows/          # CI/CD pipeline
    └── ci.yml
```

### Core Files & Directories Explained

#### **Root Level Files**

- **`environment.yml`**: Defines reproducible conda environment with pinned versions
  - Python 3.11.5, NEURON 8.2.4, BluePyOpt 1.14.18, BluePyEFE 2.4.7
  - Ensures consistent computational environment across systems
  - Includes MPI support for parallel optimization

- **`.gitignore`**: Excludes generated files from version control
  - Compiled mechanisms (`x86_64/`, `*.o`, `*.c`)
  - Optimization outputs (`optimisation/`, `*.pkl`)
  - Large data files (handled by Git LFS)

#### **`morphologies/` Directory**

Contains digitally reconstructed neuron morphologies in SWC format:

- **`720575940622093546_obaid.swc`**: Allen Brain Atlas neuron morphology
  - **SWC Format**: Standard format for neuronal morphology data
  - **Coordinates**: 3D point coordinates defining dendrites, soma, axon
  - **Topology**: Parent-child relationships between morphological segments
  - **Applications**: Realistic spatial distribution of ion channels and synapses

**SWC File Structure**:
```
# Column format: ID Type X Y Z Radius Parent
1 1 0.0 0.0 0.0 10.0 -1    # Soma
2 3 0.0 0.0 10.0 5.0 1     # Dendrite
```

#### **`mechanisms/` Directory**

Contains NEURON mechanism files (.mod) implementing ion channel kinetics:

- **`NaTa_t.mod`**: **Transient Sodium Channel (Na+ fast)**
  - Implements fast sodium current responsible for action potential upstroke
  - Hodgkin-Huxley formulation: `I_Na = gbar * m³ * h * (V - E_Na)`
  - Voltage-dependent activation (m) and inactivation (h) gates

- **`Kv3_1.mod`**: **Fast Delayed Rectifier Potassium Channel**
  - Implements fast potassium current for action potential repolarization
  - Critical for high-frequency firing in fast-spiking interneurons
  - Single activation gate: `I_K = gbar * n⁴ * (V - E_K)`

- **`Ca_HVA.mod`**: **High Voltage-Activated Calcium Channel**
  - L-type calcium channels activated at depolarized potentials
  - Provides calcium influx for calcium-dependent processes
  - Dual gating: `I_Ca = gbar * m² * h * (V - E_Ca)`

- **`Ih.mod`**: **Hyperpolarization-Activated Cation Channel (HCN)**
  - Non-selective cation channel activated by hyperpolarization
  - Contributes to resting potential and subthreshold oscillations
  - Creates "sag" response to hyperpolarizing current injection

**MOD File Compilation**:
- Run `nrnivmodl` in mechanisms directory to compile
- Creates architecture-specific shared libraries
- Must recompile when moving between systems

#### **`recordings/` Directory**

Stores experimental electrophysiological recordings:

- **Format Requirements**: CSV files with columns: `time (ms), voltage (mV), current (nA)`
- **Protocol**: Current-clamp recordings with step current injections
- **Data Types**: Voltage responses to various current amplitudes
- **Git LFS**: Large binary files (.abf) automatically handled by Git LFS

**Example Data Structure**:
```csv
time,voltage,current
0.0,-65.0,0.0      # Baseline
500.0,-65.0,0.1    # Current step onset
1500.0,-45.0,0.1   # Depolarized response
2000.0,-65.0,0.0   # Return to baseline
```

#### **`configs/` Directory**

Configuration files defining optimization parameters and feature weights:

- **`emodel.yaml`**: **Primary E-Model Configuration**
  - **Morphology**: Path to SWC file and cell name
  - **Mechanisms**: List of ion channels to include
  - **Parameters**: Optimization variables with bounds
  - **Regions**: Anatomical regions for parameter assignment
  - **Optimization**: Algorithm settings (CMA-ES, population size, generations)

- **`feature_weights.yaml`**: **Feature Importance Weights**
  - **Spike Features**: Action potential characteristics (amplitude, width, threshold)
  - **Firing Patterns**: Spike count, ISI statistics, adaptation
  - **Subthreshold**: Passive membrane properties, sag responses
  - **Protocol Weights**: Differential weighting for current step amplitudes

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd neuron-emodel-optimisation

# Create conda environment
conda env create -f environment.yml
conda activate neuron-emodel-opt

# Compile NEURON mechanisms
cd mechanisms
nrnivmodl
cd ..
```

### 2. Morphology Check

```bash
# Visualize loaded morphology
python scripts/load_morph_check.py
```

### 3. Feature Extraction (with experimental data)

```bash
# Extract features from recordings
python scripts/feature_extract.py
```

### 4. Parameter Optimization

```bash
# Run optimization
python scripts/optimise.py
```

### 5. Model Validation

```bash
# Run simulation with optimized parameters
python scripts/run_best.py

# Run original Obaid simulation with optimized parameters
python scripts/run_obaid.py
```

### 6. Sensitivity Analysis

```bash
# Analyze parameter sensitivity
python scripts/sensitivity_analysis.py
```

## Configuration

### E-Model Configuration (`configs/emodel.yaml`)

Define optimization parameters, mechanisms, and bounds:

```yaml
name: "obaid_emodel"
morphology:
  path: "../morphologies/720575940622093546_obaid.swc"
mechanisms:
  channels:
    - "NaTa_t"
    - "Kv3_1"
    - "Ca_HVA"
    - "Ih"
parameters:
  - name: "g_pas"
    bounds: [1e-6, 1e-3]
  - name: "gbar_NaTa_t"
    bounds: [0, 2.0]
```

### Feature Weights (`configs/feature_weights.yaml`)

Control optimization priorities:

```yaml
voltage_features:
  Spikecount:
    weight: 10.0
  AP_amplitude:
    weight: 8.0
  AP_threshold:
    weight: 8.0
```

## Data Requirements

### Experimental Recordings

Place voltage recordings in `recordings/` directory:
- **Format**: CSV files with columns: time (ms), voltage (mV), current (nA)
- **Protocol**: Current step protocol recommended
- **Git LFS**: Large files automatically handled

### Morphology

- **Format**: SWC format
- **Location**: `morphologies/720575940622093546_obaid.swc`
- **Source**: Allen Brain Atlas or similar

#### **`scripts/` Directory**

The core analysis scripts implementing the optimization workflow:

### **`scripts/load_morph_check.py`** - Morphology Validation
**Purpose**: Validates and visualizes neuron morphology loading
**Key Functions**:
- `load_and_plot_morphology()`: Loads SWC file using NEURON's Import3d
- `main()`: Creates XY scatter plot of all morphological segments
**Outputs**: 
- `morphology_check.png`: 2D visualization of neuron structure
- Console statistics: segment count, section count
**Usage**: `python scripts/load_morph_check.py`

### **`scripts/feature_extract.py`** - Electrophysiological Feature Extraction
**Purpose**: Extracts quantitative features from experimental recordings using BluePyEFE
**Key Functions**:
- `create_step_protocol()`: Defines current-clamp stimulus protocols
- `create_feature_configuration()`: Specifies eFEL features to extract
- `load_recordings()`: Imports CSV voltage data
- `extract_features()`: Runs BluePyEFE feature extraction pipeline
**Outputs**:
- `protocols/step_proto.json`: Stimulus protocol definition
- `protocols/step_features.json`: Feature extraction configuration  
- `features/extracted_features.csv`: Quantified electrophysiological features
- `features/feature_summary.csv`: Statistical summary of features
**Usage**: `python scripts/feature_extract.py`

### **`scripts/optimise.py`** - Parameter Optimization Engine
**Purpose**: Runs BluePyOpt multi-objective optimization to fit model parameters
**Key Functions**:
- `load_mechanisms()`: Compiles and loads NEURON mechanisms
- `create_cell_model()`: Builds BluePyOpt cell model from configuration
- `create_protocols()`: Generates stimulation protocols for optimization
- `create_objectives()`: Defines objective functions from experimental features
- `run_optimization()`: Executes CMA-ES evolutionary optimization
- `save_results()`: Stores best parameters and fitness scores
**Optimization Process**:
1. **Population Initialization**: Random parameter sampling within bounds
2. **Fitness Evaluation**: Simulate model responses and calculate feature errors
3. **Selection & Mutation**: Evolutionary operators to improve parameter sets
4. **Convergence**: Monitors fitness evolution until stopping criteria
**Outputs**:
- `optimisation/best_parameters.json`: Optimized parameter values
- `optimisation/fitness_results.json`: Optimization convergence data
**Usage**: `python scripts/optimise.py`

### **`scripts/run_best.py`** - Model Validation & Analysis
**Purpose**: Validates optimized model against experimental data
**Key Functions**:
- `load_best_parameters()`: Loads optimized parameters from JSON
- `create_optimized_cell()`: Instantiates NEURON model with best parameters
- `run_validation_simulation()`: Tests model with current-clamp protocols
- `plot_validation_results()`: Generates voltage trace and I-V plots
- `analyze_model_properties()`: Calculates rheobase, input resistance, spike counts
**Validation Metrics**:
- **Rheobase**: Minimum current to elicit action potentials
- **Input Resistance**: Slope of subthreshold I-V relationship
- **Spike Patterns**: Action potential counts vs. current amplitude
- **I-V Curves**: Steady-state voltage responses
**Outputs**:
- `validation_results.png`: Multi-panel validation plots
- Console analysis: Quantitative electrophysiological properties
**Usage**: `python scripts/run_best.py`

### **`scripts/run_obaid.py`** - Refactored Original Simulation
**Purpose**: Modernized version of original Obaid simulation using optimized parameters
**Key Architectural Changes**:
```python
# OLD: Hard-coded parameters
cell.Ra = 100
cell.g_pas = 0.0001
cell.gbar_NaTa_t = 0.5

# NEW: Load from optimization results  
cell = instantiate_emodel(project_dir)  # Uses best_parameters.json
```
**Key Functions**:
- `instantiate_emodel()`: Loads optimized e-model from results directory
- `setup_stimulation()`: Recreates original current-clamp protocol
- `run_simulation()`: Executes NEURON simulation with optimized parameters
- `plot_results()`: Generates publication-quality voltage trace plots
**Integration**: Seamlessly replaces hard-coded values with data-driven optimization
**Usage**: `python scripts/run_obaid.py`

### **`scripts/sensitivity_analysis.py`** - Parameter Sensitivity Analysis
**Purpose**: Quantifies parameter influence on model behavior using local perturbation analysis
**Key Functions**:
- `parameter_sensitivity_analysis()`: Systematically varies each parameter
- `calculate_sensitivity_indices()`: Computes normalized sensitivity measures
- `plot_sensitivity_results()`: Creates sensitivity curves and heatmaps
**Methodology**:
1. **Parameter Perturbation**: ±20% variation around optimized values
2. **Feature Response**: Measures changes in key electrophysiological features
3. **Sensitivity Index**: Normalized slope of feature vs. parameter relationship
4. **Ranking**: Identifies most influential parameters for model behavior
**Outputs**:
- `sensitivity_analysis/sensitivity_data.csv`: Raw perturbation results
- `sensitivity_analysis/sensitivity_indices.csv`: Computed sensitivity measures
- `sensitivity_analysis/sensitivity_curves.png`: Parameter-feature relationships
- `sensitivity_analysis/sensitivity_heatmap.png`: Sensitivity matrix visualization
**Usage**: `python scripts/sensitivity_analysis.py`

### **`scripts/Obaid_simulation_file.py`** - Legacy Original Code
**Purpose**: Original simulation script (preserved for reference)
**Status**: Legacy code with hard-coded parameters
**Note**: Use `run_obaid.py` for modernized version with optimized parameters

## Complete Model Execution Workflow

### **Phase 1: Environment Setup & Preparation**

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd neuron-emodel-optimisation
conda env create -f environment.yml
conda activate neuron-emodel-opt

# 2. Compile NEURON mechanisms (REQUIRED)
cd mechanisms
nrnivmodl  # Creates x86_64/ directory with compiled mechanisms
cd ..

# 3. Verify morphology loading
python scripts/load_morph_check.py
# Output: morphology_check.png, console statistics
```

### **Phase 2: Data Preparation & Feature Extraction**

```bash
# 4. Add experimental recordings (REQUIRED FOR REAL OPTIMIZATION)
# Place CSV files in recordings/ directory with format:
# time(ms), voltage(mV), current(nA)

# 5. Extract electrophysiological features
python scripts/feature_extract.py
# Outputs:
# - protocols/step_proto.json (stimulus protocols)
# - protocols/step_features.json (feature definitions)  
# - features/extracted_features.csv (quantified features)
# - features/feature_summary.csv (statistical summary)
```

### **Phase 3: Parameter Optimization**

```bash
# 6. Configure optimization parameters
# Edit configs/emodel.yaml to adjust:
# - Parameter bounds
# - Optimization algorithm settings
# - Feature weights

# 7. Run multi-objective optimization (COMPUTATIONALLY INTENSIVE)
python scripts/optimise.py
# Duration: 30 minutes - 24 hours depending on:
# - Population size (default: 50)
# - Generations (default: 100)  
# - Number of parameters (~12-15)
# - Available CPU cores

# Outputs:
# - optimisation/best_parameters.json (optimized values)
# - optimisation/fitness_results.json (convergence data)
```

### **Phase 4: Model Validation & Analysis**

```bash
# 8. Validate optimized model
python scripts/run_best.py
# Outputs:
# - validation_results.png (voltage traces, I-V curves)
# - Console analysis (rheobase, input resistance, etc.)

# 9. Run original simulation with optimized parameters
python scripts/run_obaid.py  
# Outputs:
# - obaid_simulation_results.png (voltage responses)
# - Demonstrates integration with existing code

# 10. Perform sensitivity analysis
python scripts/sensitivity_analysis.py
# Outputs:
# - sensitivity_analysis/sensitivity_data.csv
# - sensitivity_analysis/sensitivity_indices.csv
# - sensitivity_analysis/sensitivity_curves.png
# - sensitivity_analysis/sensitivity_heatmap.png
```

### **Expected Timeline & Computational Requirements**

| Phase | Duration | CPU Requirements | Key Bottlenecks |
|-------|----------|------------------|-----------------|
| Setup | 10-30 min | 1 core | Environment installation |
| Feature Extraction | 2-10 min | 1 core | Data loading, eFEL computation |
| **Optimization** | **0.5-24 hours** | **4-32 cores** | **Population evaluation** |
| Validation | 5-15 min | 1 core | Model simulation |
| Sensitivity | 15-60 min | 1-4 cores | Parameter perturbation |

### **Advanced Execution Options**

#### **Parallel Optimization (Recommended)**
```bash
# Use MPI for parallel optimization
mpirun -n 8 python scripts/optimise.py
# Scales with available CPU cores (recommended: 4-16 cores)
```

#### **Configuration Customization**
```bash
# Quick optimization (for testing)
# Edit configs/emodel.yaml:
population_size: 20    # Reduced from 50
max_generations: 25    # Reduced from 100

# High-precision optimization (for publication)
population_size: 100   # Increased from 50  
max_generations: 200   # Increased from 100
```

#### **Batch Processing Multiple Cells**
```bash
# Process multiple morphologies
for morph in morphologies/*.swc; do
    # Update configs/emodel.yaml with new morphology path
    python scripts/optimise.py
    mv optimisation/best_parameters.json "results_$(basename $morph .swc).json"
done
```

### **Quality Control & Validation Checks**

#### **Pre-Optimization Validation**
```bash
# Verify all components before optimization
python -c "
import neuron; 
neuron.load_mechanisms('./mechanisms');
print('✓ Mechanisms loaded successfully')

import bluepyopt, bluepyefe, efel;
print('✓ All packages imported successfully')

from pathlib import Path;
assert Path('morphologies/720575940622093546_obaid.swc').exists();
print('✓ Morphology file found')
"
```

#### **Post-Optimization Validation**
```bash
# Check optimization results quality
python -c "
import json
with open('optimisation/fitness_results.json') as f:
    results = json.load(f)
print(f'Final fitness: {results[\"fitness_score\"]:.4f}')
print(f'Parameters optimized: {results[\"parameter_count\"]}')
# Good fitness: < 10.0, Excellent fitness: < 5.0
"
```

### **Troubleshooting Common Issues**

#### **Mechanism Compilation Failures**
```bash
# Clean and recompile mechanisms
cd mechanisms
rm -rf x86_64/ *.o *.c nrnmech.dll
nrnivmodl
# Check for error messages and missing dependencies
```

#### **Optimization Not Converging**
```bash
# Reduce parameter space complexity
# In configs/emodel.yaml, comment out some parameters:
# parameters:
#   - name: "g_pas"
#     bounds: [1e-6, 1e-3]
#   # - name: "gbar_NaTa_t"  # Temporarily disable
#   #   bounds: [0, 2.0]
```

#### **Memory Issues During Optimization**
```bash
# Reduce population size and enable checkpointing
# Edit configs/emodel.yaml:
optimization:
  population_size: 25    # Reduced from 50
  save_checkpoints: true # Enable periodic saving
```

This comprehensive workflow ensures reproducible, high-quality e-model optimization from experimental data to validated computational models.

## Auto-Generated Directories

During pipeline execution, several directories are automatically created:

#### **`protocols/` Directory** (Auto-generated by `feature_extract.py`)
- **`step_proto.json`**: Stimulus protocol definitions for current-clamp experiments
- **`step_features.json`**: eFEL feature extraction configuration
- **Purpose**: Standardizes experimental protocols for optimization

#### **`features/` Directory** (Auto-generated by `feature_extract.py`)
- **`extracted_features.csv`**: Quantified electrophysiological features from recordings
- **`feature_summary.csv`**: Statistical summary and quality metrics
- **Purpose**: Provides optimization targets from experimental data

#### **`optimisation/` Directory** (Auto-generated by `optimise.py`)
- **`best_parameters.json`**: Optimized parameter values
- **`fitness_results.json`**: Optimization convergence and quality metrics
- **`generation_*.pkl`**: Checkpoints for resuming interrupted optimizations
- **Purpose**: Stores optimization results and enables model reconstruction

#### **`sensitivity_analysis/` Directory** (Auto-generated by `sensitivity_analysis.py`)
- **`sensitivity_data.csv`**: Raw parameter perturbation results
- **`sensitivity_indices.csv`**: Computed sensitivity measures
- **`sensitivity_curves.png`**: Parameter-feature relationship plots
- **`sensitivity_heatmap.png`**: Sensitivity matrix visualization
- **Purpose**: Quantifies parameter importance and model robustness

## Continuous Integration & Quality Assurance

### **`.github/workflows/ci.yml`** - Automated Testing Pipeline

The CI/CD workflow ensures code quality and reproducibility:

#### **Test Environment Setup**
- **Conda Environment**: Automated installation of pinned dependencies
- **Mechanism Compilation**: Validates MOD file compilation across platforms  
- **Package Verification**: Tests all required scientific computing libraries

#### **Automated Test Suite**
```yaml
# Key CI components:
1. Environment Setup:     conda env create -f environment.yml
2. Mechanism Compilation: nrnivmodl mechanisms/  
3. Morphology Loading:    python scripts/load_morph_check.py
4. Configuration Validation: YAML syntax and schema checking
5. Import Testing:        All required packages (NEURON, BluePyOpt, etc.)
6. Smoke Tests:          Basic functionality without full optimization
7. Code Quality:         Black formatting, isort, flake8 linting
```

#### **Platform Support**
- **Primary**: Ubuntu Linux (GitHub Actions)
- **Compatibility**: macOS, Windows (via conda environment)
- **Architecture**: x86_64 (Intel/AMD), ARM64 (Apple Silicon with Rosetta)

## Performance Optimization & Scalability

### **Computational Resource Requirements**

#### **Memory Requirements**
```
Minimum:  8 GB RAM (basic optimization)
Recommended: 16-32 GB RAM (full optimization)
Optimal: 64+ GB RAM (large population sizes)
```

#### **CPU Scaling**
```python
# MPI parallel optimization scaling
mpirun -n 1 python scripts/optimise.py   # ~24 hours
mpirun -n 4 python scripts/optimise.py   # ~6 hours  
mpirun -n 8 python scripts/optimise.py   # ~3 hours
mpirun -n 16 python scripts/optimise.py  # ~1.5 hours
# Scaling efficiency: ~70-80% up to 16 cores
```

#### **Storage Requirements**
```
Source Code:       ~50 MB
Morphology Data:   ~10-100 MB  
Experimental Data: ~100 MB - 10 GB (depending on recording count)
Optimization Results: ~50-500 MB
Total Project Size: ~500 MB - 15 GB
```

### **High-Performance Computing (HPC) Integration**

#### **SLURM Cluster Example**
```bash
#!/bin/bash
#SBATCH --job-name=emodel_opt
#SBATCH --ntasks=32
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute

module load conda/miniconda3
conda activate neuron-emodel-opt

cd mechanisms && nrnivmodl && cd ..
mpirun -n $SLURM_NTASKS python scripts/optimise.py
```

#### **AWS/Cloud Computing**
```bash
# Recommended instance types:
# c5.4xlarge (16 vCPUs, 32 GB RAM) - Standard optimization
# c5.9xlarge (36 vCPUs, 72 GB RAM) - Large-scale optimization  
# r5.4xlarge (16 vCPUs, 128 GB RAM) - Memory-intensive workloads
```

## Advanced Configuration & Customization

### **Parameter Space Exploration**

#### **Adaptive Bounds Strategy**
```yaml
# configs/emodel.yaml - Advanced parameter configuration
parameters:
  - name: "g_pas"
    bounds: [1e-6, 1e-3]
    distribution: "log-uniform"    # Log-scale sampling
    prior: "experimental_range"    # Use experimental constraints
    
  - name: "gbar_NaTa_t"  
    bounds: [0, 2.0]
    distribution: "uniform"
    coupling: ["gbar_Kv3_1"]      # Coupled parameter optimization
```

#### **Multi-Objective Optimization**
```yaml
# Advanced objective function configuration
objectives:
  spike_features:
    weight: 0.4
    features: ["AP_amplitude", "AP_width", "Spikecount"]
  
  subthreshold_features:
    weight: 0.3  
    features: ["input_resistance", "sag_amplitude"]
    
  firing_patterns:
    weight: 0.3
    features: ["ISI_mean", "adaptation_index"]
```

### **Experimental Protocol Customization**

#### **Custom Stimulus Protocols**
```python
# Example: Add ramp current protocol
def create_ramp_protocol(start_amp, end_amp, duration):
    return {
        "name": "ramp_current",
        "type": "current_ramp", 
        "start_amplitude": start_amp,
        "end_amplitude": end_amp,
        "duration": duration,
        "sample_rate": 40000  # 40 kHz sampling
    }
```

#### **Feature Engineering**
```python
# Custom feature extraction for specific cell types
custom_features = [
    "burst_ISI_mean",           # Burst firing patterns
    "rebound_spike_count",      # Post-hyperpolarization rebounds  
    "adaptation_index_2",       # Spike frequency adaptation
    "membrane_time_constant",   # Passive membrane properties
    "rheobase_current"          # Minimum spike threshold
]
```

This comprehensive documentation provides complete guidance for implementing, customizing, and scaling the neuron e-model optimization pipeline for diverse computational neuroscience applications.

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Description"`
4. Push branch: `git push origin feature-name`
5. Create pull request

## Dependencies

- **Python 3.11**
- **NEURON 8.2.4** (with Python interface)
- **BluePyOpt ≥1.14.18**
- **BluePyEFE 2.4.7**
- **eFEL 5.6.8**
- **singlecell-emodel-suite 0.2.0**

## License

This project is licensed under the MIT License.

## Acknowledgments

- Allen Brain Atlas for morphology data
- Blue Brain Project for optimization tools
- NEURON simulation environment

## References

1. Van Geit et al. (2016) BluePyOpt: Leveraging Open Source Software and Cloud Infrastructure to Optimise Model Parameters in Neuroscience. Front. Neuroinform.
2. Markram et al. (2015) Reconstruction and Simulation of Neocortical Microcircuitry. Cell.
3. Hines & Carnevale (1997) The NEURON Simulation Environment. Neural Comput.