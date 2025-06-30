# Model Evaluation Directory - NeuroForge-Optimizer

This directory contains comprehensive tools for evaluating, testing, and validating neuron e-models against experimental data. The scripts provide systematic approaches to assess model quality, parameter sensitivity, and electrophysiological accuracy.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Evaluation Concepts](#model-evaluation-concepts)
- [Script Documentation](#script-documentation)
- [Evaluation Workflows](#evaluation-workflows)
- [Performance Metrics](#performance-metrics)
- [Usage Examples](#usage-examples)

## 🎯 Overview

Model evaluation is crucial for ensuring that computational neuron models accurately represent biological neurons. This directory provides tools for:

- **Quantitative validation** against experimental recordings
- **Parameter sensitivity analysis** to understand model robustness
- **Stimulation protocol testing** to characterize model behavior
- **Performance benchmarking** to compare different model versions
- **Quality assessment** for model publication and sharing

## 🧠 Model Evaluation Concepts

### What is Model Evaluation?

Model evaluation in computational neuroscience involves comparing model predictions with experimental data to assess:

1. **Accuracy**: How well does the model reproduce experimental observations?
2. **Robustness**: How stable is the model behavior across parameter variations?
3. **Generalization**: Can the model predict responses to new stimuli?
4. **Biological Realism**: Do the model mechanisms reflect known biology?

### Key Evaluation Principles

- **Feature-based comparison**: Compare specific electrophysiological features (spike count, frequency, amplitude)
- **Statistical validation**: Use statistical tests to assess significance of differences
- **Protocol diversity**: Test across multiple stimulation protocols (steps, ramps, noise)
- **Parameter sensitivity**: Understand which parameters most influence model behavior

## 📁 Script Documentation

### 1. `model_evaluation.py`
**Purpose**: Comprehensive model validation against experimental recordings

**Key Features**:
- Quantitative feature comparison (experimental vs model)
- Statistical significance testing
- Publication-quality comparison plots
- Model quality scoring system

**Core Functionality**:
```python
class ModelEvaluator:
    def load_experimental_data()    # Load experimental recordings
    def run_model_protocols()       # Run model with same protocols
    def extract_features()          # Extract electrophysiological features
    def compare_features()          # Statistical comparison
    def generate_evaluation_report() # Create comprehensive report
```

**Features Evaluated**:
- **Spike Properties**: Count, frequency, amplitude, width
- **Subthreshold**: Resting potential, input resistance, membrane time constant
- **Firing Patterns**: Burst firing, adaptation, accommodation
- **Dynamics**: Rise time, decay time, after-hyperpolarization

**Usage**:
```bash
python model_evaluation.py
```

**Input**:
- `recordings/`: Experimental voltage recordings (CSV format)
- `configs/`: Model and protocol configurations
- Optimized model parameters

**Output**:
- `model_evaluation_results.png`: Comparison plots
- Quantitative evaluation metrics
- Statistical significance tests
- Model quality score (0-100%)

---

### 2. `parameter_tuning.py`
**Purpose**: Manual parameter optimization and sensitivity analysis

**Key Features**:
- Systematic parameter sweep across biologically relevant ranges
- Real-time parameter adjustment and visualization
- Optimization guidance based on experimental targets
- Parameter interaction analysis

**Core Concepts**:
- **Parameter Space Exploration**: Test combinations of key parameters
- **Gradient-based Optimization**: Find optimal parameter values
- **Sensitivity Analysis**: Identify most influential parameters
- **Constraint Satisfaction**: Ensure parameters remain biologically plausible

**Parameter Categories**:
- **Passive Properties**: Membrane resistance, capacitance, axial resistance
- **Active Conductances**: Sodium, potassium, calcium channel densities
- **Kinetics**: Activation/inactivation rates, voltage dependencies
- **Morphology**: Section lengths, diameters, spatial distributions

**Usage**:
```bash
python parameter_tuning.py --target_features features.json
```

**Optimization Methods**:
- Grid search for coarse optimization
- Random search for exploration
- Bayesian optimization for efficiency
- Evolutionary algorithms for global optimization

**Output**:
- `parameter_optimization_results.png`: Optimization progress
- Best parameter combinations
- Sensitivity analysis plots
- Parameter correlation matrices

---

### 3. `quick_model_test.py`
**Purpose**: Fast model validation with essential tests

**Key Features**:
- Minimal dependencies and fast execution
- Essential electrophysiological tests
- Quick model health checks
- Automated pass/fail criteria

**Core Tests**:
1. **Resting Potential Stability**: Verify stable resting state
2. **Rheobase Detection**: Find minimum current for spiking
3. **Action Potential Generation**: Validate spike properties
4. **Basic F-I Relationship**: Test frequency-current relationship

**Test Protocol**:
```python
def test_sequence():
    1. Initialize model
    2. Check resting potential (-70 to -60 mV)
    3. Find rheobase (binary search)
    4. Test suprathreshold responses
    5. Validate spike properties
    6. Generate summary report
```

**Usage**:
```bash
python quick_model_test.py
```

**Success Criteria**:
- Stable resting potential (< 1 mV variation)
- Rheobase within biological range (0.05-0.5 nA)
- Action potential amplitude > 80 mV
- Spike width 1-3 ms
- Monotonic F-I relationship

**Output**:
- `quick_model_test_results.png`: Test results summary
- Pass/fail status for each test
- Key model parameters
- Performance benchmarks

---

### 4. `advanced_stimulation_tests.py`
**Purpose**: Comprehensive stimulation protocol testing

**Key Features**:
- Complex stimulation patterns (bursts, ramps, noise)
- Temporal dynamics analysis
- Adaptation and facilitation tests
- Nonlinear response characterization

**Stimulation Protocols**:

1. **Current Steps**: Traditional step current injections
   - Duration: 1000ms
   - Amplitudes: -0.2 to 0.5 nA
   - Purpose: Basic excitability testing

2. **Ramp Stimulation**: Linearly increasing current
   - Duration: 2000ms
   - Slope: 0.1 nA/s
   - Purpose: Threshold dynamics

3. **Sinusoidal Stimulation**: Oscillatory current injection
   - Frequencies: 1-100 Hz
   - Amplitudes: 0.05-0.2 nA
   - Purpose: Frequency response

4. **Burst Protocols**: High-frequency pulse trains
   - Pulse width: 2ms
   - Inter-pulse intervals: 10-100ms
   - Purpose: Short-term plasticity

5. **Noise Stimulation**: Stochastic current injection
   - Gaussian white noise
   - Different variance levels
   - Purpose: Stochastic resonance

**Analysis Methods**:
- Spike train analysis (ISI, CV, burstiness)
- Frequency domain analysis (power spectra)
- Nonlinear dynamics (phase space plots)
- Information theoretic measures

**Usage**:
```bash
python advanced_stimulation_tests.py --protocol all
```

**Output**:
- `advanced_stimulation_results.png`: Protocol responses
- Spike train statistics
- Frequency response curves
- Adaptation indices

---

### 5. `test_model_stimulation.py`
**Purpose**: Comprehensive model testing framework

**Key Features**:
- Unified testing interface
- Automated test suite execution
- Customizable test protocols
- Detailed performance reporting

**Test Categories**:

1. **Basic Functionality**:
   - Model initialization
   - Mechanism loading
   - Simulation stability

2. **Electrophysiological Properties**:
   - Membrane properties
   - Action potential characteristics
   - Synaptic responses

3. **Protocol Responses**:
   - Current clamp responses
   - Voltage clamp responses
   - Dynamic clamp responses

4. **Robustness Tests**:
   - Parameter perturbations
   - Noise tolerance
   - Temperature sensitivity

**Usage**:
```bash
python test_model_stimulation.py --suite comprehensive
```

**Reporting**:
- Automated test reports
- Performance benchmarks
- Comparison with reference models
- Quality assurance metrics

## 🔄 Evaluation Workflows

### 1. Basic Model Validation
```bash
# Quick health check
python quick_model_test.py

# Comprehensive evaluation
python model_evaluation.py

# Generate report
python test_model_stimulation.py --report
```

### 2. Parameter Optimization Workflow
```bash
# Initial parameter sweep
python parameter_tuning.py --mode sweep

# Focused optimization
python parameter_tuning.py --mode optimize --target experimental_features.json

# Validation
python model_evaluation.py --params optimized_params.json
```

### 3. Advanced Characterization
```bash
# Full protocol testing
python advanced_stimulation_tests.py --protocol all

# Sensitivity analysis
python parameter_tuning.py --mode sensitivity

# Robustness testing
python test_model_stimulation.py --suite robustness
```

## 📊 Performance Metrics

### Quantitative Metrics

1. **Feature Accuracy**:
   - Mean squared error (MSE)
   - Pearson correlation coefficient
   - Normalized root mean square error (NRMSE)

2. **Spike Train Metrics**:
   - Victor-Purpura distance
   - Van Rossum distance
   - Spike train correlation

3. **Model Quality Score**:
   - Weighted sum of feature accuracies
   - Biological plausibility score
   - Robustness index

### Biological Validation

1. **Physiological Ranges**:
   - Resting potential: -70 ± 10 mV
   - Input resistance: 50-500 MΩ
   - Membrane time constant: 5-50 ms

2. **Action Potential Properties**:
   - Threshold: -50 ± 10 mV
   - Amplitude: 80-120 mV
   - Width: 1-3 ms

3. **Firing Properties**:
   - Rheobase: 0.05-0.5 nA
   - Maximum frequency: 50-200 Hz
   - Adaptation ratio: 0.1-0.9

## 🚀 Usage Examples

### Example 1: Basic Model Evaluation
```python
from model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Load experimental data
exp_data = evaluator.load_experimental_data("recordings/")

# Run model protocols
model_data = evaluator.run_model_protocols()

# Compare and generate report
report = evaluator.compare_features(exp_data, model_data)
print(f"Model quality score: {report['quality_score']:.1f}%")
```

### Example 2: Parameter Optimization
```python
from parameter_tuning import ParameterOptimizer

# Initialize optimizer
optimizer = ParameterOptimizer()

# Define parameter space
param_space = {
    'gbar_NaTa_t': (0.05, 0.15),
    'gbar_Kv3_1': (0.08, 0.12),
    'g_pas': (0.0001, 0.001)
}

# Run optimization
best_params = optimizer.optimize(param_space, target_features)
print(f"Best parameters: {best_params}")
```

### Example 3: Advanced Testing
```python
from advanced_stimulation_tests import AdvancedTester

# Initialize tester
tester = AdvancedTester()

# Run full protocol suite
results = tester.run_all_protocols()

# Analyze results
analysis = tester.analyze_responses(results)
print(f"Adaptation index: {analysis['adaptation_index']:.2f}")
```

## 🔧 Troubleshooting

### Common Issues

1. **Model fails to spike**:
   - Check sodium channel density
   - Verify membrane properties
   - Adjust stimulus strength

2. **Unstable simulations**:
   - Reduce time step (dt)
   - Check mechanism implementations
   - Verify parameter ranges

3. **Poor feature matching**:
   - Adjust feature weights
   - Check experimental data quality
   - Refine parameter bounds

### Performance Optimization

1. **Speed up simulations**:
   - Use adaptive time steps
   - Optimize mechanism code
   - Parallel protocol execution

2. **Memory management**:
   - Clear unused sections
   - Batch processing
   - Efficient data structures

## 📚 References

- [Neuron Model Validation Methods](https://doi.org/10.1371/journal.pcbi.1004720)
- [BluePyOpt Parameter Optimization](https://doi.org/10.3389/fninf.2016.00017)
- [Electrophysiological Feature Extraction](https://doi.org/10.1371/journal.pcbi.1005071)
- [Model Evaluation Best Practices](https://doi.org/10.1371/journal.pcbi.1006023)

---

**Last Updated**: June 2025  
**Version**: 1.0  
**Maintainer**: NeuroForge-Optimizer Team