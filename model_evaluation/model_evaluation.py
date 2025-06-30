#!/usr/bin/env python3
"""
Model Evaluation Against Experimental Data for NeuroForge-Optimizer.

This script compares model predictions with experimental recordings to validate
the optimization results and assess model quality.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import neuron
from neuron import h
from scipy import stats
from scipy.signal import find_peaks


class ModelEvaluator:
    """Comprehensive model evaluation against experimental data."""
    
    def __init__(self, mechanisms_dir="./mechanisms"):
        """Initialize the evaluator."""
        self.mechanisms_dir = mechanisms_dir
        self.setup_neuron()
        
    def setup_neuron(self):
        """Initialize NEURON environment."""
        print("🔧 Setting up NEURON for model evaluation...")
        neuron.load_mechanisms(self.mechanisms_dir)
        h.load_file("stdrun.hoc")
        
    def load_experimental_data(self, recordings_dir="./recordings"):
        """Load experimental recordings from CSV files."""
        print("📊 Loading experimental data...")
        
        recordings_path = Path(recordings_dir)
        data = {}
        
        if not recordings_path.exists():
            print(f"⚠️  Recordings directory not found: {recordings_path}")
            return self.create_synthetic_experimental_data()
        
        # Load all CSV files
        csv_files = list(recordings_path.glob("*.csv"))
        if not csv_files:
            print("⚠️  No CSV files found, creating synthetic experimental data")
            return self.create_synthetic_experimental_data()
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Assume format: time, voltage, current
                if len(df.columns) >= 2:
                    data[csv_file.stem] = {
                        'time': df.iloc[:, 0].values,
                        'voltage': df.iloc[:, 1].values,
                        'current': df.iloc[:, 2].values if len(df.columns) > 2 else None
                    }
                    print(f"✅ Loaded: {csv_file.name}")
                    
            except Exception as e:
                print(f"❌ Error loading {csv_file}: {e}")
        
        if not data:
            print("⚠️  No valid recordings loaded, creating synthetic data")
            return self.create_synthetic_experimental_data()
            
        return data
    
    def create_synthetic_experimental_data(self):
        """Create realistic synthetic experimental data for demonstration."""
        print("🧪 Creating synthetic experimental data for evaluation...")
        
        # Simulate experimental patch-clamp recordings
        data = {}
        
        # Current step protocol
        currents = [0.0, 0.1, 0.2, 0.3]
        
        for i, current in enumerate(currents):
            # Create realistic experimental data with noise
            time = np.arange(0, 1000, 0.1)  # 10 kHz sampling
            
            # Simulate realistic voltage response
            if current == 0.0:
                # Resting potential with noise
                voltage = -65 + np.random.normal(0, 0.5, len(time))
            else:
                # Action potentials with realistic kinetics and noise
                voltage = np.full_like(time, -65)
                
                # Add stimulus response
                stim_start = 200
                stim_end = 700
                stim_mask = (time >= stim_start) & (time <= stim_end)
                
                if current >= 0.1:
                    # Generate spikes
                    n_spikes = int(current * 5)  # Frequency increases with current
                    spike_times = np.linspace(stim_start + 50, stim_end - 50, n_spikes)
                    
                    for spike_time in spike_times:
                        spike_mask = (time >= spike_time) & (time <= spike_time + 2)
                        voltage[spike_mask] = 40 + np.random.normal(0, 2)  # Peak voltage
                        
                        # Afterhyperpolarization
                        ahp_mask = (time >= spike_time + 2) & (time <= spike_time + 10)
                        voltage[ahp_mask] = -70 + np.random.normal(0, 1)
                
                # Depolarization during stimulus
                voltage[stim_mask] = np.maximum(voltage[stim_mask], -65 + current * 20)
                
                # Add realistic noise
                voltage += np.random.normal(0, 0.8, len(time))
            
            data[f"step_{current:.1f}nA"] = {
                'time': time,
                'voltage': voltage,
                'current': np.full_like(time, current) if current > 0 else np.zeros_like(time),
                'protocol': 'current_step',
                'amplitude': current
            }
            
        print(f"✅ Created {len(data)} synthetic experimental traces")
        return data
    
    def load_optimized_model(self, params_file="./optimisation/best_parameters.json"):
        """Load optimized model parameters."""
        print("🔧 Loading optimized model...")
        
        # Create model cell
        soma = h.Section(name='soma')
        soma.L = 20
        soma.diam = 20
        
        # Insert passive properties
        soma.insert('pas')
        soma.g_pas = 0.0001
        soma.e_pas = -65
        soma.Ra = 100
        soma.cm = 1
        
        # Load optimized parameters if available
        params_path = Path(params_file)
        if params_path.exists():
            print(f"📁 Loading parameters from: {params_path}")
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                # Apply optimized parameters
                for param_name, param_value in params.items():
                    try:
                        if param_name == "g_pas":
                            soma.g_pas = param_value
                        elif param_name == "Ra":
                            soma.Ra = param_value
                        elif param_name == "cm":
                            soma.cm = param_value
                        elif param_name.startswith("gbar_"):
                            mech_name = param_name.replace("gbar_", "")
                            soma.insert(mech_name)
                            setattr(soma, param_name, param_value)
                    except Exception as e:
                        print(f"⚠️  Could not set {param_name}: {e}")
                
                print(f"✅ Applied {len(params)} optimized parameters")
                
            except Exception as e:
                print(f"❌ Error loading parameters: {e}")
                print("Using default parameters")
        else:
            print("⚠️  No optimized parameters found, using default values")
            # Add some active channels with default values
            try:
                soma.insert('NaTa_t')
                soma.gbar_NaTa_t = 0.15
                soma.insert('Kv3_1')
                soma.gbar_Kv3_1 = 0.08
            except:
                pass
        
        return soma
    
    def simulate_model_response(self, cell, current_amplitude, delay=200, duration=500, total_time=1000):
        """Simulate model response to current injection."""
        # Create stimulus
        stim = h.IClamp(cell(0.5))
        stim.delay = delay
        stim.dur = duration
        stim.amp = current_amplitude
        
        # Set up recordings
        t_vec = h.Vector()
        v_vec = h.Vector()
        t_vec.record(h._ref_t)
        v_vec.record(cell(0.5)._ref_v)
        
        # Run simulation
        h.dt = 0.025
        h.tstop = total_time
        h.v_init = -65
        h.finitialize(h.v_init)
        h.run()
        
        return np.array(t_vec), np.array(v_vec)
    
    def extract_features(self, time, voltage, current_amplitude=None):
        """Extract electrophysiological features from voltage trace."""
        features = {}
        
        # Basic statistics
        features['resting_potential'] = np.mean(voltage[:int(len(voltage)*0.2)])  # First 20%
        features['max_voltage'] = np.max(voltage)
        features['min_voltage'] = np.min(voltage)
        features['voltage_range'] = features['max_voltage'] - features['min_voltage']
        
        # Spike detection
        spikes, spike_props = find_peaks(voltage, height=-20, distance=20)
        features['spike_count'] = len(spikes)
        
        if len(spikes) > 0:
            features['spike_amplitude'] = np.mean(voltage[spikes])
            features['first_spike_time'] = time[spikes[0]] if len(spikes) > 0 else None
            
            # Inter-spike intervals
            if len(spikes) > 1:
                isis = np.diff(time[spikes])
                features['isi_mean'] = np.mean(isis)
                features['isi_std'] = np.std(isis)
                features['isi_cv'] = features['isi_std'] / features['isi_mean'] if features['isi_mean'] > 0 else 0
            else:
                features['isi_mean'] = None
                features['isi_std'] = None
                features['isi_cv'] = None
        else:
            features['spike_amplitude'] = None
            features['first_spike_time'] = None
            features['isi_mean'] = None
            features['isi_std'] = None
            features['isi_cv'] = None
        
        # Subthreshold features
        if current_amplitude and current_amplitude > 0:
            # Input resistance (for subthreshold responses)
            baseline = features['resting_potential']
            
            # Find steady-state response
            steady_start = int(len(voltage) * 0.7)  # Last 30% of trace
            steady_voltage = np.mean(voltage[steady_start:])
            
            if features['spike_count'] == 0:  # Only for subthreshold
                voltage_deflection = steady_voltage - baseline
                features['input_resistance'] = voltage_deflection / current_amplitude * 1000  # MΩ
            else:
                features['input_resistance'] = None
        else:
            features['input_resistance'] = None
        
        return features
    
    def compare_features(self, exp_features, model_features):
        """Compare experimental and model features."""
        comparison = {}
        
        # Define important features for comparison
        key_features = [
            'resting_potential', 'max_voltage', 'spike_count', 'spike_amplitude',
            'isi_mean', 'input_resistance'
        ]
        
        for feature in key_features:
            exp_val = exp_features.get(feature)
            model_val = model_features.get(feature)
            
            if exp_val is not None and model_val is not None:
                # Calculate relative error
                if exp_val != 0:
                    rel_error = abs(model_val - exp_val) / abs(exp_val) * 100
                else:
                    rel_error = abs(model_val - exp_val)
                
                comparison[feature] = {
                    'experimental': exp_val,
                    'model': model_val,
                    'absolute_error': abs(model_val - exp_val),
                    'relative_error_percent': rel_error,
                    'match_quality': 'excellent' if rel_error < 10 else 'good' if rel_error < 25 else 'poor'
                }
            else:
                comparison[feature] = {
                    'experimental': exp_val,
                    'model': model_val,
                    'absolute_error': None,
                    'relative_error_percent': None,
                    'match_quality': 'no_data'
                }
        
        return comparison
    
    def evaluate_model_vs_experiment(self, experimental_data):
        """Complete model evaluation against experimental data."""
        print("\n🔬 Evaluating model against experimental data...")
        
        # Load optimized model
        model_cell = self.load_optimized_model()
        
        evaluation_results = {}
        
        for exp_name, exp_data in experimental_data.items():
            print(f"\n📊 Evaluating protocol: {exp_name}")
            
            # Extract experimental current amplitude if available
            current_amp = exp_data.get('amplitude', 0.0)
            
            # Simulate model response
            model_time, model_voltage = self.simulate_model_response(
                model_cell, current_amp
            )
            
            # Extract features from both
            exp_features = self.extract_features(
                exp_data['time'], exp_data['voltage'], current_amp
            )
            model_features = self.extract_features(
                model_time, model_voltage, current_amp
            )
            
            # Compare features
            feature_comparison = self.compare_features(exp_features, model_features)
            
            evaluation_results[exp_name] = {
                'experimental_data': exp_data,
                'model_time': model_time,
                'model_voltage': model_voltage,
                'experimental_features': exp_features,
                'model_features': model_features,
                'feature_comparison': feature_comparison,
                'current_amplitude': current_amp
            }
            
            # Print summary for this protocol
            self.print_protocol_summary(exp_name, feature_comparison)
        
        return evaluation_results
    
    def print_protocol_summary(self, protocol_name, comparison):
        """Print summary of feature comparison for one protocol."""
        print(f"\n  📋 {protocol_name} Summary:")
        
        excellent_count = 0
        good_count = 0
        poor_count = 0
        
        for feature, comp in comparison.items():
            if comp['match_quality'] == 'excellent':
                excellent_count += 1
                print(f"    ✅ {feature}: {comp['match_quality']} ({comp['relative_error_percent']:.1f}% error)")
            elif comp['match_quality'] == 'good':
                good_count += 1
                print(f"    🟨 {feature}: {comp['match_quality']} ({comp['relative_error_percent']:.1f}% error)")
            elif comp['match_quality'] == 'poor':
                poor_count += 1
                print(f"    ❌ {feature}: {comp['match_quality']} ({comp['relative_error_percent']:.1f}% error)")
            else:
                print(f"    ⚪ {feature}: no data")
        
        total_scored = excellent_count + good_count + poor_count
        if total_scored > 0:
            quality_score = (excellent_count * 2 + good_count) / (total_scored * 2) * 100
            print(f"    🏆 Overall quality score: {quality_score:.1f}%")
    
    def plot_evaluation_results(self, evaluation_results):
        """Create comprehensive evaluation plots."""
        print("\n📈 Creating evaluation plots...")
        
        n_protocols = len(evaluation_results)
        fig, axes = plt.subplots(n_protocols + 1, 2, figsize=(16, 4*(n_protocols + 1)))
        
        if n_protocols == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each protocol comparison
        protocol_names = list(evaluation_results.keys())
        
        for i, (protocol_name, results) in enumerate(evaluation_results.items()):
            exp_data = results['experimental_data']
            
            # Left plot: Voltage traces comparison
            axes[i, 0].plot(exp_data['time'], exp_data['voltage'], 'b-', 
                           linewidth=2, label='Experimental', alpha=0.8)
            axes[i, 0].plot(results['model_time'], results['model_voltage'], 'r--', 
                           linewidth=2, label='Model', alpha=0.8)
            
            axes[i, 0].set_xlabel('Time (ms)')
            axes[i, 0].set_ylabel('Voltage (mV)')
            axes[i, 0].set_title(f'{protocol_name} - Voltage Traces')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Right plot: Feature comparison
            comparison = results['feature_comparison']
            features = []
            exp_values = []
            model_values = []
            
            for feature, comp in comparison.items():
                if comp['experimental'] is not None and comp['model'] is not None:
                    features.append(feature.replace('_', '\n'))
                    exp_values.append(comp['experimental'])
                    model_values.append(comp['model'])
            
            if features:
                x = np.arange(len(features))
                width = 0.35
                
                axes[i, 1].bar(x - width/2, exp_values, width, label='Experimental', alpha=0.7)
                axes[i, 1].bar(x + width/2, model_values, width, label='Model', alpha=0.7)
                
                axes[i, 1].set_xlabel('Features')
                axes[i, 1].set_ylabel('Values')
                axes[i, 1].set_title(f'{protocol_name} - Feature Comparison')
                axes[i, 1].set_xticks(x)
                axes[i, 1].set_xticklabels(features, rotation=45, ha='right')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
        
        # Summary plot
        self.plot_overall_summary(evaluation_results, axes[-1, :])
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=150, bbox_inches='tight')
        print("✅ Evaluation results saved: model_evaluation_results.png")
        
    def plot_overall_summary(self, evaluation_results, axes):
        """Plot overall evaluation summary."""
        # Left: Quality scores
        protocol_names = []
        quality_scores = []
        
        for protocol_name, results in evaluation_results.items():
            comparison = results['feature_comparison']
            
            excellent_count = sum(1 for comp in comparison.values() 
                                if comp['match_quality'] == 'excellent')
            good_count = sum(1 for comp in comparison.values() 
                           if comp['match_quality'] == 'good')
            poor_count = sum(1 for comp in comparison.values() 
                           if comp['match_quality'] == 'poor')
            
            total_scored = excellent_count + good_count + poor_count
            if total_scored > 0:
                quality_score = (excellent_count * 2 + good_count) / (total_scored * 2) * 100
                protocol_names.append(protocol_name)
                quality_scores.append(quality_score)
        
        if quality_scores:
            axes[0].bar(range(len(protocol_names)), quality_scores, alpha=0.7, color='green')
            axes[0].set_xlabel('Protocols')
            axes[0].set_ylabel('Quality Score (%)')
            axes[0].set_title('Model Quality by Protocol')
            axes[0].set_xticks(range(len(protocol_names)))
            axes[0].set_xticklabels(protocol_names, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 100)
        
        # Right: Overall statistics
        axes[1].axis('off')
        
        # Calculate overall statistics
        all_errors = []
        feature_counts = {'excellent': 0, 'good': 0, 'poor': 0}
        
        for results in evaluation_results.values():
            for comp in results['feature_comparison'].values():
                if comp['relative_error_percent'] is not None:
                    all_errors.append(comp['relative_error_percent'])
                    feature_counts[comp['match_quality']] += 1
        
        if all_errors:
            mean_error = np.mean(all_errors)
            median_error = np.median(all_errors)
            
            summary_text = f"""
MODEL EVALUATION SUMMARY
========================

📊 Overall Performance:
  • Mean Error: {mean_error:.1f}%
  • Median Error: {median_error:.1f}%
  
🎯 Feature Matching:
  • Excellent: {feature_counts['excellent']} features
  • Good: {feature_counts['good']} features  
  • Poor: {feature_counts['poor']} features
  
🏆 Model Status:
  • Total Protocols: {len(evaluation_results)}
  • Average Quality: {np.mean(quality_scores):.1f}%
  
✅ EVALUATION COMPLETE
"""
        else:
            summary_text = "No quantitative comparison available"
        
        axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
                    fontfamily='monospace', fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    def generate_evaluation_report(self, evaluation_results):
        """Generate a comprehensive evaluation report."""
        print("\n📋 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\n🔬 Evaluation Overview:")
        print(f"  • Number of protocols tested: {len(evaluation_results)}")
        print(f"  • Model type: Optimized single-compartment neuron")
        print(f"  • Evaluation date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Detailed results for each protocol
        for protocol_name, results in evaluation_results.items():
            print(f"\n📊 Protocol: {protocol_name}")
            print("-" * 40)
            
            comparison = results['feature_comparison']
            current_amp = results['current_amplitude']
            
            print(f"  Current amplitude: {current_amp:.3f} nA")
            
            for feature, comp in comparison.items():
                if comp['experimental'] is not None and comp['model'] is not None:
                    exp_val = comp['experimental']
                    model_val = comp['model']
                    error = comp['relative_error_percent']
                    quality = comp['match_quality']
                    
                    print(f"  {feature}:")
                    print(f"    Experimental: {exp_val:.3f}")
                    print(f"    Model: {model_val:.3f}")
                    print(f"    Error: {error:.1f}% ({quality})")
        
        # Overall assessment
        print(f"\n🏆 OVERALL MODEL ASSESSMENT:")
        print("=" * 40)
        
        total_features = 0
        excellent_features = 0
        good_features = 0
        
        for results in evaluation_results.values():
            for comp in results['feature_comparison'].values():
                if comp['match_quality'] in ['excellent', 'good', 'poor']:
                    total_features += 1
                    if comp['match_quality'] == 'excellent':
                        excellent_features += 1
                    elif comp['match_quality'] == 'good':
                        good_features += 1
        
        if total_features > 0:
            success_rate = (excellent_features + good_features) / total_features * 100
            print(f"  Model success rate: {success_rate:.1f}%")
            print(f"  Excellent matches: {excellent_features}/{total_features}")
            print(f"  Good matches: {good_features}/{total_features}")
            
            if success_rate >= 80:
                print("  ✅ MODEL QUALITY: EXCELLENT")
            elif success_rate >= 60:
                print("  🟨 MODEL QUALITY: GOOD")
            else:
                print("  ❌ MODEL QUALITY: NEEDS IMPROVEMENT")
        
        print(f"\n📁 Results saved in: model_evaluation_results.png")


def main():
    """Main evaluation function."""
    print("🔬 NeuroForge-Optimizer Model Evaluation")
    print("=" * 50)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Load experimental data
    experimental_data = evaluator.load_experimental_data()
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.evaluate_model_vs_experiment(experimental_data)
    
    # Create plots
    evaluator.plot_evaluation_results(evaluation_results)
    
    # Generate report
    evaluator.generate_evaluation_report(evaluation_results)
    
    print("\n🎉 MODEL EVALUATION COMPLETED!")
    print("📊 Check model_evaluation_results.png for detailed comparison")
    
    return evaluation_results


if __name__ == "__main__":
    results = main()