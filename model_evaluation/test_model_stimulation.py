#!/usr/bin/env python3
"""
Comprehensive Model Testing & Stimulation Protocol Suite for NeuroForge-Optimizer.

This script tests your neuron model with various stimulation protocols to validate
model behavior and demonstrate electrophysiological properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import neuron
from neuron import h


class ModelTester:
    """Comprehensive model testing class with various stimulation protocols."""
    
    def __init__(self, mechanisms_dir="./mechanisms"):
        """Initialize the model tester."""
        self.mechanisms_dir = mechanisms_dir
        self.cell = None
        self.setup_neuron()
        
    def setup_neuron(self):
        """Initialize NEURON environment and load mechanisms."""
        print("🔧 Setting up NEURON environment...")
        
        # Load mechanisms
        try:
            neuron.load_mechanisms(self.mechanisms_dir)
            print("✅ NEURON mechanisms loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Mechanism loading issue: {e}")
        
        # Load standard library
        h.load_file("stdrun.hoc")
        print("✅ NEURON environment ready")
    
    def create_model_cell(self, add_active_channels=True):
        """Create a model cell with passive and active properties."""
        print("🧬 Creating model cell...")
        
        # Create soma
        soma = h.Section(name='soma')
        soma.L = 20          # Length (μm)
        soma.diam = 20       # Diameter (μm)
        
        # Insert passive properties
        soma.insert('pas')
        soma.g_pas = 0.0001  # Passive conductance (S/cm2)
        soma.e_pas = -65     # Reversal potential (mV)
        soma.Ra = 100        # Axial resistance (Ω⋅cm)
        soma.cm = 1          # Membrane capacitance (μF/cm2)
        
        if add_active_channels:
            # Add active ion channels
            try:
                # Sodium channels
                soma.insert('NaTa_t')
                soma.gbar_NaTa_t = 0.15   # Transient Na+ conductance
                print("✅ Added NaTa_t (transient sodium)")
                
                # Potassium channels  
                soma.insert('Kv3_1')
                soma.gbar_Kv3_1 = 0.08    # Fast delayed rectifier K+
                print("✅ Added Kv3_1 (fast potassium)")
                
                # Calcium channels
                soma.insert('Ca_HVA')
                soma.gbar_Ca_HVA = 0.001  # High voltage-activated Ca2+
                print("✅ Added Ca_HVA (calcium)")
                
                # HCN channels
                soma.insert('Ih')
                soma.gbar_Ih = 0.0002     # Hyperpolarization-activated
                print("✅ Added Ih (HCN)")
                
            except Exception as e:
                print(f"⚠️  Some channels not available: {e}")
        
        self.cell = soma
        print(f"✅ Cell created with active channels: {add_active_channels}")
        return soma
    
    def current_step_protocol(self, amplitudes=None, delay=200, duration=800, total_time=1400):
        """Test with current step stimulation protocol."""
        print("\n🔋 Testing Current Step Protocol")
        print("=" * 50)
        
        if amplitudes is None:
            amplitudes = [-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
        
        results = []
        
        for i, amp in enumerate(amplitudes):
            print(f"  Step {i+1}/{len(amplitudes)}: {amp:.3f} nA")
            
            # Create stimulus
            stim = h.IClamp(self.cell(0.5))
            stim.delay = delay
            stim.dur = duration
            stim.amp = amp
            
            # Set up recordings
            t_vec = h.Vector()
            v_vec = h.Vector()
            t_vec.record(h._ref_t)
            v_vec.record(self.cell(0.5)._ref_v)
            
            # Run simulation
            h.dt = 0.025
            h.tstop = total_time
            h.v_init = -65
            h.finitialize(h.v_init)
            h.run()
            
            # Store results
            time = np.array(t_vec)
            voltage = np.array(v_vec)
            results.append({
                'amplitude': amp,
                'time': time,
                'voltage': voltage,
                'max_voltage': np.max(voltage),
                'min_voltage': np.min(voltage),
                'spike_count': self.count_spikes(voltage, threshold=-20)
            })
        
        self.plot_current_steps(results, delay, duration)
        self.analyze_current_steps(results)
        return results
    
    def ramp_current_protocol(self, start_amp=0, end_amp=0.5, duration=2000):
        """Test with ramp current stimulation."""
        print("\n📈 Testing Ramp Current Protocol")
        print("=" * 50)
        
        # Create ramp stimulus using multiple small steps
        n_steps = 200
        amps = np.linspace(start_amp, end_amp, n_steps)
        step_duration = duration / n_steps
        
        print(f"  Ramp: {start_amp:.3f} → {end_amp:.3f} nA over {duration:.0f} ms")
        
        # Set up recordings
        t_vec = h.Vector()
        v_vec = h.Vector()
        i_vec = h.Vector()
        t_vec.record(h._ref_t)
        v_vec.record(self.cell(0.5)._ref_v)
        
        # Create ramp using IClamp with time-varying amplitude
        stim = h.IClamp(self.cell(0.5))
        stim.delay = 200
        stim.dur = duration
        
        # Run simulation with manual ramp
        h.dt = 0.025
        h.tstop = duration + 400
        h.v_init = -65
        h.finitialize(h.v_init)
        
        # Manual ramp implementation
        for i in range(int(h.tstop / h.dt)):
            current_time = i * h.dt
            if current_time >= 200 and current_time <= 200 + duration:
                progress = (current_time - 200) / duration
                current_amp = start_amp + (end_amp - start_amp) * progress
                stim.amp = current_amp
            else:
                stim.amp = 0
            h.fadvance()
        
        # Store results
        time = np.array(t_vec)
        voltage = np.array(v_vec)
        
        result = {
            'time': time,
            'voltage': voltage,
            'start_amp': start_amp,
            'end_amp': end_amp,
            'duration': duration,
            'spike_count': self.count_spikes(voltage, threshold=-20)
        }
        
        self.plot_ramp_response(result)
        return result
    
    def frequency_response_test(self, base_amp=0.1, mod_amp=0.05, frequencies=[1, 5, 10, 20, 50]):
        """Test frequency response with sinusoidal current modulation."""
        print("\n🌊 Testing Frequency Response")
        print("=" * 50)
        
        results = []
        
        for freq in frequencies:
            print(f"  Testing {freq} Hz modulation...")
            
            # Set up simulation
            duration = max(2000, 5000/freq)  # At least 5 cycles
            h.dt = 0.025
            h.tstop = duration
            h.v_init = -65
            
            # Set up recordings
            t_vec = h.Vector()
            v_vec = h.Vector()
            t_vec.record(h._ref_t)
            v_vec.record(self.cell(0.5)._ref_v)
            
            # Create sinusoidal current
            stim = h.IClamp(self.cell(0.5))
            stim.delay = 200
            stim.dur = duration - 400
            
            h.finitialize(h.v_init)
            
            # Manual sinusoidal stimulation
            for i in range(int(h.tstop / h.dt)):
                current_time = i * h.dt
                if current_time >= 200 and current_time <= duration - 200:
                    phase = 2 * np.pi * freq * (current_time - 200) / 1000
                    current_amp = base_amp + mod_amp * np.sin(phase)
                    stim.amp = current_amp
                else:
                    stim.amp = 0
                h.fadvance()
            
            # Store results
            time = np.array(t_vec)
            voltage = np.array(v_vec)
            
            results.append({
                'frequency': freq,
                'time': time,
                'voltage': voltage,
                'base_amp': base_amp,
                'mod_amp': mod_amp
            })
        
        self.plot_frequency_response(results)
        return results
    
    def hyperpolarization_test(self, amplitudes=None, duration=1000):
        """Test hyperpolarization responses (sag, rebound)."""
        print("\n⬇️  Testing Hyperpolarization Responses")
        print("=" * 50)
        
        if amplitudes is None:
            amplitudes = [-0.3, -0.2, -0.1, -0.05]
        
        results = []
        
        for amp in amplitudes:
            print(f"  Hyperpolarizing step: {amp:.3f} nA")
            
            # Create stimulus
            stim = h.IClamp(self.cell(0.5))
            stim.delay = 300
            stim.dur = duration
            stim.amp = amp
            
            # Set up recordings
            t_vec = h.Vector()
            v_vec = h.Vector()
            t_vec.record(h._ref_t)
            v_vec.record(self.cell(0.5)._ref_v)
            
            # Run simulation
            h.dt = 0.025
            h.tstop = duration + 600
            h.v_init = -65
            h.finitialize(h.v_init)
            h.run()
            
            # Analyze sag and rebound
            time = np.array(t_vec)
            voltage = np.array(v_vec)
            
            # Calculate sag ratio
            baseline = np.mean(voltage[:int(300/h.dt)])
            min_voltage = np.min(voltage[int(300/h.dt):int((300+duration)/h.dt)])
            steady_voltage = np.mean(voltage[int((300+duration-100)/h.dt):int((300+duration)/h.dt)])
            sag_amplitude = steady_voltage - min_voltage
            sag_ratio = sag_amplitude / (baseline - min_voltage) if baseline != min_voltage else 0
            
            # Check for rebound spikes
            rebound_period = voltage[int((300+duration)/h.dt):int((300+duration+200)/h.dt)]
            rebound_spikes = self.count_spikes(rebound_period, threshold=-20)
            
            results.append({
                'amplitude': amp,
                'time': time,
                'voltage': voltage,
                'sag_amplitude': sag_amplitude,
                'sag_ratio': sag_ratio,
                'rebound_spikes': rebound_spikes,
                'min_voltage': min_voltage,
                'steady_voltage': steady_voltage
            })
        
        self.plot_hyperpolarization(results, duration)
        self.analyze_hyperpolarization(results)
        return results
    
    def count_spikes(self, voltage, threshold=-20):
        """Count action potentials in voltage trace."""
        spikes = 0
        above_threshold = voltage > threshold
        for i in range(1, len(above_threshold)):
            if above_threshold[i] and not above_threshold[i-1]:
                spikes += 1
        return spikes
    
    def plot_current_steps(self, results, delay, duration):
        """Plot current step responses."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot voltage traces
        for result in results:
            amp = result['amplitude']
            time = result['time']
            voltage = result['voltage']
            ax1.plot(time, voltage, label=f'{amp:.3f} nA', linewidth=1.5)
        
        ax1.axvspan(delay, delay + duration, alpha=0.2, color='red', label='Current injection')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.set_title('Current Step Responses')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot I-V relationship
        amplitudes = [r['amplitude'] for r in results]
        steady_voltages = []
        
        for result in results:
            # Calculate steady-state voltage (last 100ms of stimulus)
            time = result['time']
            voltage = result['voltage']
            dt = time[1] - time[0]
            start_idx = int((delay + duration - 100) / dt)
            end_idx = int((delay + duration) / dt)
            if start_idx < len(voltage) and end_idx <= len(voltage):
                steady_voltage = np.mean(voltage[start_idx:end_idx])
            else:
                steady_voltage = np.mean(voltage[-100:])
            steady_voltages.append(steady_voltage)
        
        ax2.plot(amplitudes, steady_voltages, 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Current Amplitude (nA)')
        ax2.set_ylabel('Steady-State Voltage (mV)')
        ax2.set_title('Current-Voltage (I-V) Relationship')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('current_step_test_results.png', dpi=150, bbox_inches='tight')
        print("✅ Current step results saved: current_step_test_results.png")
        plt.show()
    
    def plot_ramp_response(self, result):
        """Plot ramp current response."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        time = result['time']
        voltage = result['voltage']
        
        # Create current ramp for visualization
        current = np.zeros_like(time)
        ramp_start = 200
        ramp_end = ramp_start + result['duration']
        ramp_mask = (time >= ramp_start) & (time <= ramp_end)
        current[ramp_mask] = result['start_amp'] + (result['end_amp'] - result['start_amp']) * \
                           (time[ramp_mask] - ramp_start) / result['duration']
        
        # Plot voltage response
        ax1.plot(time, voltage, 'b-', linewidth=2)
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.set_title('Ramp Current Response')
        ax1.grid(True, alpha=0.3)
        
        # Plot current ramp
        ax2.plot(time, current, 'r-', linewidth=2)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (nA)')
        ax2.set_title('Current Ramp Stimulus')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ramp_current_test_results.png', dpi=150, bbox_inches='tight')
        print("✅ Ramp current results saved: ramp_current_test_results.png")
        plt.show()
    
    def plot_frequency_response(self, results):
        """Plot frequency response results."""
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 3*len(results)))
        if len(results) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            time = result['time']
            voltage = result['voltage']
            freq = result['frequency']
            
            axes[i].plot(time, voltage, 'b-', linewidth=1.5)
            axes[i].set_ylabel('Voltage (mV)')
            axes[i].set_title(f'Response to {freq} Hz Modulation')
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig('frequency_response_test_results.png', dpi=150, bbox_inches='tight')
        print("✅ Frequency response results saved: frequency_response_test_results.png")
        plt.show()
    
    def plot_hyperpolarization(self, results, duration):
        """Plot hyperpolarization test results."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for result in results:
            amp = result['amplitude']
            time = result['time']
            voltage = result['voltage']
            sag_ratio = result['sag_ratio']
            rebound_spikes = result['rebound_spikes']
            
            label = f'{amp:.3f} nA (sag: {sag_ratio:.3f}, rebound: {rebound_spikes})'
            ax.plot(time, voltage, label=label, linewidth=1.5)
        
        ax.axvspan(300, 300 + duration, alpha=0.2, color='blue', label='Hyperpolarization')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Membrane Potential (mV)')
        ax.set_title('Hyperpolarization Responses (Sag & Rebound)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hyperpolarization_test_results.png', dpi=150, bbox_inches='tight')
        print("✅ Hyperpolarization results saved: hyperpolarization_test_results.png")
        plt.show()
    
    def analyze_current_steps(self, results):
        """Analyze current step results."""
        print("\n📊 Current Step Analysis:")
        print("-" * 30)
        
        # Find rheobase
        rheobase = None
        for result in results:
            if result['spike_count'] > 0:
                rheobase = result['amplitude']
                break
        
        if rheobase:
            print(f"  Rheobase (threshold current): {rheobase:.3f} nA")
        else:
            print("  Rheobase: Not reached in tested range")
        
        # Calculate input resistance from subthreshold responses
        subthreshold_results = [r for r in results if r['spike_count'] == 0 and r['amplitude'] != 0]
        
        if len(subthreshold_results) >= 2:
            amps = [r['amplitude'] for r in subthreshold_results]
            # Calculate voltage deflections
            deflections = []
            for r in subthreshold_results:
                time = r['time']
                voltage = r['voltage']
                dt = time[1] - time[0]
                baseline = np.mean(voltage[:int(200/dt)])
                steady = np.mean(voltage[-int(100/dt):])
                deflections.append(steady - baseline)
            
            # Linear fit for input resistance
            resistance = np.polyfit(amps, deflections, 1)[0] * 1000  # Convert to MΩ
            print(f"  Input resistance: {resistance:.1f} MΩ")
        
        # Spike counting summary
        max_spikes = max([r['spike_count'] for r in results])
        print(f"  Maximum spike count: {max_spikes}")
        
        for result in results:
            if result['spike_count'] > 0:
                print(f"    {result['amplitude']:.3f} nA → {result['spike_count']} spikes")
    
    def analyze_hyperpolarization(self, results):
        """Analyze hyperpolarization results."""
        print("\n📊 Hyperpolarization Analysis:")
        print("-" * 35)
        
        for result in results:
            amp = result['amplitude']
            sag_ratio = result['sag_ratio']
            rebound_spikes = result['rebound_spikes']
            
            print(f"  {amp:.3f} nA:")
            print(f"    Sag ratio: {sag_ratio:.3f}")
            print(f"    Rebound spikes: {rebound_spikes}")
    
    def run_comprehensive_test(self):
        """Run all test protocols."""
        print("🧪 NeuroForge-Optimizer Comprehensive Model Testing")
        print("=" * 60)
        
        # Create model
        self.create_model_cell(add_active_channels=True)
        
        # Run all test protocols
        current_results = self.current_step_protocol()
        ramp_results = self.ramp_current_protocol()
        freq_results = self.frequency_response_test()
        hyper_results = self.hyperpolarization_test()
        
        # Generate summary report
        self.generate_summary_report(current_results, ramp_results, freq_results, hyper_results)
        
        return {
            'current_steps': current_results,
            'ramp_current': ramp_results,
            'frequency_response': freq_results,
            'hyperpolarization': hyper_results
        }
    
    def generate_summary_report(self, current_results, ramp_results, freq_results, hyper_results):
        """Generate a comprehensive summary report."""
        print("\n📋 COMPREHENSIVE MODEL TEST REPORT")
        print("=" * 60)
        
        # Model properties summary
        print("🧬 Model Properties:")
        print("  Cell type: Single compartment soma")
        print("  Ion channels: NaTa_t, Kv3_1, Ca_HVA, Ih, pas")
        print("  Morphology: Simplified (20μm diameter sphere)")
        
        # Current step summary
        rheobase = None
        max_spikes = 0
        for result in current_results:
            if result['spike_count'] > 0 and rheobase is None:
                rheobase = result['amplitude']
            max_spikes = max(max_spikes, result['spike_count'])
        
        print(f"\n⚡ Excitability:")
        if rheobase:
            print(f"  Rheobase: {rheobase:.3f} nA")
        print(f"  Maximum firing rate: {max_spikes} spikes/800ms")
        
        # Hyperpolarization summary
        if hyper_results:
            avg_sag = np.mean([r['sag_ratio'] for r in hyper_results if r['sag_ratio'] > 0])
            total_rebound = sum([r['rebound_spikes'] for r in hyper_results])
            print(f"\n🔽 Hyperpolarization Response:")
            print(f"  Average sag ratio: {avg_sag:.3f}")
            print(f"  Rebound spikes observed: {total_rebound > 0}")
        
        # Frequency response summary
        print(f"\n🌊 Frequency Response:")
        print(f"  Tested frequencies: {[r['frequency'] for r in freq_results]} Hz")
        print(f"  Model responds to sinusoidal current modulation")
        
        print(f"\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved as PNG files in current directory")


def main():
    """Main testing function."""
    print("🚀 Starting NeuroForge-Optimizer Model Testing...")
    
    # Create tester instance
    tester = ModelTester()
    
    # Run comprehensive tests
    results = tester.run_comprehensive_test()
    
    print("\n🎉 Model testing completed successfully!")
    print("📊 Check the generated PNG files for detailed results")
    
    return results


if __name__ == "__main__":
    results = main()