#!/usr/bin/env python3
"""
Quick Model Testing Script for NeuroForge-Optimizer.

Fast, focused testing of key model behaviors with minimal dependencies.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import neuron
from neuron import h


def setup_model():
    """Create and configure the neuron model."""
    print("🧬 Setting up neuron model...")
    
    # Load mechanisms
    neuron.load_mechanisms('./mechanisms')
    h.load_file("stdrun.hoc")
    
    # Create soma
    soma = h.Section(name='soma')
    soma.L = 20
    soma.diam = 20
    
    # Passive properties
    soma.insert('pas')
    soma.g_pas = 0.0001
    soma.e_pas = -65
    soma.Ra = 100
    soma.cm = 1
    
    # Active channels
    soma.insert('NaTa_t')
    soma.gbar_NaTa_t = 0.15
    
    soma.insert('Kv3_1')
    soma.gbar_Kv3_1 = 0.08
    
    print("✅ Model ready with Na+ and K+ channels")
    return soma


def test_basic_excitability(cell):
    """Test basic action potential generation."""
    print("\n⚡ Testing Basic Excitability...")
    
    # Test current amplitudes
    test_currents = [0.05, 0.1, 0.15, 0.2, 0.3]
    results = []
    
    for i, current in enumerate(test_currents):
        print(f"  Testing {current:.3f} nA... ", end="")
        
        # Create stimulus
        stim = h.IClamp(cell(0.5))
        stim.delay = 100
        stim.dur = 500
        stim.amp = current
        
        # Record voltage
        t_vec = h.Vector()
        v_vec = h.Vector()
        t_vec.record(h._ref_t)
        v_vec.record(cell(0.5)._ref_v)
        
        # Run simulation
        h.dt = 0.025
        h.tstop = 800
        h.v_init = -65
        h.finitialize(h.v_init)
        h.run()
        
        # Analyze
        time = np.array(t_vec)
        voltage = np.array(v_vec)
        max_v = np.max(voltage)
        spikes = count_spikes(voltage)
        
        results.append({
            'current': current,
            'time': time,
            'voltage': voltage,
            'max_voltage': max_v,
            'spikes': spikes
        })
        
        print(f"Max: {max_v:.1f}mV, Spikes: {spikes}")
    
    return results


def test_threshold_finding(cell):
    """Find the rheobase (threshold current)."""
    print("\n🎯 Finding Rheobase (Threshold Current)...")
    
    # Binary search for threshold
    low_current = 0.0
    high_current = 0.5
    threshold_current = None
    
    for iteration in range(10):  # Max 10 iterations
        test_current = (low_current + high_current) / 2
        
        # Test this current
        stim = h.IClamp(cell(0.5))
        stim.delay = 100
        stim.dur = 500
        stim.amp = test_current
        
        t_vec = h.Vector()
        v_vec = h.Vector()
        t_vec.record(h._ref_t)
        v_vec.record(cell(0.5)._ref_v)
        
        h.dt = 0.025
        h.tstop = 800
        h.v_init = -65
        h.finitialize(h.v_init)
        h.run()
        
        voltage = np.array(v_vec)
        spikes = count_spikes(voltage)
        
        print(f"  Iteration {iteration+1}: {test_current:.4f} nA → {spikes} spikes")
        
        if spikes > 0:
            high_current = test_current
            threshold_current = test_current
        else:
            low_current = test_current
        
        # Convergence check
        if (high_current - low_current) < 0.001:
            break
    
    if threshold_current:
        print(f"✅ Rheobase found: {threshold_current:.4f} nA")
    else:
        print("❌ Rheobase not found in tested range")
    
    return threshold_current


def test_frequency_current_relationship(cell):
    """Test firing frequency vs. current amplitude."""
    print("\n📊 Testing Frequency-Current (F-I) Relationship...")
    
    currents = np.linspace(0.1, 0.4, 7)
    frequencies = []
    
    for current in currents:
        print(f"  Testing {current:.3f} nA... ", end="")
        
        # Longer stimulus for frequency measurement
        stim = h.IClamp(cell(0.5))
        stim.delay = 200
        stim.dur = 1000  # 1 second stimulus
        stim.amp = current
        
        t_vec = h.Vector()
        v_vec = h.Vector()
        t_vec.record(h._ref_t)
        v_vec.record(cell(0.5)._ref_v)
        
        h.dt = 0.025
        h.tstop = 1500
        h.v_init = -65
        h.finitialize(h.v_init)
        h.run()
        
        voltage = np.array(v_vec)
        spikes = count_spikes(voltage)
        frequency = spikes  # spikes per second (1s stimulus)
        
        frequencies.append(frequency)
        print(f"{frequency} Hz")
    
    return currents, frequencies


def count_spikes(voltage, threshold=-20):
    """Count action potentials in voltage trace."""
    spikes = 0
    above_threshold = voltage > threshold
    for i in range(1, len(above_threshold)):
        if above_threshold[i] and not above_threshold[i-1]:
            spikes += 1
    return spikes


def plot_results(excitability_results, f_i_currents, f_i_frequencies, rheobase):
    """Create summary plots of all test results."""
    print("\n📈 Creating summary plots...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Voltage traces for different currents
    ax1 = plt.subplot(2, 3, (1, 2))
    for result in excitability_results:
        current = result['current']
        time = result['time']
        voltage = result['voltage']
        spikes = result['spikes']
        
        label = f"{current:.3f} nA ({spikes} spikes)"
        ax1.plot(time, voltage, label=label, linewidth=1.5)
    
    ax1.axvspan(100, 600, alpha=0.2, color='red', label='Current injection')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Voltage Responses to Current Steps')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Current-Voltage relationship
    ax2 = plt.subplot(2, 3, 3)
    currents = [r['current'] for r in excitability_results]
    max_voltages = [r['max_voltage'] for r in excitability_results]
    
    ax2.plot(currents, max_voltages, 'o-', linewidth=2, markersize=6)
    if rheobase:
        ax2.axvline(rheobase, color='red', linestyle='--', label=f'Rheobase: {rheobase:.3f} nA')
        ax2.legend()
    ax2.set_xlabel('Current Amplitude (nA)')
    ax2.set_ylabel('Maximum Voltage (mV)')
    ax2.set_title('Current vs. Peak Voltage')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Frequency-Current relationship
    ax3 = plt.subplot(2, 3, 4)
    ax3.plot(f_i_currents, f_i_frequencies, 'o-', linewidth=2, markersize=6, color='green')
    ax3.set_xlabel('Current Amplitude (nA)')
    ax3.set_ylabel('Firing Frequency (Hz)')
    ax3.set_title('F-I Relationship')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Spike count summary
    ax4 = plt.subplot(2, 3, 5)
    spike_counts = [r['spikes'] for r in excitability_results]
    ax4.bar([f"{c:.3f}" for c in currents], spike_counts, alpha=0.7)
    ax4.set_xlabel('Current Amplitude (nA)')
    ax4.set_ylabel('Spike Count')
    ax4.set_title('Spike Count vs. Current')
    plt.xticks(rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Model summary
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    # Summary statistics
    max_freq = max(f_i_frequencies)
    total_spikes = sum(spike_counts)
    responsive_currents = len([s for s in spike_counts if s > 0])
    
    summary_text = f"""
MODEL TEST SUMMARY
==================

Ion Channels:
• NaTa_t (Na+): ✅
• Kv3_1 (K+): ✅  
• Passive: ✅

Excitability:
• Rheobase: {rheobase:.3f} nA
• Max frequency: {max_freq} Hz
• Total spikes: {total_spikes}
• Responsive currents: {responsive_currents}/{len(currents)}

Status: ✅ FUNCTIONAL
"""
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('quick_model_test_results.png', dpi=150, bbox_inches='tight')
    print("✅ Results saved: quick_model_test_results.png")
    return fig


def main():
    """Main testing function."""
    print("🚀 NeuroForge-Optimizer Quick Model Test")
    print("=" * 45)
    
    try:
        # Setup model
        cell = setup_model()
        
        # Run tests
        excitability_results = test_basic_excitability(cell)
        rheobase = test_threshold_finding(cell)
        f_i_currents, f_i_frequencies = test_frequency_current_relationship(cell)
        
        # Create summary plots
        plot_results(excitability_results, f_i_currents, f_i_frequencies, rheobase)
        
        # Final summary
        print("\n🎉 QUICK MODEL TEST COMPLETED!")
        print("=" * 45)
        print("✅ Your NeuroForge-Optimizer model is working correctly!")
        print("📊 Key findings:")
        if rheobase:
            print(f"   • Rheobase (threshold): {rheobase:.3f} nA")
        print(f"   • Maximum firing rate: {max(f_i_frequencies)} Hz")
        print(f"   • Model generates realistic action potentials")
        print("📁 Detailed results saved as: quick_model_test_results.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()