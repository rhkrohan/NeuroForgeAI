#!/usr/bin/env python3
"""
Simple demo simulation script for NeuroForge-Optimizer.

This script demonstrates basic simulation without requiring the full optimization pipeline.
Run this to see if your basic setup is working.
"""

def test_imports():
    """Test if required packages are available."""
    print("🧪 Testing package imports...")
    
    try:
        import sys
        print(f"✅ Python {sys.version.split()[0]}")
    except:
        print("❌ Python issue")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not available - install with: pip install numpy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib available")
    except ImportError:
        print("❌ Matplotlib not available - install with: pip install matplotlib")
        return False
    
    try:
        import neuron
        from neuron import h
        print(f"✅ NEURON available")
        return True
    except ImportError:
        print("❌ NEURON not available - install with: pip install neuron")
        return False


def test_mechanisms():
    """Test if NEURON mechanisms are compiled and loadable."""
    print("\n🔧 Testing NEURON mechanisms...")
    
    try:
        import neuron
        neuron.load_mechanisms('./mechanisms')
        print("✅ NEURON mechanisms loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Mechanism loading failed: {e}")
        print("💡 Fix: cd mechanisms && nrnivmodl")
        return False


def simple_simulation():
    """Run a simple NEURON simulation."""
    print("\n🚀 Running simple simulation...")
    
    try:
        import neuron
        from neuron import h
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Load mechanisms
        neuron.load_mechanisms('./mechanisms')
        
        # Initialize NEURON
        h.load_file("stdrun.hoc")
        
        # Create simple cell
        print("Creating cell...")
        soma = h.Section(name='soma')
        soma.L = 20          # Length (μm)
        soma.diam = 20       # Diameter (μm)
        
        # Insert passive properties
        soma.insert('pas')
        soma.g_pas = 0.0001  # Passive conductance (S/cm2)
        soma.e_pas = -65     # Reversal potential (mV)
        soma.Ra = 100        # Axial resistance (Ω⋅cm)
        soma.cm = 1          # Membrane capacitance (μF/cm2)
        
        # Try to insert active mechanisms (if available)
        try:
            soma.insert('NaTa_t')
            soma.gbar_NaTa_t = 0.1
            print("✅ Added NaTa_t (sodium channel)")
        except:
            print("⚠️  NaTa_t mechanism not available")
        
        try:
            soma.insert('Kv3_1')
            soma.gbar_Kv3_1 = 0.05
            print("✅ Added Kv3_1 (potassium channel)")
        except:
            print("⚠️  Kv3_1 mechanism not available")
        
        # Create current clamp stimulus
        print("Setting up stimulus...")
        stim = h.IClamp(soma(0.5))
        stim.delay = 100     # Start time (ms)
        stim.dur = 500       # Duration (ms)
        stim.amp = 0.1       # Current amplitude (nA)
        
        # Set up recordings
        print("Setting up recordings...")
        t_vec = h.Vector()
        v_vec = h.Vector()
        t_vec.record(h._ref_t)
        v_vec.record(soma(0.5)._ref_v)
        
        # Run simulation
        print("Running simulation...")
        h.dt = 0.025         # Time step (ms)
        h.tstop = 800        # Stop time (ms)
        h.v_init = -65       # Initial voltage (mV)
        
        h.finitialize(h.v_init)
        h.run()
        
        # Convert to numpy arrays
        time = np.array(t_vec)
        voltage = np.array(v_vec)
        
        print("✅ Simulation completed successfully!")
        
        # Plot results
        print("Creating plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(time, voltage, linewidth=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title('NeuroForge-Optimizer Demo Simulation')
        plt.grid(True, alpha=0.3)
        
        # Add stimulus indicator
        plt.axvspan(100, 600, alpha=0.2, color='red', label='Current injection (0.1 nA)')
        plt.legend()
        
        # Save plot
        output_file = 'demo_simulation_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved as: {output_file}")
        
        # Show plot
        plt.show()
        
        # Print results summary
        print(f"\n📊 Simulation Results:")
        print(f"   Resting potential: {voltage[0]:.1f} mV")
        print(f"   Maximum voltage: {np.max(voltage):.1f} mV")
        print(f"   Minimum voltage: {np.min(voltage):.1f} mV")
        print(f"   Simulation duration: {time[-1]:.1f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo function."""
    print("🧬 NeuroForge-Optimizer Demo Simulation")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Basic packages missing. Install with:")
        print("   pip install neuron numpy matplotlib")
        return
    
    # Test mechanisms
    if not test_mechanisms():
        print("\n❌ NEURON mechanisms not ready. Run:")
        print("   cd mechanisms && nrnivmodl")
        return
    
    # Run simulation
    if simple_simulation():
        print("\n🎉 Demo simulation successful!")
        print("✅ Your NeuroForge-Optimizer setup is working!")
        print("\n🚀 Next steps:")
        print("   1. Add experimental data to recordings/")
        print("   2. Run: python scripts/feature_extract.py")
        print("   3. Run: python scripts/optimise.py")
    else:
        print("\n❌ Demo simulation failed")
        print("💡 Check error messages above for troubleshooting")


if __name__ == "__main__":
    main()