#!/usr/bin/env python3
"""
Morphology sanity check script.

Loads the SWC file and generates an XY scatter plot of all segment coordinates
to verify the morphology is loaded correctly.
"""

import matplotlib.pyplot as plt
import neuron
from pathlib import Path


def load_and_plot_morphology(swc_path):
    """
    Load morphology from SWC file and create XY scatter plot.
    
    Args:
        swc_path (Path): Path to the SWC morphology file
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Initialize NEURON
    neuron.h.load_file("stdrun.hoc")
    
    # Create cell and load morphology
    cell = neuron.h.Section(name='soma')
    morph = neuron.h.Import3d_SWC_read()
    morph.input(str(swc_path))
    
    imprt = neuron.h.Import3d_GUI(morph, 0)
    imprt.instantiate(cell)
    
    # Extract all segment coordinates
    x_coords = []
    y_coords = []
    
    # Iterate through all sections
    for section in neuron.h.allsec():
        # Get 3D coordinates for each segment
        for i in range(int(neuron.h.n3d(sec=section))):
            x_coords.append(neuron.h.x3d(i, sec=section))
            y_coords.append(neuron.h.y3d(i, sec=section))
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_coords, y_coords, s=2, alpha=0.7)
    ax.set_xlabel('X coordinate (μm)')
    ax.set_ylabel('Y coordinate (μm)')
    ax.set_title(f'Morphology: {swc_path.name}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add statistics
    stats_text = f'Points: {len(x_coords)}\nSections: {len(list(neuron.h.allsec()))}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def main():
    """Main function to run morphology check."""
    # Define morphology path relative to script location
    script_dir = Path(__file__).parent
    morph_path = script_dir.parent / "morphologies" / "720575940622093546_obaid.swc"
    
    # Check if morphology file exists
    if not morph_path.exists():
        print(f"Error: Morphology file not found at {morph_path}")
        print("Please ensure the SWC file is in the morphologies/ directory")
        return
    
    print(f"Loading morphology from: {morph_path}")
    
    try:
        fig, ax = load_and_plot_morphology(morph_path)
        
        # Save plot
        output_path = script_dir.parent / "morphology_check.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Morphology plot saved to: {output_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"Error loading morphology: {e}")
        raise


if __name__ == "__main__":
    main()