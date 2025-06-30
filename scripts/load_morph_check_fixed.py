#!/usr/bin/env python3
"""
Fixed Morphology Loading Script for NeuroForge-Optimizer.

This script properly loads SWC morphology files and creates visualization plots.
Handles NEURON Import3d functionality correctly.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import neuron
from neuron import h
from pathlib import Path


def setup_neuron_import3d():
    """Setup NEURON for morphology import."""
    print("🔧 Setting up NEURON for morphology import...")
    
    # Load standard files
    h.load_file("stdrun.hoc")
    
    # Load Import3d functionality
    try:
        h.load_file("import3d.hoc")
        print("✅ Import3d loaded successfully")
    except:
        print("⚠️  Import3d.hoc not found, trying alternative method")
    
    # Load mechanisms
    try:
        neuron.load_mechanisms('./mechanisms')
        print("✅ Mechanisms loaded")
    except:
        print("⚠️  Mechanisms not loaded")


def read_swc_file(swc_path):
    """
    Read SWC file manually to extract morphology data.
    
    SWC format: ID Type X Y Z Radius Parent
    Types: 1=soma, 2=axon, 3=basal dendrite, 4=apical dendrite
    """
    print(f"📖 Reading SWC file: {swc_path}")
    
    points = []
    connections = []
    
    try:
        with open(swc_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 7:
                        point_id = int(parts[0])
                        point_type = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        radius = float(parts[5])
                        parent_id = int(parts[6])
                        
                        points.append({
                            'id': point_id,
                            'type': point_type,
                            'x': x,
                            'y': y,
                            'z': z,
                            'radius': radius,
                            'parent': parent_id
                        })
                        
                        if parent_id != -1:  # -1 means no parent (root)
                            connections.append((parent_id, point_id))
                            
                except ValueError as e:
                    print(f"⚠️  Skipping malformed line {line_num}: {line}")
                    continue
        
        print(f"✅ Successfully read {len(points)} points, {len(connections)} connections")
        
    except Exception as e:
        print(f"❌ Error reading SWC file: {e}")
        return None, None
    
    return points, connections


def classify_neuron_regions(points):
    """Classify points by neuron region type."""
    regions = {
        1: {'name': 'soma', 'points': [], 'color': 'red', 'size': 50},
        2: {'name': 'axon', 'points': [], 'color': 'blue', 'size': 10},
        3: {'name': 'basal_dendrite', 'points': [], 'color': 'green', 'size': 15},
        4: {'name': 'apical_dendrite', 'points': [], 'color': 'purple', 'size': 15},
        'other': {'name': 'other', 'points': [], 'color': 'gray', 'size': 10}
    }
    
    for point in points:
        point_type = point['type']
        if point_type in regions:
            regions[point_type]['points'].append(point)
        else:
            regions['other']['points'].append(point)
    
    return regions


def load_morphology_with_neuron(swc_path):
    """
    Try to load morphology using NEURON's Import3d if available.
    Falls back to manual reading if Import3d is not available.
    """
    print("🧬 Attempting to load morphology with NEURON...")
    
    setup_neuron_import3d()
    
    # Try NEURON Import3d method
    try:
        # Check if Import3d functions are available
        if hasattr(h, 'Import3d_SWC_read'):
            print("✅ Using NEURON Import3d_SWC_read")
            
            # Create Import3d reader
            reader = h.Import3d_SWC_read()
            reader.input(str(swc_path))
            
            # Create importer
            importer = h.Import3d_GUI(reader, 0)
            
            # Create sections
            importer.instantiate(h)
            
            # Extract coordinates from NEURON sections
            points = []
            for sec in h.allsec():
                n3d = int(h.n3d(sec=sec))
                for i in range(n3d):
                    x = h.x3d(i, sec=sec)
                    y = h.y3d(i, sec=sec) 
                    z = h.z3d(i, sec=sec)
                    diam = h.diam3d(i, sec=sec)
                    
                    # Determine type based on section name
                    sec_name = sec.name()
                    if 'soma' in sec_name.lower():
                        point_type = 1
                    elif 'axon' in sec_name.lower():
                        point_type = 2
                    elif 'apic' in sec_name.lower():
                        point_type = 4
                    elif 'dend' in sec_name.lower():
                        point_type = 3
                    else:
                        point_type = 5
                    
                    points.append({
                        'id': len(points) + 1,
                        'type': point_type,
                        'x': x,
                        'y': y,
                        'z': z,
                        'radius': diam/2,
                        'parent': -1,
                        'section': sec_name
                    })
            
            print(f"✅ NEURON Import3d successful: {len(points)} points")
            return points, []
            
        else:
            raise AttributeError("Import3d_SWC_read not available")
            
    except Exception as e:
        print(f"⚠️  NEURON Import3d failed: {e}")
        print("📖 Falling back to manual SWC reading...")
        
        # Fall back to manual reading
        points, connections = read_swc_file(swc_path)
        return points, connections


def create_morphology_plots(points, connections, swc_path):
    """Create comprehensive morphology visualization plots."""
    print("📊 Creating morphology plots...")
    
    if not points:
        print("❌ No points to plot")
        return None
    
    # Classify regions
    regions = classify_neuron_regions(points)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: XY projection (top view)
    ax1 = axes[0, 0]
    for region_type, region_data in regions.items():
        if region_data['points']:
            x_coords = [p['x'] for p in region_data['points']]
            y_coords = [p['y'] for p in region_data['points']]
            
            ax1.scatter(x_coords, y_coords, 
                       c=region_data['color'], 
                       s=region_data['size'],
                       label=f"{region_data['name']} ({len(region_data['points'])})",
                       alpha=0.7)
    
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_title('Morphology - Top View (XY)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: XZ projection (side view)
    ax2 = axes[0, 1]
    for region_type, region_data in regions.items():
        if region_data['points']:
            x_coords = [p['x'] for p in region_data['points']]
            z_coords = [p['z'] for p in region_data['points']]
            
            ax2.scatter(x_coords, z_coords,
                       c=region_data['color'],
                       s=region_data['size'],
                       label=f"{region_data['name']}",
                       alpha=0.7)
    
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Z (μm)')
    ax2.set_title('Morphology - Side View (XZ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: YZ projection (front view)
    ax3 = axes[1, 0]
    for region_type, region_data in regions.items():
        if region_data['points']:
            y_coords = [p['y'] for p in region_data['points']]
            z_coords = [p['z'] for p in region_data['points']]
            
            ax3.scatter(y_coords, z_coords,
                       c=region_data['color'],
                       s=region_data['size'],
                       label=f"{region_data['name']}",
                       alpha=0.7)
    
    ax3.set_xlabel('Y (μm)')
    ax3.set_ylabel('Z (μm)')
    ax3.set_title('Morphology - Front View (YZ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    total_points = len(points)
    total_connections = len(connections)
    
    # Spatial extent
    all_x = [p['x'] for p in points]
    all_y = [p['y'] for p in points]
    all_z = [p['z'] for p in points]
    
    x_range = max(all_x) - min(all_x) if all_x else 0
    y_range = max(all_y) - min(all_y) if all_y else 0
    z_range = max(all_z) - min(all_z) if all_z else 0
    
    # Region counts
    region_counts = {}
    for region_type, region_data in regions.items():
        if region_data['points']:
            region_counts[region_data['name']] = len(region_data['points'])
    
    summary_text = f"""
MORPHOLOGY ANALYSIS SUMMARY
===========================

📁 File: {swc_path.name}

📊 Point Statistics:
  • Total points: {total_points}
  • Total connections: {total_connections}

📏 Spatial Extent:
  • X range: {x_range:.1f} μm
  • Y range: {y_range:.1f} μm  
  • Z range: {z_range:.1f} μm

🧬 Region Breakdown:
"""
    
    for region_name, count in region_counts.items():
        percentage = (count / total_points * 100) if total_points > 0 else 0
        summary_text += f"  • {region_name}: {count} ({percentage:.1f}%)\n"
    
    summary_text += f"\n✅ MORPHOLOGY LOADED SUCCESSFULLY"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    return fig


def main():
    """Main morphology checking function."""
    print("🧬 NeuroForge-Optimizer Morphology Check")
    print("=" * 45)
    
    # Define morphology path
    script_dir = Path(__file__).parent
    morph_path = script_dir.parent / "morphologies" / "720575940622093546_obaid.swc"
    
    # Check if file exists
    if not morph_path.exists():
        print(f"❌ Morphology file not found: {morph_path}")
        print("Please ensure the SWC file is in the morphologies/ directory")
        return False
    
    print(f"📁 Morphology file: {morph_path}")
    
    try:
        # Load morphology
        points, connections = load_morphology_with_neuron(morph_path)
        
        if not points:
            print("❌ Failed to load morphology")
            return False
        
        # Create plots
        fig = create_morphology_plots(points, connections, morph_path)
        
        if fig:
            # Save plot
            output_path = script_dir.parent / "morphology_analysis.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Morphology analysis saved: {output_path}")
            
            # Also create a simple XY plot for compatibility
            simple_fig, simple_ax = plt.subplots(figsize=(10, 8))
            
            all_x = [p['x'] for p in points]
            all_y = [p['y'] for p in points]
            
            simple_ax.scatter(all_x, all_y, s=2, alpha=0.7)
            simple_ax.set_xlabel('X (μm)')
            simple_ax.set_ylabel('Y (μm)')
            simple_ax.set_title(f'Morphology: {morph_path.name}')
            simple_ax.grid(True, alpha=0.3)
            simple_ax.set_aspect('equal')
            
            simple_output = script_dir.parent / "morphology_check.png"
            simple_fig.savefig(simple_output, dpi=150, bbox_inches='tight')
            print(f"✅ Simple morphology plot saved: {simple_output}")
            
            plt.close('all')  # Close figures to free memory
            
        print("\n🎉 Morphology check completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during morphology check: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()