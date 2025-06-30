# NEURON Mechanisms Directory

This directory contains NEURON mechanism files (.mod) for ion channels and other cellular mechanisms.

## Compilation

To compile the mechanisms, run:

```bash
cd mechanisms
nrnivmodl
```

This will create compiled mechanism files in the `x86_64/` subdirectory (architecture-dependent).

## Available Mechanisms

The following mechanisms are included:

- **NaTa_t**: Transient sodium channel (Traub-type)
- **NaP_Et2**: Persistent sodium channel
- **Kv3_1**: Kv3.1 potassium channel (fast delayed rectifier)
- **K_T**: Transient potassium channel (A-type)
- **SK_E2**: Small conductance calcium-activated potassium channel
- **Ca_HVA**: High voltage-activated calcium channel
- **Ca_LVAst**: Low voltage-activated calcium channel (T-type)
- **CaDynamics_E2**: Calcium dynamics mechanism
- **Ih**: Hyperpolarization-activated cation channel (HCN)

## Usage

After compilation, the mechanisms can be loaded in Python:

```python
import neuron
neuron.load_mechanisms("./mechanisms")

# Insert mechanisms into sections
soma.insert('NaTa_t')
soma.insert('Kv3_1')
# etc.
```

## Notes

- These are simplified example mechanisms
- For production use, replace with validated mechanisms from ModelDB or other sources
- Parameters should be adjusted based on experimental data
- Mechanisms are compatible with NEURON 8.2+