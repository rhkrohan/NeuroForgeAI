TITLE Hyperpolarization-activated cation channel (Ih)

COMMENT
HCN channel (hyperpolarization-activated, cyclic nucleotide-gated)
Simplified model for e-model optimization
ENDCOMMENT

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT i
    RANGE gbar, g, i, ehcn
    RANGE minf, mtau
    GLOBAL vshift
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
}

PARAMETER {
    gbar = 0.0001 (S/cm2)
    ehcn = -45 (mV)
    vshift = 0 (mV)
}

STATE {
    m
}

ASSIGNED {
    v (mV)
    i (mA/cm2)
    g (S/cm2)
    minf
    mtau (ms)
}

INITIAL {
    rates(v)
    m = minf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * m
    i = g * (v - ehcn)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
}

PROCEDURE rates(v(mV)) {
    LOCAL alpha, beta, sum, q10
    
    q10 = 4.5^((34-21)/10)
    
    UNITSOFF
    
    : m activation (hyperpolarization-activated)
    minf = 1 / (1 + exp((v + 91)/10))
    mtau = 1 / (q10 * (exp(-14.59 - 0.086*v) + exp(-1.87 + 0.0701*v)))
    
    if (mtau < 2) {
        mtau = 2
    }
    
    UNITSON
}