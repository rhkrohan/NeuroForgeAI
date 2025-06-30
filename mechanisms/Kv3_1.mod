TITLE Kv3.1 potassium channel (Kv3_1)

COMMENT
Fast delayed rectifier potassium channel
Simplified model for e-model optimization
ENDCOMMENT

NEURON {
    SUFFIX Kv3_1
    USEION k READ ek WRITE ik
    RANGE gbar, g, ik
    RANGE ninf, ntau
    GLOBAL vshift
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
}

PARAMETER {
    gbar = 0.01 (S/cm2)
    vshift = 0 (mV)
}

STATE {
    n
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    g (S/cm2)
    ninf
    ntau (ms)
}

INITIAL {
    rates(v)
    n = ninf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * n^4
    ik = g * (v - ek)
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n) / ntau
}

PROCEDURE rates(v(mV)) {
    LOCAL alpha, beta, sum, q10
    
    q10 = 2.3^((34-21)/10)
    
    UNITSOFF
    
    : n activation
    alpha = 0.02 * (v - 25) / (1 - exp(-(v - 25)/9))
    beta = 0.002 * exp(-(v - 25)/35)
    sum = alpha + beta
    ntau = 1 / (q10 * sum)
    ninf = alpha / sum
    
    UNITSON
}