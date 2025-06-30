TITLE High voltage activated calcium channel (Ca_HVA)

COMMENT
High voltage activated calcium channel
Simplified model for e-model optimization
ENDCOMMENT

NEURON {
    SUFFIX Ca_HVA
    USEION ca READ eca WRITE ica
    RANGE gbar, g, ica
    RANGE minf, hinf, mtau, htau
    GLOBAL vshift
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
}

PARAMETER {
    gbar = 0.001 (S/cm2)
    vshift = 0 (mV)
}

STATE {
    m h
}

ASSIGNED {
    v (mV)
    eca (mV)
    ica (mA/cm2)
    g (S/cm2)
    minf
    hinf
    mtau (ms)
    htau (ms)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * m^2 * h
    ica = g * (v - eca)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
}

PROCEDURE rates(v(mV)) {
    LOCAL alpha, beta, sum, q10
    
    q10 = 2.3^((34-21)/10)
    
    UNITSOFF
    
    : m activation
    alpha = 0.055 * (-27 - v) / (exp((-27-v)/3.8) - 1)
    beta = (0.94 * exp((-75-v)/17))
    sum = alpha + beta
    mtau = 1 / (q10 * sum)
    minf = alpha / sum
    
    : h inactivation
    alpha = 0.000457 * exp((-13-v)/50)
    beta = 0.0065 / (exp((-v-15)/28) + 1)
    sum = alpha + beta
    htau = 1 / (q10 * sum)
    hinf = alpha / sum
    
    UNITSON
}