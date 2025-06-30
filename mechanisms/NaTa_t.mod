TITLE Sodium transient current (NaTa_t)

COMMENT
Transient sodium channel (Traub-type)
Simplified model for e-model optimization
ENDCOMMENT

NEURON {
    SUFFIX NaTa_t
    USEION na READ ena WRITE ina
    RANGE gbar, g, ina
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
    gbar = 0.1 (S/cm2)
    vshift = 0 (mV)
}

STATE {
    m h
}

ASSIGNED {
    v (mV)
    ena (mV)
    ina (mA/cm2)
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
    g = gbar * m^3 * h
    ina = g * (v - ena)
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
    alpha = 0.182 * (v + 38) / (1 - exp(-(v + 38)/6))
    beta = 0.124 * (-v - 38) / (1 - exp((v + 38)/6))
    sum = alpha + beta
    mtau = 1 / (q10 * sum)
    minf = alpha / sum
    
    : h inactivation
    alpha = 0.015 * exp(-(v + 60)/28)
    beta = 0.015 * exp((v + 60)/28)
    sum = alpha + beta
    htau = 1 / (q10 * sum)
    hinf = alpha / sum
    
    UNITSON
}