import sympy as sp
from particles import *
from chiral_Lagrangian import *

cG = sp.Symbol(r'c_G', real=True)
cu = sp.Symbol(r'c_u', real=True)
cd = sp.Symbol(r'c_d', real=True)
cs = sp.Symbol(r'c_s', real=True)
cds = sp.Symbol(r'c_{ds}', real=False)
csd = sp.Symbol(r'c_{sd}', real=False)
kappau = sp.Symbol(r'\kappa_u', real=True)
kappad = sp.Symbol(r'\kappa_d', real=True)
kappas = sp.Symbol(r'\kappa_s', real=True)
kappa = sp.diag(kappau, kappad, kappas)
cq = sp.Matrix([[cu, 0, 0], 
                 [0, cd, cds],
                 [0, csd, cs]])
cqhat = cq + 2*cG*kappa
fa = sp.Symbol(r'f_a', real=True)

def alp_chi_lagrangian(nfields: int) -> sp.Expr:
    """
    The chiral Lagrangian for the light quarks and the axion.
    """
    smfields = nfields - 1
    U = field_to_symbol(expansion_U(smfields))
    chi_alp = -2*sp.I*cG*alp/fa * (kappa @ chi)
    lag = F0**2*sp.Rational(1,4)*sp.trace(chi_alp.H @ U + chi_alp @ U.H)
    return sp.expand(sp.simplify(rotate_eta(conjugated_fields(lag))))

def alp_kinetic_lagrangian(nfields: int) -> sp.Expr:
    terms = []
    Phi = sp.MatrixSymbol(r'\Phi', 3, 3)
    dPhi = sp.MatrixSymbol(r'\partial_{\mu} \Phi', 3, 3)
    for i in range(1, nfields-1):
        j = nfields - i-1
        Usymb = sp.I**j/F0**j * Phi**j/sp.factorial(j)
        Udsymb = (-sp.I)**j/F0**j * Phi**j/sp.factorial(j)
        s = []
        for k in range(i):
            s.append(Phi**k @ dPhi @ Phi**(i-k-1))
        s0 = sum(s[1:], s[0])
        dUsymb = sp.I**i/F0**i * s0/sp.factorial(i)
        dUdsymb = (-sp.I)**i/F0**i * s0/sp.factorial(i)
        
        terms.append(field_to_symbol(sp.expand(Udsymb*dUsymb + dUsymb*Udsymb - Usymb*dUdsymb - dUdsymb*Usymb).subs({Phi: pseudoscalar_matrix, dPhi: sp.diff(pseudoscalar_matrix, x)})))
    if len(terms) == 0:
        dUsymb = sp.I/F0 * dPhi
        dUdsymb = (-sp.I)/F0 * dPhi
        term = field_to_symbol(sp.expand(2*dUsymb-2*dUdsymb).subs({Phi: pseudoscalar_matrix, dPhi: sp.diff(pseudoscalar_matrix, x)}))
    elif len(terms) == 1:
        term = terms[0]
    else:
        term = sum(terms[1:], terms[0])
    return sp.expand(F0**2*sp.Rational(1,8)*sp.I*dalp/fa*rotate_eta(sp.trace(cqhat @ term)))