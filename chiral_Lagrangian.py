from particles import *
import sympy as sp
from itertools import combinations

def two_derivatives(n):
    if n < 2:
        raise ValueError("n must be greater than 2.")
    results = []
    for combo in combinations(range(n), 2):
        lst = [0]*n
        lst[combo[0]] = 1
        lst[combo[1]] = 1
        results.append(lst)
    return results

F0 = sp.Symbol('F_0', real=True)
mu = sp.Symbol('m_u', real=True)
md = sp.Symbol('m_d', real=True)
ms = sp.Symbol('m_s', real=True)
mq = sp.diag(mu, md, ms)
mpi = sp.Symbol(r'm_{\pi^0}', real=True)
meta0 = sp.Symbol('m_{\eta^0}', real=True)
B0 = mpi**2/(mu+md)
chi = 2*B0*mq
th_pi_eta = sp.Symbol(r'\theta_{\pi^0\eta}', real=True)
th_pi_etap = sp.Symbol(r"\theta_{\pi^0\eta'}", real=True)
th_pi_alp = sp.Symbol(r'\theta_{\pi^0a}', real=True)
th_eta_alp = sp.Symbol(r'\theta_{\eta a}', real=True)
th_etap_alp = sp.Symbol(r"\theta_{\eta' a}", real=True)
th_K0_alp = sp.Symbol(r'\theta_{K^0 a}', real=True)
th_K0bar_alp = sp.Symbol(r'\theta_{\bar{K}^0 a}', real=True)

pseudoscalar_matrix = pi0F * sp.diag(1, -1, 0) + \
                      eta0F *sp.sqrt(2)/sp.sqrt(3) *sp.eye(3) + \
                      eta8F /sp.sqrt(3) *sp.diag(1, 1, -2) + \
                      pi_plusF * sp.Matrix([[0, sp.sqrt(2), 0], [0, 0, 0], [0, 0, 0]]) + \
                      pi_minusF * sp.Matrix([[0, 0, 0], [sp.sqrt(2), 0, 0], [0, 0, 0]]) + \
                      K_minusF * sp.Matrix([[0, 0, 0], [0, 0, 0], [sp.sqrt(2), 0, 0]]) + \
                      K_plusF * sp.Matrix([[0, 0, sp.sqrt(2)], [0, 0, 0], [0, 0, 0]]) + \
                      K0F * sp.Matrix([[0, 0, 0], [0, 0, sp.sqrt(2)], [0, 0, 0]]) + \
                      K0_barF * sp.Matrix([[0, 0, 0], [0, 0, 0], [0, sp.sqrt(2), 0]])

def expansion_U(order: int) -> sp.Expr:
    if order == 0:
        return sp.eye(3)
    return (sp.I/F0)**order*sp.Rational(1,sp.factorial(order)) * pseudoscalar_matrix**order

def expansion_dU(order: int) -> sp.Expr:
    if order == 0:
        return sp.zeros(3)
    return sp.diff(expansion_U(order), x)

def sm_chi_lagrangian(nfields: int) -> sp.Expr:
    if nfields == 2:
        lag = - meta0**2*sp.Rational(1,2)*eta0**2
    else:
        lag = 0
    U = field_to_symbol(expansion_U(nfields))
    lag += F0**2*sp.Rational(1,4)*sp.trace(chi @ U + chi @ U.H)
    return sp.expand(sp.simplify(rotate_eta(conjugated_fields(lag))))

def sm_kinetic_lagrangian(nfields: int) -> sp.Expr:
    lag = 0
    for i in range(1, nfields):
        j = nfields - i
        dU = rotate_eta(field_to_symbol(expansion_dU(i)))
        dU_dagger = rotate_eta(conjugated_fields(field_to_symbol(expansion_dU(j)).H))
        lag += sp.trace(dU @ dU_dagger)
    return sp.expand(sp.simplify(F0**2*sp.Rational(1,4)*lag))

def rotated_lagrangian(lagr: sp.Expr) -> sp.Expr:
    eps_eta = sp.Symbol(r'\epsilon_\eta', real=True)
    eps_alp = sp.Symbol(r'\epsilon_a', real=True)
    lagr_rot = sp.expand(lagr.subs({
        pi0: pi0 + eps_eta*(th_pi_eta * eta + th_pi_etap * etap) + eps_alp*th_pi_alp * alp,
        eta: eta - eps_eta*th_pi_eta * pi0 + eps_alp*th_eta_alp * alp,
        etap: etap - eps_eta*th_pi_etap * pi0 + eps_alp*th_etap_alp * alp,
        K0: K0  + eps_alp*th_K0_alp * alp,
        K0_bar: K0_bar  + eps_alp*th_K0bar_alp * alp,
        dpi0: dpi0 + eps_eta*(th_pi_eta * deta + th_pi_etap * detap) + eps_alp*th_pi_alp * dalp,
        deta: deta - eps_eta*th_pi_eta * dpi0 + eps_alp*th_eta_alp * dalp,
        detap: detap - eps_eta*th_pi_etap * dpi0 + eps_alp*th_etap_alp * dalp,
        dK0: dK0  + eps_alp*th_K0_alp * dalp,
        dK0_bar: dK0_bar  + eps_alp*th_K0bar_alp * dalp,
    }))
    return sp.series(sp.series(lagr_rot, eps_eta, 0, 2).removeO(), eps_alp, 0, 2).removeO().subs({eps_eta: 1, eps_alp:1})

def extract_coefficient(lagrangian, fields):
    result = {}
    derivatives = two_derivatives(len(fields))
    for d in derivatives:
        prod = 1
        for i, f in enumerate(fields):
            if d[i] == 0:
                prod *= symbol_to_field(f)
            else:
                prod *= sp.diff(symbol_to_field(f), x)
        operator = field_to_symbol(prod)
        result[operator] = lagrangian.coeff(operator)
    prod = 1
    for i, f in enumerate(fields):
        prod *= f
    result[prod] = lagrangian.coeff(prod)
    return result