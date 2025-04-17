import sympy as sp

x = sp.Symbol('x', real=True)
th_eta = sp.Symbol(r"\theta_{\eta\eta'}", real=True)

alp = sp.Symbol('a', real=True)
pi0 = sp.Symbol(r'\pi^0', real=True)
eta = sp.Symbol(r'\eta', real=True)
etap = sp.Symbol(r"\eta'", real=True)
eta0 = sp.Symbol(r'\eta_0', real=True)
eta8 = sp.Symbol(r"\eta_8'", real=True)
pi_plus = sp.Symbol(r'\pi^+', real=False)
pi_minus = sp.Symbol(r'\pi^-', real=False)
K_plus = sp.Symbol(r'K^+', real=False)
K_minus = sp.Symbol(r'K^-', real=False)
K0 = sp.Symbol(r'K^0', real=False)
K0_bar = sp.Symbol(r'\bar{K}^0', real=False)

alpF = sp.Function('a', real=True)(x)
pi0F = sp.Function(r'\pi^0', real=True)(x)
etaF = sp.Function(r'\eta', real=True)(x)
etapF = sp.Function(r"\eta'", real=True)(x)
eta0F = sp.Function(r'\eta_0', real=True)(x)
eta8F = sp.Function(r"\eta_8'", real=True)(x)
pi_plusF = sp.Function(r'\pi^+', real=False)(x)
pi_minusF = sp.Function(r'\pi^-', real=False)(x)
K_plusF = sp.Function(r'K^+', real=False)(x)
K_minusF = sp.Function(r'K^-', real=False)(x)
K0F = sp.Function(r'K^0', real=False)(x)
K0_barF = sp.Function(r'\bar{K}^0', real=False)(x)

dalp = sp.Symbol(r'\partial_{\mu} a', real=True)
dpi0 = sp.Symbol(r'\partial_{\mu} \pi^0', real=True)
deta = sp.Symbol(r'\partial_{\mu} \eta', real=True)
detap = sp.Symbol(r"\partial_{\mu} \eta'", real=True)
deta0 = sp.Symbol(r'\partial_{\mu} \eta_0', real=True)
deta8 = sp.Symbol(r"\partial_{\mu} \eta_8'", real=True)
dpi_plus = sp.Symbol(r'\partial_{\mu} \pi^+', real=False)
dpi_minus = sp.Symbol(r'\partial_{\mu} \pi^-', real=False)
dK_plus = sp.Symbol(r'\partial_{\mu} K^+', real=False)
dK_minus = sp.Symbol(r'\partial_{\mu} K^-', real=False)
dK0 = sp.Symbol(r'\partial_{\mu} K^0', real=False)
dK0_bar = sp.Symbol(r'\partial_{\mu} \bar{K}^0', real=False)

def conjugated_fields(expr: sp.Expr) -> sp.Expr:
    return expr.subs({
        sp.conjugate(pi_plus) :pi_minus,
        sp.conjugate(pi_minus) :pi_plus,
        sp.conjugate(K_plus) :K_minus,
        sp.conjugate(K_minus) :K_plus,
        sp.conjugate(K0) :K0_bar,
        sp.conjugate(K0_bar) :K0,
        sp.conjugate(dpi_plus) :dpi_minus,
        sp.conjugate(dpi_minus) :dpi_plus,
        sp.conjugate(dK_plus) :dK_minus,
        sp.conjugate(dK_minus) :dK_plus,
        sp.conjugate(dK0) :dK0_bar,
        sp.conjugate(dK0_bar) :dK0,
    })

def field_to_symbol(expr: sp.Expr) -> sp.Expr:
    return expr.subs({
        pi_plusF : pi_plus,
        pi_minusF : pi_minus,
        K_plusF : K_plus,
        K_minusF : K_minus,
        K0F : K0,
        K0_barF : K0_bar,
        alpF : alp,
        pi0F : pi0,
        etaF : eta,
        etapF : etap,
        eta0F : eta0,
        eta8F : eta8,
        sp.diff(pi_plusF, x) : dpi_plus,
        sp.diff(pi_minusF, x) : dpi_minus,
        sp.diff(K_plusF, x) : dK_plus,
        sp.diff(K_minusF, x) : dK_minus,
        sp.diff(K0F, x) : dK0,
        sp.diff(K0_barF, x) : dK0_bar,
        sp.diff(alpF, x) : dalp,
        sp.diff(pi0F, x) : dpi0,
        sp.diff(etaF, x) : deta,
        sp.diff(etapF, x) : detap,
        sp.diff(eta0F, x) : deta0,
        sp.diff(eta8F, x) : deta8,
    })

def symbol_to_field(expr: sp.Expr) -> sp.Expr:
    return expr.subs({
        pi_plus : pi_plusF,
        pi_minus : pi_minusF,
        K_plus : K_plusF,
        K_minus : K_minusF,
        K0 : K0F,
        K0_bar : K0_barF,
        alp : alpF,
        pi0 : pi0F,
        eta : etaF,
        etap : etapF,
        eta0 : eta0F,
        eta8 : eta8F,
        dpi_plus : sp.diff(pi_plusF, x),
        dpi_minus : sp.diff(pi_minusF, x),
        dK_plus : sp.diff(K_plusF, x),
        dK_minus : sp.diff(K_minusF, x),
        dK0 : sp.diff(K0F, x),
        dK0_bar : sp.diff(K0_barF, x),
        dalp : sp.diff(alpF, x),
        dpi0 : sp.diff(pi0F, x),
        deta : sp.diff(etaF, x),
        detap : sp.diff(etapF, x),
        deta0 : sp.diff(eta0F, x),
        deta8 : sp.diff(eta8F, x),
    })

def rotate_eta(expr: sp.Expr) -> sp.Expr:
    return expr.subs({
        eta8: eta * sp.cos(th_eta) + etap * sp.sin(th_eta),
        eta0: -eta * sp.sin(th_eta) + etap * sp.cos(th_eta),
        eta8F: etaF * sp.cos(th_eta) + etapF * sp.sin(th_eta),
        eta0F: -etaF * sp.sin(th_eta) + etapF * sp.cos(th_eta),
        sp.diff(eta8F, x): sp.diff(etaF, x) * sp.cos(th_eta) + sp.diff(etapF, x) * sp.sin(th_eta),
        sp.diff(eta0F, x): -sp.diff(etaF, x) * sp.sin(th_eta) + sp.diff(etapF, x) * sp.cos(th_eta),
        deta8: deta * sp.cos(th_eta) + detap * sp.sin(th_eta),
        deta0: -deta * sp.sin(th_eta) + detap * sp.cos(th_eta),
    })