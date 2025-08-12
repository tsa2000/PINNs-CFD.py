"""physics_model.py
Governing equations helper: property queries and residual building blocks.
The functions are kept modular for clarity and easy unit testing.
"""
import math
import numpy as np

# Try optional backends
try:
    from CoolProp.CoolProp import PropsSI
    _HAVE_COOLPROP = True
except Exception:
    PropsSI = None
    _HAVE_COOLPROP = False

try:
    import cantera as ct
    _HAVE_CAN
except Exception:
    ct = None
    _HAVE_CAN = False

def query_properties(T, P, fluid='Air'):
    """Return a dict with keys rho, mu, cp, k_th for scalar T(K), P(Pa).
    Falls back to ideal-gas / simple correlations if CoolProp/Cantera are not available.
    """
    try:
        if _HAVE_COOLPROP:
            rho = float(PropsSI('D','T',float(T),'P',float(P),fluid))
            mu = float(PropsSI('V','T',float(T),'P',float(P),fluid))
            cp = float(PropsSI('C','T',float(T),'P',float(P),fluid))
            k_th = float(PropsSI('L','T',float(T),'P',float(P),fluid))
            return {'rho':rho, 'mu':mu, 'cp':cp, 'k_th':k_th}
    except Exception:
        # continue to fallback

        pass
    # Simple air-like fallback (SI units)
    R = 287.058
    rho = float(P) / (R * max(float(T), 1.0))
    mu = 1.8e-5 * (float(T) / 300.0) ** 0.7
    cp = 1005.0
    k_th = 0.026
    return {'rho':rho, 'mu':mu, 'cp':cp, 'k_th':k_th}

def prepare_property_arrays(T_arr, P_arr, fluid='Air'):
    """Query property arrays for vectors of T and P (numpy arrays)."""
    out = {'rho':[], 'mu':[], 'cp':[], 'k_th':[]}
    for T,P in zip(np.asarray(T_arr), np.asarray(P_arr)):
        q = query_properties(float(T), float(P), fluid=fluid)
        out['rho'].append(q['rho'])
        out['mu'].append(q['mu'])
        out['cp'].append(q['cp'])
        out['k_th'].append(q['k_th'])
    for k in out:
        out[k] = np.asarray(out[k], dtype=np.float64)
    return out

# Residual helper placeholders (suitable for integration with PINN)
def momentum_residual_placeholder(u, v, w, p, rho, mu):
    """Return zero residual placeholders shaped like inputs. Replace with autograd-enabled versions in PINN code."""
    return np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)

def energy_residual_placeholder(T, u, v, w, rho, cp, k_th):
    return np.zeros_like(T)
