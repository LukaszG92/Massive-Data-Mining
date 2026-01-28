import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.special import erf


def calculate_p(w, c=1.0):
    # The formula is undefined for w=0; we only consider w > 0.
    if w <= 0:
        return np.nan

    u = w / c

    # The expression 1 - 2*normaldist(0,1).cdf(-u) is equivalent to erf(u/sqrt(2)).
    # Using erf can be more direct and numerically stable.
    term1 = erf(u / np.sqrt(2))

    # This is the second term in the formula for p.
    term2 = (np.sqrt(2 / np.pi) / u) * (1 - np.exp(-u**2 / 2))
    
    p_val = term1 - term2

    # The value of p must be in the interval (0, 1) for the logs in T_comp to be valid.
    if not (0 < p_val < 1):
        return np.nan # Return NaN for invalid probability values

    return p_val


def objective_function(w, c, n):
    p1 = calculate_p(w, c=1.0)
    p2 = calculate_p(w, c=c)

    # If w leads to invalid p values, return infinity to guide the optimizer away.
    if np.isnan(p1) or np.isnan(p2):
        return np.inf

    log_inv_p1 = np.log(1 / p1)
    log_inv_p2 = np.log(1 / p2)

    # If p2 is close to 1, log(1/p2) is close to 0, causing division by zero.
    if np.isclose(log_inv_p2, 0):
        return np.inf

    rho = log_inv_p1 / log_inv_p2
    
    # The formula for T_comp as given in the image.
    T_comp = n**rho * (np.log(n) / log_inv_p2)
    
    return T_comp


def find_optimal_w(c, n, verbose=False):
    # if verbose:
    #     print(f"Finding optimal w for c = {c} and n = {n:.2e}...")
    
    # Use a lambda function to pass the fixed parameters c and n to the objective function.
    # The solver will only vary w.
    cost_function = lambda w: objective_function(w, c, n)
    
    # Use a large upper bound instead of None to avoid type issues
    result = minimize_scalar(
        cost_function,
        bounds=(1, 1000),  # Use large upper bound instead of None
        method='bounded'
    )

    if result.success:
        optimal_w = result.x
        # print(f"✅ Optimization Successful!")
        # print(f"  Optimal w:      {optimal_w:.6f}")
        # print(f"  Minimum T_comp: {result.fun:.6e}")
        
        # Display intermediate values at the optimal w for verification
        p1_opt = calculate_p(optimal_w, c=1.0)
        p2_opt = calculate_p(optimal_w, c=c)
        rho_opt = np.log(1/p1_opt) / np.log(1/p2_opt)
        
        # print(f"  Values at optimal w:")
        # print(f"    p₁ = {p1_opt:.6f}")
        # print(f"    p₂ = {p2_opt:.6f}")
        # print(f"    ρ  = {rho_opt:.6f}")
    
        return optimal_w
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")


def get_optimal_w(c, n, verbose=False):
    return find_optimal_w(c, n, verbose=verbose)


# Cache for computed w values to avoid repeated optimization
_w_cache = {}

def get_cached_optimal_w(c, n, verbose=False):
    cache_key = (c, n)
    
    if cache_key in _w_cache:
        if verbose:
            print(f"Using cached optimal w = {_w_cache[cache_key]:.6f} for c={c}, n={n}")
        return _w_cache[cache_key]
    
    optimal_w = find_optimal_w(c, n, verbose=verbose)
    _w_cache[cache_key] = optimal_w
    
    return optimal_w
