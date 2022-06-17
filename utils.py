import warnings


class AsymtoteException(Exception):
    pass


class ConvergenceWarning(Warning):
    def __init__(self, msg):
        self.msg = msg
        
    def __str__(self):
        return repr(self.msg)
    

class IncompleteSolution(Warning):
    def __init__(self, msg):
        self.msg = msg
        
    def __str__(self):
        return repr(self.msg)
    

def convergencewarning(eps, max_iter, x, f):
    # throwing ConvergenceWarning.
    msg = (
        f'Could not converge within error margin `eps={eps}` '
        f'for maximum number of iterations `max_iter={max_iter}` '
        f'[x={x:.2e}, error={f(x):.2e}]'
    )
    
    warnings.warn(msg, ConvergenceWarning, stacklevel=2)
    
def incompletesolution(x0, x_max, i, n):
    # throwing IncompleteSolution warning.
    msg = (
        f'Could not find all solutions within interval `x0={x0:.3e}` '
        f'to `x_max{x_max:.3e}`. '
        f'[{i}/{n} solutions found]'
    )
    
    warnings.warn(msg, IncompleteSolution, stacklevel=2)


def bisect(f, a, b, eps=1e-6, max_iter=100):
    ''' Root finding, bisection method.
    
    Args:
        f (function): Function to evaluate.
        a (float): Lower interval boundary.
        b (float): Upper interval boundary.
        eps (float): Error margin.
        max_iter (int): Maximum no. iterations.
    Returns:
        m (float): Found root.
    Warns:
        ConvergenceWarning: If can't converge within error margin `eps` for maximum number of iterations `max_iter`.
        
    '''
    for _ in range(max_iter):
        # interval midpoint `m`
        m = (a + b)/2
        
        # if root found within error margin `eps` return `m`
        if abs(f(m)) < eps:
            return m
        # else update interval
        else:
            # root between `a` and `m` (checking by sign change)
            if f(a)*f(m) < 0:
                b = m
            # otherwise root is between `m` and `b`
            else:
                a = m

    # check if we hit an asymtote if we can't converge
    if -f(a)*f(b) > 1/eps:
        raise AsymtoteException(f'Asymtote at {m:.3e}. f(a)={f(a):.3e} and f(b)={f(b):.3e} for [a, b]=[{a:.3e},{b:.3e}]')
    # otherwise throw warning if can't find solution wihtin maximum no. iterations
    else:                
        convergencewarning(eps, max_iter, m, f)
        return m


def newton_raphson(x0, f, df, eps=1e-6, max_iter=100):
    ''' Root finding, Newton Raphson
    
    Args:
        x0 (float): Initial guess.
        f (function): Function to evaluate.
        df (function): Differential of function to evaluate with respect to x.
        eps (float): Error margin.
        max_iter (int): Maximum no. iterations.
    Returns:
        x0 (float): Found root.
    Warns:
        ConvergenceWarning: If can't converge within error margin `eps` for maximum number of iterations `max_iter`.
        
    '''
    for _ in range(max_iter):
        # update x0
        x0 = x0 - f(x0)/df(x0)
        
        # if root found within error margin `eps` return `x0`
        if abs(f(x0)) < eps:
            return x0
        
    # throw warning if can't find solution wihtin maximum no. iterations
    convergencewarning(eps, max_iter, x0, f)
    return x0


def rootwalk(f, a, b, N=1000, n=1, eps=1e-6, max_iter=100):
    ''' Walking along function to find roots using bisect method.
    
    Args:
        f (function): Function to evaluate.
        a (float): Lower boundary.
        b (float): Upper boundary.
        N (float): No. points in interval.
        n (int): No. roots to find.
        eps (float): Error margin used in bisection method.
        max_iter (int): Maximum no. iterations used in bisection method.
    Returns:
        r (list): List of found roots.
    Warns:
        ConvergenceWarning: During the bisection method,
            if can't converge within error margin `eps` for maximum number of iterations `max_iter`.
        IncompleteSolution: If can't find `n` specified roots in domain `[a,b]`.
    '''
    dx = (b - a)/N
    x = a
       
    i = 0
    r = [float('nan')] * n
        
    for _ in range(N):
        try:
            if f(x)*f(x + dx) < 0:
                r[i] = (bisect(f, x, x + dx, eps, max_iter))
                i += 1
        except AsymtoteException: 
            pass # pass asymtotes, continue
        
        if i == n:
            return r
        
        x += dx
    
    incompletesolution(a, b, i, n)
    return r