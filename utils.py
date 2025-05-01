from typing import Callable, Any, Tuple
import functools

import jax
import jax.numpy as jnp
from jax import flatten_util, tree_util

Pytree = Any

def flatten_output(afunc, unravel_first_arg):
    @functools.wraps(afunc)
    def _afunc(*args):
        x, *args = args
        return jax.flatten_util.ravel_pytree(afunc(unravel_first_arg(x), *args))[0]
    return _afunc

def conjugate_gradient(f : Callable, z_guess : jnp.ndarray) -> jnp.ndarray :
    # solves root finding problem : f(z) = 0 using conjugate gradient method. 
    # Every linear solve uses conjugate gradient method and therefore only uses hvp/vhp

    vjp_f = lambda tangents, primals : jax.vjp(f, primals)[-1](tangents)[0]
    
    def body_fun(val):
        dval, _ = jax.scipy.sparse.linalg.cg(functools.partial(vjp_f, primals = val), f(val))
        return val - dval
    
    def cond_fun(val):
        return jnp.linalg.norm(f(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    z, _ = jax.lax.scan(scan_fun, z_guess, xs = None, length = 20.)
    return z

def newton_method(f : Callable, z_guess : jnp.ndarray) -> jnp.ndarray :
    # solves root finding problem : f(z) = 0 using newtons method 
    # Every linear solve uses explicit hessian 

    # function is only reverse mode autodiff compatible
    grad_f = jax.jacfwd(f)
    
    def body_fun(val):
        dval = jnp.linalg.solve(grad_f(val), f(val))
        return val - dval
    
    def cond_fun(val):
        return jnp.linalg.norm(f(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    z, _ = jax.lax.scan(scan_fun, z_guess, xs = None, length = 20.)
    return z

def newton_method_inverse(f : Callable, z_guess : jnp.ndarray) -> Tuple[jnp.ndarray] :
    # Solves root finding problem : f(z) = 0 using newtons method 
    # SVD decomposition of the Hessian is calculated and returned for reuse in reverse-mode autodiff

    grad_f = jax.jacfwd(f)
    inverse = jnp.linalg.svd(grad_f(z_guess), full_matrices = False)    

    def body_fun(val):
        val, (u, s, vh) = val
        new_val = val - vh.T @ ((u.T @ f(val)) / s)
        inverse = jnp.linalg.svd(grad_f(new_val), full_matrices = False)
        return new_val, inverse
    
    def cond_fun(val):
        val, _ = val
        return jnp.linalg.norm(f(val)) > 1e-5
    
    def scan_fun(carry, xs):
        carry = jax.lax.cond(cond_fun(carry), body_fun, lambda carry : carry, carry)
        return carry, None

    (z, inverse), _ = jax.lax.scan(scan_fun, (z_guess, inverse), xs = None, length = 20.)
    return z, inverse


@functools.partial(jax.custom_vjp, nondiff_argnums = (0, 1))
def _root_finding_rev(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> jnp.ndarray :
    # Reverse mode auto diff compatible root finding problem with f : Rn -> Rn
    _f = lambda z : f(z, p)
    return solver(_f, z)

def _root_finding_rev_fwd(solver, f, z, p):
    z_star = _root_finding_rev(solver, f, z, p)
    return z_star, (z_star, p)

def _root_finding_rev_bwd(solver, f, res, gdot):
    
    z_star, p = res
    _, vjp_x = jax.vjp(lambda x : f(x, p), z_star)
    _, vjp_p = jax.vjp(lambda p : f(z_star, p), p)
    
    return None, *vjp_p(solver(lambda x : vjp_x(x)[0] + gdot, jnp.zeros_like(z_star)))

_root_finding_rev.defvjp(_root_finding_rev_fwd, _root_finding_rev_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums = (0, 1))
def _root_finding_reuse(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> jnp.ndarray :
    # Reverse mode auto diff compatible root finding problem
    _f = lambda z : f(z, p) 
    return solver(_f, z)

def _root_finding_reuse_fwd(solver, f, z, p):
    z_star, inverse = _root_finding_reuse(solver, f, z, p)
    return (z_star, inverse), (z_star, p, inverse)

def _root_finding_reuse_bwd(solver, f, res, gdot):
    
    gdot, _ = gdot
    z_star, p, (u, s, vh) = res
    _, vjp_x = jax.vjp(lambda x : f(x, p), z_star)
    _, vjp_p = jax.vjp(lambda p : f(z_star, p), p)
    
    return None, *vjp_p( - ((gdot @ vh.T) / s) @ u.T)

_root_finding_reuse.defvjp(_root_finding_reuse_fwd, _root_finding_reuse_bwd)

def root_finding_rev(f : Callable, z : Pytree, p : Pytree, reuse_inverse : bool = False) -> Pytree : 
    # Reverse-mode autodiff compatible root finding problem. 
    # Note that reusing inverse incorrectly predicts higher order derivatives (> 1). 
    # Use reuse_inverse = False for computing higher oder derivatives

    z_flat, unravel = flatten_util.ravel_pytree(z)
    _f = flatten_output(f, unravel_first_arg = unravel)
    
    z_opt = _root_finding_reuse(newton_method_inverse, _f, z_flat, p)[0] if reuse_inverse else _root_finding_rev(newton_method, _f, z_flat, p)
    return unravel(z_opt)


@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def _root_finding_fwd(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> Pytree :
    # Forward mode auto diff compatible root finding problem with f : Rn -> Rn
    return solver(lambda z : f(z, p), z)

@_root_finding_fwd.defjvp
def _root_finding_fwd_fwd(solver, f, primals, tangents):
    z, p = primals
    zdot, pdot = tangents
    
    zstar = _root_finding_fwd(solver, f, z, p)
    # tangents_out = solver(lambda v : jax.jvp(lambda z, p : f(z, p), (zstar, p), (v, pdot))[-1], jnp.zeros_like(zdot))
    tangents_out = jnp.linalg.solve(jax.jacfwd(f)(zstar, p), - jax.jvp(lambda p : f(zstar, p), (p, ), (pdot, ))[-1])
    return zstar, tangents_out


@functools.partial(jax.custom_jvp, nondiff_argnums = (0, 1))
def _root_finding_fwd_reuse(solver : Callable, f : Callable, z : jnp.ndarray, p : Pytree) -> Pytree :
    # Forward mode auto diff compatible root finding problem with f : Rn -> Rn
    return solver(lambda z : f(z, p), z)

@_root_finding_fwd_reuse.defjvp
def _root_finding_fwd_reuse_fwd(solver, f, primals, tangents):
    z, p = primals
    _, pdot = tangents
    
    solution = zstar, (u, s, vh) = _root_finding_fwd_reuse(solver, f, z, p)
    tangents_out = - vh.T @ ((u.T @ jax.jvp(lambda p : f(zstar, p), (p, ), (pdot, ))[-1]) / s)
    return solution, (tangents_out, tree_util.tree_map(jnp.zeros_like, (u, s, vh)))


def root_finding_fwd(f : Callable, z : Pytree, p : Pytree, reuse_inverse : bool = False) -> Pytree : 
    # Forward- and reverse-mode autodiff compatible root finding problem. 
    # Note that reusing inverse incorrectly predicts higher order derivatives (> 1).
    # Use reuse_inverse = False for computing higher oder derivatives

    z_flat, unravel = flatten_util.ravel_pytree(z)
    _f = flatten_output(f, unravel_first_arg = unravel)
    
    z_opt = _root_finding_fwd_reuse(newton_method_inverse, _f, z_flat, p)[0] if reuse_inverse else _root_finding_fwd(newton_method, _f, z_flat, p)
    return unravel(z_opt)

