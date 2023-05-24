import taichi as ti

## @param x, y denotes the parameter to compare
#  @param tolerance denotes the tolerance of comparison
#  @detail: Returns if |x - y| < tolerance
@ti.func
def approx_equal(x, y, tolerance) -> bool:
    return ti.abs(x - y) < tolerance


@ti.func
def householder_vector_decomposition(x: ti.template()):
    x2 = x[1:]
    X2 = 0.0
    if ti.static(x.n > 1):
        X2 = x2.norm()

    alpha = x.norm()
    rho = -ti.math.sign(x[0]) * alpha
    v1 = x[0] - rho
    u2 = x2 / v1
    X2 = X2 / ti.abs(v1)
    tau = (1 + X2 * X2) / 2
    u = ti.Vector.one(dt=ti.f32, n=ti.static(u2.n + 1))
    return rho, u, tau

@ti.func
def householder_qr_decomposition(M: ti.template()):
    Q = ti.Matrix.identity(ti.f32, M.n)
    R = M

    for j in ti.static(range(M.m)):
        col = ti.Vector.zero(dt=ti.f32, n=ti.static(M.n - j))
        for k in ti.static(range(M.n - j)):
            col[k] = R[k + j, j]
        rho, u, tau = householder_vector_decomposition(col)
        Hk = ti.Matrix.identity(ti.f32, M.n)
        v = ti.Vector.zero(dt=ti.f32, n=ti.static(j + u.n))
        for k in ti.static(range(u.n)):
            v[k + j] = u[k]
        Hk -= (1.0 / tau) * (v.outer_product(v))
        R = Hk @ R
        Q = Q @ Hk

    return Q, R

@ti.func
def solve_qef(A: ti.template()):
    # SVD decomposition
    U, Sigma, V = ti.svd(A[:3, :3])
    # Truncate result
    for i in ti.static(range(3)):
        Sigma[i, i] = ti.math.sign(Sigma[i, i]) * ti.max(0.1, ti.abs(Sigma[i, i]))
    # Calculate pseudo-inverse matrix
    pseudoInv = V @ Sigma.inverse() @ U.transpose()
    b = A[:3, 3]
    # Return result
    res = pseudoInv @ b
    return res

