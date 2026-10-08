import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        xi = np.linspace(0, 1, N + 1)
        return np.meshgrid(xi, xi, indexing="ij", sparse=sparse)


    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D /= (1/N)**2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
            """
        xij, yij = self.create_mesh(N)

        # Initial condition u(x,y,0)
        ue0 = self.ue(mx, my).subs(t, 0)
        f = sp.lambdify((x, y), ue0, "numpy")
        un = f(xij, yij)

        D = self.D2(N)
        lap_un = D @ un + un @ D.T

        # Construct U^{-1} numerically from the PDE and u_t(x,y,0)=0
        unm1 = un + 0.5 * self.c**2 * self.dt**2 * lap_un

        self.apply_bcs(un)
        self.apply_bcs(unm1)

        return np.array([un, unm1])

    @property
    def dt(self) -> float:
        """Return the time step"""
        dt = self.cfl * self.h / self.c
        return(dt)

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = u.shape[0] - 1
        h = 1 / N

        xij, yij = self.create_mesh(N)

        ue_t = self.ue(self.mx, self.my).subs(t, t0)
        f = sp.lambdify((x, y), ue_t, "numpy")
        ue_vals = f(xij, yij)

        return np.sqrt(h**2 * np.sum((u - ue_vals)**2))

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.h = 1 / N

        D = self.D2(N)

        U = self.initialize(N, mx, my)
        un = U[0]      # U^0
        unm1 = U[1]    # U^{-1}

        errors = []
        data = {}

        if store_data > 0:
            data[0] = un.copy()

        for n in range(Nt):
            lap_un = D @ un + un @ D.T

            unp1 = (
                2 * un
                - unm1
                + self.c**2 * self.dt**2 * lap_un
            )

            self.apply_bcs(unp1)

            # shift time levels
            unm1 = un
            un = unp1

            current_time = (n + 1) * self.dt

            if store_data == -1:
                errors.append(self.l2_error(un, current_time))

            elif store_data > 0 and (n + 1) % store_data == 0:
                data[n + 1] = un.copy()

        if store_data == -1:
            return self.h, np.array(errors)

        return data

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        h = 1 / N

        D = sparse.diags(
            [1, -2, 1],
            [-1, 0, 1],
            shape=(N + 1, N + 1),
            format="lil",
        )

        D[0, 1] = 2
        D[-1, -2] = 2

        return D / h**2

    def ue(self, mx: int, my: int) -> sp.Expr:
        raise NotImplementedError("The ue method is not implemented yet.")

    def apply_bcs(self, u: np.ndarray):
        raise NotImplementedError("The apply_bcs method is not implemented yet.")


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    raise NotImplementedError("The test_exact_wave2d function is not implemented yet.")

if __name__ == "__main__":
    test_convergence_wave2d()
    #test_convergence_wave2d_neumann()

    print("All tests passed!")

