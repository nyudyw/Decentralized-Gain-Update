import numpy as np
from scipy.optimize import root


class DistFlow:
    def __init__(self, case):
        """Load the network used by the power-flow solver."""
        self.n_bus = len(case['bus'])
        self.n_branch = len(case['branch'])
        self.from_bus = case['branch'][:, 0].astype(int) - 1
        self.to_bus = case['branch'][:, 1].astype(int) - 1
        self.r = case['branch'][:, 2]
        self.x = case['branch'][:, 3]

    def runpf(self, p, q):
        """Solve one nonlinear DistFlow step and return column-vector results."""
        s_inj = (p + 1j*q).reshape(-1)

        def equations(x):
            """Return the DistFlow equation errors."""
            n = self.n_bus
            m = self.n_branch

            v = x[:n]
            S_real = x[n:n+m]
            S_imag = x[n+m:n+2*m]
            l = x[n+2*m:]
            S = S_real + 1j*S_imag

            residual = [v[0]-1]

            for j in range(1, n):
                out_branches = [i for i in range(m) if self.from_bus[i] == j]
                in_branches = [i for i in range(m) if self.to_bus[i] == j]

                sum_out = sum(S[k] for k in out_branches)
                sum_in = sum(S[i] - (self.r[i] + 1j*self.x[i])*l[i]
                            for i in in_branches)
                res = sum_out - sum_in - s_inj[j-1]
                residual.extend([res.real, res.imag])

            for i in range(m):
                j = self.from_bus[i]
                k = self.to_bus[i]
                z = self.r[i] + 1j*self.x[i]

                res_v = v[j] - v[k] - 2*np.real(np.conj(z)*S[i]) + abs(z)**2*l[i]
                res_l = v[j]*l[i] - abs(S[i])**2

                residual.extend([res_v, res_l])

            return residual

        n = self.n_bus
        m = self.n_branch
        x0 = np.ones(3*self.n_branch + self.n_bus)
        x0[n:] = 0

        solution = root(equations, x0, method='hybr')

        v = solution.x[:n]
        S = solution.x[n:n+m] + 1j*solution.x[n+m:n+2*m]
        l = solution.x[n+2*m:]

        return {
            'v': np.sqrt(v)[:, None],
            'S': S[:, None],
            'l': l[:, None],
        }
