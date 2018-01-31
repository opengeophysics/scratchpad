from SimPEG.Problem import BaseProblem
from SimPEG import Props, Survey


class EmpymodProblem(BaseProblem):
    """
    Problem class for a simulation conducted with empymod
    """

    rho, rho_map, rho_deriv = Props.Invertible(
        "Electrical Resistivity (Ohm m)"
    )

    deleteTheseOnModelUpdate = ['_Japprox']

    # Stuff we don't touch at the moment and keep fixed
    _empymod_settings = []
    empymod_settings = {
        'xdirect': False,
        'ht': 'fht',
        'htarg': None,
        'opt': 'spline',
        'loop': False,
        'verb': 1
    }

    def __init__(self, mesh, **kwargs):
        assert mesh.dim == 1, "only 1D modelling supported"
        super(EmpymodProblem, self).__init__(mesh, **kwargs)

        # set air at infinity
        depth = self.mesh.gridN.copy()
        depth[0] = -np.inf

        # Set the empymod_fixedparams
        # Most of this was defined in point 1.2
        self.fixed_params = {
            'src': src,
            'rec': rec,
            'depth': depth,
            'freqtime': freq,
            'aniso': None,
            'epermH': None,
            'epermV': None,
            'mpermH': None,
            'mpermV': None,
            'msrc': False,
            'mrec': False,
            'srcpts': 1,
            'recpts': 1,
            'strength': 0,
        }

    def _calc_fm(self, rho):
        """
        compute data using empymod. The real and imaginary parts are separated so that
        we are always working with real values
        """

        # Calculate result
        # out = empymod.bipole(res=rho, **self.fixed_params, **self.empymod_settings)
        # Ensure dimensionality, because empymod.dipole squeezes the output
        if len(freq) == 1:
            out = out[None, :]
        out = np.ravel(out, order='F')

        return np.hstack([out.real, out.imag])

    def fields(self, m):
        """
        Computes the fields
        """

        # set the model (this performs the mappings)
        self.model = m
        return self._calc_fm(self.rho)

    def Japprox(self, m, perturbation=0.1, min_perturbation=1e-3):
        """
        Approximate sensitivity computed using a finite difference approach
        """
        if getattr(self, '_Japprox', None) is None:
            self.model = m
            delta_m = min_perturbation # np.max([perturbation*m.mean(), min_perturbation])

            J = []

            for i, entry in enumerate(m):
                mpos = m.copy()
                mpos[i] = entry + delta_m

                mneg = m.copy()
                mneg[i] = entry - delta_m

                pos = self._calc_fm(self.rho_map * mpos)
                neg = self._calc_fm(self.rho_map * mneg)
                J.append((pos - neg) / (2.*delta_m))

            self._Japprox = np.vstack(J).T

        return self._Japprox

    def Jvec(self, m, v, f=None):
        """
        Sensitivity times a vector
        """
        self.model = m
        return self.Japprox(m).dot(v)

    def Jtvec(self, m, v, f=None):
        """
        Adjoint Sensitivity times a vector
        """
        self.model = m
        return self.Japprox(m).T.dot(v)


class EmpymodSurvey(Survey.BaseSurvey):
    """
    Survey class for a simulation conducted with empymod
    """

    @property
    def nD(self):
        # this will likely need to be generalized
        nsrc = len(self.prob.fixed_params['src'][0])
        nrec = len(self.prob.fixed_params['rec'][0])
        nfreq = len(self.prob.fixed_params['freqtime'])
        return nsrc * nrec * nfreq * 2

    def eval(self, f):
        return f

def re_field(inp):
    inp = inp.reshape(2, -1)
    inp = inp[0, :] + 1j*inp[1, :]
    return inp.reshape((-1, len(x), len(sx)), order='F')
