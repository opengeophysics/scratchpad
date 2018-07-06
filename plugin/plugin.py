import numpy as np
import properties
from scipy.constants import mu_0

from SimPEG.Problem import BaseProblem
from SimPEG import Props, Survey, Mesh

from empymod import utils


class EmpymodProblem(BaseProblem):
    """
    Problem class for a simulation conducted with empymod
    """

    # Invertible properties
    rho, rho_map, rho_deriv = Props.Invertible(
        "Electrical Resistivity (Ohm m)"
    )

    # Properties
    src = properties.List(
        "list of source locations", prop=properties.Array(
            "source location", shape=("*",)
        )
    )

    rec = properties.List(
        "list of reciever locations", prop=properties.Array(
            "receiver location", shape=("*",)
        )
    )

    depth = properties.Array("depth of the layers")

    freq = properties.Array("frequencies of the source")

    msrc = properties.Bool("docstring here", default=False) #TODO: what is this? docstring?

    mrec = properties.Bool("docstring here", default=False) #TODO: what is this? docstring?

    strength = properties.Float("docstring here", default=0.)

    _fixed_params = [
        'src', 'rec', 'depth', 'freqtime', 'aniso', 'epermH', 'epermV',
        'mpermH', 'mpermV', 'msrc', 'mrec', 'srcpts', 'recpts', 'strength',
    ]

    deleteTheseOnModelUpdate = ['_Japprox']

    _empymod_settings = [ 'xdirect', 'ht', 'htarg', 'opt', 'loop', 'verb']

    def __init__(self, **kwargs):
        # assert mesh.dim == 1, "only 1D modelling supported"
        super(EmpymodProblem, self).__init__(mesh=Mesh.BaseMesh(), **kwargs)

    @properties.validator('src')
    def _validate_src(self, change):
        value = change['value']
        src, nsrc, nsrcz, srcdipole = empymod.utils.check_bipole(value, 'src')
        self._nsrc = nsrc
        self._nsrcz = nsrcz
        self._srcdipole = srcdipole
        change["value"] = value

    @property
    def nsrc(self):
        """Number of sources"""
        return getattr(self, '_nsrc', None)

    @property
    def nsrcz(self):
        """Number of unique z-locations of the sources"""
        return getattr(self, '_nsrcz', None)

    @property
    def srcdipole(self):
        """are there dipole sources present in the source list?"""
        return getattr(self, '_srcdipole', None)

    @property
    def isrc(self):
        """this is either 1 or nsrc"""
        return int(self.nsrc/self.nsrcz)

    @property
    def isrz(self):
        """this is either 1, nsrc, nrec, or nsrc*nrec"""
        return int(self.isrc*self.irec)

    @properties.validator('rec')
    def _validate_rec(self, change):
        value = change['value']
        rec, nrec, nrecz, recdipole = empymod.utils.check_bipole(value, 'rec')
        self._nrec = nrec
        self._nrecz = nrecz
        self._recdipole = recdipole
        change["value"] = value

    @property
    def nrec(self):
        """Number of receivers"""
        return getattr(self, '_nrec', None)

    @property
    def nrecz(self):
        """Number of unique z-locations of the receivers"""
        return getattr(self, '_nrecz', None)

    @property
    def recdipole(self):
        """are there dipole receivers present in the receiver list?"""
        return getattr(self, '_recdipole', None)

    @property
    def irec(self):
        """this is either 1 or nrec"""
        return int(self.nrec/self.nrecz)

    @property
    def zeta(self):
        return np.outer(2j*np.pi*self.freq, np.ones(self.depth.shape)*mu_0)

    # TODO: make the following settable?
    @property
    def zetaH(self):
        return self.zeta

    @property
    def zetaV(self):

        return self.zeta

    @property
    def isfullspace(self):
        return False

    @property
    def xdirect(self):
        """Direct field in the wavenumber domain"""
        return False

    @property
    def ht(self):
        """FHT (digital linear filter)"""
        return "fht"

    @property
    def htarg(self):
        """Default FHT arguments"""
        return (empymod.filters.key_201_2009(), None)

    @property
    def use_spline(self):
        """Lagged convolution"""
        return True

    @property
    def use_ne_eval(self):
        """Do not use `numexpr`"""
        return False

    @property
    def loop_freq(self):
        """If use_spline=True, loop_freq has to be True too"""
        return True

    @property
    def loop_off(self):
        return False

    @property
    def conv(self):
        return True

    @property
    def fixed_params(self):
        return dict(zip(key, getattr(self, key)) for key in self._fixed_params)

    @property
    def empymod_settings(self):
        return dict(
            zip(key, getattr(self, key)) for key in self._empymod_settings
        )

    # Forward Simulation
    def _calc_fm(self, rho):
        """
        compute data using empymod. The real and imaginary parts are separated
        so that we are always working with real values
        """

        # Calculate result
        out = empymod.bipole(res=rho, **self.fixed_params, **self.empymod_settings)
        # Ensure dimensionality, because empymod.dipole squeezes the output
        if len(self.freq) == 1:
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
