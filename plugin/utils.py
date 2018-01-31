import numpy as np
import properties
import types


class SensitivityFiniteDifference(properties.HasProperties):
    """
    A class for computing Finite Difference sensitivities

    .. code::python
        function = lambda x: x**3
        x = np.random.rand(10)
        sensitivity = SensitivityFiniteDifference(function=function)
        J = sensitivity(x)
    """

    perturbation = properties.Float(
        "perturbation on the model used for the finite difference sensitivity "
        "calculation. "
        ":code:`delta_m = np.maximum(perturbation * np.absolute(m)), min_perturbation)`",
        default=0.001,
        min=0.0
    )

    min_perturbation = properties.Float(
        "minimum perturbation on the model",
        default=1e-3,
        min=0.0
    )

    stencil = properties.StringChoice(
        "stencil used for the finite difference calculation: forward, "
        "backward, or centered",
        choices=['forward', 'centered', 'backward'],
        default='centered'
    )

    function = properties.Instance(
        "function to compute the finite difference sensitivity of",
        instance_class=types.FunctionType,
        required=True
    )

    def _coerce_delta_m(self, m, delta_m):
        """
        Some sanity checks on delta_m, if not set, create it based on defaults
        """

        if isinstance(delta_m, float):
            assert delta_m > 0, (
                "delta_m must be positive, not {}".format(delta_m)
            )

        delta_m = np.maximum(
            self.perturbation*np.max(np.absolute(m)), self.min_perturbation
        )

        return delta_m

    def forward_difference(self, m, delta_m=None, function_m=None):
        """compute the forward difference sensitivity"""
        J = []

        for i, entry in enumerate(m):
            mpos = m.copy()
            mneg = m.copy()

            # get the perturbed models
            mpos[i] = entry + delta_m

            # get solution at preterbed model
            pos = self.function(mpos)
            neg = function_m if function_m is not None else self.function(mneg)

            J.append((pos - neg) / (delta_m))

        return np.vstack(J).T

    def centered_difference(self, m, delta_m=None, function_m=None):
        """compute the centered difference sensitivity"""
        J = []

        for i, entry in enumerate(m):
            mpos = m.copy()
            mneg = m.copy()

            # get the perturbed models
            mpos[i] = entry + delta_m
            mneg[i] = entry - delta_m

            # get solution at preterbed model
            pos = self.function(mpos)
            neg = self.function(mneg)

            J.append((pos - neg) / (2.*delta_m))

        return np.vstack(J).T

    def backward_difference(self, m, delta_m=None, function_m=None):
        """compute the backward difference sensitivity"""
        J = []

        for i, entry in enumerate(m):
            mpos = m.copy()
            mneg = m.copy()

            # get the perturbed models
            mneg[i] = entry - delta_m

            # get solution at preterbed model
            pos = function_m if function_m is not None else self.function(mpos)
            neg = self.function(mneg)

            J.append((pos - neg) / (delta_m))

        return np.vstack(J).T

    def __call__(self, m, delta_m=None, function_m=None):

        delta_m = self._coerce_delta_m(m, delta_m)
        sensitivity_calc = getattr(self, '{}_difference'.format(self.stencil))
        return sensitivity_calc(m, delta_m, function_m)
