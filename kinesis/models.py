import os
import logging
import pickle
import numpy as np
import pystan

logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.dirname(__file__))

__all__ = ["get_model", "Fitter"]


def model_path(model_name):
    return os.path.join(ROOT, "stan", model_name + ".stan")


def model_cache_path(model_name):
    return os.path.join(ROOT, "stan", model_name + ".pkl")


def get_model(model_name, recompile=False):
    """ Get compiled StanModel
    This will compile the stan model if a cached pickle does not exist.

    model_name : str
        model name without `.stan`
    recompile : bool
        If True, force recompilation even when cached model exists.
    """
    model_file = model_path(model_name)
    cache_file = model_cache_path(model_name)
    if (not os.path.exists(cache_file)) or recompile:
        model = pystan.StanModel(file=model_file)
        logger.info("Compiling {:s}".format(model_name))
        with open(cache_file, "wb") as f:
            pickle.dump(model, f)
    else:
        logger.info("Reading model from disk")
        model = pickle.load(open(cache_file, "rb"))
    return model


class Fitter(object):
    """class to facilitate cluster fitting
    
    include_T : bool
        True to include linear velocity gradient in the model.
    recompile : bool
        True to force recompilation of the stan model.
    
    Attributes
    ----------
    include_T : bool
        True if linear velocity gradient is included in the model.
    model : pystan.StanModel
        compiled stan model
    """

    def __init__(self, include_T=True, recompile=False):
        self.include_T = include_T
        self.model = get_model("general_model", recompile=recompile)

        # default parameters to query
        self._pars = ["v0", "sigv", "a_model", "rv_model"]
        if include_T:
            self._pars += ["T_param"]

    def validate_dataframe(self, df):
        """Validate that the dataframe has required columns
        
        Parameters
        ----------
        df : pandas.DataFrame
            data
        
        Raises
        ------
        ValueError
            if any column is missing.
        """
        required_columns = [
            "ra",
            "dec",
            "parallax",
            "pmra",
            "pmdec",
            "parallax_error",
            "pmra_error",
            "pmdec_error",
            "parallax_pmra_corr",
            "parallax_pmdec_corr",
            "pmra_pmdec_corr",
        ]
        for col in required_columns:
            if col not in df:
                raise ValueError(f"Dataframe is missing {col}")
        if self.include_T:
            for col in ["radial_velocity", "radial_velocity_error"]:
                if col not in df:
                    raise ValueError(f"`include_T` is True but df is missing {col}")

    def fit(self, df, sample=True, b0=None, **kwargs):
        """Fit model to the given data
        
        Parameters
        ----------
        df : pandas.DataFrame
            astrometry + RV data with Gaia-like column names
        sample : bool, optional
            draw mcmc samples, by default True
            If False, this will do opimization.
        b0 : np.array
            [x, y, z] defining cluster center
            If include_T is True, b0 must be specified.
        
        Returns
        -------
        OrderedDict or pystan.StanFit4Model
            If sample is False, the return type is OrderedDict from pystan.StanModel.optimizing.
        """

        if "data" in kwargs:
            raise ValueError("`data` should be specified as pandas.DataFrame.")
        self.validate_dataframe(df)

        N = len(df)
        if ("radial_velocity" not in df) or (df["radial_velocity"].notna().sum() == 0):
            rv = np.empty(0, float)
            rv_error = np.empty(0, float)
            irv = np.empty(0, int)
            Nrv = 0
        else:
            irv = np.arange(N)[df["radial_velocity"].notna()]
            rv = df["radial_velocity"].values[irv]
            rv_error = df["radial_velocity_error"].values[irv]
            Nrv = len(irv)
        data = dict(
            N=len(df),
            ra=df["ra"].values,
            dec=df["dec"].values,
            a=df[["parallax", "pmra", "pmdec"]].values,
            C=df.g.make_cov(),
            rv=rv,
            rv_error=rv_error,
            irv=irv,
            Nrv=Nrv,
            include_T=int(self.include_T),
        )
        # b0_default = np.median(df.g.icrs.cartesian.xyz.value, axis=1)
        # b0 = kwargs.pop("b0", b0_default)
        # data["b0"] = b0
        if b0 is None:
            if self.include_T:
                raise ValueError("`b0` must be given if include_T=True")
            b0 = np.array([0.0, 0.0, 0.0])  # this does not matter
        data["b0"] = b0

        # TODO: init with MAP
        def stan_init():
            return dict(
                d=1e3 / df["parallax"].values,
                sigv=1.5,
                v0=np.random.normal(scale=50, size=3),
                T=np.zeros(shape=(int(self.include_T), 3, 3)),
            )

        if sample:
            pars = kwargs.pop("pars", self._pars)
            return self.model.sampling(data=data, init=stan_init, pars=pars, **kwargs)
        else:
            return self.model.optimizing(data=data, init=stan_init, **kwargs)
