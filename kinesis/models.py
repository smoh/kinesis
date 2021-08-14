import os
from abc import ABC, abstractmethod
import pathlib
import logging
import pickle
import numpy as np
import pystan
import arviz as az

from kinesis.analysis import decompose_T

logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.dirname(__file__))
STANDIR = os.path.join(ROOT, "stan")


__all__ = ["get_model", "Basic", "AllCombined"]

available = [
    os.path.basename(str(path)).split(".")[0]
    for path in pathlib.Path(STANDIR).glob("*.stan")
]


def model_path(model_name):
    return os.path.join(ROOT, "stan", model_name + ".stan")


def model_cache_path(model_name):
    return os.path.join(ROOT, "stan", model_name + ".pkl")


def get_model(model_name, recompile=False):
    """Get compiled StanModel
    This will compile the stan model if a cached pickle does not exist.

    Args:
        model_name (str): model name without `.stan`
        recompile (bool): Force force recompilation.
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


class KinesisModelBase(ABC):
    pars_to_query = None
    required_columns = None
    model_name = None
    _additional_data_required = None

    def __init__(self, recompile=False):
        """Load/Compile StanModel to memory.

        Args:
            recompile (bool): True to force recompilation of the stan model.

        Attributes:
            include_T :(bool): True if linear velocity gradient is included in the model.
            model (StanModel): compiled stan model
        """
        self.model = get_model(self.model_name, recompile=recompile)

    def validate_dataframe(self, df):
        """Validate that the dataframe has required columns

        Args:
            df (DataFrame): data

        Raises:
            ValueError: if any column is missing.
        """
        for col in self.required_columns:
            if col not in df:
                raise ValueError(f"Dataframe is missing {col}")

    @abstractmethod
    def _prepare_standata(self, df):
        pass

    @abstractmethod
    def _default_init(self, df):
        pass

    def fit(self, df, sample=True, **kwargs):

        if "data" in kwargs:
            raise ValueError("`data` should be specified as pandas.DataFrame.")
        self.validate_dataframe(df)
        data = self._prepare_standata(df)
        if "b0" in kwargs:
            data["b0"] = kwargs.pop("b0")

        init = kwargs.pop("init", self._default_init(data))

        if sample:
            pars = kwargs.pop("pars", self.pars_to_query)
            stanfit = self.model.sampling(data=data, init=init, pars=pars, **kwargs)
            return stanfit
        else:
            return self.model.optimizing(data=data, init=init, **kwargs)


class AllCombined(KinesisModelBase):
    pars_to_query = [
        "v0",
        "sigv",
        "Omega",
        "f_mem",
        "v0_bg",
        "sigv_bg",
        "T_param",
        "a_model",
        "rv_model",
        "Omega",
        "probmem",
    ]
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
        "radial_velocity",
        "radial_velocity_error",
    ]
    model_name = "allcombined"

    def _prepare_standata(self, df):

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
            include_T=1,
        )

        return data

    def _default_init(self, data):
        # data is stan data
        def init_func():
            return dict(
                d=1e3 / data["a"][:, 0],
                # sigv=np.random.normal(size=3),
                v0=np.random.normal(scale=50, size=3),
                T=np.zeros(shape=(data["include_T"], 3, 3)),
            )

        return init_func


class Basic(KinesisModelBase):
    pars_to_query = ["d", "v0", "sigv", "a_model", "rv_model"]
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
        # "radial_velocity",
        # "radial_velocity_error",
    ]
    model_name = "general_model"

    def _prepare_standata(self, df):

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
            include_T=0,
        )

        return data

    def _default_init(self, df):
        def init_func():
            return dict(
                d=1e3 / df["parallax"].values,
                sigv=1.5,
                v0=np.random.normal(scale=50, size=3),
                T=np.zeros(shape=(0, 3, 3)),
            )

        return init_func