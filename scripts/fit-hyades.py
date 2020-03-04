"""
Fit and save hyades cluster data
"""
import os
import pandas as pd
import numpy as np
import click

import kinesis as kn
import gapipes as gp

# hard-coded center value
b_c_icrs = np.array([17.15474298, 41.28962638, 13.69105771])


def fit_and_save(srcdf, outfile, v0=None):
    necessary_columns = [
        "ra",
        "dec",
        "phot_g_mean_mag",
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
    data = srcdf[
        necessary_columns + ["radial_velocity", "radial_velocity_error"]
    ].copy()
    b0 = b_c_icrs
    print(f"{len(data)} rows")
    print(f"b0 = {b0}")

    N = len(data)
    irv = np.arange(N)[data["radial_velocity"].notna()]
    rv = data["radial_velocity"].values[irv]
    rv_error = data["radial_velocity_error"].values[irv]
    data_dict = {
        "N": N,
        "Nrv": data["radial_velocity"].notna().sum(),
        "ra": data["ra"].values,
        "dec": data["dec"].values,
        "a": data[["parallax", "pmra", "pmdec"]].values,
        "C": data.g.make_cov(),
        "irv": irv,
        "rv": rv,
        "rv_error": rv_error,
        "include_T": 1,
        "b0": b0,
    }

    if v0 is None:

        def stan_init():
            return dict(
                d=1e3 / data["parallax"].values,
                sigv=[0.5, 0.5, 0.5],
                Omega=np.eye(3),
                v0=[-5, 45, 5],
                T=np.zeros(shape=(1, 3, 3)),
                v0_bg=[0, 0, 0],
                sigv_bg=50.0,
                f_mem=0.95,
            )

        stanmodel = kn.get_model("allcombined")
        fit = stanmodel.sampling(
            data=data_dict,
            init=stan_init,
            pars=[
                "v0",
                "sigv",
                "Omega",
                "T_param",
                "v0_bg",
                "sigv_bg",
                "f_mem",
                "probmem",
                "a_model",
                "rv_model",
            ],
        )
    else:
        data_dict["v0"] = v0

        def stan_init():
            return dict(
                d=1e3 / data["parallax"].values,
                sigv=[0.5, 0.5, 0.5],
                Omega=np.eye(3),
                T=np.zeros(shape=(1, 3, 3)),
                v0_bg=[0, 0, 0],
                sigv_bg=50.0,
                f_mem=0.95,
            )

        stanmodel = kn.get_model("allcombined-fixed-v0")
        fit = stanmodel.sampling(
            data=data_dict,
            init=stan_init,
            pars=[
                "sigv",
                "Omega",
                "T_param",
                "v0_bg",
                "sigv_bg",
                "f_mem",
                "probmem",
                "a_model",
                "rv_model",
            ],
        )
    kn.save_stanfit(fit, outfile)


def validate_output_path(ctx, param, path):
    """validate output path"""
    # make sure the directory part of path already exists
    output_directory = os.path.dirname(path)
    if output_directory == "":
        output_directory = "."
    if not os.path.exists(output_directory):
        raise click.BadParameter("{} does not exist".format(output_directory))

    # make sure the file does not exist
    if os.path.exists(path):
        raise click.BadParameter("{} already exists.".format(path))

    # make sure the file is .pickle
    ext = os.path.splitext(path)[1]
    if ext == "":
        path += ".pickle"
    elif ext != ".pickle":
        raise click.BadParameter("output_path must have .pickle extension")
    return path


def validate_rbin(ctx, param, value):
    # value will be empty tuple () if not given
    if value == ():
        return None
    if (value[0] >= 0) & (value[1] >= 0) & (value[0] < value[1]):
        return value
    else:
        raise click.BadParameter("r1, r2 must be positive and r1 < r2.")


@click.command()
@click.option(
    "--rbin",
    "-r",
    type=float,
    nargs=2,
    callback=validate_rbin,
    help="fit only stars within (r1, r2) from the cluster center",
)
@click.option(
    "--brightcorr",
    help="correct bright source (G<12) proper motion for frame rotation according to Lindegren formula",
    is_flag=True,
)
@click.option("--v0", help="fixed value of v0 = (vx, vy, vz)", type=float, nargs=3)
@click.argument(
    "output_path", type=click.Path(writable=True), callback=validate_output_path
)
def main(output_path, rbin, brightcorr=False, v0=None):
    """
    Fit and save sample pickle to OUTPUT_PATH.
    """
    df = pd.read_csv("../data/hyades_full.csv")

    df_fit = None
    if rbin is None:
        df_fit = df.copy()
    else:
        r1, r2 = rbin
        xyz = df.g.icrs.cartesian.xyz.value
        r_c = np.linalg.norm(xyz - b_c_icrs[:, None], axis=0)
        df_fit = df.loc[(r_c > r1) & (r_c < r2)].copy()
        if len(df_fit) < 100:
            click.echo(
                "Only {} stars at r={}; doing nothing".format(len(df_fit), (r1, r2))
            )
            return

    if brightcorr:
        click.echo("Applying proper motion correction to bright sources")
        df_fit = df_fit.g.correct_brightsource_pm()

    if v0 == ():
        v0 = None
    fit_and_save(df_fit, output_path, v0=v0)


if __name__ == "__main__":
    main()
