"""
Make latex table of fit summary using arviz and pandas to_latex method.
"""
import pandas as pd
import numpy as np
import arviz as az
import click

import kinesis as kn

kn.set_mpl_style()

df = kn.data.load_hyades_dataset()
b0 = np.array([17.15474298, 41.28962638, 13.69105771])

# map arviz summary index -> latex column name
_column_name_dict = {
    "f_mem": r"\fmem",
    "v0[0]": "$v_{0,x}$ (ICRS)",
    "v0[1]": "$v_{0,y}$ (ICRS)",
    "v0[2]": "$v_{0,z}$ (ICRS)",
    "sigv[0]": "$\sigma_{x}$ (ICRS)",
    "sigv[1]": "$\sigma_{y}$ (ICRS)",
    "sigv[2]": "$\sigma_{z}$ (ICRS)",
    "sigv_gal[0]": "$\sigma_{x}$",
    "sigv_gal[1]": "$\sigma_{y}$",
    "sigv_gal[2]": "$\sigma_{z}$",
    "Sigma_gal[0,0]": "$\Sigma_{xx}$",
    "Sigma_gal[0,1]": "$\Sigma_{xy}$",
    "Sigma_gal[0,2]": "$\Sigma_{xz}$",
    "Sigma_gal[1,1]": "$\Sigma_{yy}$",
    "Sigma_gal[1,2]": "$\Sigma_{yz}$",
    "Sigma_gal[2,2]": "$\Sigma_{zz}$",
    "Omega_gal[0,1]": "$\Omega_{xy}$",
    "Omega_gal[0,2]": "$\Omega_{xz}$",
    "Omega_gal[1,2]": "$\Omega_{yz}$",
    "w1_gal": "$w_1$",
    "w2_gal": "$w_2$",
    "w3_gal": "$w_3$",
    "w4_gal": "$w_4$",
    "w5_gal": "$w_5$",
    "kappa": "$\kappa$",
    "omegax_gal": "$\omega_x$",
    "omegay_gal": "$\omega_y$",
    "omegaz_gal": "$\omega_z$",
    "kappa_gal": "$\kappa$",
    "v0_bg[0]": "$v_{\rm{bg},x}$ (ICRS)",
    "v0_bg[1]": "$v_{\rm{bg},y}$ (ICRS)",
    "v0_bg[2]": "$v_{\rm{bg},z}$ (ICRS)",
    "sigv_bg": "$\sigma_{\rm bg}$",
}

_columns_to_remove = [
    "Sigma_gal[1,0]",
    "Sigma_gal[2,0]",
    "Sigma_gal[2,1]",
    # lower triangle of Omega
    "Omega[0,0]",
    "Omega[1,1]",
    "Omega[2,2]",
    "Omega[1,0]",
    "Omega[2,0]",
    "Omega[2,1]",
    "Omega_gal[0,0]",
    "Omega_gal[1,1]",
    "Omega_gal[2,2]",
    "Omega_gal[1,0]",
    "Omega_gal[2,0]",
    "Omega_gal[2,1]",
]


def make_summary_table(azfit):
    pars = [
        "f_mem",
        "v0",
        "sigv_gal",
        "Omega_gal",
        "omegax_gal",
        "omegay_gal",
        "omegaz_gal",
        "w1_gal",
        "w2_gal",
        "w3_gal",
        "w4_gal",
        "w5_gal",
        "kappa_gal",
        "v0_bg",
        "sigv_bg",
    ]

    azfit = kn.add_transformed_posterior(azfit)
    # To report everything else except v0/v0_bg in Galactic,
    # convert covariance to sigv and correlation matrix from Sigma_gal
    Sigma = azfit.posterior["Sigma_gal"].values
    diag = np.sqrt(Sigma[:, :, [0, 1, 2], [0, 1, 2]])
    # ij component is sigma_i * sigma_j
    denom = np.einsum("mnij,mnjk->mnik", diag[:, :, :, None], diag[:, :, None, :])
    Omega = Sigma / denom
    azfit.posterior["Omega_gal"] = (
        ("chain", "draw", "Omega_dim_0", "Omega_dim_1"),
        Omega,
    )
    azfit.posterior["sigv_gal"] = (("chain", "draw", "sigv_dim"), diag)

    # az.summary does not allow missing pars
    pars_for_az = []
    pars_missing = []
    for p in pars:
        if p in azfit.posterior:
            pars_for_az.append(p)
        else:
            pars_missing.append(p)
    summary_table = (
        az.summary(azfit, pars_for_az).drop(_columns_to_remove, errors="ignore")
        # .rename(index=_column_name_dict)
    )[["mean", "sd", "hpd_3%", "hpd_97%"]]
    return summary_table


@click.command()
@click.option(
    "--fit", "-f", type=(str, str), multiple=True, help="(key, path/to/pickle)"
)
@click.option("--latex", is_flag=True)
def main(fit, latex):
    frames = [make_summary_table(az.from_pystan(kn.load_stanfit(v))) for k, v in fit]
    keys = [k for k, v in fit]
    if latex:
        merged = (
            pd.concat(frames, keys=keys, axis=1, sort=True)
            .reindex(frames[0].index)
            .rename(
                index=_column_name_dict,
                columns={"hpd_3%": "hpd 3\%", "hpd_97%": "hpd 97\%"},
            )
        )
        output_file = "../report/fit-summary.tex"
        merged.to_latex(output_file, na_rep="", escape=False, multicolumn_format="c")
        click.echo("output written to {}".format(output_file))
    else:
        merged = pd.concat(frames, keys=keys, axis=1, sort=True).reindex(frames[0].index)
        click.echo(merged)


if __name__ == "__main__":
    main()
