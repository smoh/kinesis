"""A closer look at the Hyades residual velocities"""

# To ignore all the votable warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import palettable

from astropy.table import Table
import astropy.units as u
import astropy.coordinates as coord
from astroquery.gaia import Gaia

import kinesis


plt.rc('figure', dpi=150)



# %% Data
reino = pd.read_csv('data/Reino2018_supp.txt', comment='#',
                    delim_whitespace=True,
                    dtype={'HIP':str, 'TYC':str, 'source_id':str})
print(len(reino), "rows")
reino = reino.reset_index()  # keep row index
# Query full tgas table for tgas sources only
reino_tgas = reino.loc[reino.source_id.notnull()]

@kinesis.cache_to("data/reino_tgas_full.csv")
def download_reino_tgas_full():
    r = Gaia.launch_job("""
select * from TAP_UPLOAD.reino
left join gaiadr1.tgas_source tgas
  on tgas.source_id = reino.source_id""",
        upload_resource=Table.from_pandas(reino_tgas[['index','source_id']].astype(int)),
        upload_table_name='reino').get_results()
    return r.to_pandas()
hy_tgas = download_reino_tgas_full()
# full DR2 table for Hyades stars from DR2 van Leeuwen 2018
hy_dr2 = pd.read_csv("/Users/semyeong/projects/spelunky/oh17-dr2/dr2_vL_clusters_full.csv")\
    .groupby('cluster').get_group('Hyades')
hy_tgas = hy_tgas.pipe(kinesis.add_vtan).pipe(kinesis.add_vtan_errors)
hy_dr2 = hy_dr2.pipe(kinesis.add_vtan).pipe(kinesis.add_vtan_errors)

hy_tgas_coords = kinesis.make_icrs(hy_tgas, include_pm_rv=False)
hy_dr2_coords = kinesis.make_icrs(hy_dr2)

# %% Plot tangential velocity errors
plt.figure()
plt.axes(aspect='equal')
plt.scatter(hy_dr2.vra_error, hy_dr2.vdec_error, s=4, label='DR2')
plt.scatter(hy_tgas.vra_error, hy_tgas.vdec_error, s=4, label='TGAS')
plt.loglog();
plt.xlabel(r"$\sigma(v_\alpha)$ [km/s]")
plt.ylabel(r"$\sigma(v_\delta)$ [km/s]");
plt.axvline(0.3, c='k', lw=1)
plt.axhline(0.3, c='k', lw=1);
plt.legend(loc='upper left');
plt.savefig('plots/hyades/tangential_velocity_errors.pdf')

def project_mean_velocity(vmean, position):
    """
    Project vmean at `position` and get predicted proper motions

    vmean : quantity (v_x, v_y, v_z)
        in cartesian equatorial coordinates, km/s
    position :

    Returns (pmra, pmdec) in units of mas/yr
    """
    if isinstance(position, pd.DataFrame):
        icrs = kinesis.make_icrs(position, include_pm_rv=False)
    cartesian_coords = coord.ICRS(
        icrs.cartesian.xyz,
        v_x=[vmean[0]]*len(icrs),
        v_y=[vmean[1]]*len(icrs),
        v_z=[vmean[2]]*len(icrs),
        representation_type='cartesian',
        differential_type='cartesian')
    pmra = cartesian_coords.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value
    pmdec = cartesian_coords.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value
    rv = cartesian_coords.spherical.differentials['s'].d_distance.to(u.km/u.s).value
    return pmra, pmdec, rv


# Cluster parameters from Reino et al. 2018
vmean = [-5.96, 45.60, 5.57] * u.km/u.s   # cartesian equatorial
sigmav = 0.3 * u.km/u.s
# vmean = [-6.14, 45.740104, 5.510726] * u.km/u.s
# vmean = [5.96, 35.60, -2.32] * u.km/u.s   # cartesian equatorial
pmra0_dr2, pmdec0_dr2, rv0_dr2 = project_mean_velocity(vmean, hy_dr2)
pmra0_tgas, pmdec0_tgas, rv0_tgas = project_mean_velocity(vmean, hy_tgas)
# Add residual proper motions subtracting bulk motion
hy_tgas['dpmra'], hy_tgas['dpmdec'] = hy_tgas.pmra - pmra0_tgas, hy_tgas.pmdec - pmdec0_tgas
hy_dr2['dpmra'], hy_dr2['dpmdec'] = hy_dr2.pmra - pmra0_dr2, hy_dr2.pmdec - pmdec0_dr2

# Make a mock hyades at the locations of observed DR2 hyades stars
mockhy = kinesis.Cluster(vmean.value, sigmav.value,
                         omegas=[0,0,0],
                         ws=[0,0,0,0,0],
                         b0=[17.2, 41.6, 13.6])
mockhy_members = mockhy.sample_at(kinesis.make_icrs(hy_dr2, coord.ICRS))
mockhy_members_pmra = mockhy_members.spherical.differentials['s'].d_lon.to(u.mas/u.yr).value
mockhy_members_pmdec = mockhy_members.spherical.differentials['s'].d_lat.to(u.mas/u.yr).value
mockhy_members_dpmra = mockhy_members_pmra - pmra0_dr2
mockhy_members_dpmdec = mockhy_members_pmdec - pmdec0_dr2


# %% Plot residual velocities
fig, ax = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)
ax[0].errorbar(
    hy_dr2.dpmra/hy_dr2.parallax*4.74,
    hy_dr2.dpmdec/hy_dr2.parallax*4.74,
    # xerr=hy_dr2.vra_error, yerr=hy_dr2.vdec_error,
    ls='None', marker='.', ms=1, elinewidth=.5,
    label='DR2');
ax[0].errorbar(
    hy_tgas.dpmra/hy_tgas.parallax*4.74,
    hy_tgas.dpmdec/hy_tgas.parallax*4.74,
    xerr=hy_tgas.vra_error, yerr=hy_tgas.vdec_error,
    ls='None', marker='.', ms=1, elinewidth=.5,
    label='TGAS');
ax[1].scatter(mockhy_members_dpmra/hy_dr2.parallax*4.74,
              mockhy_members_dpmdec/hy_dr2.parallax*4.74, s=1)
ax[0].set_xlabel(r'$\Delta v_\alpha$ [km/s]')
ax[0].set_ylabel(r'$\Delta v_\delta$ [km/s]')
ax[1].set_xlabel(r'$\Delta v_\alpha$ [km/s]')
for cax in ax: cax.axvline(0, c='gray', lw=1);
for cax in ax: cax.axhline(0, c='gray', lw=1);
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_xlim([-5.5, 3.5])
ax[0].set_ylim([-4.5, 4.5])
ax[0].set_title("Data")
ax[1].set_title("Mock cluster with isotropic dispersion");
fig.savefig('plots/hyades/residual_tangential_velocity.pdf')

# %% Residual velocities vx cartesian coordinates
fig, ax = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
ax[0].errorbar(
    hy_dr2_coords.cartesian.x.value,
    hy_dr2.dpmra/hy_dr2.parallax*4.74,
    yerr=hy_dr2.vra_error,
    ls='None', marker='.', ms=1, elinewidth=.5);
ax[0].errorbar(
    hy_tgas_coords.cartesian.x.value,
    hy_tgas.dpmra/hy_tgas.parallax*4.74,
    # yerr=hy_tgas.vra_error,
    ls='None', marker='.', ms=1, elinewidth=.5);

ax[1].errorbar(
    hy_dr2_coords.cartesian.x.value,
    hy_dr2.dpmdec/hy_dr2.parallax*4.74,
    ls='None',
    yerr=hy_dr2.vdec_error, marker='.', ms=1, elinewidth=.5);
ax[1].errorbar(
    hy_tgas_coords.cartesian.x.value,
    hy_tgas.dpmdec/hy_tgas.parallax*4.74,
    ls='None',
    yerr=hy_tgas.vdec_error, marker='.', ms=1, elinewidth=.5);
plt.xlim(4,32);
plt.ylim(-5.5,5.5)
ax[0].axhline(0, lw=1, c='gray');
ax[1].axhline(0, lw=1, c='gray');
ax[0].set_xlabel("ICRS $X$")
ax[1].set_xlabel("ICRS $X$")
ax[0].set_title("R.A.")
ax[1].set_title("Decl.")
ax[0].set_ylabel(r'$\mu$ [mas/yr]');

# %% Residual velocities vx cartesian coordinates
fig, ax = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
ax[0].errorbar(
    hy_dr2_coords.cartesian.y.value,
    hy_dr2.dpmra/hy_dr2.parallax*4.74,
    yerr=hy_dr2.vra_error,
    ls='None', marker='.', ms=1, elinewidth=.5);
ax[0].errorbar(
    hy_tgas_coords.cartesian.y.value,
    hy_tgas.dpmra/hy_tgas.parallax*4.74,
    # yerr=hy_tgas.vra_error,
    ls='None', marker='.', ms=1, elinewidth=.5);

ax[1].errorbar(
    hy_dr2_coords.cartesian.y.value,
    hy_dr2.dpmdec/hy_dr2.parallax*4.74,
    ls='None',
    yerr=hy_dr2.vdec_error, marker='.', ms=1, elinewidth=.5);
ax[1].errorbar(
    hy_tgas_coords.cartesian.y.value,
    hy_tgas.dpmdec/hy_tgas.parallax*4.74,
    ls='None',
    yerr=hy_tgas.vdec_error, marker='.', ms=1, elinewidth=.5);
plt.xlim(18,62);
plt.ylim(-5.5,5.5)
ax[0].axhline(0, lw=1, c='gray');
ax[1].axhline(0, lw=1, c='gray');
ax[0].set_xlabel("ICRS $Y$")
ax[1].set_xlabel("ICRS $Y$")
ax[0].set_title("R.A.")
ax[1].set_title("Decl.")
ax[0].set_ylabel(r'$\mu$ [mas/yr]');

# %% Residual velocities vx cartesian coordinates
fig, ax = plt.subplots(1, 2, figsize=(10,4), sharex=True, sharey=True)
ax[0].errorbar(
    hy_dr2_coords.cartesian.z.value,
    hy_dr2.dpmra/hy_dr2.parallax*4.74,
    yerr=hy_dr2.vra_error,
    ls='None', marker='.', ms=1, elinewidth=.5);
ax[0].errorbar(
    hy_tgas_coords.cartesian.z.value,
    hy_tgas.dpmra/hy_tgas.parallax*4.74,
    # yerr=hy_tgas.vra_error,
    ls='None', marker='.', ms=1, elinewidth=.5);

ax[1].errorbar(
    hy_dr2_coords.cartesian.z.value,
    hy_dr2.dpmdec/hy_dr2.parallax*4.74,
    ls='None',
    yerr=hy_dr2.vdec_error, marker='.', ms=1, elinewidth=.5);
ax[1].errorbar(
    hy_tgas_coords.cartesian.z.value,
    hy_tgas.dpmdec/hy_tgas.parallax*4.74,
    ls='None',
    yerr=hy_tgas.vdec_error, marker='.', ms=1, elinewidth=.5);
# plt.xlim(18,62);
plt.ylim(-5.5,5.5)
ax[0].axhline(0, lw=1, c='gray');
ax[1].axhline(0, lw=1, c='gray');
ax[0].set_xlabel("ICRS $Y$")
ax[1].set_xlabel("ICRS $Y$")
ax[0].set_title("R.A.")
ax[1].set_title("Decl.")
ax[0].set_ylabel(r'$\mu$ [mas/yr]');


# %% DR2 RV subset
hy_dr2_rv = hy_dr2.loc[hy_dr2.radial_velocity_error<1]
hy_dr2_rv = kinesis.add_xv(hy_dr2_rv, coord.ICRS)
hy_dr2_rv[['dx','dy','dz']] = hy_dr2_rv[['x','y','z']] - hy_dr2_rv[['x','y','z']].mean()
hy_dr2_rv[['dvx','dvy','dvz']] = hy_dr2_rv[['vx','vy','vz']] - hy_dr2_rv[['vx','vy','vz']].mean()

fig, ax = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
for ipos, X in enumerate(['dx', 'dy', 'dz']):
    for ivel, V in enumerate(['dvx', 'dvy', 'dvz']):
        ax[2-ivel, ipos].scatter(hy_dr2_rv[X], hy_dr2_rv[V], s=4)
        ax[2-ivel, ipos].axhline(0, lw=1, c='gray')
        ax[2-ivel, ipos].axvline(0, lw=1, c='gray')
ax[0,0].set_ylim(-2,2)
ax[2,0].set_xlabel('x')
ax[2,1].set_xlabel('y')
ax[2,2].set_xlabel('z')
ax[0,0].set_ylabel('vz')
ax[1,0].set_ylabel('vy')
ax[2,0].set_ylabel('vz');
fig.savefig('plots/hyades/dx_dv.pdf')

def get_slope(x, y):
    A = np.vander(x, 2)
    ATA = np.dot(A.T, A/1.)
    mean_w = np.linalg.solve(ATA, np.dot(A.T, y))
    sigma_w = np.linalg.inv(ATA)
    return mean_w[0], np.sqrt(sigma_w[0,0])
