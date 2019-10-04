// NOTE: this model should be equivalent to anisotropic_rv.stan,
// but I tried the more straightforward way of having latent individual
// velocities `vi`. 
// The model as is does not work because `vi`, whether it is in `transformed_parameter`
// or in `model` block, is never initialized, and although the model will compile,
// it will error to having 'nan' in all initialization attempts.
// TODO: I think the way is to put them in `paramters` block and initialize explicitly,
// but moving on for now as the other implementation works.
data {
  int<lower=2> N;     // number of stars
  int<lower=0> Nrv;     // number of stars with RVs

  // Units:
  //  ra [deg]
  //  dec [deg]
  //  parallax [mas]
  //  pmra [mas/yr]
  //  pmdec [mas/yr]
  real ra[N];
  real dec[N];
  vector[3] a[N];        // parallax, pmra, pmdec, rv
  cov_matrix[3] C[N];    // covariance matrix

  int<lower=0, upper=N-1> irv[Nrv];          // zero-based index of RV stars
  real rv[Nrv];          // radial velocity [km/s]
  real rv_error[Nrv];    // radial velocity error [km/s]
}

transformed data {
  matrix[3,3] M[N];      // to equitorial rectangular coordinates
  real ra_rad[N];
  real dec_rad[N];
  for (i in 1:N) {
    ra_rad[i] = ra[i] * pi() / 180.;
    dec_rad[i] = dec[i] * pi() / 180.;
  }

  for(i in 1:N) {
    M[i,1,1] = -sin(ra_rad[i]);
    M[i,1,2] =  cos(ra_rad[i]);
    M[i,1,3] = 0.;
    M[i,2,1] = -sin(dec_rad[i])*cos(ra_rad[i]);
    M[i,2,2] = -sin(dec_rad[i])*sin(ra_rad[i]);
    M[i,2,3] = cos(dec_rad[i]);
    M[i,3,1] = cos(dec_rad[i])*cos(ra_rad[i]);
    M[i,3,2] = cos(dec_rad[i])*sin(ra_rad[i]);
    M[i,3,3] = sin(dec_rad[i]);
  }
}

parameters {
  vector<lower=0>[N] d;     // true distances [pc]
  vector[3] v0;             // mean velocity  [km/s]
  vector<lower=0>[3] sigv;             // dispersion     [km/s]
  corr_matrix[3] Omega;                // correlation
}

transformed parameters {
  matrix[3,3] Sigma;
  vector[3] vi[N];  // individaul velocities

  vector[3] a_model[N];
  real rv_model[Nrv];

  Sigma = quad_form_diag(Omega, sigv);

  for(i in 1:N) {
    a_model[i][1] = 1000./d[i];
    a_model[i][2] = M[i,1] * vi[i] / (d[i]/1000.) / 4.74;
    a_model[i][3] = M[i,2] * vi[i] / (d[i]/1000.) / 4.74;
  }

  if (Nrv > 0) {
    for(i in 1:Nrv) {
      rv_model[i] = M[irv[i]+1, 3] * vi[irv[i]+1];
    }
  }
}

model {
  // priors
  v0 ~ normal(0, 100);
  sigv ~ cauchy(0, 2.5);
  Omega ~ lkj_corr(2);
  print(v0);
  print(sigv);
  print(Omega);

  for(i in 1:N) {
    vi[i] ~ multi_normal(v0, Sigma);
  }

  // likelihood
  for(i in 1:N) {
    a[i] ~ multi_normal(a_model[i], C[i]);
  }

  if(Nrv > 0)
    for(i in 1:Nrv) {
      rv[i] ~ normal(rv_model[i], rv_error[i]);
    }
}
