data {
  int<lower=2> N;     // number of stars

  // Units:
  //  ra [deg]
  //  dec [deg]
  //  parallax [mas]
  //  pmra [mas/yr]
  //  pmdec [mas/yr]
  real ra[N];
  real dec[N];
  vector[4] a[N];        // parallax, pmra, pmdec, rv
  cov_matrix[4] C[N];    // covariance matrix
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
  vector[4] a_model[N];

  Sigma = quad_form_diag(Omega, sigv);

  for(i in 1:N) {
    a_model[i][1] = 1000./d[i];
    a_model[i][2] = M[i,1] * v0 / (d[i]/1000.) / 4.74;
    a_model[i][3] = M[i,2] * v0 / (d[i]/1000.) / 4.74;
    a_model[i][4] = M[i,3] * v0;
  }
}

model {
  matrix[4,4] D[N];       // modified covariance matrix

  // priors
  v0 ~ normal(0, 100);
  sigv ~ uniform(0., 50);

  for(i in 1:N) {
    D[i] = C[i];
  }
  for(i in 1:N) {
    D[i,2:4,2:4] = D[i,2:4,2:4]
      + M[i] * quad_form_diag(Omega, sigv) * transpose(M[i]) / (d[i]/1000.)^2 / 4.74^2;
  }

  for(i in 1:N) {
    a[i] ~ multi_normal(a_model[i], D[i]);
  }
}
