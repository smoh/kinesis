data {
  int<lower=2> N;       // number of stars
  int<lower=0> Nrv;     // number of stars with RVs

  // Units:
  //  ra [deg]
  //  dec [deg]
  //  parallax [mas]
  //  pmra [mas/yr]
  //  pmdec [mas/yr]
  real ra[N];
  real dec[N];
  vector[3] a[N];        // parallax, pmra, pmdec
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
  real<lower=0> sigv;       // dispersion     [km/s]
}

transformed parameters {
  vector[3] a_model[N];
  real rv_model[Nrv];

  for(i in 1:N) {
    a_model[i][1] = 1000./d[i];
    a_model[i][2] = M[i,1] * v0 / (d[i]/1000.) / 4.74;
    a_model[i][3] = M[i,2] * v0 / (d[i]/1000.) / 4.74;
  }

  if (Nrv > 0)
    for(i in 1:Nrv) {
      rv_model[irv[i]+1] = M[irv[i]+1, 3] * v0;
    }
}

model {
  matrix[3,3] D[N];       // modified covariance matrix


  // v0 ~ normal(0, 100);
  // sigv ~ normal(0, 10);

  // likelihood
  for(i in 1:N) {
    D[i] = C[i];
    D[i,2,2] += sigv^2 / (d[i]/1000.)^2 / 4.74^2;
    D[i,3,3] += sigv^2 / (d[i]/1000.)^2 / 4.74^2;
    // print(i, D[i])
  }

  for(i in 1:N) {
    a[i] ~ multi_normal(a_model[i], D[i]);
  }

  if(Nrv > 0)
    for(i in 1:Nrv) {
      rv[irv[i]+1] ~ normal(rv_model[irv[i]+1], sqrt(sigv^2 + (rv_error[irv[i]+1])^2));
    }
}
