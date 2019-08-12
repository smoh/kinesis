// A model of cluster proper motions including linear velocity field
//
// NOTE: See https://dev.to/martinmodrak/optional-parametersdata-in-stan-4o33
//       for a trick to handle optional inputs in stan
// TODO: clean up and make note of units
// TODO: priors?
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

  // 1 if include T matrix
  int<lower=0, upper=1> include_T;
  // b0 is the reference position where v = v0
  // equatorial [x, y, z] in pc
  vector[3] b0;
  // b0 only used when include_T == 1.
  // vector[3] b0[include_T ? 0 : 1];     // reference position where v = v0

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

  matrix[3, 3] T[include_T ? 1 : 0];    // km/s/kpc = m/s/pc

  // real rv_offset;
  // real<lower=0> rv_extra_dispersion;
}

// transformed parameters are stored, too.
transformed parameters {
  vector[3] a_model[N];
  real rv_model[Nrv];
  vector[3] b[N];

  matrix[3, 3] T_param;
  if (include_T) {
    T_param = T[1];
  } else {
    T_param = rep_matrix(0., 3, 3);
  }
  // print(T_param);


  for (j in 1:N) {
    b[j][1] = d[j] * cos(dec_rad[j]) * cos(ra_rad[j]);
    b[j][2] = d[j] * cos(dec_rad[j]) * sin(ra_rad[j]);
    b[j][3] = d[j] * sin(dec_rad[j]);
  }

  for(i in 1:N) {
    a_model[i][1] = 1000./d[i];
    a_model[i][2] = M[i,1] * (v0 + T_param/1000.*(b[i] - b0)) / (d[i]/1000.) / 4.74;
    a_model[i][3] = M[i,2] * (v0 + T_param/1000.*(b[i] - b0)) / (d[i]/1000.) / 4.74;
  }

  if (Nrv > 0) {
    for(i in 1:Nrv) {
      rv_model[i] = M[irv[i]+1, 3] * (v0 + T_param/1000.*(b[irv[i]+1] - b0));
    }
  }
}

model {
  matrix[3,3] D[N];       // modified covariance matrix


  // v0 ~ normal(0, 100);
  // sigv ~ normal(0, 10);

  // likelihood
  for(i in 1:N) {
    D[i] = C[i];
    // D[i,2,2] += sigv^2 / (d[i]/1000.)^2 / 4.74^2;
    // D[i,3,3] += sigv^2 / (d[i]/1000.)^2 / 4.74^2;
    D[i,2,2] = D[i,2,2] + sigv^2 / (d[i]/1000.)^2 / 4.74^2;
    D[i,3,3] = D[i,3,3] + sigv^2 / (d[i]/1000.)^2 / 4.74^2;
    // print(i, D[i])
  }

  for(i in 1:N) {
    a[i] ~ multi_normal(a_model[i], D[i]);
  }

  if(Nrv > 0)
    for(i in 1:Nrv) {
      // rv[i] ~ normal(rv_model[i], sqrt(sigv^2 + (rv_error[i])^2 + rv_extra_dispersion^2));
      rv[i] ~ normal(rv_model[i], sqrt(sigv^2 + (rv_error[i])^2));
    }
}

// generated quantities {
//   vector[3] a_hat[N];
//   real rv_hat[Nrv];

//   for(i in 1:N) {
//     // matrix[3, 3] D_tmp;
//     // D_tmp = C[i];
//     // D_tmp[2,2] = D_tmp[2,2] + sigv^2 / (d[i]/1000.)^2 / 4.74^2;
//     // D_tmp[3,3] = D_tmp[3,3] + sigv^2 / (d[i]/1000.)^2 / 4.74^2;
//     a_hat[i] = multi_normal_rng(a_model[i], C[i]);
//   }
//   if (Nrv > 0) {
//     for(i in 1:Nrv) {
//       rv_hat[i] = normal_rng(rv_model[i], rv_error[i]);
//     }
//   }
// }