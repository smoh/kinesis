// A model of proper motions and RVs of stars in a cluster
// including linear velocity field and anisotropic velocity dispersion.
// Contamination is taken into account by two-component mixture model.
//
// NOTE: See https://dev.to/martinmodrak/optional-parametersdata-in-stan-4o33
//       for a trick to handle optional inputs in stan
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
  vector<lower=0>[3] sigv;             // dispersion     [km/s]
  corr_matrix[3] Omega;                // correlation

  matrix[3, 3] T[include_T ? 1 : 0];    // km/s/kpc = m/s/pc

  // Contamination from the "background" population
  real<lower=0,upper=1> f_mem;  // mixing proportion for background
  vector[3] v0_bg;
  real<lower=0> sigv_bg;

  // real rv_offset;
  // real<lower=0> rv_extra_dispersion;
}

// transformed parameters are stored, too.
transformed parameters {
  vector[3] b[N];

  vector[3] a_model[N];
  real rv_model[Nrv];
  matrix[3,3] D[N];       // modified covariance matrix

  vector[3] a_bg[N];
  real rv_bg[Nrv];
  matrix[3,3] D_bg[N];

  matrix[3,3] Sigma;
  
  matrix[3,3] T_param;

  Sigma = quad_form_diag(Omega, sigv);

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

    a_bg[i][1] = 1000./d[i];
    a_bg[i][2] = M[i,1] * (v0_bg) / (d[i]/1000.) / 4.74;
    a_bg[i][3] = M[i,2] * (v0_bg) / (d[i]/1000.) / 4.74;
  }

  for(i in 1:N) {
    D[i] = C[i];
    D[i,2:3,2:3] = D[i,2:3,2:3] + M[i,1:2] * Sigma * transpose(M[i,1:2]) / (d[i]/1000.)^2 / 4.74^2;

    D_bg[i] = C[i];
    D_bg[i,2,2] +=  sigv_bg^2 / (d[i]/1000.)^2 / 4.74^2;
    D_bg[i,3,3] +=  sigv_bg^2 / (d[i]/1000.)^2 / 4.74^2;
  }

  if (Nrv > 0) {
    for(i in 1:Nrv) {
      rv_model[i] = M[irv[i]+1, 3] * (v0 + T_param/1000.*(b[irv[i]+1] - b0));
      rv_bg[i] = M[irv[i]+1, 3] * (v0_bg);
    }
  }
}

model {
  // temporary var to hold logP for RV part
  real lpsi[N];
  real lpsi_bg[N];
  for(i in 1:N) {
    lpsi[i] = 0.;
    lpsi_bg[i] = 0.;
  }
  //--------------------------------------------------------------------------
  // Priors
  v0 ~ normal(0, 50);
  sigv ~ cauchy(0, 2.5);
  Omega ~ lkj_corr(2);

  sigv_bg ~ normal(30, 20);
  v0_bg ~ normal(0, 50);

  f_mem ~ uniform(0, 1);

  //TODO: is this the correct way?
  for (i in 1:3) {
    for (j in 1:3) {
      T_param[i,j] ~ normal(0, 50);  # Note: in m/s/pc
    }
  }
  // end of Priors ------------------------------------------------------------

  if(Nrv > 0)
    for(i in 1:Nrv) {
      lpsi[irv[i]+1] += normal_lpdf(rv[i] | rv_model[i], sqrt(M[i,3] * Sigma * transpose(M[i,3]) + (rv_error[i])^2));
      lpsi_bg[irv[i]+1] += normal_lpdf(rv[i] | rv_bg[i], sqrt(sigv_bg^2 + (rv_error[i])^2));
    }
  for(i in 1:N) {
    target += log_mix(f_mem,
      multi_normal_lpdf(a[i] | a_model[i], D[i]) + lpsi[i],
      multi_normal_lpdf(a[i] | a_bg[i], D_bg[i]) + lpsi_bg[i]
      );
  }
}


generated quantities {
  // temporary var to hold logP for RV part
  real lpsi[N];
  real lpsi_bg[N];
  real probmem[N];
  real lps[2];
  for(i in 1:N) {
    lpsi[i] = 0.;
    lpsi_bg[i] = 0.;
  }

  if(Nrv > 0)
    for(i in 1:Nrv) {
      lpsi[irv[i]+1] += normal_lpdf(rv[i] | rv_model[i], sqrt(M[i,3] * Sigma * transpose(M[i,3]) + (rv_error[i])^2));
      lpsi_bg[irv[i]+1] += normal_lpdf(rv[i] | rv_bg[i], sqrt(sigv_bg^2 + (rv_error[i])^2));
    }
  for(i in 1:N) {
    lps[1] = log(f_mem) + multi_normal_lpdf(a[i] | a_model[i], D[i]) + lpsi[i];
    lps[2] = log(1-f_mem) + multi_normal_lpdf(a[i] | a_bg[i], D_bg[i]) + lpsi_bg[i];
    probmem[i] = exp(lps[1]-log_sum_exp(lps[1], lps[2]));
  }
}

// generated quantities {
//   vector[3] a_hat[N];
//   real log_lik_a[N];
//   real rv_hat[Nrv];
//   real log_lik_rv[Nrv];


//   for(i in 1:N) {
//     a_hat[i] = multi_normal_rng(a_model[i], D[i]);
//     log_lik_a[i] = multi_normal_lpdf(a[i] | a_model[i], D[i]);
//   }
//   if (Nrv > 0) {
//     for(i in 1:Nrv) {
//       rv_hat[i] = normal_rng(rv_model[i], sqrt(sigv^2 + (rv_error[i])^2));
//       log_lik_rv[i] = normal_lpdf(rv[i] | rv_model[i], sqrt(sigv^2 + (rv_error[i])^2));
//     }
//   }
// }
