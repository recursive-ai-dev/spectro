/*
 * ssgs_native.c - rudimentary, compiled numeric kernels for SSGS.
 *
 * Implements the C ABI documented in ssgs_core.py:
 *   ssgs_lpc_levinson, ssgs_gaussian_logprobs,
 *   ssgs_forward_backward, ssgs_decode
 *
 * Flat row-major float64 arrays, no allocation inside hot loops.
 * Build: gcc -shared -fPIC -O3 -o libssgs_native.so ssgs_native.c -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double logsumexp(const double *x, int n) {
    if (n <= 0) return -1e300;
    double m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    if (isinf(m)) return m;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp(x[i] - m);
    return m + log(s);
}

/* ---- LPC via Levinson-Durbin (mirrors ssgs_core.lpc_coefficients) ---- */
void ssgs_lpc_levinson(const double *r, int p, double *a) {
    for (int i = 0; i <= p; i++) a[i] = 0.0;
    a[0] = 1.0;
    if (r[0] <= 1e-8 || p <= 0) return;

    double *upd = (double *)malloc((p + 1) * sizeof(double));
    if (!upd) return;

    double err = r[0];
    for (int i = 1; i <= p; i++) {
        double acc = r[i];
        for (int j = 1; j < i; j++) acc += a[j] * r[i - j];
        double k = -acc / err;
        upd[0] = 1.0;
        for (int j = 1; j < i; j++) upd[j] = a[j] + k * a[i - j];
        upd[i] = k;
        for (int j = 0; j <= i; j++) a[j] = upd[j];
        err *= (1.0 - k * k);
        if (err <= 1e-12) break;
    }
    free(upd);
}

/* ---- Gaussian emission: hand Cholesky + forward solve ---- */
static int chol_lower(const double *A, int p, double *L) {
    for (int i = 0; i < p * p; i++) L[i] = 0.0;
    for (int i = 0; i < p; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i * p + j];
            for (int k = 0; k < j; k++) s -= L[i * p + k] * L[j * p + k];
            if (i == j) {
                if (s <= 0.0) return 0;
                L[i * p + i] = sqrt(s);
            } else {
                L[i * p + j] = s / L[j * p + j];
            }
        }
    }
    return 1;
}

void ssgs_gaussian_logprobs(const double *means, const double *covs,
                            const double *obs, int S, int p, int T,
                            double *out) {
    if (S <= 0 || p <= 0 || T <= 0) return;
    double c = p * log(2.0 * M_PI);
    double *L = (double *)malloc(p * p * sizeof(double));
    double *d = (double *)malloc(p * sizeof(double));
    double *y = (double *)malloc(p * sizeof(double));
    double *cj = (double *)malloc(p * p * sizeof(double));
    if (!L || !d || !y || !cj) {
        free(L); free(d); free(y); free(cj);
        return;
    }
    for (int s = 0; s < S; s++) {
        const double *cov = covs + s * p * p;
        int ok = chol_lower(cov, p, L);
        if (!ok) {
            memcpy(cj, cov, p * p * sizeof(double));
            for (int i = 0; i < p; i++) cj[i * p + i] += 1e-4;
            if (!chol_lower(cj, p, L)) {
                for (int t = 0; t < T; t++) out[t * S + s] = NAN;
                continue;
            }
        }
        double logdet = 0.0;
        for (int i = 0; i < p; i++) logdet += log(L[i * p + i]);
        logdet *= 2.0;
        const double *mean = means + s * p;
        for (int t = 0; t < T; t++) {
            const double *o = obs + t * p;
            for (int i = 0; i < p; i++) d[i] = o[i] - mean[i];
            for (int i = 0; i < p; i++) {
                double s2 = d[i];
                for (int k = 0; k < i; k++) s2 -= L[i * p + k] * y[k];
                y[i] = s2 / L[i * p + i];
            }
            double quad = 0.0;
            for (int i = 0; i < p; i++) quad += y[i] * y[i];
            out[t * S + s] = -0.5 * (quad + c + logdet);
        }
    }
    free(L); free(d); free(y); free(cj);
}

/* ---- Forward-backward (EM expectation step) ---- */
void ssgs_forward_backward(const double *emit, const double *init,
                           const double *trans, int T, int S,
                           double *gamma, double *xi, double *loglik) {
    if (T <= 0 || S <= 0) return;
    double *alpha = (double *)malloc(T * S * sizeof(double));
    double *beta = (double *)malloc(T * S * sizeof(double));
    double *tmp = (double *)malloc(S * sizeof(double));
    double *row = (double *)malloc(S * sizeof(double));
    if (!alpha || !beta || !tmp || !row) {
        free(alpha); free(beta); free(tmp); free(row);
        return;
    }
    for (int s = 0; s < S; s++) alpha[s] = init[s] + emit[s];
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < S; j++) {
            for (int i = 0; i < S; i++) tmp[i] = alpha[(t - 1) * S + i] + trans[i * S + j];
            alpha[t * S + j] = logsumexp(tmp, S) + emit[t * S + j];
        }
    }
    for (int s = 0; s < S; s++) beta[(T - 1) * S + s] = 0.0;
    for (int t = T - 2; t >= 0; t--) {
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++)
                tmp[j] = trans[i * S + j] + emit[(t + 1) * S + j] + beta[(t + 1) * S + j];
            beta[t * S + i] = logsumexp(tmp, S);
        }
    }
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < S; s++) row[s] = alpha[t * S + s] + beta[t * S + s];
        double ls = logsumexp(row, S);
        for (int s = 0; s < S; s++)
            gamma[t * S + s] = exp(alpha[t * S + s] + beta[t * S + s] - ls);
    }
    for (int t = 0; t < T - 1; t++) {
        double *xr = xi + t * S * S;
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++)
                xr[i * S + j] = alpha[t * S + i] + trans[i * S + j] +
                                emit[(t + 1) * S + j] + beta[(t + 1) * S + j];
        double ls2 = logsumexp(xr, S * S);
        for (int k = 0; k < S * S; k++) xr[k] = exp(xr[k] - ls2);
    }
    *loglik = logsumexp(alpha + (T - 1) * S, S);
    free(alpha); free(beta); free(tmp); free(row);
}

/* ---- Smoothness-first lexicographic Viterbi decode ---- */
void ssgs_decode(const double *trans, const double *smooth,
                 const double *init, double weight, int S, int nframes,
                 int *path) {
    if (S <= 0 || nframes <= 0) return;
    if (weight <= 0.0) {
        double *dp = (double *)malloc(nframes * S * sizeof(double));
        int *back = (int *)malloc(nframes * S * sizeof(int));
        if (!dp || !back) { free(dp); free(back); return; }
        for (int s = 0; s < S; s++) dp[s] = init[s];
        for (int t = 1; t < nframes; t++) {
            for (int j = 0; j < S; j++) {
                double best = 1e300; int bi = 0;
                for (int i = 0; i < S; i++) {
                    double c = dp[(t - 1) * S + i] + trans[i * S + j];
                    if (c < best) { best = c; bi = i; }
                }
                dp[t * S + j] = best; back[t * S + j] = bi;
            }
        }
        double best = 1e300; int be = 0;
        for (int s = 0; s < S; s++)
            if (dp[(nframes - 1) * S + s] < best) { best = dp[(nframes - 1) * S + s]; be = s; }
        path[nframes - 1] = be;
        for (int t = nframes - 1; t > 0; t--) path[t - 1] = back[t * S + path[t]];
        free(dp); free(back);
        return;
    }
    double *dps = (double *)malloc(nframes * S * sizeof(double));
    double *dpt = (double *)malloc(nframes * S * sizeof(double));
    int *back = (int *)malloc(nframes * S * sizeof(int));
    if (!dps || !dpt || !back) { free(dps); free(dpt); free(back); return; }
    for (int s = 0; s < S; s++) { dps[s] = 0.0; dpt[s] = init[s]; }
    double tol = 1e-9;
    for (int t = 1; t < nframes; t++) {
        for (int j = 0; j < S; j++) {
            double minsm = 1e300;
            for (int i = 0; i < S; i++) {
                double sm = dps[(t - 1) * S + i] + smooth[i * S + j];
                if (sm < minsm) minsm = sm;
            }
            double total_best = 1e300; int best_i = 0;
            for (int i = 0; i < S; i++) {
                double sm = dps[(t - 1) * S + i] + smooth[i * S + j];
                if (sm <= minsm + tol) {
                    double tot = dpt[(t - 1) * S + i] + trans[i * S + j];
                    if (tot < total_best) { total_best = tot; best_i = i; }
                }
            }
            dps[t * S + j] = minsm;
            dpt[t * S + j] = total_best;
            back[t * S + j] = best_i;
        }
    }
    double minsm = 1e300;
    for (int s = 0; s < S; s++) if (dps[(nframes - 1) * S + s] < minsm) minsm = dps[(nframes - 1) * S + s];
    double total_best = 1e300; int be = 0;
    for (int s = 0; s < S; s++) {
        if (dps[(nframes - 1) * S + s] <= minsm + tol) {
            double tot = dpt[(nframes - 1) * S + s];
            if (tot < total_best) { total_best = tot; be = s; }
        }
    }
    path[nframes - 1] = be;
    for (int t = nframes - 1; t > 0; t--) path[t - 1] = back[t * S + path[t]];
    free(dps); free(dpt); free(back);
}
