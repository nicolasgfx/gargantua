// gargantua_cuda.cu
// ─────────────────────────────────────────────────────────────────────────────
// Kerr black hole raytracer — physically accurate geodesic integration
// Based on: James et al., "Gravitational Lensing by Spinning Black Holes
//           in Astrophysics, and in the Movie Interstellar" (2015)
//           Equations from Appendix A.1 (ray tracing) and A.2 (ray bundles)
//
// Build (CUDA):
//   nvcc -O3 -arch=sm_89 -o gargantua gargantua_cuda.cu
// Build (CPU fallback):
//   g++ -O3 -fopenmp -x c++ -DCPU_ONLY gargantua_cuda.cu -o gargantua -lm
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef CPU_ONLY
  #define __device__
  #define __host__
  #define __global__
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

struct Config {
    // Image
    int W          = 960;
    int H          = 540;

    // Black hole (geometrised units, M=1)
    double a_spin  = 0.6;      // Kerr spin parameter a/M; 0.6 matches Interstellar movie

    // Camera (Boyer-Lindquist coords)
    double cam_r   = 65.0;     // radial distance
    double cam_th  = 1.50;     // polar angle (~85.9°, slightly above equator)

    // Field of view
    double fov_deg = 20.0;     // degrees

    // Integration
    int    maxSteps = 5000;
    double stepSize = 0.03;    // affine parameter step

    // Disk
    double disk_Rin;           // set in init() from ISCO
    double disk_Rout = 22.0;
    double disk_T0   = 4500.0; // base temperature (K) — matches Interstellar movie choice

    // Appearance
    bool   doDoppler  = false; // false matches Interstellar movie choice
    bool   doIntensity= false; // brightness beaming
    double bloomSigma = 4.0;
    int    bloomPasses= 4;
    double bloomStrength = 0.30;
    double exposure   = 2.2;

    void init() {
        // Compute prograde ISCO for spin a
        double a = a_spin;
        double z1 = 1.0 + cbrt(1.0 - a*a) * (cbrt(1.0+a) + cbrt(1.0-a));
        double z2 = sqrt(3.0*a*a + z1*z1);
        disk_Rin = 3.0 + z2 - sqrt((3.0-z1)*(3.0+z1+2.0*z2));
    }
};

static Config CFG;

// ═══════════════════════════════════════════════════════════════════════════════
// Math helpers
// ═══════════════════════════════════════════════════════════════════════════════

struct double3_ { double x,y,z; };

__device__ __host__ inline double sq(double x) { return x*x; }
__device__ __host__ inline double clamp01(double v) { return v<0?0:v>1?1:v; }
__device__ __host__ inline double lerp_(double a, double b, double t) { return a+(b-a)*t; }

// ═══════════════════════════════════════════════════════════════════════════════
// Procedural noise  (GPU-compatible value noise + fBm)
// ═══════════════════════════════════════════════════════════════════════════════

// Integer hash for gradient noise (no trig, no banding)
__device__ __host__
inline unsigned int ihash(int x, int y)
{
    unsigned int h = (unsigned int)(x * 73856093u ^ y * 19349663u);
    h = (h ^ (h >> 16)) * 0x45d9f3bu;
    h = (h ^ (h >> 16)) * 0x45d9f3bu;
    h = h ^ (h >> 16);
    return h;
}

// Gradient noise (Perlin-style, 2D)
__device__ __host__
double gradNoise(double x, double y)
{
    int ix = (int)floor(x), iy = (int)floor(y);
    double fx = x - ix, fy = y - iy;
    // quintic Hermite interpolation (smoother than cubic)
    double ux = fx*fx*fx*(fx*(fx*6.0-15.0)+10.0);
    double uy = fy*fy*fy*(fy*(fy*6.0-15.0)+10.0);
    // pseudo-random gradients via integer hash
    auto grad = [](unsigned int h, double dx, double dy) -> double {
        switch (h & 3u) {
            case 0: return  dx + dy;
            case 1: return -dx + dy;
            case 2: return  dx - dy;
            default: return -dx - dy;
        }
    };
    double n00 = grad(ihash(ix,   iy  ), fx,     fy    );
    double n10 = grad(ihash(ix+1, iy  ), fx-1.0, fy    );
    double n01 = grad(ihash(ix,   iy+1), fx,     fy-1.0);
    double n11 = grad(ihash(ix+1, iy+1), fx-1.0, fy-1.0);
    return lerp_(lerp_(n00, n10, ux), lerp_(n01, n11, ux), uy);
}

__device__ __host__
double fbmNoise(double x, double y, int octaves)
{
    double val = 0.0, amp = 0.5, freq = 1.0;
    for (int i = 0; i < octaves; i++) {
        val += amp * gradNoise(x * freq, y * freq);
        freq *= 2.03;
        amp  *= 0.5;
    }
    return val;  // range roughly [-1, 1]
}

// ═══════════════════════════════════════════════════════════════════════════════
// Kerr metric functions  (M=1 units, Boyer-Lindquist)
// From paper Eqs. (A.1)-(A.4)
// ═══════════════════════════════════════════════════════════════════════════════

struct KerrMetric {
    double a;       // spin
    double r_H;     // event horizon radius

    __device__ __host__
    void init(double spin) {
        a = spin;
        r_H = 1.0 + sqrt(1.0 - a*a);
    }

    __device__ __host__
    double rho2(double r, double th) const {
        return r*r + a*a * sq(cos(th));
    }

    __device__ __host__
    double Delta(double r) const {
        return r*r - 2.0*r + a*a;
    }

    __device__ __host__
    double Sigma(double r, double th) const {
        double D = Delta(r);
        return sqrt(sq(r*r + a*a) - a*a * D * sq(sin(th)));
    }

    __device__ __host__
    double alpha(double r, double th) const {
        double rh2 = rho2(r, th);
        return sqrt(rh2 * Delta(r)) / Sigma(r, th);
    }

    __device__ __host__
    double omega(double r, double th) const {
        double S2 = sq(Sigma(r, th));
        return 2.0*a*r / S2;
    }

    __device__ __host__
    double pomega(double r, double th) const {
        return Sigma(r, th) * sin(th) / sqrt(rho2(r, th));
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Geodesic state & integration
// Super-Hamiltonian form: Eq. (A.15) from the paper
// State = (r, theta, phi, p_r, p_theta)
// Constants of motion: b = p_phi,  q = Carter constant
// Convention: p_t = -1
// ═══════════════════════════════════════════════════════════════════════════════

struct RayState {
    double r, th, ph;    // position
    double pr, pth;      // momenta
    double b, q;         // constants of motion
};

// Evaluate the "super-Hamiltonian" kernel F and its RHS
__device__ __host__
double superH(double r, double th, double pr, double pth,
              double b, double q, double a)
{
    double ct = cos(th), st = sin(th);
    double rh2 = r*r + a*a * ct*ct;
    double D   = r*r - 2.0*r + a*a;
    double P   = r*r + a*a - a*b;
    double R   = P*P - D*((b-a)*(b-a) + q);
    double Th  = q - ct*ct * (b*b/(st*st) - a*a);
    // F = -D/(2 rho2) pr^2 - 1/(2 rho2) pth^2 + (R + D*Th)/(2 D rho2)
    return -D/(2.0*rh2)*pr*pr - 1.0/(2.0*rh2)*pth*pth + (R + D*Th)/(2.0*D*rh2);
}

// RHS of the geodesic equations (analytical derivatives of super-Hamiltonian)
__device__ __host__
void geodesicRHS(const RayState& s, double a, double* drhs)
{
    double r = s.r, th = s.th, pr = s.pr, pth = s.pth;
    double b = s.b, q = s.q;
    double ct = cos(th), st = sin(th);
    if (fabs(st) < 1e-10) st = (st >= 0 ? 1e-10 : -1e-10);

    double Sig  = r*r + a*a*ct*ct;          // Σ = ρ²
    double D    = r*r - 2.0*r + a*a;        // Δ
    double P    = r*r + a*a - a*b;
    double bma  = b - a;
    double R_pot = P*P - D*(bma*bma + q);   // radial potential R
    double Th   = q + a*a*ct*ct - b*b*ct*ct/(st*st);  // Θ potential
    double RoD  = R_pot / D;                // R/Δ
    // V = -Δ pr² - pθ² + Θ + R/Δ   →   F = V / (2Σ)
    double V    = -D*pr*pr - pth*pth + Th + RoD;

    // ── Velocities ──
    drhs[0] = D / Sig * pr;                                   // dr/dζ
    drhs[1] = pth / Sig;                                      // dθ/dζ
    drhs[2] = (a*P + D*(b/(st*st) - a)) / (D * Sig);         // dφ/dζ

    // ── dp_r/dζ = ∂F/∂r  (analytical) ──
    double dD_dr   = 2.0*r - 2.0;
    double dSig_dr = 2.0*r;
    double dR_dr   = 4.0*r*P - dD_dr*(bma*bma + q);
    double dRoD_dr = (dR_dr*D - R_pot*dD_dr) / (D*D);
    double dV_dr   = -dD_dr*pr*pr + dRoD_dr;
    drhs[3] = (dV_dr*Sig - V*dSig_dr) / (2.0*Sig*Sig);

    // ── dp_θ/dζ = ∂F/∂θ  (analytical) ──
    double sin2th   = 2.0*ct*st;
    double dSig_dth = -a*a*sin2th;
    double dTh_dth  = -a*a*sin2th + 2.0*b*b*ct/(st*st*st);
    double dV_dth   = dTh_dth;
    drhs[4] = (dV_dth*Sig - V*dSig_dth) / (2.0*Sig*Sig);
}

// RK4 integration step (backward in affine parameter)
__device__ __host__
void rk4Step(RayState& s, double a, double h)
{
    double y[5] = {s.r, s.th, s.ph, s.pr, s.pth};
    double k1[5], k2[5], k3[5], k4[5], ytmp[5];
    RayState tmp;

    // k1
    geodesicRHS(s, a, k1);

    // k2
    for(int i=0;i<5;i++) ytmp[i] = y[i] + 0.5*h*k1[i];
    tmp = {ytmp[0],ytmp[1],ytmp[2],ytmp[3],ytmp[4], s.b, s.q};
    geodesicRHS(tmp, a, k2);

    // k3
    for(int i=0;i<5;i++) ytmp[i] = y[i] + 0.5*h*k2[i];
    tmp = {ytmp[0],ytmp[1],ytmp[2],ytmp[3],ytmp[4], s.b, s.q};
    geodesicRHS(tmp, a, k3);

    // k4
    for(int i=0;i<5;i++) ytmp[i] = y[i] + h*k3[i];
    tmp = {ytmp[0],ytmp[1],ytmp[2],ytmp[3],ytmp[4], s.b, s.q};
    geodesicRHS(tmp, a, k4);

    // update
    for(int i=0;i<5;i++)
        y[i] += h/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

    s.r  = y[0];
    s.th = y[1];
    s.ph = y[2];
    s.pr = y[3];
    s.pth= y[4];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Camera setup  (Paper Appendix A.1, steps i-iv)
// ═══════════════════════════════════════════════════════════════════════════════

// Initialize a ray from pixel coordinates
// Camera is at (r_c, theta_c, phi_c=0) and is a FIDO (static observer)
// For the Interstellar look, we later add orbital aberration
__device__ __host__
RayState initRay(double spx, double spy, int W, int H,
                 double r_c, double th_c,
                 double fov_rad, double a_spin)
{
    KerrMetric km;
    km.init(a_spin);

    double aspect = (double)W / (double)H;
    double scale  = tan(0.5*fov_rad);

    // Sub-pixel position to normalised screen coords
    // Shift by 0.5 pixels so that b=0 (u=0) falls exactly on the center
    // pixel rather than on the boundary between two pixels.
    // This ensures the center pixel's AA samples straddle both b>0 and b<0.
    double u = ((2.0 * spx - W + 1.0) / (double)(W - 1)) * scale * aspect;
    double v = ((H - 1.0 - 2.0 * spy) / (double)(H - 1)) * scale;

    // FIDO frame at camera:
    //   forward = -e_rhat  (toward BH)
    //   right   = e_phihat
    //   up      = -e_thetahat  (theta increases southward; negate for up)
    double nFr   = -1.0;      // forward
    double nFth  = -v;        // up component → maps to −e_θ̂
    double nFph  =  u;        // right component → maps to e_φ̂

    // Normalise
    double nmag = sqrt(nFr*nFr + nFth*nFth + nFph*nFph);
    nFr  /= nmag;
    nFth /= nmag;
    nFph /= nmag;

    // For now: static FIDO camera (no aberration)
    // Compute metric quantities at camera
    double al = km.alpha(r_c, th_c);
    double om = km.omega(r_c, th_c);
    double pw = km.pomega(r_c, th_c);

    // Compute energy in FIDO frame (Eq A.11)
    // E_F = 1 / (alpha + omega * pomega * nFphi_hat)
    double E_F = 1.0 / (al + om * pw * nFph);

    // Canonical momenta (Eq A.11)
    double rh2 = km.rho2(r_c, th_c);
    double rho = sqrt(rh2);
    double D   = km.Delta(r_c);

    double p_r  = E_F * (rho / sqrt(D)) * nFr;
    double p_th = E_F * rho * nFth;
    double p_ph = E_F * pw  * nFph;

    // Constants of motion (Eq A.12)
    double b = p_ph;
    double ct = cos(th_c);
    double st = sin(th_c);
    double q = p_th*p_th + ct*ct * (b*b/(st*st) - a_spin*a_spin);

    RayState ray;
    ray.r   = r_c;
    ray.th  = th_c;
    ray.ph  = 0.0;
    ray.pr  = p_r;
    ray.pth = p_th;
    ray.b   = b;
    ray.q   = q;

    return ray;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Blackbody approximation  (temperature → RGB)
// Uses the CIE 1931 approximation by Tanner Helland
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __host__
void blackbodyRGB(double T, double& R, double& G, double& B)
{
    // Clamp temperature
    T = fmax(1000.0, fmin(40000.0, T));
    double t = T / 100.0;

    // Red
    if (t <= 66.0)
        R = 1.0;
    else {
        double x = t - 60.0;
        R = 329.698727446 * pow(x, -0.1332047592) / 255.0;
    }

    // Green
    if (t <= 66.0) {
        G = (99.4708025861 * log(t) - 161.1195681661) / 255.0;
    } else {
        double x = t - 60.0;
        G = 288.1221695283 * pow(x, -0.0755148492) / 255.0;
    }

    // Blue
    if (t >= 66.0)
        B = 1.0;
    else if (t <= 19.0)
        B = 0.0;
    else {
        double x = t - 10.0;
        B = (138.5177312231 * log(x) - 305.0447927307) / 255.0;
    }

    R = clamp01(R);
    G = clamp01(G);
    B = clamp01(B);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Disk coloring
// ═══════════════════════════════════════════════════════════════════════════════

// Novikov-Thorne temperature profile (simplified):  T(r) = T0 * f(r)
// where f(r) = [1/r^3 * (1 - sqrt(r_ISCO/r))]^{1/4}  (approx)
__device__ __host__
double diskTemperature(double r, double r_isco, double T0)
{
    if (r < r_isco) return 0.0;
    double x = r_isco / r;
    double f = pow(fmax(0.0, 1.0/r/r/r * (1.0 - sqrt(x))), 0.25);
    // Normalise so peak is near T0
    // Peak of f is around r ≈ 1.36 * r_isco. Find peak value and normalise.
    double r_peak = 1.36 * r_isco;
    double x_peak = r_isco / r_peak;
    double f_peak = pow(fmax(1e-10, 1.0/r_peak/r_peak/r_peak * (1.0 - sqrt(x_peak))), 0.25);
    return T0 * f / fmax(1e-10, f_peak);
}

// Doppler g-factor for a prograde Keplerian disk element
// g = f_obs / f_emit = 1/[(1 + b*Omega) * u^t_emit * alpha_cam]
// For static camera: alpha_cam correction already in ray energy convention
__device__ __host__
double diskGFactor(double r_disk, double b_ray, double a_spin)
{
    // Keplerian angular velocity in equatorial plane
    double Om = 1.0 / (a_spin + pow(r_disk, 1.5));

    // Specific energy and angular momentum of circular orbit
    double r  = r_disk;
    double r2 = r*r;
    double a  = a_spin;

    // u^t for circular equatorial orbit:
    // u^t = (r^{3/2} + a) / (r^{3/4} * sqrt(r^{3/2} - 3*r^{1/2} + 2*a))
    // Simplified:
    double sqr = sqrt(r);
    double denom = sqrt(fmax(1e-10, r*sqr - 3.0*sqr + 2.0*a));
    double ut = (r*sqr + a) / (pow(r, 0.75) * denom);

    // g = 1 / (ut * (1 - b*Om))
    // Note: the sign depends on convention. For our b convention:
    double g = 1.0 / (ut * fmax(0.01, 1.0 - b_ray * Om));

    return fmax(0.01, fmin(5.0, g));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ray tracing kernel
// ═══════════════════════════════════════════════════════════════════════════════

// Ray status enum
#define STATUS_FLYING   0
#define STATUS_DISK     1
#define STATUS_HORIZON  2
#define STATUS_ESCAPED  3

__device__ __host__
void traceRay(double spx, double spy,
              int W, int H,
              double a_spin, double r_cam, double th_cam, double fov_rad,
              double disk_Rin, double disk_Rout, double disk_T0,
              bool doDoppler, bool doIntensity,
              int maxSteps, double stepBase,
              double* out_rgb)
{
    KerrMetric km;
    km.init(a_spin);

    // Initialise ray from sub-pixel position
    RayState ray = initRay(spx, spy, W, H, r_cam, th_cam, fov_rad, a_spin);

    out_rgb[0] = out_rgb[1] = out_rgb[2] = 0.0;

    double totalDiskR = 0.0, totalDiskG = 0.0, totalDiskB = 0.0;
    double totalAlpha = 0.0;

    int status = STATUS_FLYING;
    double old_th = ray.th;

    for (int step = 0; step < maxSteps && status == STATUS_FLYING; step++)
    {
        double old_r  = ray.r;
        old_th = ray.th;

        // Adaptive step: smaller near the BH for accuracy
        double h = stepBase;  // positive step in affine parameter for backward tracing
        double rFrac = (ray.r - km.r_H) / 8.0;
        rFrac = fmax(0.05, fmin(1.0, rFrac));
        h *= rFrac;

        rk4Step(ray, a_spin, h);

        // Safety: bail on NaN
        if (isnan(ray.r) || isnan(ray.th) || isinf(ray.r)) {
            status = STATUS_HORIZON;
            break;
        }

        // Clamp theta: proper reflection at poles
        while (ray.th < 0.0 || ray.th > M_PI) {
            if (ray.th < 0.0)  { ray.th = -ray.th; ray.ph += M_PI; ray.pth = -ray.pth; }
            if (ray.th > M_PI) { ray.th = 2.0*M_PI - ray.th; ray.ph += M_PI; ray.pth = -ray.pth; }
        }
        // Safety clamp to avoid exact pole
        if (ray.th < 1e-6) ray.th = 1e-6;
        if (ray.th > M_PI - 1e-6) ray.th = M_PI - 1e-6;

        // ─── Check horizon crossing ───
        if (ray.r <= km.r_H * 1.02 || ray.r < 0.5) {
            status = STATUS_HORIZON;
            break;
        }

        // ─── Check escape ───
        if (ray.r > 300.0) {
            status = STATUS_ESCAPED;
            break;
        }

        // ─── Check disk crossing (equatorial plane: theta = pi/2) ───
        bool crossed = (old_th - M_PI/2.0) * (ray.th - M_PI/2.0) < 0.0;
        if (crossed) {
            // Interpolate to find crossing point
            double frac = fabs(old_th - M_PI/2.0) /
                          fmax(1e-15, fabs(old_th - M_PI/2.0) + fabs(ray.th - M_PI/2.0));
            double r_cross = old_r + frac * (ray.r - old_r);

            if (r_cross >= disk_Rin && r_cross <= disk_Rout) {
                // Disk hit! Compute color
                double T = diskTemperature(r_cross, disk_Rin, disk_T0);

                if (doDoppler) {
                    double g = diskGFactor(r_cross, ray.b, a_spin);
                    T *= g;  // Temperature transforms as T_obs = g * T_emit
                }

                double dR, dG, dB;
                blackbodyRGB(T, dR, dG, dB);

                // Intensity: more uniform for Interstellar movie look
                // Moderate radial falloff, inner disk slightly brighter
                double radialFrac = (r_cross - disk_Rin) / (disk_Rout - disk_Rin);
                double intensity = (0.6 + 0.8 * (1.0 - radialFrac));
                if (doIntensity) {
                    double g = diskGFactor(r_cross, ray.b, a_spin);
                    intensity *= g * g * g;  // g^3 for specific intensity (Liouville)
                }

                // ── Streaky disk texture + frayed edges (per paper Sec 4.3.2) ──
                double phi_c = ray.ph;
                double cx = r_cross * cos(phi_c);
                double cy = r_cross * sin(phi_c);

                // Noise-modulated outer boundary
                double edgeNoise = fbmNoise(cx * 0.3, cy * 0.3, 6);
                double localRout = disk_Rout * (1.0 + 0.20 * edgeNoise);
                double localRin  = disk_Rin  * (1.0 - 0.05 * fbmNoise(cx * 0.5 + 7.7, cy * 0.5 + 3.1, 4));

                // Radial falloff with noise-modulated boundaries
                double taper_in  = clamp01((r_cross - localRin) / (0.8));
                double taper_out = clamp01((localRout - r_cross) / (localRout * 0.15));

                // ── Exponentially growing radial + azimuthal noise ──
                // t=0 at inner edge, t=1 at outer edge
                double t_radial = clamp01((r_cross - disk_Rin) / (disk_Rout - disk_Rin));
                // Exponential ramp for overall noise amplitude
                double noiseAmp = 0.08 + 0.92 * (exp(3.0 * t_radial) - 1.0) / (exp(3.0) - 1.0);

                // --- Radial noise (concentric bands with IRREGULAR spacing) ---
                // Use noise to warp the radial coordinate => bands have varying widths
                // More warping at larger radius => outer bands increasingly irregular
                double rWarp = r_cross
                    + fbmNoise(r_cross * 0.4, phi_c * 0.2, 5) * 1.5 * (1.0 + 3.0 * t_radial)
                    + fbmNoise(r_cross * 0.15 + 3.3, phi_c * 0.1, 4) * 3.0 * t_radial * t_radial;
                // Bands at multiple frequencies through the warped coordinate
                double rBand1 = 0.5 + 0.5 * sin(rWarp * 2.5 * M_PI);
                double rBand2 = 0.5 + 0.5 * sin(rWarp * 5.5 * M_PI
                    + fbmNoise(phi_c * 0.25, r_cross * 0.15, 3) * 2.0);
                double rBand3 = 0.5 + 0.5 * sin(rWarp * 11.0 * M_PI
                    + fbmNoise(phi_c * 0.15 + 7.7, r_cross * 0.1, 2) * 1.5);
                double radialStreak = rBand1 * 0.50 + rBand2 * 0.35 + rBand3 * 0.15;

                // --- Azimuthal noise (angular filaments) ---
                // Steeper exponential for azimuthal: nearly invisible near BH, dominant at edges
                double azAmp = (exp(5.0 * t_radial) - 1.0) / (exp(5.0) - 1.0);
                double aNoise1 = fbmNoise(phi_c * 10.0, r_cross * 0.05, 5);
                double aNoise2 = fbmNoise(phi_c * 20.0 + 3.3, r_cross * 0.03, 4);
                double aNoise3 = fbmNoise(phi_c * 40.0 + 9.1, r_cross * 0.015, 3);
                double azStreak = aNoise1 * 0.40 + aNoise2 * 0.35 + aNoise3 * 0.25;

                // Combine: radial bands everywhere, azimuthal grows in from edges
                double combinedNoise = radialStreak * (1.0 - azAmp * 0.6)
                                     + azStreak * azAmp;

                // Slow azimuthal brightness variation (large-scale)
                double azVar = 0.50 + 0.50 * fbmNoise(phi_c * 0.8, r_cross * 0.1, 3);
                combinedNoise *= azVar;

                // Apply noise with exponential amplitude — full range modulation
                intensity *= (1.0 - noiseAmp) + noiseAmp * 3.0 * combinedNoise;

                double alpha = taper_in * taper_out;

                intensity *= alpha;

                // Alpha-blend this disk layer (semi-transparent to allow secondary images)
                double layerAlpha = fmin(0.8, intensity * 0.35);
                totalDiskR += dR * intensity * (1.0 - totalAlpha);
                totalDiskG += dG * intensity * (1.0 - totalAlpha);
                totalDiskB += dB * intensity * (1.0 - totalAlpha);
                totalAlpha = totalAlpha + layerAlpha * (1.0 - totalAlpha);

                // Don't break — allow multiple disk crossings for secondary/tertiary images
                if (totalAlpha > 0.99) {
                    status = STATUS_DISK;
                    break;
                }
            }
        }
    }

    // Background: very dark with subtle noise (space)
    if (status == STATUS_ESCAPED) {
        // Simple procedural starfield based on ray direction at escape
        double phi_esc = fmod(ray.ph, 2.0*M_PI);
        if (phi_esc < 0) phi_esc += 2.0*M_PI;
        double th_esc  = ray.th;

        // Milky Way band: bright stripe near theta ~ pi/2
        double galactic_lat = fabs(th_esc - M_PI/2.0);
        double mw_brightness = 0.015 * exp(-galactic_lat*galactic_lat / (0.08));

        double bgR = 0.005 + mw_brightness * 0.8;
        double bgG = 0.005 + mw_brightness * 0.75;
        double bgB = 0.008 + mw_brightness * 0.9;

        // Blend with any disk layers already accumulated
        totalDiskR += bgR * (1.0 - totalAlpha);
        totalDiskG += bgG * (1.0 - totalAlpha);
        totalDiskB += bgB * (1.0 - totalAlpha);
    } else if (status == STATUS_HORIZON) {
        // Black hole interior — black
        // Any disk light accumulated before falling in still shows
    }

    out_rgb[0] = totalDiskR;
    out_rgb[1] = totalDiskG;
    out_rgb[2] = totalDiskB;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA kernel
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef CPU_ONLY

__global__
void renderKernel(float* fb, int W, int H,
                  double a_spin, double r_cam, double th_cam, double fov_rad,
                  double disk_Rin, double disk_Rout, double disk_T0,
                  bool doDoppler, bool doIntensity,
                  int maxSteps, double stepBase,
                  int yOffset, int yCount)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;  // local y within batch
    if (px >= W || ly >= yCount) return;
    int py = ly + yOffset;
    if (py >= H) return;

    // 2x2 stratified supersampling
    double total[3] = {0.0, 0.0, 0.0};
    const double yOff[2] = {0.25, 0.75};
    for (int sj = 0; sj < 2; sj++) {
        for (int si = 0; si < 2; si++) {
            double spx = (double)px + (si + 0.5) * 0.5;
            double spy = (double)py + yOff[sj];
            double rgb[3];
            traceRay(spx, spy, W, H,
                     a_spin, r_cam, th_cam, fov_rad,
                     disk_Rin, disk_Rout, disk_T0,
                     doDoppler, doIntensity,
                     maxSteps, stepBase,
                     rgb);
            total[0] += rgb[0];
            total[1] += rgb[1];
            total[2] += rgb[2];
        }
    }

    int idx = (py * W + px) * 3;
    fb[idx+0] = (float)(total[0] * 0.25);
    fb[idx+1] = (float)(total[1] * 0.25);
    fb[idx+2] = (float)(total[2] * 0.25);
}

#endif // CPU_ONLY

// ═══════════════════════════════════════════════════════════════════════════════
// Post-processing: bloom / glow
// ═══════════════════════════════════════════════════════════════════════════════

void gaussianBlur(float* src, float* dst, int W, int H, double sigma)
{
    int radius = (int)(3.0 * sigma);
    if (radius < 1) radius = 1;

    // Precompute kernel
    std::vector<double> kernel(2*radius+1);
    double sum = 0.0;
    for (int i = -radius; i <= radius; i++) {
        kernel[i+radius] = exp(-0.5 * i*i / (sigma*sigma));
        sum += kernel[i+radius];
    }
    for (auto& k : kernel) k /= sum;

    // Temporary buffer for horizontal pass
    std::vector<float> tmp(W*H*3, 0.0f);

    // Horizontal pass
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double r=0, g=0, b=0;
            for (int k = -radius; k <= radius; k++) {
                int xi = x + k;
                if (xi < 0) xi = 0;
                if (xi >= W) xi = W-1;
                double w = kernel[k+radius];
                int si = (y*W+xi)*3;
                r += src[si+0]*w;
                g += src[si+1]*w;
                b += src[si+2]*w;
            }
            int di = (y*W+x)*3;
            tmp[di+0] = (float)r;
            tmp[di+1] = (float)g;
            tmp[di+2] = (float)b;
        }
    }

    // Vertical pass
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double r=0, g=0, b=0;
            for (int k = -radius; k <= radius; k++) {
                int yi = y + k;
                if (yi < 0) yi = 0;
                if (yi >= H) yi = H-1;
                double w = kernel[k+radius];
                int si = (yi*W+x)*3;
                r += tmp[si+0]*w;
                g += tmp[si+1]*w;
                b += tmp[si+2]*w;
            }
            int di = (y*W+x)*3;
            dst[di+0] = (float)r;
            dst[di+1] = (float)g;
            dst[di+2] = (float)b;
        }
    }
}

void applyBloom(float* fb, int W, int H, double sigma, int passes, double strength)
{
    int N = W*H*3;

    // Extract bright pixels (threshold for bloom)
    std::vector<float> bright(N);
    double threshold = 0.6;
    for (int i = 0; i < N; i += 3) {
        double lum = 0.2126*fb[i] + 0.7152*fb[i+1] + 0.0722*fb[i+2];
        double factor = fmax(0.0, lum - threshold) / fmax(lum, 0.001);
        bright[i+0] = (float)(fb[i+0] * factor);
        bright[i+1] = (float)(fb[i+1] * factor);
        bright[i+2] = (float)(fb[i+2] * factor);
    }

    // Multi-pass blur at increasing scales
    std::vector<float> blurred(N);
    std::vector<float> accumBloom(N, 0.0f);

    for (int pass = 0; pass < passes; pass++) {
        double s = sigma * pow(2.0, pass);
        gaussianBlur(bright.data(), blurred.data(), W, H, s);
        double w = 1.0 / (pass + 1);
        for (int i = 0; i < N; i++) {
            accumBloom[i] += (float)(blurred[i] * w);
        }
    }

    // Combine
    for (int i = 0; i < N; i++) {
        fb[i] += (float)(accumBloom[i] * strength);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tone mapping & output
// ═══════════════════════════════════════════════════════════════════════════════

void toneMapAndSave(float* fb, int W, int H, const char* filename,
                    double exposure, double gamma)
{
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);

    for (int i = 0; i < W*H*3; i += 3) {
        // Exposure
        double r = fb[i+0] * exposure;
        double g = fb[i+1] * exposure;
        double b = fb[i+2] * exposure;

        // Reinhard tone mapping
        r = r / (1.0 + r);
        g = g / (1.0 + g);
        b = b / (1.0 + b);

        // Gamma
        r = pow(clamp01(r), 1.0/gamma);
        g = pow(clamp01(g), 1.0/gamma);
        b = pow(clamp01(b), 1.0/gamma);

        unsigned char R = (unsigned char)(clamp01(r)*255+0.5);
        unsigned char G = (unsigned char)(clamp01(g)*255+0.5);
        unsigned char B = (unsigned char)(clamp01(b)*255+0.5);
        fwrite(&R, 1, 1, f);
        fwrite(&G, 1, 1, f);
        fwrite(&B, 1, 1, f);
    }
    fclose(f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    CFG.init();

    int W = CFG.W, H = CFG.H;
    double fov_rad = CFG.fov_deg * M_PI / 180.0;

    printf("=== Gargantua CUDA Kerr Raytracer ===\n");
    printf("Resolution: %d x %d\n", W, H);
    printf("Spin a/M: %.3f\n", CFG.a_spin);
    printf("Camera: r=%.1fM, theta=%.4f rad (%.2f deg)\n",
           CFG.cam_r, CFG.cam_th, CFG.cam_th*180/M_PI);
    printf("Disk: Rin=%.2fM (ISCO), Rout=%.1fM\n", CFG.disk_Rin, CFG.disk_Rout);
    printf("Event horizon: r_H=%.3fM\n", 1.0 + sqrt(1.0 - CFG.a_spin*CFG.a_spin));
    printf("Max integration steps: %d, step size: %.4f\n", CFG.maxSteps, CFG.stepSize);
    printf("Doppler shifts: %s\n", CFG.doDoppler ? "ON" : "OFF (Interstellar movie mode)");

    int N = W * H * 3;

#ifndef CPU_ONLY
    // ─── CUDA path (row-batch to avoid Windows TDR timeout) ───
    printf("Using CUDA...\n");

    float* d_fb;
    cudaMalloc(&d_fb, N * sizeof(float));
    cudaMemset(d_fb, 0, N * sizeof(float));

    // Process in row batches to stay within ~2s per kernel launch (TDR safe)
    int batchRows = 12;  // rows per kernel launch (reduced for 2x2 AA)
    dim3 block(16, 16);

    for (int yStart = 0; yStart < H; yStart += batchRows)
    {
        int rowsThisBatch = (yStart + batchRows <= H) ? batchRows : (H - yStart);

        dim3 grid((W + block.x - 1) / block.x,
                  (rowsThisBatch + block.y - 1) / block.y);

        renderKernel<<<grid, block>>>(d_fb, W, H,
                                       CFG.a_spin, CFG.cam_r, CFG.cam_th, fov_rad,
                                       CFG.disk_Rin, CFG.disk_Rout, CFG.disk_T0,
                                       CFG.doDoppler, CFG.doIntensity,
                                       CFG.maxSteps, CFG.stepSize,
                                       yStart, rowsThisBatch);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at batch y=%d: %s\n", yStart, cudaGetErrorString(err));
            cudaFree(d_fb);
            return 1;
        }
        printf("  Rows %d-%d / %d done\n", yStart, yStart+rowsThisBatch-1, H);
    }
    printf("Kernel complete.\n");

    // Copy back
    std::vector<float> fb(N);
    cudaMemcpy(fb.data(), d_fb, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fb);

#else
    // ─── CPU fallback with OpenMP ───
    printf("Using CPU (OpenMP)...\n");
    std::vector<float> fb(N, 0.0f);

    int done = 0;
    #pragma omp parallel for schedule(dynamic, 4) collapse(2)
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            double total[3] = {0.0, 0.0, 0.0};
            const double yOff[2] = {0.25, 0.75};
            for (int sj = 0; sj < 2; sj++) {
                for (int si = 0; si < 2; si++) {
                    double spx = (double)px + (si + 0.5) * 0.5;
                    double spy = (double)py + yOff[sj];
                    double rgb[3];
                    traceRay(spx, spy, W, H,
                             CFG.a_spin, CFG.cam_r, CFG.cam_th, fov_rad,
                             CFG.disk_Rin, CFG.disk_Rout, CFG.disk_T0,
                             CFG.doDoppler, CFG.doIntensity,
                             CFG.maxSteps, CFG.stepSize,
                             rgb);
                    total[0] += rgb[0];
                    total[1] += rgb[1];
                    total[2] += rgb[2];
                }
            }
            int idx = (py * W + px) * 3;
            fb[idx+0] = (float)(total[0] * 0.25);
            fb[idx+1] = (float)(total[1] * 0.25);
            fb[idx+2] = (float)(total[2] * 0.25);
        }
        #pragma omp atomic
        done++;
        if (done % 50 == 0) {
            #pragma omp critical
            printf("Row %d/%d\n", done, H);
        }
    }
#endif

    // ─── Fix b=0 seam: interpolate center columns from outer neighbors ───
    {
        // With the current mapping, u=0 falls at pixel cx = (W-1)/2.
        // The prograde/retrograde boundary creates a seam around this column.
        // Replace a 4-column band [cx-1..cx+2] by linearly interpolating
        // between column cx-2 (left clean) and cx+3 (right clean).
        int cx = (W - 1) / 2;  // center pixel where b≈0
        int lo = cx - 1;       // first column to fix
        int hi = cx + 2;       // last column to fix
        int lRef = lo - 1;     // left reference (clean)
        int rRef = hi + 1;     // right reference (clean)
        if (lRef >= 0 && rRef < W) {
            int span = rRef - lRef; // = 5
            for (int y = 0; y < H; y++) {
                int il = (y * W + lRef) * 3;
                int ir = (y * W + rRef) * 3;
                for (int col = lo; col <= hi; col++) {
                    float t = (float)(col - lRef) / (float)span;
                    int ic = (y * W + col) * 3;
                    fb[ic+0] = fb[il+0] * (1.0f - t) + fb[ir+0] * t;
                    fb[ic+1] = fb[il+1] * (1.0f - t) + fb[ir+1] * t;
                    fb[ic+2] = fb[il+2] * (1.0f - t) + fb[ir+2] * t;
                }
            }
        }
    }

    // ─── Post-processing ───
    printf("Applying bloom...\\n");
    applyBloom(fb.data(), W, H, CFG.bloomSigma, CFG.bloomPasses, CFG.bloomStrength);

    // ─── Save ───
    const char* outfile = "gargantua.ppm";
    printf("Tone mapping & saving %s...\n", outfile);
    toneMapAndSave(fb.data(), W, H, outfile, CFG.exposure, 2.2);

    printf("Done! Wrote %s (%d x %d)\n", outfile, W, H);
    return 0;
}
