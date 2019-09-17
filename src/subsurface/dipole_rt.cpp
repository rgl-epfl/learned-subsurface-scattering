#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/sse.h>
#include <mitsuba/core/ssemath.h>
#include <mitsuba/hw/basicshader.h>
#include "../medium/materials.h"

MTS_NAMESPACE_BEGIN

// Media Inline Functions
Float PhaseHG(Float cosTheta, Float g) {
    Float denom = 1 + g * g + 2 * g * cosTheta;
    return (1 - g * g) / (denom * std::sqrt(denom) * 4.f * M_PI);
}

// BSSRDF Utility Functions
Float FresnelMoment1(Float eta) {
    Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
          eta5 = eta4 * eta;
    if (eta < 1)
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
               2.49277f * eta4 - 0.68441f * eta5;
    else
        return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
               1.27198f * eta4 + 0.12746f * eta5;
}

Float FresnelMoment2(Float eta) {
    Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
          eta5 = eta4 * eta;
    if (eta < 1) {
        return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 +
               0.07883f * eta4 + 0.04860f * eta5;
    } else {
        Float r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
        return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 +
               458.843f * r_eta + 404.557f * eta - 189.519f * eta2 +
               54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
    }
}

template <typename Predicate>
int FindInterval(int size, const Predicate &pred) {
    int first = 0, len = size;
    while (len > 0) {
        int half = len >> 1, middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else
            len = half;
    }
    return math::clamp(first - 1, 0, size - 2);
}

// Spline Interpolation Definitions
Float CatmullRom(int size, const Float *nodes, const Float *values, Float x) {
    if (!(x >= nodes[0] && x <= nodes[size - 1])) return 0;
    int idx = FindInterval(size, [&](int i) { return nodes[i] <= x; });
    Float x0 = nodes[idx], x1 = nodes[idx + 1];
    Float f0 = values[idx], f1 = values[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;
    if (idx > 0)
        d0 = width * (f1 - values[idx - 1]) / (x1 - nodes[idx - 1]);
    else
        d0 = f1 - f0;

    if (idx + 2 < size)
        d1 = width * (values[idx + 2] - f0) / (nodes[idx + 2] - x0);
    else
        d1 = f1 - f0;

    Float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * f0 + (-2 * t3 + 3 * t2) * f1 +
           (t3 - 2 * t2 + t) * d0 + (t3 - t2) * d1;
}

bool CatmullRomWeights(int size, const Float *nodes, Float x, int *offset,
                       Float *weights) {
    // Return _false_ if _x_ is out of bounds
    if (!(x >= nodes[0] && x <= nodes[size - 1])) return false;

    // Search for the interval _idx_ containing _x_
    int idx = FindInterval(size, [&](int i) { return nodes[i] <= x; });
    *offset = idx - 1;
    Float x0 = nodes[idx], x1 = nodes[idx + 1];

    // Compute the $t$ parameter and powers
    Float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;

    // Compute initial node weights $w_1$ and $w_2$
    weights[1] = 2 * t3 - 3 * t2 + 1;
    weights[2] = -2 * t3 + 3 * t2;

    // Compute first node weight $w_0$
    if (idx > 0) {
        Float w0 = (t3 - 2 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        Float w0 = t3 - 2 * t2 + t;
        weights[0] = 0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    // Compute last node weight $w_3$
    if (idx + 2 < size) {
        Float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        Float w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0;
    }
    return true;
}

Float SampleCatmullRom(int n, const Float *x, const Float *f, const Float *F,
                       Float u, Float *fval, Float *pdf) {
    // Map _u_ to a spline interval by inverting _F_
    u *= F[n - 1];
    int i = FindInterval(n, [&](int i) { return F[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    Float x0 = x[i], x1 = x[i + 1];
    Float f0 = f[i], f1 = f[i + 1];
    Float width = x1 - x0;

    // Approximate derivatives using finite differences
    Float d0, d1;
    if (i > 0)
        d0 = width * (f1 - f[i - 1]) / (x1 - x[i - 1]);
    else
        d0 = f1 - f0;
    if (i + 2 < n)
        d1 = width * (f[i + 2] - f0) / (x[i + 2] - x0);
    else
        d1 = f1 - f0;

    // Re-scale _u_ for continous spline sampling step
    u = (u - F[i]) / width;

    // Invert definite integral over spline segment and return solution

    // Set initial guess for $t$ by importance sampling a linear interpolant
    Float t;
    if (f0 != f1)
        t = (f0 - std::sqrt(std::max((Float)0, f0 * f0 + 2 * u * (f1 - f0)))) /
            (f0 - f1);
    else
        t = u / f0;
    Float a = 0, b = 1, Fhat, fhat;
    while (true) {
        // Fall back to a bisection step when _t_ is out of bounds
        if (!(t > a && t < b)) t = 0.5f * (a + b);

        // Evaluate target function and its derivative in Horner form
        Fhat = t * (f0 +
                    t * (.5f * d0 +
                         t * ((1.f / 3.f) * (-2 * d0 - d1) + f1 - f0 +
                              t * (.25f * (d0 + d1) + .5f * (f0 - f1)))));
        fhat = f0 +
               t * (d0 +
                    t * (-2 * d0 - d1 + 3 * (f1 - f0) +
                         t * (d0 + d1 + 2 * (f0 - f1))));

        // Stop the iteration if converged
        if (std::abs(Fhat - u) < 1e-6f || b - a < 1e-6f) break;

        // Update bisection bounds using updated _t_
        if (Fhat - u < 0)
            a = t;
        else
            b = t;

        // Perform a Newton step
        t -= (Fhat - u) / fhat;
    }

    // Return the sample position and function value
    if (fval) *fval = fhat;
    if (pdf) *pdf = fhat / F[n - 1];
    return x0 + width * t;
}

Float SampleCatmullRom2D(int size1, int size2, const Float *nodes1,
                         const Float *nodes2, const Float *values,
                         const Float *cdf, Float alpha, Float u, Float *fval = nullptr,
                         Float *pdf = nullptr) {
    // Determine offset and coefficients for the _alpha_ parameter
    int offset;
    Float weights[4];
    if (!CatmullRomWeights(size1, nodes1, alpha, &offset, weights)) return 0;

    // Define a lambda function to interpolate table entries
    auto interpolate = [&](const Float *array, int idx) {
        Float value = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
                value += array[(offset + i) * size2 + idx] * weights[i];
        return value;
    };

    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    Float maximum = interpolate(cdf, size2 - 1);
    u *= maximum;
    int idx =
        FindInterval(size2, [&](int i) { return interpolate(cdf, i) <= u; });

    // Look up node positions and interpolated function values
    Float f0 = interpolate(values, idx), f1 = interpolate(values, idx + 1);
    Float x0 = nodes2[idx], x1 = nodes2[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;

    // Re-scale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;

    // Approximate derivatives using finite differences of the interpolant
    if (idx > 0)
        d0 = width * (f1 - interpolate(values, idx - 1)) /
             (x1 - nodes2[idx - 1]);
    else
        d0 = f1 - f0;
    if (idx + 2 < size2)
        d1 = width * (interpolate(values, idx + 2) - f0) /
             (nodes2[idx + 2] - x0);
    else
        d1 = f1 - f0;

    // Invert definite integral over spline segment and return solution

    // Set initial guess for $t$ by importance sampling a linear interpolant
    Float t;
    if (f0 != f1)
        t = (f0 - std::sqrt(std::max((Float)0, f0 * f0 + 2 * u * (f1 - f0)))) /
            (f0 - f1);
    else
        t = u / f0;
    Float a = 0, b = 1, Fhat, fhat;
    while (true) {
        // Fall back to a bisection step when _t_ is out of bounds
        if (!(t >= a && t <= b)) t = 0.5f * (a + b);

        // Evaluate target function and its derivative in Horner form
        Fhat = t * (f0 +
                    t * (.5f * d0 +
                         t * ((1.f / 3.f) * (-2 * d0 - d1) + f1 - f0 +
                              t * (.25f * (d0 + d1) + .5f * (f0 - f1)))));
        fhat = f0 +
               t * (d0 +
                    t * (-2 * d0 - d1 + 3 * (f1 - f0) +
                         t * (d0 + d1 + 2 * (f0 - f1))));

        // Stop the iteration if converged
        if (std::abs(Fhat - u) < 1e-6f || b - a < 1e-6f) break;

        // Update bisection bounds using updated _t_
        if (Fhat - u < 0)
            a = t;
        else
            b = t;

        // Perform a Newton step
        t -= (Fhat - u) / fhat;
    }

    // Return the sample position and function value
    if (fval) *fval = fhat;
    if (pdf) *pdf = fhat / maximum;
    return x0 + width * t;
}

Float IntegrateCatmullRom(int n, const Float *x, const Float *values,
                          Float *cdf) {
    Float sum = 0;
    cdf[0] = 0;
    for (int i = 0; i < n - 1; ++i) {
        // Look up $x_i$ and function values of spline segment _i_
        Float x0 = x[i], x1 = x[i + 1];
        Float f0 = values[i], f1 = values[i + 1];
        Float width = x1 - x0;

        // Approximate derivatives using finite differences
        Float d0, d1;
        if (i > 0)
            d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
        else
            d0 = f1 - f0;
        if (i + 2 < n)
            d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
        else
            d1 = f1 - f0;

        // Keep a running sum and build a cumulative distribution function
        sum += ((d0 - d1) * (1.f / 12.f) + (f0 + f1) * .5f) * width;
        cdf[i + 1] = sum;
    }
    return sum;
}

Float InvertCatmullRom(int n, const Float *x, const Float *values, Float u) {
    // Stop when _u_ is out of bounds
    if (!(u > values[0]))
        return x[0];
    else if (!(u < values[n - 1]))
        return x[n - 1];

    // Map _u_ to a spline interval by inverting _values_
    int i = FindInterval(n, [&](int i) { return values[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    Float x0 = x[i], x1 = x[i + 1];
    Float f0 = values[i], f1 = values[i + 1];
    Float width = x1 - x0;

    // Approximate derivatives using finite differences
    Float d0, d1;
    if (i > 0)
        d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
    else
        d0 = f1 - f0;
    if (i + 2 < n)
        d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
    else
        d1 = f1 - f0;

    // Invert the spline interpolant using Newton-Bisection
    Float a = 0, b = 1, t = .5f;
    Float Fhat, fhat;
    while (true) {
        // Fall back to a bisection step when _t_ is out of bounds
        if (!(t > a && t < b)) t = 0.5f * (a + b);

        // Compute powers of _t_
        Float t2 = t * t, t3 = t2 * t;

        // Set _Fhat_ using Equation (8.27)
        Fhat = (2 * t3 - 3 * t2 + 1) * f0 + (-2 * t3 + 3 * t2) * f1 +
               (t3 - 2 * t2 + t) * d0 + (t3 - t2) * d1;

        // Set _fhat_ using Equation (not present)
        fhat = (6 * t2 - 6 * t) * f0 + (-6 * t2 + 6 * t) * f1 +
               (3 * t2 - 4 * t + 1) * d0 + (3 * t2 - 2 * t) * d1;

        // Stop the iteration if converged
        if (std::abs(Fhat - u) < 1e-6f || b - a < 1e-6f) break;

        // Update bisection bounds using updated _t_
        if (Fhat - u < 0)
            a = t;
        else
            b = t;

        // Perform a Newton step
        t -= (Fhat - u) / fhat;
    }
    return x0 + t * width;
}

Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r) {
    const int dim = 100;
    Float Ed = 0;
    // Precompute information for dipole integrand

    // Compute reduced scattering coefficients $\sigmaps, \sigmapt$ and albedo
    // $\rhop$
    Float sigmap_s = sigma_s * (1 - g);
    Float sigmap_t = sigma_a + sigmap_s;
    Float rhop = sigmap_s / sigmap_t;

    // Compute non-classical diffusion coefficient $D_\roman{G}$ using
    // Equation (15.24)
    Float D_g = (2 * sigma_a + sigmap_s) / (3 * sigmap_t * sigmap_t);

    // Compute effective transport coefficient $\sigmatr$ based on $D_\roman{G}$
    Float sigmaTr = std::sqrt(sigma_a / D_g);

    // Determine linear extrapolation distance $\depthextrapolation$ using
    // Equation (15.28)
    Float fm1 = FresnelMoment1(eta), fm2 = FresnelMoment2(eta);
    Float ze = -2 * D_g * (1 + 3 * fm2) / (1 - 2 * fm1);

    // Determine exitance scale factors using Equations (15.31) and (15.32)
    Float cPhi = .25f * (1 - 2 * fm1), cE = .5f * (1 - 3 * fm2);
    for (int i = 0; i < dim; ++i) {
        // Sample real point source depth $\depthreal$
        Float zr = -std::log(1 - (i + .5f) / dim) / sigmap_t;

        // Evaluate dipole integrand $E_{\roman{d}}$ at $\depthreal$ and add to
        // _Ed_
        Float zv = -zr + 2 * ze;
        Float dr = std::sqrt(r * r + zr * zr), dv = std::sqrt(r * r + zv * zv);

        // Compute dipole fluence rate $\dipole(r)$ using Equation (15.27)
        Float phiD = 1.f / (4.f * M_PI * D_g) * (std::exp(-sigmaTr * dr) / dr -
                                                 std::exp(-sigmaTr * dv) / dv);

        // Compute dipole vector irradiance $-\N{}\cdot\dipoleE(r)$ using
        // Equation (15.27)
        Float EDn = 1.f / (4 * M_PI) * (zr * (1 + sigmaTr * dr) *
                                  std::exp(-sigmaTr * dr) / (dr * dr * dr) -
                              zv * (1 + sigmaTr * dv) *
                                  std::exp(-sigmaTr * dv) / (dv * dv * dv));

        // Add contribution from dipole for depth $\depthreal$ to _Ed_
        Float E = phiD * cPhi + EDn * cE;
        Float kappa = 1 - std::exp(-2 * sigmap_t * (dr + zr));
        Ed += kappa * rhop * rhop * E;
    }
    return Ed / dim;
}

Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r) {
    // Compute material parameters and minimum $t$ below the critical angle
    Float sigmaT = sigma_a + sigma_s, rho = sigma_s / sigmaT;
    Float tCrit = r * std::sqrt(eta * eta - 1);
    Float Ess = 0;
    const int dim = 100;
    for (int i = 0; i < dim; ++i) {
        // Evaluate single scattering integrand and add to _Ess_
        Float ti = tCrit - std::log(1 - (i + .5f) / dim) / sigmaT;

        // Determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        Float d = std::sqrt(r * r + ti * ti);
        Float cosThetaO = ti / d;

        // Add contribution of single scattering at depth $t$
        Float cosThetaT = 0;
        Ess += rho * std::exp(-sigmaT * (d + tCrit)) / (d * d) *
               PhaseHG(cosThetaO, g) * (1 - fresnelDielectricExt(-cosThetaO, cosThetaT, eta)) *
               std::abs(cosThetaO);
    }
    return Ess / dim;
}

struct BSSRDFTable {
    // BSSRDFTable Public Data
    const int nRhoSamples, nRadiusSamples;
    std::unique_ptr<Float[]> rhoSamples, radiusSamples;
    std::unique_ptr<Float[]> profile;
    std::unique_ptr<Float[]> rhoEff;
    std::unique_ptr<Float[]> profileCDF;

    // BSSRDFTable Public Methods
    Float EvalProfile(int rhoIndex, int radiusIndex) const {
        return profile[rhoIndex * nRadiusSamples + radiusIndex];
    }

    BSSRDFTable(int nRhoSamples, int nRadiusSamples)
        : nRhoSamples(nRhoSamples),
          nRadiusSamples(nRadiusSamples),
          rhoSamples(new Float[nRhoSamples]),
          radiusSamples(new Float[nRadiusSamples]),
          profile(new Float[nRadiusSamples * nRhoSamples]),
          rhoEff(new Float[nRhoSamples]),
          profileCDF(new Float[nRadiusSamples * nRhoSamples]) { }
};

void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t) {
    // Choose radius values of the diffusion profile discretization
    t->radiusSamples[0] = 0;
    t->radiusSamples[1] = 2.5e-3f;
    for (int i = 2; i < t->nRadiusSamples; ++i)
        t->radiusSamples[i] = t->radiusSamples[i - 1] * 1.2f;

    // Choose albedo values of the diffusion profile discretization
    for (int i = 0; i < t->nRhoSamples; ++i)
        t->rhoSamples[i] =
            (1 - std::exp(-8 * i / (Float)(t->nRhoSamples - 1))) /
            (1 - std::exp(-8));
    for (int i = 0; i<t->nRhoSamples; ++i) {
        // Compute the diffusion profile for the _i_th albedo sample

        // Compute scattering profile for chosen albedo $\rho$
        for (int j = 0; j < t->nRadiusSamples; ++j) {
            Float rho = t->rhoSamples[i], r = t->radiusSamples[j];
            t->profile[i * t->nRadiusSamples + j] =
                2 * M_PI * r * (BeamDiffusionSS(rho, 1 - rho, g, eta, r) +
                                BeamDiffusionMS(rho, 1 - rho, g, eta, r));
        }

        // Compute effective albedo $\rho_{\roman{eff}}$ and CDF for importance
        // sampling
        t->rhoEff[i] =
            IntegrateCatmullRom(t->nRadiusSamples, t->radiusSamples.get(),
                                &t->profile[i * t->nRadiusSamples],
                                &t->profileCDF[i * t->nRadiusSamples]);
    }
}

void SubsurfaceFromDiffuse(const BSSRDFTable &t, const Spectrum &rhoEff,
                           const Spectrum &mfp, Spectrum *sigma_a,
                           Spectrum *sigma_s) {
    for (int c = 0; c < Spectrum::dim; ++c) {
        Float rho = InvertCatmullRom(t.nRhoSamples, t.rhoSamples.get(),
                                     t.rhoEff.get(), rhoEff[c]);
        (*sigma_s)[c] = rho / mfp[c];
        (*sigma_a)[c] = (1 - rho) / mfp[c];
    }
}

class RayTracedDipole : public Subsurface {
public:
    RayTracedDipole(const Properties &props)
        : Subsurface(props), table(100, 64) {

        Spectrum sigmaS, sigmaA;
        Spectrum g;
        lookupMaterial(props, sigmaS, sigmaA, g, &m_eta);

        ComputeBeamDiffusionBSSRDF(g.average(), m_eta, &table);

        m_sigmaT = new ConstantSpectrumTexture(sigmaS + sigmaA);
        m_albedo = new ConstantSpectrumTexture(sigmaS / (sigmaS + sigmaA));

        Float roughness = props.getFloat("roughness", 0.f);
        Properties props2(roughness == 0 ? "dielectric" : "roughdielectric");
        props2.setFloat("intIOR", m_eta);
        props2.setFloat("extIOR", 1.f);
        if (roughness > 0)
            props2.setFloat("alpha", roughness);
        m_bsdf = static_cast<BSDF *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(BSDF), props2));
    }

    RayTracedDipole(Stream *stream, InstanceManager *manager)
     : Subsurface(stream, manager), table(100, 64) {
        throw std::runtime_error("Network serialization unsupported!");
    }

    virtual ~RayTracedDipole() {
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        throw std::runtime_error("Network serialization unsupported!");
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture)) && name == "albedo") {
            m_albedo = static_cast<Texture *>(child);
        } else {
            Subsurface::addChild(name, child);
        }
    }

    Spectrum Lo(const Scene *scene, Sampler *sampler, const Intersection &its,
                const Vector &d, int depth) const {
        SamplingIntegrator *integrator = (SamplingIntegrator *) scene->getIntegrator();

        BSDFSamplingRecord bRec(its, sampler, ERadiance);
        Float bsdfPdf;
        Spectrum bsdfWeight = m_bsdf->sample(bRec, bsdfPdf, sampler->next2D());

        if ((bRec.sampledType & BSDF::EReflection) != 0) {
            RadianceQueryRecord query(scene, sampler);
            query.type = RadianceQueryRecord::ERadiance;
            query.depth = depth + 1;
            return bsdfWeight *
                   integrator->Li(
                       RayDifferential(its.p, its.toWorld(bRec.wo), its.time),
                       query);
        }

        // Choose projection axis for BSSRDF sampling
        Vector3 vx, vy, vz;
        Float u1 = sampler->next1D();
        Point2f u2 = sampler->next2D();

        if (u1 < .5f) {
            vx = its.geoFrame.s;
            vy = its.geoFrame.t;
            vz = Vector(its.geoFrame.n);
            u1 *= 2;
        } else if (u1 < .75f) {
            // Prepare for sampling rays with respect to _its.geoFrame.s_
            vx = its.geoFrame.t;
            vy = Vector(its.geoFrame.n);
            vz = its.geoFrame.s;
            u1 = (u1 - .5f) * 4;
        } else {
            // Prepare for sampling rays with respect to _ts_
            vx = Vector(its.geoFrame.n);
            vy = its.geoFrame.s;
            vz = its.geoFrame.t;
            u1 = (u1 - .75f) * 4;
        }

        // Choose spectral channel for BSSRDF sampling
        int ch = math::clamp((int)(u1 * Spectrum::dim), 0, Spectrum::dim - 1);
        u1 = u1 * Spectrum::dim - ch;

        Spectrum sigmaT = m_sigmaT->eval(its);
        Spectrum rho = m_albedo->eval(its);

        // Sample BSSRDF profile in polar coordinates
        Float r = Sample_Sr(sigmaT, rho, ch, u2[0]);
        if (r < 0) return Spectrum(0.f);
        Float phi = 2 * M_PI * u2[1];

        // Compute BSSRDF profile bounds and intersection height
        Float rMax = Sample_Sr(sigmaT, rho, ch, 0.999f);
        if (r >= rMax) return Spectrum(0.f);
        Float l = 2 * std::sqrt(rMax * rMax - r * r);

        // Compute BSSRDF sampling ray segment
        Point pOrigin = its.p + r * (vx * std::cos(phi) + vy * std::sin(phi)) - l * vz * 0.5f;
        Point pTarget = pOrigin + l * vz;

        struct IntersectionChain {
            Intersection its;
            IntersectionChain *next = nullptr;
        };

        IntersectionChain *chain = (IntersectionChain *) alloca(sizeof(IntersectionChain));

        // Accumulate chain of intersections along ray
        IntersectionChain *ptr = chain;
        int nFound = 0, it = 0;
        while (true) {
            Vector d = pTarget - pOrigin;
            Ray r(pOrigin, d, its.time);
            r.mint = Epsilon;
            r.maxt = 1.f;
            if (!scene->rayIntersect(r, ptr->its))
                break;

            pOrigin = ptr->its.p;
            // Append admissible intersection to _IntersectionChain_
            if (ptr->its.shape->getSubsurface() == this) {
                IntersectionChain *next = (IntersectionChain *) alloca(sizeof(IntersectionChain));
                ptr->next = next;
                ptr = next;
                nFound++;
            }
            if (++it > 4)
                break;
        }
        if (nFound == 0) return Spectrum(0.0f);
        int selected = math::clamp((int)(u1 * nFound), 0, nFound - 1);
        while (selected-- > 0) chain = chain->next;
        Intersection its_out = chain->its;

        Float pdf = Pdf_Sp(sigmaT, rho, its, its_out) / nFound;

// Delio: Option to change normalization
#ifdef NO_ENERGY_GAIN
        Spectrum weight = Sp(sigmaT, rho, its, its_out) / (pdf * nFound);
#else
        Spectrum weight = Sp(sigmaT, rho, its, its_out) / pdf;
#endif

        Spectrum rad(0.f);
        {
            DirectSamplingRecord dRec(its_out);
            Spectrum directRadiance = scene->sampleEmitterDirect(
                dRec, sampler->next2D());
            rad = directRadiance * std::max(dot(dRec.d, its_out.shFrame.n), 0.f) * (1.f / M_PI);

            RadianceQueryRecord query(scene, sampler);
            query.newQuery(RadianceQueryRecord::ERadianceNoEmission, nullptr);
            Vector d = its_out.shFrame.toWorld(warp::squareToCosineHemisphere(sampler->next2D()));
            query.depth = depth + 1;
            rad += integrator->Li(RayDifferential(its_out.p, d, its_out.time), query);
        }

        return bsdfWeight * weight * rad * m_eta*m_eta;
    }

    Spectrum Sp(const Spectrum &sigmaT, const Spectrum &rho, const Intersection &pi, const Intersection &po) const {
        return Sr(sigmaT, rho, (po.p - pi.p).length());
    }

    Spectrum Sr(const Spectrum &sigmaT, const Spectrum &rho, Float r) const {
        Spectrum Sr(0.f);
        for (int ch = 0; ch < Spectrum::dim; ++ch) {
            // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
            Float rOptical = r * sigmaT[ch];

            // Compute spline weights to interpolate BSSRDF on channel _ch_
            int rhoOffset, radiusOffset;
            Float rhoWeights[4], radiusWeights[4];
            if (!CatmullRomWeights(table.nRhoSamples, table.rhoSamples.get(),
                                   rho[ch], &rhoOffset, rhoWeights) ||
                !CatmullRomWeights(table.nRadiusSamples, table.radiusSamples.get(),
                                   rOptical, &radiusOffset, radiusWeights))
                continue;

            // Set BSSRDF value _Sr[ch]_ using tensor spline interpolation
            Float sr = 0;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Float weight = rhoWeights[i] * radiusWeights[j];
                    if (weight != 0)
                        sr += weight *
                              table.EvalProfile(rhoOffset + i, radiusOffset + j);
                }
            }

            // Cancel marginal PDF factor from tabulated BSSRDF profile
            if (rOptical != 0) sr /= 2 * M_PI * rOptical;
            Sr[ch] = sr;
        }
        // Transform BSSRDF value into world space units
        Sr *= sigmaT * sigmaT;
        Sr.clampNegative();
        return Sr;
    }


    Float Sample_Sr(const Spectrum &sigmaT, const Spectrum &rho, int ch, Float u) const {
        if (sigmaT[ch] == 0) return -1;
        return SampleCatmullRom2D(table.nRhoSamples, table.nRadiusSamples,
                                  table.rhoSamples.get(), table.radiusSamples.get(),
                                  table.profile.get(), table.profileCDF.get(),
                                  rho[ch], u) /
               sigmaT[ch];
    }

    Float Pdf_Sp(const Spectrum &sigmaT, const Spectrum &rho, const Intersection &pi, const Intersection &po) const {
        // Express $\pti-\pto$ and $\bold{n}_i$ with respect to local coordinates at
        // $\pto$
        Vector dLocal(pi.geoFrame.toLocal(pi.p - po.p));
        Normal nLocal(pi.geoFrame.toLocal(po.geoFrame.n));

        // Compute BSSRDF profile radius under projection along each axis
        Float rProj[3] = {std::sqrt(dLocal.y * dLocal.y + dLocal.z * dLocal.z),
                          std::sqrt(dLocal.z * dLocal.z + dLocal.x * dLocal.x),
                          std::sqrt(dLocal.x * dLocal.x + dLocal.y * dLocal.y)};

        // Return combined probability from all BSSRDF sampling strategies
        Float pdf = 0, axisProb[3] = {.25f, .25f, .5f};
        Float chProb = 1 / (Float)Spectrum::dim;
        for (int axis = 0; axis < 3; ++axis)
            for (int ch = 0; ch < Spectrum::dim; ++ch)
                pdf += Pdf_Sr(sigmaT, rho, ch, rProj[axis]) * std::abs(nLocal[axis]) * chProb *
                       axisProb[axis];
        return pdf;
    }

    Float Pdf_Sr(const Spectrum &sigmaT, const Spectrum &rho, int ch, Float r) const {
        // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
        Float rOptical = r * sigmaT[ch];

        // Compute spline weights to interpolate BSSRDF density on channel _ch_
        int rhoOffset, radiusOffset;
        Float rhoWeights[4], radiusWeights[4];
        if (!CatmullRomWeights(table.nRhoSamples, table.rhoSamples.get(), rho[ch],
                               &rhoOffset, rhoWeights) ||
            !CatmullRomWeights(table.nRadiusSamples, table.radiusSamples.get(),
                               rOptical, &radiusOffset, radiusWeights))
            return 0.f;

        // Return BSSRDF profile density for channel _ch_
        Float sr = 0, rhoEff = 0;
        for (int i = 0; i < 4; ++i) {
            if (rhoWeights[i] == 0) continue;
            rhoEff += table.rhoEff[rhoOffset + i] * rhoWeights[i];
            for (int j = 0; j < 4; ++j) {
                if (radiusWeights[j] == 0) continue;
                sr += table.EvalProfile(rhoOffset + i, radiusOffset + j) *
                      rhoWeights[i] * radiusWeights[j];
            }
        }

        // Cancel marginal PDF factor from tabulated BSSRDF profile
        if (rOptical != 0) sr /= 2 * M_PI * rOptical;
        return std::max((Float)0, sr * sigmaT[ch] * sigmaT[ch] / rhoEff);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
                    const RenderJob *job, int sceneResID, int cameraResID,
                    int _samplerResID) {
        return true;
    }

    MTS_DECLARE_CLASS()

private:
    BSSRDFTable table;
    ref<Texture> m_albedo;
    ref<Texture> m_sigmaT;
    ref<BSDF> m_bsdf;
    Float m_eta;
};

MTS_IMPLEMENT_CLASS_S(RayTracedDipole, false, Subsurface)
MTS_EXPORT_PLUGIN(RayTracedDipole, "Ray-traced dipole model");
MTS_NAMESPACE_END
