#include <mitsuba/render/sss_particle_tracer.h>

#include <array>
#include <iostream>

#include <mitsuba/core/statistics.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/sampler.h>

#include <time.h>

MTS_NAMESPACE_BEGIN


inline Vector reflect(const Vector &wi) { return Vector(-wi.x, -wi.y, wi.z); }

/// Refraction in local coordinates
inline Vector refract(const Vector &wi, Float cosThetaT, Float eta) {
    Float scale = -(cosThetaT < 0 ? 1.0f / eta : eta);
    return Vector(scale * wi.x, scale * wi.y, cosThetaT);
}

std::tuple<Ray, Vector3> sampleShape(const Shape *shape, const MediumParameters &medium, Sampler *sampler) {

    for (int i = 0; i < 1000; ++i) { // After 1000 trials we should have sampled something inside the object...
        PositionSamplingRecord pRec(0.0f);
        shape->samplePosition(pRec, sampler->next2D());

        Point3 o       = pRec.p;
        Vector3 normal = pRec.n;

        Vector3 localDir = warp::squareToCosineHemisphere(sampler->next2D());
        // Evaluate fresnel term to decide whether to use direction or not
        Float cosThetaT;
        Float F = fresnelDielectricExt(Frame::cosTheta(localDir), cosThetaT, medium.eta);
        if (sampler->next1D() <= F)
            continue;
        else
            localDir = refract(localDir, cosThetaT, medium.eta);
        // Rotate to world coordinates
        Vector3 sampleD = Frame(normal).toWorld(localDir);

        return std::make_tuple(Ray(o + Epsilon * sampleD, sampleD, 0.0f), normal);
    }
    return std::make_tuple(Ray(), Vector3());
}

std::tuple<Ray, Vector3> sampleShapeFixedInDir(const Shape *shape, Sampler *sampler, const Vector3f inDirLocal) {
    PositionSamplingRecord pRec(0.0f);
    shape->samplePosition(pRec, sampler->next2D());

    Point3 o       = pRec.p;
    Vector3 normal = pRec.n;

    // Rotate to world coordinates
    Vector3 sampleD = Frame(normal).toWorld(inDirLocal);
    return std::make_tuple(Ray(o + Epsilon * sampleD, sampleD, 0.0f), normal);
}

void generateStartingConfiguration(const Vector *inDir, const Shape *shape, Sampler *sampler,
                                   const MediumParameters &medium, const ConstraintKdTree *kdtree,
                                   bool importanceSamplePolynomials, Ray &sampledRay, Vector &normal,
                                   PolyUtils::Polynomial &poly, const Volpath3D::SamplingConfig &samplingConfig) {
    assert(shape);
    if (kdtree) {
        for (int i = 0; i < 10; ++i) {
            if (inDir) {
                std::tie(sampledRay, normal) = sampleShapeFixedInDir(shape, sampler, *inDir);
            } else {
                std::tie(sampledRay, normal) = sampleShape(shape, medium, sampler);
            }
            PolyUtils::PolyFitRecord pfRec;
            pfRec.p         = sampledRay.o;
            pfRec.d         = -sampledRay.d;
            pfRec.n         = normal;
            pfRec.kernelEps = PolyUtils::getKernelEps(medium);
            pfRec.config    = samplingConfig.polyCfg;
            std::vector<Point> pos;
            std::vector<Vector> dirs;
            std::tie(poly, pos, dirs) = PolyUtils::fitPolynomial(pfRec, kdtree);
            if (!importanceSamplePolynomials || Volpath3D::acceptPolynomial(poly, sampler)) {
                if (samplingConfig.polyCfg.hardSurfaceConstraint) {
                    Vector polyNormal = PolyUtils::adjustRayForPolynomialTracing(sampledRay, poly, normal);
                    normal = polyNormal; // Return the polynomial normal
                    return;
                } else {
                    std::cout << "CURRENTLY ONLY POLYS WITH HARD SURFACE CONSTRAINT = TRUE SUPPORTED (in generateStartingConfiguration)\n";
                    exit(1);
                }
            }
        }
        std::cout << "Failed to sample location on the surface\n";
    } else {
        if (inDir) {
            std::tie(sampledRay, normal) = sampleShapeFixedInDir(shape, sampler, *inDir);
        } else {
            std::tie(sampledRay, normal) = sampleShape(shape, medium, sampler);
        }
        return;
    }
}

inline Vector sampleHG(const Vector &d, const Point2 &sample, float g) {
    Float cosTheta;
    if (std::abs(g) < Epsilon) {
        cosTheta = 1 - 2 * sample.x;
    } else {
        Float sqrTerm = (1 - g * g) / (1 - g + 2 * g * sample.x);
        cosTheta      = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
    }
    Float sinTheta = math::safe_sqrt(1.0f - cosTheta * cosTheta);
    Float sinPhi, cosPhi;
    math::sincos(2 * M_PI * sample.y, &sinPhi, &cosPhi);
    return Frame(d).toWorld(Vector(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
}

Volpath3D::PathSampleResult Volpath3D::samplePath(const Scene *scene, const PolyUtils::Polynomial *polynomial, Sampler *sampler,
                                                  Ray ray, const MediumParameters &m,
                                                  const SamplingConfig &samplingConfig) {

    PathSampleResult r;
    r.throughput = Spectrum(1.0f);
    r.status     = PathSampleResult::EStatus::EInvalid;

    Float sigmaT = m.sigmaT.average();
    for (size_t bounces = 0; bounces < samplingConfig.maxBounces; ++bounces) {
        if (samplingConfig.ignoreZeroScatter && bounces == 0) { // Trace the first ray segment such that there is no
                                                                // intersection
            Intersection its;
            if (scene)
                scene->rayIntersect(ray, its);
            else
                its = PolyUtils::intersectPolynomial(ray, *polynomial, samplingConfig.polynomialStepSize, false, bounces);

            if ((scene && !its.isValid()) ||
                (its.isValid() && dot(its.shFrame.n, ray.d) <= 0)) { // discard illegal paths
                // scene && !its.isValid(): For polynomials its common to create "infinite objects where there wont be
                // any intersection"
                r.bounces = bounces;
                return r;
            }
            Float t;
            if (!its.isValid())
                t = -log(1 - sampler->next1D()) / sigmaT;
            else
                t = -log(1 - sampler->next1D() * (1 - std::exp(-sigmaT * its.t))) / sigmaT;

            r.throughput *= m.albedo;
            ray = Ray(ray.o + t * ray.d, sampleHG(ray.d, sampler->next2D(), m.g), 0.0f,
                      std::numeric_limits<float>::infinity(), 0.0f);
        } else {
            Float t  = -log(1 - sampler->next1D()) / sigmaT;
            ray.maxt = t;
            Intersection its;
            if (scene)
                scene->rayIntersect(ray, its);
            else
                its = PolyUtils::intersectPolynomial(ray, *polynomial, samplingConfig.polynomialStepSize, false, bounces);

            if (its.isValid() && dot(its.shFrame.n, ray.d) <= 0) { // break if somehow hit object from outside
                return r;
            }
            if (its.isValid()) { // If we hit the object, potentially exit
                // Check if light gets reflected internally
                Float cosThetaT;
                Float F = fresnelDielectricExt(Frame::cosTheta(its.wi), cosThetaT, m.eta);
                if (sampler->next1D() > F) { // refract
                    // Technically, radiance should be scaled here by 1/eta**2
                    // => Since we dont scale when entering medium, we dont need it
                    r.dOut      = its.shFrame.toWorld(refract(its.wi, cosThetaT, m.eta)); // s.dOut = ray.d;
                    r.pOut      = its.p;
                    r.outNormal = its.shFrame.n;
                    r.bounces   = bounces;
                    r.status    = PathSampleResult::EStatus::EValid;
                    return r;
                } else {
                    ray = Ray(its.p, its.shFrame.toWorld(reflect(its.wi)), 0.0f);
                }
            } else {
                r.throughput *= m.albedo; // At every medium interaction, multiply by the m.albedo of the current sample
                ray = Ray(ray.o + t * ray.d, sampleHG(ray.d, sampler->next2D(), m.g), 0.0f,
                          std::numeric_limits<float>::infinity(), 0.0f);
            }
        }
        // if the throughput is too small, perform russion roulette
        // Float rrProb = 1 - m.albedo.average();
        Float rrProb = 1.0f - r.throughput.max();
        if (samplingConfig.disableRR)
            rrProb = 0.0f;
        if (sampler->next1D() < rrProb) {
            r.status  = PathSampleResult::EStatus::EAbsorbed;
            r.bounces = bounces;
            return r;
        } else {
            r.throughput *= 1 / (1 - rrProb);
        }
    }
    r.status = PathSampleResult::EStatus::EAbsorbed; // count paths exceeding max bounce as absorbed
    return r;
}

std::vector<Volpath3D::TrainingSample>
Volpath3D::samplePathsBatch(const Scene *scene, const Shape *shape, const MediumParameters &medium,
                            const SamplingConfig &samplingConfig, size_t batchSize, size_t nAbsSamples,
                            const Point3f *inPos, const Vector3f *inDir, Sampler *sampler, const PolyUtils::Polynomial *polynomial,
                            const ConstraintKdTree *kdtree, int polyOrder) {

    size_t maxBounces = 10000;
    PolyUtils::Polynomial fitPolynomial;
    fitPolynomial.order          = polyOrder;
    const PolyUtils::Polynomial *tracedPoly = polynomial ? polynomial : &fitPolynomial;

    Ray sampledRay;
    Vector3 normal(1, 0, 0);
    std::vector<float> shCoefficients;
    if (inPos && inDir) {
        sampledRay = Ray(*inPos, *inDir, 0.0f);
    } else {
        generateStartingConfiguration(inDir, shape, sampler, medium, kdtree, samplingConfig.importanceSamplePolynomials,
                                      sampledRay, normal, fitPolynomial, samplingConfig);
    }
    size_t nAbsorbed     = 0;
    size_t nEscaped      = 0;
    size_t nIter         = 0;
    size_t nValidSamples = 0;
    // Sample until we have enough samples to fill the current batch
    std::vector<TrainingSample> batchTrainSamples;
    while (batchTrainSamples.size() < batchSize || nValidSamples < nAbsSamples) {
        // Resample the inpos/indir if many samples are not valid
        if ((nEscaped > 2 * nAbsSamples) || (nAbsorbed > 100 * std::max(batchSize, nAbsSamples))) {
            nIter         = 0;
            nAbsorbed     = 0;
            nEscaped      = 0;
            nValidSamples = 0;
            batchTrainSamples.clear();
            if (inPos && inDir) {
                break; // If inpos and direction are invalid, we can just break and not return any samples
            } else {
                generateStartingConfiguration(inDir, shape, sampler, medium, kdtree,
                                              samplingConfig.importanceSamplePolynomials, sampledRay, normal,
                                              fitPolynomial, samplingConfig);
            }
        }
        // Regenerate random ray direction (for now we average this out!)
        PathSampleResult r = Volpath3D::samplePath(scene, tracedPoly, sampler, sampledRay, medium, samplingConfig);
        switch (r.status) {
            case PathSampleResult::EStatus::EInvalid:
                nEscaped++;
                break;
            case PathSampleResult::EStatus::EAbsorbed:
                nValidSamples++;
                if (nIter < nAbsSamples) // Only compute stats for the nAbsSamples samples
                    nAbsorbed++;
                break;
            case PathSampleResult::EStatus::EValid:
                nValidSamples++;
                TrainingSample s;
                s.pIn        = sampledRay.o;
                s.dIn        = sampledRay.d;
                s.dOut       = r.dOut;
                s.pOut       = r.pOut;
                s.inNormal   = normal;
                s.outNormal  = r.outNormal;
                s.throughput = r.throughput;
                s.albedo     = medium.albedo;
                s.sigmaT     = medium.sigmaT;
                s.g          = medium.g;
                s.ior        = medium.eta;
                s.bounces    = r.bounces;

                if (tracedPoly) {
                    s.shapeCoeffs = tracedPoly->coeffs;
                }

                if (batchTrainSamples.size() < batchSize)
                    batchTrainSamples.push_back(s);
                break;
        }
        nIter++;
    }

    // For all the samples in the current batch, multiply their contribution by the probability of a sample being
    // absorbed
    Float absorptionProb = (Float) nAbsorbed / (Float) nAbsSamples;
    for (size_t k = 0; k < batchTrainSamples.size(); ++k) {
        batchTrainSamples[k].absorptionProb    = absorptionProb;
        batchTrainSamples[k].absorptionProbVar = absorptionProb * (1 - absorptionProb) / (Float)(nAbsSamples - 1);
    }
    return batchTrainSamples;
}

std::vector<Volpath3D::TrainingSample>
Volpath3D::samplePaths(const Scene *scene, const Shape *shape, const std::vector<MediumParameters> &medium,
                       const SamplingConfig &samplingConfig, size_t nSamples, size_t batchSize, size_t nAbsSamples,
                       const Point3f *inPos, const Vector3f *inDir, Sampler *sampler, const PolyUtils::Polynomial *polynomial,
                       const ConstraintKdTree *kdtree) {

    std::vector<TrainingSample> trainingSamples;
    for (size_t i = 0; i < nSamples / batchSize; ++i) {
        // Get the parameters for the current batch
        size_t paramIdx = std::min(i, medium.size() - 1);
        auto batchTrainSamples =
            samplePathsBatch(scene, shape, medium[paramIdx], samplingConfig, batchSize, nAbsSamples, inPos, inDir,
                             sampler, polynomial, kdtree, samplingConfig.polyCfg.order);
        for (auto &b : batchTrainSamples)
            trainingSamples.push_back(b);
    }
    return trainingSamples;
}

MTS_NAMESPACE_END
