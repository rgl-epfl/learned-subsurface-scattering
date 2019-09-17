#include <mitsuba/render/polynomials.h>

#include <array>
#include <iostream>
#include <time.h>
#include <chrono>

#include <mitsuba/render/sampler.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/util.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <mitsuba/render/sss_particle_tracer.h>


MTS_NAMESPACE_BEGIN



size_t numPolynomialCoefficients(size_t deg) {
    return (deg + 1) * (deg + 2) * (deg + 3) / 6;
}

Eigen::VectorXi derivPermutationEigen(size_t degree, size_t axis) {
    auto numCoeffs = numPolynomialCoefficients(degree);
    Eigen::VectorXi permutation = Eigen::VectorXi::Constant(numCoeffs, -1);
    for (size_t d = 0; d <= degree; ++d) {
        for (size_t i = 0; i <= d; ++i) {
            size_t dx = d - i;
            for (size_t j = 0; j <= i; ++j) {
                size_t dy = d - dx - j;
                size_t dz = d - dx - dy;
                Vector3i deg(dx, dy, dz);
                Vector3i derivDeg = deg;
                derivDeg[axis] -= 1;
                if (derivDeg[0] < 0 || derivDeg[1] < 0 || derivDeg[2] < 0) {
                    continue;
                }
                // For a valid derivative: add entry to matrix
                permutation[PolyUtils::powerToIndex(derivDeg[0], derivDeg[1], derivDeg[2])] = PolyUtils::powerToIndex(dx, dy, dz);
            }
        }
    }
    return permutation;
}


const static int PERMX2[10] = {1,  4,  5,  6, -1, -1, -1, -1, -1, -1};
const static int PERMY2[10] = {2,  5,  7,  8, -1, -1, -1, -1, -1, -1};
const static int PERMZ2[10] =  {3,  6,  8,  9, -1, -1, -1, -1, -1, -1};
const static int PERMX3[20] = {1,  4,  5,  6, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,-1, -1};
const static int PERMY3[20] = {2,  5,  7,  8, 11, 13, 14, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
const static int PERMZ3[20] = {3,  6,  8,  9, 12, 14, 15, 17, 18, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

const int *derivPermutation(size_t degree, size_t axis) {
    switch (axis) {
        case 0:
            return degree == 2 ? PERMX2 : PERMX3;
        case 1:
            return degree == 2 ? PERMY2 : PERMY3;
        case 2:
            return degree == 2 ? PERMZ2 : PERMZ3;
        default:
            return nullptr;
    }
}


float PolyUtils::getKernelEps(const MediumParameters &medium, int channel, Float kernel_multiplier) {
    Float sigmaT    = medium.sigmaT[channel];
    Float albedo    = medium.albedo[channel];
    Float g         = medium.g;
    Float sigmaS    = albedo * sigmaT;
    Float sigmaa    = sigmaT - sigmaS;
    Float sigmaSp   = (1 - g) * sigmaS;
    Float sigmaTp   = sigmaSp + sigmaa;
    Float alphaP    = sigmaSp / sigmaTp;
    Float effAlphaP = Volpath3D::effectiveAlbedo(alphaP);
    Float val       = 0.25f * g + 0.25 * alphaP + 1.0 * effAlphaP;
    // Float d         = alphaP - 0.5f;
    // Float val = 0.474065f * alphaP + 0.578414f * g + 0.000028448817f * std::exp(d * d / 0.025f); //
    // Submission version
    return kernel_multiplier * 4.0f * val * val / (sigmaTp * sigmaTp);
}


template<size_t degree>
float evalPolyImpl(const Point &pos,
 const Point &evalP, Float scaleFactor, bool useLocalDir, const Vector &refDir,
    const std::vector<float> &coeffs) {
    assert(degree <= 3);

    Vector relPos;
    if (useLocalDir) {
        Vector s, t;
        Volpath3D::onbDuff(refDir, s, t);
        Frame local(s, t, refDir);
        relPos = local.toLocal(evalP - pos) * scaleFactor;
    } else {
        relPos = (evalP - pos) * scaleFactor;
    }

    size_t termIdx = 4;
    float value = coeffs[0] + coeffs[1] * relPos.x + coeffs[2] * relPos.y + coeffs[3] * relPos.z;
    float xPowers[4] = {1.0, relPos.x, relPos.x * relPos.x, relPos.x * relPos.x * relPos.x};
    float yPowers[4] = {1.0, relPos.y, relPos.y * relPos.y, relPos.y * relPos.y * relPos.y};
    float zPowers[4] = {1.0, relPos.z, relPos.z * relPos.z, relPos.z * relPos.z * relPos.z};

    for (size_t d = 2; d <= degree; ++d) {
        for (size_t i = 0; i <= d; ++i) {
            size_t dx = d - i;
            for (size_t j = 0; j <= i; ++j) {
                size_t dy = d - dx - j;
                size_t dz = d - dx - dy;
                value += coeffs[termIdx] * xPowers[dx] * yPowers[dy] * zPowers[dz];
                ++termIdx;
            }
        }
    }
    return value;
}



float evalPoly(const Point &pos,
 const Point &evalP, size_t degree,
    Float scaleFactor, bool useLocalDir, const Vector &refDir,
    const std::vector<float> &coeffs) {
    assert(degree <= 3 && degree >= 2);
    if (degree == 3)
        return evalPolyImpl<3>(pos, evalP, scaleFactor, useLocalDir, refDir, coeffs);
    else
        return evalPolyImpl<2>(pos, evalP, scaleFactor, useLocalDir, refDir, coeffs);

}

template<typename T>
std::pair<float, Vector> evalPolyGrad(const Point &pos,
 const Point &evalP, size_t degree,
    const int *permX, const int *permY, const int *permZ,
    Float scaleFactor, bool useLocalDir, const Vector &refDir,
    const T &coeffs) {

    Vector relPos;
    if (useLocalDir) {
        Vector s, t;
        Volpath3D::onbDuff(refDir, s, t);
        Frame local(s, t, refDir);
        relPos = local.toLocal(evalP - pos) * scaleFactor;
    } else {
        relPos = (evalP - pos) * scaleFactor;
    }

    size_t termIdx = 0;
    Vector deriv(0.0f, 0.0f, 0.0f);
    float value = 0.0f;

    float xPowers[4] = {1.0, relPos.x, relPos.x * relPos.x, relPos.x * relPos.x * relPos.x};
    float yPowers[4] = {1.0, relPos.y, relPos.y * relPos.y, relPos.y * relPos.y * relPos.y};
    float zPowers[4] = {1.0, relPos.z, relPos.z * relPos.z, relPos.z * relPos.z * relPos.z};
    for (size_t d = 0; d <= degree; ++d) {
        for (size_t i = 0; i <= d; ++i) {
            size_t dx = d - i;
            for (size_t j = 0; j <= i; ++j) {
                size_t dy = d - dx - j;
                size_t dz = d - dx - dy;
                float t = xPowers[dx] * yPowers[dy] * zPowers[dz];
                value += coeffs[termIdx] * t;

                int pX = permX[termIdx];
                int pY = permY[termIdx];
                int pZ = permZ[termIdx];
                if (pX > 0)
                    deriv.x += (dx + 1) * t * coeffs[pX];
                if (pY > 0)
                    deriv.y += (dy + 1) * t * coeffs[pY];
                if (pZ > 0)
                    deriv.z += (dz + 1) * t * coeffs[pZ];
                ++termIdx;
            }
        }
    }
    return std::make_pair(value, deriv);
}

Vector evalGradient(const Point &pos,
 const Eigen::VectorXf &coeffs,
 const ScatterSamplingRecord &sRec, size_t degree, Float scaleFactor, bool useLocalDir, const Vector &refDir) {
    const int *permX = derivPermutation(degree, 0);
    const int *permY = derivPermutation(degree, 1);
    const int *permZ = derivPermutation(degree, 2);
    float polyValue;
    Vector gradient;
    std::tie(polyValue, gradient) = evalPolyGrad(pos, sRec.p, degree, permX, permY, permZ, scaleFactor, useLocalDir, refDir, coeffs);
    if (useLocalDir) {
        Vector s, t;
        Volpath3D::onbDuff(refDir, s, t);
        Frame local(s, t, refDir);
        gradient = local.toWorld(gradient);
    }
    return gradient;
}


static constexpr int nChooseK(int n, int k) {
    return (k == 0 || n == k) ? 1 : nChooseK(n - 1, k - 1) + nChooseK(n - 1, k);
}

static constexpr int nPolyCoeffs(int polyOrder, bool hardSurfaceConstraint) {
    return hardSurfaceConstraint ? nChooseK(3 + polyOrder, polyOrder) - 1 : nChooseK(3 + polyOrder, polyOrder);
}

template <size_t polyOrder, bool hardSurfaceConstraint = true>
void basisFunFitBuildA(const Point &pos, const Vector &inDir, const std::vector<Point> &evalP,
                       const Eigen::VectorXi &permX, const Eigen::VectorXi &permY, const Eigen::VectorXi &permZ,
                       const Eigen::VectorXf *weights,
                       Eigen::Matrix<float, Eigen::Dynamic, nPolyCoeffs(polyOrder, hardSurfaceConstraint)> &A,
                       const Eigen::VectorXf &weightedB, Float scaleFactor) {

    size_t n = evalP.size();
    Eigen::Matrix<float, Eigen::Dynamic, 3 * (polyOrder + 1)> relPosPow(n, 3 * (polyOrder + 1));
    for (size_t i = 0; i < n; ++i) {
        Vector rel = (evalP[i] - pos) * scaleFactor;
        for (size_t d = 0; d <= polyOrder; ++d) {
            relPosPow(i, d * 3 + 0) = PolyUtils::powi(rel.x, d);
            relPosPow(i, d * 3 + 1) = PolyUtils::powi(rel.y, d);
            relPosPow(i, d * 3 + 2) = PolyUtils::powi(rel.z, d);
        }
    }

    constexpr int nCoeffs = nPolyCoeffs(polyOrder, false);
    Eigen::Matrix<float, Eigen::Dynamic, nCoeffs> fullA =
        Eigen::Matrix<float, Eigen::Dynamic, nCoeffs>::Zero(n * 4, nCoeffs);
    size_t termIdx = 0;
    for (size_t d = 0; d <= polyOrder; ++d) {
        for (size_t i = 0; i <= d; ++i) {
            size_t dx = d - i;
            for (size_t j = 0; j <= i; ++j) {
                size_t dy           = d - dx - j;
                size_t dz           = d - dx - dy;
                Eigen::VectorXf col = relPosPow.col(0 + 3 * dx).array() * relPosPow.col(1 + 3 * dy).array() *
                                      relPosPow.col(2 + 3 * dz).array();
                if (weights) {
                    col = col.array() * (*weights).array();
                }
                fullA.block(0, termIdx, n, 1) = col;
                const int pX = permX[termIdx];
                const int pY = permY[termIdx];
                const int pZ = permZ[termIdx];
                if (pX > 0) {
                    fullA.block(n, pX, n, 1) = (dx + 1) * col;
                }
                if (pY > 0) {
                    fullA.block(2 * n, pY, n, 1) = (dy + 1) * col;
                }
                if (pZ > 0) {
                    fullA.block(3 * n, pZ, n, 1) = (dz + 1) * col;
                }
                ++termIdx;
            }
        }
    }
    A = fullA.block(0, 1, fullA.rows(), fullA.cols() - 1);
}

std::tuple<std::vector<std::vector<Point>>, std::vector<std::vector<Vector>>, std::vector<std::vector<float>>>
PolyUtils::getLocalPoints(const std::vector<Point3> &queryLocations, Float kernelEps, const std::string &kernel, const ConstraintKdTree *kdtree) {
    std::function<Float(Float, Float)> kernelFun;
    kernelFun = gaussianKernel;
    std::vector<std::vector<Point>> allPositionConstraints;
    std::vector<std::vector<Vector>> allNormalConstraints;
    std::vector<std::vector<float>> allConstraintWeights;

    for (size_t k = 0; k < queryLocations.size(); ++k) {
        std::vector<Point> positionConstraints;
        std::vector<Vector> normalConstraints;
        std::vector<Float> sampleWeights;
        std::tie(positionConstraints, normalConstraints, sampleWeights) =
            kdtree->getConstraints(queryLocations[k], kernelEps, kernelFun);

        allPositionConstraints.push_back(positionConstraints);
        allNormalConstraints.push_back(normalConstraints);
        allConstraintWeights.push_back(sampleWeights);
    }
    return std::make_tuple(allPositionConstraints, allNormalConstraints, allConstraintWeights);
}



template<size_t polyOrder, bool hardSurfaceConstraint = true>
std::tuple<PolyUtils::Polynomial,
std::vector<Point>,
std::vector<Vector>>
fitPolynomialsImpl(const PolyUtils::PolyFitRecord &pfRec, const ConstraintKdTree *kdtree) {

    float kernelEps = pfRec.kernelEps;
    std::function<Float(Float, Float)> kernelFun = PolyUtils::gaussianKernel;

    std::vector<Point> positionConstraints;
    std::vector<Vector> normalConstraints;
    std::vector<Float> sampleWeights;
    std::tie(positionConstraints, normalConstraints, sampleWeights) = kdtree->getConstraints(pfRec.p, kernelEps, kernelFun);

    size_t n = positionConstraints.size();
    float invSqrtN = 1.0f / std::sqrt(n);
    Eigen::VectorXf weights(n);

    Vector s, t;
    Volpath3D::onbDuff(pfRec.d, s, t);
    Frame local(s, t, pfRec.d);
    bool useLightSpace = pfRec.config.useLightspace;
    for (size_t i = 0; i < n; ++i) {

        Float d2 = distanceSquared(pfRec.p, positionConstraints[i]);
        Float w;
        if (sampleWeights[i] < 0){ // Special constraint
            // w = std::sqrt(kernelFun(d2, kernelEps * 4.0) * 1.0f) * 4.0f;
            w = pfRec.config.globalConstraintWeight * std::sqrt(1.0f / 32.0f);
        } else {
            w = std::sqrt(kernelFun(d2, kernelEps) * sampleWeights[i]) * invSqrtN;
        }
        weights[i] = w;
        if (useLightSpace) {
            auto localPos = local.toLocal(positionConstraints[i] - pfRec.p);
            positionConstraints[i] = Point(localPos.x, localPos.y, localPos.z);
        }
    }
    Eigen::VectorXf weightedB(4 * n);
    for (size_t i = 0; i < n; ++i) {
        Vector normal = normalConstraints[i];
        if (useLightSpace) {
            normal = local.toLocal(normal);
        }
        weightedB[i + 0 * n] = 0.0f;
        weightedB[i + 1 * n] = normal.x * weights[i];
        weightedB[i + 2 * n] = normal.y * weights[i];
        weightedB[i + 3 * n] = normal.z * weights[i];
    }

    // Evaluate derivatives
    Eigen::VectorXi pX = derivPermutationEigen(polyOrder, 0);
    Eigen::VectorXi pY = derivPermutationEigen(polyOrder, 1);
    Eigen::VectorXi pZ = derivPermutationEigen(polyOrder, 2);

    constexpr size_t nCoeffs = nPolyCoeffs(polyOrder, hardSurfaceConstraint);
    Eigen::Matrix<float, Eigen::Dynamic, nCoeffs> A(4 * n, nCoeffs);
    Eigen::Matrix<float, nCoeffs, nCoeffs> AtA(nCoeffs, nCoeffs);

    // This scale factor seems to lead to a well behaved fit in many different settings
    float fitScaleFactor = PolyUtils::getFitScaleFactor(kernelEps);
    Vector usedRefDir = useLightSpace ? local.toLocal(pfRec.n) : pfRec.n;

    if (useLightSpace) {
        basisFunFitBuildA<polyOrder, hardSurfaceConstraint>(Point(0.0f), usedRefDir, positionConstraints,
                        pX, pY, pZ, &weights, A, weightedB, fitScaleFactor);
    } else {
        basisFunFitBuildA<polyOrder, hardSurfaceConstraint>(pfRec.p, usedRefDir, positionConstraints, pX, pY, pZ, &weights, A, weightedB, fitScaleFactor);
    }
    Eigen::Matrix<float, nCoeffs, 1> Atb = A.transpose() * weightedB;

    Eigen::VectorXf coeffs;
    if (pfRec.config.useSvd) {
        Eigen::MatrixXf ADyn = A;
        Eigen::BDCSVD<Eigen::MatrixXf> svd = ADyn.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXf &sVal = svd.singularValues();
        float eps = 0.01f;
        coeffs = svd.matrixV() * ((sVal.array() > eps).select(sVal.array().inverse(), 0)).matrix().asDiagonal() * svd.matrixU().transpose() * weightedB;
    } else {
        Eigen::MatrixXf reg = Eigen::MatrixXf::Identity(A.cols(), A.cols()) * pfRec.config.regularization;
        reg(0, 0) = 0.0f;
        reg(1, 1) = 0.0f;
        reg(2, 2) = 0.0f;
        AtA = A.transpose() * A + reg;
        coeffs = AtA.ldlt().solve(Atb);
        // coeffs = A.householderQr().solve(weightedB);
    }

    std::vector<Float> coeffsVec(numPolynomialCoefficients(polyOrder));
    if (hardSurfaceConstraint) {
        coeffsVec[0] = 0.0f;
        for (size_t i = 1; i < coeffs.size(); ++i)
            coeffsVec[i] = coeffs[i - 1];
    } else {
        for (size_t i = 0; i < coeffs.size(); ++i)
            coeffsVec[i] = coeffs[i];
    }

    PolyUtils::Polynomial poly;
    poly.coeffs      = coeffsVec;
    poly.refPos      = pfRec.p;
    poly.refDir      = pfRec.d;
    poly.useLocalDir = pfRec.config.useLightspace;
    poly.scaleFactor = PolyUtils::getFitScaleFactor(kernelEps);
    poly.order = polyOrder;
    return std::make_tuple(poly, positionConstraints, normalConstraints);
}

std::tuple<
PolyUtils::Polynomial,
std::vector<Point>,
std::vector<Vector>
 > PolyUtils::fitPolynomial(const PolyFitRecord &pfRec, const ConstraintKdTree *kdtree) {
    if (pfRec.config.hardSurfaceConstraint) {
        if (pfRec.config.order == 2)
            return fitPolynomialsImpl<2, true>(pfRec, kdtree);
        else
            return fitPolynomialsImpl<3, true>(pfRec, kdtree);
    } else {
        std::cout << "UNSUPPORTED: hardSurfaceConstraint = false\n";
    }
}

Vector PolyUtils::adjustRayDirForPolynomialTracing(Vector &inDir, const Intersection &its, int polyOrder, float polyScaleFactor, int channel) {
    const int *pX = derivPermutation(polyOrder, 0);
    const int *pY = derivPermutation(polyOrder, 1);
    const int *pZ = derivPermutation(polyOrder, 2);

    float polyValue;
    Vector polyNormal;
    std::tie(polyValue, polyNormal) = evalPolyGrad(its.p, its.p, polyOrder, pX, pY, pZ,
                                                            polyScaleFactor, false,
                                                            inDir, its.polyCoeffs[channel]);
    polyNormal = normalize(polyNormal);
    Vector rotationAxis = cross(its.shFrame.n, polyNormal);
    if (rotationAxis.length() < 1e-8f) {
        return polyNormal;
    }
    Vector normalizedTarget = normalize(its.shFrame.n);
    float angle = acos(std::max(std::min(dot(polyNormal, normalizedTarget), 1.0f), -1.0f));
    Transform transf = Transform::rotate(rotationAxis, radToDeg(angle));
    inDir = transf(inDir);
    return polyNormal;
}

Vector PolyUtils::adjustRayForPolynomialTracing(Ray &ray, const PolyUtils::Polynomial &polynomial, const Vector &targetNormal) {
    // Transforms a given ray such that it always comes from the upper hemisphere wrt. to the polynomial normal
    const int *pX = derivPermutation(polynomial.order, 0);
    const int *pY = derivPermutation(polynomial.order, 1);
    const int *pZ = derivPermutation(polynomial.order, 2);
    float polyValue;
    Vector polyNormal;
    assert(pX && pY && pZ);
    assert(polynomial.coeffs.size() > 0);

    std::tie(polyValue, polyNormal) = evalPolyGrad(polynomial.refPos, ray.o, polynomial.order, pX, pY, pZ,
                                                            polynomial.scaleFactor, polynomial.useLocalDir,
                                                            polynomial.refDir, polynomial.coeffs);
    polyNormal = normalize(polyNormal);
    Vector rotationAxis = cross(targetNormal, polyNormal);
    if (rotationAxis.length() < 1e-8f)
        return polyNormal;
    Vector normalizedTarget = normalize(targetNormal);
    float angle = acos(std::max(std::min(dot(polyNormal, normalizedTarget), 1.0f), -1.0f));
    Transform transf = Transform::rotate(rotationAxis, radToDeg(angle));
    ray.d = transf(ray.d);
    return polyNormal;
}


bool PolyUtils::adjustRayForPolynomialTracingFull(Ray &ray, const PolyUtils::Polynomial &polynomial, const Vector &targetNormal) {
    const int *pX = derivPermutation(polynomial.order, 0);
    const int *pY = derivPermutation(polynomial.order, 1);
    const int *pZ = derivPermutation(polynomial.order, 2);

    float polyValue;
    Vector polyGradient;
    std::tie(polyValue, polyGradient) = evalPolyGrad(polynomial.refPos, ray.o, polynomial.order, pX, pY, pZ,
                                                              polynomial.scaleFactor, polynomial.useLocalDir,
                                                              polynomial.refDir, polynomial.coeffs);
    polyGradient = normalize(polyGradient);
    Vector rayD = polyValue > 0 ? -polyGradient : polyGradient;

    // 1. Trace ray along
    float polyStepSize = 0.1f; // TODO: Set the stepsize based on the bounding box of the object
    Ray projectionRay(ray.o - rayD * 0.5f * polyStepSize, rayD, 0.0f);
    Intersection polyIts = intersectPolynomial(projectionRay, polynomial, polyStepSize, false);

    if (!polyIts.isValid()) {
        std::cout << "polyValue: " << polyValue << std::endl;
        std::cout << "polyGradient.toString(): " << polyGradient.toString() << std::endl;
        std::cout << "rayD.toString(): " << rayD.toString() << std::endl;
        return false;
    }
    // ray.o = polyIts.p + (polyValue > 0 ? rayD * ShadowEpsilon : -rayD * ShadowEpsilon); // ray now has a new origin
    ray.o = polyIts.p; // ray now has a new origin

    // 2. Evaluate the normal
    Vector polyNormal;
    std::tie(polyValue, polyNormal) = evalPolyGrad(polynomial.refPos, ray.o, polynomial.order, pX, pY, pZ,
                                                              polynomial.scaleFactor, polynomial.useLocalDir,
                                                              polynomial.refDir, polynomial.coeffs);

    polyNormal = normalize(polyNormal);
    Vector rotationAxis = cross(targetNormal, polyNormal);
    if (rotationAxis.length() < 1e-8f) {
        return true;
    }

    Vector normalizedTarget = normalize(targetNormal);
    float angle = acos(std::max(std::min(dot(polyNormal, normalizedTarget), 1.0f), -1.0f));

    Transform transf = Transform::rotate(rotationAxis, radToDeg(angle));
    ray.d = transf(ray.d);
    return true;
}






StatsCounter missedProjection("Projection",
    "Missed projection", EPercentage);




void PolyUtils::projectPointsToSurface(const Scene *scene,
                                    const Point &refPoint, const Vector &refDir,
                                    ScatterSamplingRecord &sRec,
                                    const Eigen::VectorXf &polyCoefficients,
                                    size_t polyOrder, bool useLocalDir, Float scaleFactor, Float kernelEps) {

    if (!sRec.isValid)
        return;

    Vector dir = evalGradient(refPoint, polyCoefficients, sRec, polyOrder, scaleFactor, useLocalDir, refDir);
    dir = normalize(dir);
    // Float dists[5] = {0.0, kernelEps, 2 * kernelEps, 3 * kernelEps, std::numeric_limits<Float>::infinity()};
    // Float dists[3] = {0.0, kernelEps, std::numeric_limits<Float>::infinity()};
    Float dists[2] = {2 * kernelEps, std::numeric_limits<Float>::infinity()};
    missedProjection.incrementBase();
    for (int i = 0; i < 2; ++i) {
        Float maxProjDist = dists[i];
        Ray ray1 = Ray(sRec.p, dir, -Epsilon, maxProjDist, 0);
        Intersection its;
        Point projectedP;
        Vector normal;
        Float pointDist = -1;
        bool itsFoundCurrent = false;
        if (scene->rayIntersect(ray1, its)) {
            projectedP = its.p;
            normal = its.shFrame.n;
            pointDist = its.t;
            itsFoundCurrent = true;
        }
        float maxT = itsFoundCurrent ? its.t : maxProjDist;
        Ray ray2 = Ray(sRec.p, -dir, -Epsilon, maxT, 0);
        Intersection its2;
        if (scene->rayIntersect(ray2, its2)) {
            if (pointDist < 0 || pointDist > its2.t) {
                projectedP = its2.p;
                normal = its2.shFrame.n;
            }
            itsFoundCurrent = true;
        }
        sRec.isValid = itsFoundCurrent;
        if (itsFoundCurrent) {
            sRec.p = projectedP;
            sRec.n = normal;
        }
        if (itsFoundCurrent)
            return;
    }
    ++missedProjection;
}


Intersection PolyUtils::intersectPolynomial(const Ray &ray, const Polynomial &polynomial, Float stepSize, bool printDebug, int nBounce) {

    const int *pX = derivPermutation(polynomial.order, 0);
    const int *pY = derivPermutation(polynomial.order, 1);
    const int *pZ = derivPermutation(polynomial.order, 2);

    // Naive ray marching (for testing)
    float initVal;
    for (int i = 0; i < 50 / stepSize; ++i) {
        Float t = stepSize * i + ray.mint;
        bool maxTReached = false;
        if (t > ray.maxt) {
            // If we stepped too far, check if we maybe intersect on remainder of segment
            t = ray.maxt;
            maxTReached = true;
        }

        float polyValue = evalPoly(polynomial.refPos, ray.o + ray.d * t, polynomial.order,
                            polynomial.scaleFactor, polynomial.useLocalDir, polynomial.refDir, polynomial.coeffs);
        if (i == 0) {
            initVal = polyValue;
            // if (initVal == 0.0f) {
            //     std::cout << "Self intersection (increase ray epsilon?), bounce: " << nBounce << std::endl;
            // }
        } else {
            if (polyValue * initVal <= 0) {
                // We observed a sign change between two variables
                // Assuming we are in a regime where the derivative makes sense, perform newton-bisection iterations to find the actual intersection location


                // [a, b] = interval of potential solution
                float a = stepSize * (i - 1) + ray.mint;
                float b = stepSize * i + ray.mint;
                if (printDebug) {
                    std::cout << "i - 1: " << i - 1 << std::endl;
                    std::cout << "i: " << i << std::endl;
                    std::cout << "a: " << (ray.o + ray.d * a).toString() << std::endl;
                    std::cout << "b: " << (ray.o + ray.d * b).toString() << std::endl;
                }
                t = 0.5 * (a + b);
                Vector deriv;
                for (int j = 0; j < 5; ++j) {
                    if (!(t >= a && t <= b)) // if t is out of bounds, go back to bisection method
                        t = 0.5f * (a + b);

                    std::tie(polyValue, deriv) = evalPolyGrad(polynomial.refPos, ray.o + ray.d * t, polynomial.order, pX, pY, pZ,
                                                              polynomial.scaleFactor, polynomial.useLocalDir,
                                                              polynomial.refDir, polynomial.coeffs);
                    if ((initVal < 0 && polyValue < 0) || (initVal > 0 && polyValue > 0))
                        a = t;
                    else
                        b = t;

                    t = t - polyValue / (dot(deriv, ray.d) * polynomial.scaleFactor); // update t with newton
                    if (std::abs(polyValue) < 1e-5)
                        break;
                }
                if (t > ray.maxt) {
                    break;
                }
                // deriv = Vector(1,0,0);
                Intersection its;
                its.p = ray.o + ray.d * t;
                its.shFrame = Frame(normalize(deriv));
                its.geoFrame = its.shFrame;
                its.t = t;
                its.wi = its.shFrame.toLocal(-ray.d);
                return its;
            }
        }
        if (maxTReached)
            break;
    }
    Intersection its;
    its.t = std::numeric_limits<Float>::infinity();
    return its;
}


std::tuple<Point, Vector, size_t> ConstraintKdTree::avgValues(TreeNode &node, TreeNode::IndexType index) {
    Point rp(0.0f), lp(0.0f);
    Vector rn(0.0f), ln(0.0f);
    size_t rsamples = 0, lsamples = 0;
    if (m_tree.hasRightChild(index)) {
        auto rightIdx = node.getRightIndex(index);
        std::tie(rp, rn, rsamples) = avgValues(m_tree[rightIdx], rightIdx);
    }

    //TODO: Does the node always have a left child?
    if (!node.isLeaf()) {
        auto leftIdx = node.getLeftIndex(index);
        std::tie(lp, ln, lsamples) = avgValues(m_tree[leftIdx], leftIdx);
    }

    size_t totalSamples = 1 + rsamples + lsamples;
    Point avgP = (node.getData().p + ((float) rsamples) * rp + ((float) lsamples) * lp) / ((float) totalSamples);
    Vector avgN = (node.getData().n + ((float) rsamples) * rn + ((float) lsamples) * ln) / ((float) totalSamples);
    if (!avgN.isZero()) {
        avgN = normalize(avgN);
    }

    // Set the current nodes values
    node.getData().avgP = avgP;
    node.getData().avgN = avgN;
    node.getData().sampleCount = totalSamples;

    return std::make_tuple(avgP, avgN, totalSamples);
}


std::pair<Float, Float> ConstraintKdTree::getMinMaxDistanceSquared(const Point &p, const AABB &bounds) const {

    // Projection of p onto AABB
    Float minDist2 = bounds.squaredDistanceTo(p);
    Float maxDist2 = 0.0f;
    for (size_t i = 0; i < 8; ++i) {
        maxDist2 = std::max(maxDist2, distanceSquared(bounds.getCorner(i), p));
    }
    return std::make_pair(minDist2, maxDist2);
}


void ConstraintKdTree::getConstraints(const Point &p, TreeNode node, TreeNode::IndexType index, const AABB &aabb,
                                          std::vector<Point> &points, std::vector<Vector> &normals,
                                          std::vector<Float> &sampleWeights, Float kernelEps,
                                          const std::function<Float(Float, Float)> &kernel) const {
    Float distance2Threshold = 9.0 * kernelEps;

    if (aabb.squaredDistanceTo(p) > distance2Threshold) {
        return;
    }

    if (distanceSquared(node.getData().p, p) < distance2Threshold) {
        points.push_back(node.getData().p);
        normals.push_back(node.getData().n);
        sampleWeights.push_back(1.0f);
    }

    if (node.isLeaf())
        return;

    // Compute bounding box of child nodes
    size_t ax      = node.getAxis();
    Point leftMin  = aabb.min;
    Point leftMax  = aabb.max;
    leftMax[ax]    = node.getPosition()[ax];
    Point rightMin = aabb.min;
    Point rightMax = aabb.max;
    rightMin[ax]   = node.getPosition()[ax];
    AABB leftBounds(leftMin, leftMax);
    AABB rightBounds(rightMin, rightMax);
    if (m_tree.hasRightChild(index)) {
        auto rightIdx = node.getRightIndex(index);
        getConstraints(p, m_tree[rightIdx], rightIdx, rightBounds, points, normals, sampleWeights, kernelEps,
                           kernel);
    }
    // Node always has left child by construction
    auto leftIdx = node.getLeftIndex(index);
    getConstraints(p, m_tree[leftIdx], leftIdx, leftBounds, points, normals, sampleWeights, kernelEps, kernel);
}

MTS_NAMESPACE_END


