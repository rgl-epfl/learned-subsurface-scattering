/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include <fstream>
#include <utility>
#include <vector>
#include <chrono>


#include "vaehelper.h"
#include "vaehelpereigen.h"
#include "vaehelperpt.h"

#ifdef USETF
#include "vaehelpertf.h"
#endif

#include <mitsuba/core/statistics.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include "../medium/materials.h"
#include <mitsuba/hw/basicshader.h>




#include "vaeconfig.h"

#include <fstream>



MTS_NAMESPACE_BEGIN


void logError (int line, const Spectrum& value) {
    if (!(std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2])))
        std::cout << "invalid sample line " << line << " tag " << value[0] << " " << value[1] << " " << value[2] << std::endl;
}


void logError (int line, const Float& value) {
    if (!std::isfinite(value))
        std::cout << "invalid float " << line << std::endl;
}

#define CHECK_VALID( value ) logError(__LINE__, value)



static StatsCounter avgPathLength("PTracer", "Average path length", EAverage);

class VaeScatter : public Subsurface {
public:
    VaeScatter(const Properties &props)
        : Subsurface(props) {

        m_scatter_model = props.getString("vaescatter", "");
        m_use_ptracer = props.getBoolean("bruteforce", false);
        m_use_ptracer_direction = props.getBoolean("useptracerdirection", false);
        m_use_polynomials = props.getBoolean("usepolynomials", false);
        m_use_difftrans = props.getBoolean("difftrans", false);
        m_use_mis = props.getBoolean("usemis", false);
        m_disable_absorption = props.getBoolean("disableabsorption", false);
        m_disable_projection = props.getBoolean("disableprojection", false);
        m_visualize_invalid_samples = props.getBoolean("showinvalidsamples", false);
        m_visualize_absorption = props.getBoolean("visualizeabsorption", false);
        // m_ignoreavgconstraints = props.getBoolean("ignoreavgconstraints", false);
        // m_low_kdtree_threshold = props.getBoolean("lowkdtreethreshold", false);

        Spectrum sigmaS, sigmaA;
        Spectrum g;
        lookupMaterial(props, sigmaS, sigmaA, g, &m_eta);

        if (props.hasProperty("forceG")) {
            g = Spectrum(props.getFloat("forceG"));
            sigmaS = sigmaS / (Spectrum(1) - g);
        }

        Spectrum sigmaT = sigmaS + sigmaA;
        Spectrum albedo = sigmaS / sigmaT;

        m_albedo = props.getSpectrum("albedo", albedo);
        m_albedoTexture = new ConstantSpectrumTexture(m_albedo);

        m_sigmaT = props.getSpectrum("sigmaT", sigmaT);
        m_g = props.getFloat("g", g.average());


        m_medium.albedo = m_albedo;
        m_medium.sigmaT = m_sigmaT;
        m_medium.g = m_g;
        m_medium.eta = m_eta;

        m_use_rgb = m_medium.isRgb();

        m_modelName = props.getString("modelname", "0029_mlpshapefeaturesdeg3");
        m_absModelName = props.getString("absmodelname", "None");
        m_angularModelName = props.getString("angularmodelname", "None");
        m_outputDir = props.getString("outputdir", "/hdd/code/mitsuba-ml/pysrc/outputs/vae3d/");
        m_sssSamples = props.getInteger("sampleCount", 1);
        m_polyOrder = props.getInteger("polyOrder", 3);

        m_polyGlobalConstraintWeight = props.getFloat("polyGlobalConstraintWeight", -1.0f);
        m_polyRegularization = props.getFloat("polyRegularization", -1.0f);
        m_kernelEpsScale = props.getFloat("kernelEpsScale", 1.0f);

        PolyUtils::PolyFitConfig pfConfig;
        m_polyGlobalConstraintWeight = m_polyGlobalConstraintWeight < 0.0f ? pfConfig.globalConstraintWeight : m_polyGlobalConstraintWeight;
        m_polyRegularization = m_polyRegularization < 0.0f ? pfConfig.regularization : m_polyRegularization;


        Float roughness = props.getFloat("roughness", 0.f);
        Properties props2(roughness == 0 ? "dielectric" : "roughdielectric");
        props2.setFloat("intIOR", m_eta);
        props2.setFloat("extIOR", 1.f);
        if (roughness > 0)
            props2.setFloat("alpha", roughness);
        m_bsdf = static_cast<BSDF *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(BSDF), props2));


        // m_vaehelper = new VaeHelperTf();

        if (m_use_ptracer)
            m_vaehelper = new VaeHelperPtracer(m_use_polynomials, m_use_difftrans, m_disable_projection, m_kernelEpsScale);
        else
            m_vaehelper = new VaeHelperEigen(m_kernelEpsScale);
    }

    VaeScatter(Stream *stream, InstanceManager *manager)
     : Subsurface(stream, manager) {
        m_scatter_model = stream->readString();
        configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture)) && name == "albedo") {
            m_albedoTexture = static_cast<Texture *>(child);
            m_albedo = m_albedoTexture->getAverage();
            m_medium.albedo = m_albedo;
            m_use_rgb = m_medium.isRgb();
            std::cout << "m_use_rgb: " << m_use_rgb << std::endl;
        } else {
            Subsurface::addChild(name, child);
        }
    }

    virtual ~VaeScatter() {
        Log(EInfo, "Done rendering VaeScatter, printing stats. Only accurate for SINGLE THREADED execution!");
        std::cout << "numScatterEvaluations: " << numScatterEvaluations << std::endl;
        std::cout << "totalScatterTime: " << totalScatterTime << std::endl;
        std::cout << "totalScatterTime / numScatterEvaluations: " << totalScatterTime / numScatterEvaluations << std::endl;
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Subsurface::serialize(stream, manager);
        stream->writeString(m_scatter_model);
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    inline Vector refract(const Vector &wi, Float cosThetaT, Float eta) const {
        Float scale = -(cosThetaT < 0 ? 1.0f / eta : eta);
        return Vector(scale*wi.x, scale*wi.y, cosThetaT);
    }

    Float FresnelMoment1(Float eta) const {
        Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
            eta5 = eta4 * eta;
        if (eta < 1)
            return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
                2.49277f * eta4 - 0.68441f * eta5;
        else
            return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
                1.27198f * eta4 + 0.12746f * eta5;
    }

    // Factor to correct throughput after refraction on outgoing direction
    Float Sw(const Vector3f &w) const { //TODO: When does this make a difference
        Float c = 1 - 2 * FresnelMoment1(1 / m_eta);
        Float cosThetaT;
        // return (1 - fresnelDielectricExt(Frame::cosTheta(w), cosThetaT, m_eta)) / (c * M_PI);
        return (1 - fresnelDielectricExt(Frame::cosTheta(w), cosThetaT, m_eta)) / (c);
    }

    std::pair<Ray, Intersection> escapeObject(const Ray &ray, const Scene *scene) const {
        Ray escapeRay(ray.o, ray.d, ray.time);
        Intersection rayIts;
        // Continue the ray until we eventually either hit an object or dont hit anything
        Float eps = ShadowEpsilon * (1.0f + std::max(std::abs(ray.o.x), std::max(std::abs(ray.o.y), std::abs(ray.o.z))));
        for (int i = 0; i < m_maxSelfIntersections; ++i) {
            if (scene->rayIntersect(escapeRay, rayIts) && rayIts.hasSubsurface() && Frame::cosTheta(rayIts.wi) < 0) {
                escapeRay = Ray(rayIts.p, ray.d, eps, 100000.0f, ray.time);
            } else {
                break;
            }
        }
        return std::make_pair(escapeRay, rayIts);
    }

    bool isShadowedIgnoringSssObject(Ray ray, const Scene *scene, Float emitterDist) const {
        Float eps = ShadowEpsilon * (1.0f + std::max(std::abs(ray.o.x), std::max(std::abs(ray.o.y), std::abs(ray.o.z))));
        Intersection shadowIts;
        for (int i = 0; i < m_maxSelfIntersections; ++i) {
            if (scene->rayIntersect(ray, shadowIts)) {
                if (shadowIts.hasSubsurface() && Frame::cosTheta(shadowIts.wi) < 0) {
                    ray = Ray(shadowIts.p, ray.d, eps, (1 - eps) * (emitterDist - shadowIts.t), shadowIts.time);
                } else {
                    return true;
                }
            } else {
                return false;
            }
        }
        return true;
    }


    static inline Vector reflect(const Vector &wi) {
        return Vector(-wi.x, -wi.y, wi.z);
    }


    Spectrum LoImpl(const Scene *scene, Sampler *sampler,
            const Intersection &its, const Vector &d, int depth, bool recursiveCall) const {

        its.predAbsorption = Spectrum(0.0f);

        // If we use the multichannel integrator to render, we need to use its child integrator here
        const MonteCarloIntegrator *integrator = dynamic_cast<const MonteCarloIntegrator *>(
            scene->getIntegrator());
        if (!integrator || !integrator->getClass()->derivesFrom(MTS_CLASS(SamplingIntegrator))) {
            integrator = dynamic_cast<const MonteCarloIntegrator *>(scene->getIntegrator()->getSubIntegrator(0));
            if (!integrator || !integrator->getClass()->derivesFrom(MTS_CLASS(SamplingIntegrator)))
                Log(EError, "Requires a sampling-based surface integrator!");
        }
        if (!sRecSingleTls.get()) {
            sRecSingleTls.set(new ScatterSamplingRecordArray(1));
        }
        if (!sRecBatchedTls.get()) {
            sRecBatchedTls.set(new ScatterSamplingRecordArray(m_sssSamples));
        }
        if (!sRecRgbTls.get()) {
            sRecRgbTls.set(new ScatterSamplingRecordArray(3));
        }

        // {
        //     if (!recursiveCall) {
        //         BSDFSamplingRecord bRec(its, sampler, ERadiance);
        //         bRec.typeMask = BSDF::ETransmission;
        //         Float bsdfPdf;
        //         Spectrum bsdfWeight = m_bsdf->sample(bRec, bsdfPdf, sampler->next2D());
        //         auto &sRecSingle2 = sRecSingleTls.get()->data;
        //         auto &sRecBatched2 = sRecBatchedTls.get()->data;
        //         auto &sRecRgb2 = sRecRgbTls.get()->data;
        //         auto &sRec2 = (depth == 1 && m_use_rgb) ? sRecRgb2 : sRecSingle2;
        //         int nSamples2 = sRec2.size();
        //         Vector refractedD = -its.toWorld(bRec.wo);

        //         sampleOutgoingPosition(scene, its, refractedD, sampler, sRec2, nSamples2);
        //         for (int i = 0; i < nSamples2; ++i) {
        //             its.predAbsorption += sRec2[i].throughput / nSamples2 / 3.0;
        //         }
        //     }
        // }

        BSDFSamplingRecord bRec(its, sampler, ERadiance);
        Float bsdfPdf;
        Spectrum bsdfWeight = m_bsdf->sample(bRec, bsdfPdf, sampler->next2D());
        if ((!recursiveCall && ((bRec.sampledType & BSDF::EReflection) != 0)) ||
            (recursiveCall && ((bRec.sampledType & BSDF::ETransmission) != 0))) {
            RadianceQueryRecord query(scene, sampler);
            query.type = RadianceQueryRecord::ERadiance;
            query.depth = depth + 1;
            query.its.sampledColorChannel = its.sampledColorChannel;
            return bsdfWeight * integrator->Li(RayDifferential(its.p, its.toWorld(bRec.wo), its.time), query);
        }

        Vector refractedD = -its.toWorld(bRec.wo);
        // Trace a ray to determine depth through object, then decide whether we should use 0-scattering or multiple scattering
        Ray zeroScatterRay(its.p, -refractedD, its.time);
        Intersection zeroScatterIts;
        if (!scene->rayIntersect(zeroScatterRay, zeroScatterIts)) {
            return Spectrum(0.0f);
        }

        if (sampler->next1D() > 1 - std::exp(-m_sigmaT.average() * zeroScatterIts.t)) { // Ray passes through object without scattering
            RadianceQueryRecord query(scene, sampler);
            // query.newQuery(RadianceQueryRecord::ERadiance | RadianceQueryRecord::EIntersection, its.shape->getExteriorMedium());
            query.newQuery(RadianceQueryRecord::ERadiance, its.shape->getExteriorMedium());
            query.depth = depth + 1;

            if (depth > 10)
                return Spectrum(0.0f);

            if (zeroScatterIts.hasSubsurface())
                return bsdfWeight * LoImpl(scene, sampler, zeroScatterIts, refractedD, depth + 1, true);
            else
                return Spectrum(0.0f);
        }


        ScatterSamplingRecord sRecRgb[3];
        ScatterSamplingRecord sRecSingle[1];

        int nSamples = (depth == 1 && m_use_rgb) ? 3 : 1;
        ScatterSamplingRecord *sRec = nSamples == 3 ? sRecRgb : sRecSingle;
        sampleOutgoingPosition(scene, its, refractedD, sampler, sRec, nSamples);
        Spectrum result(0.0f);
        Spectrum resultNoAbsorption(0.0f);
        int nMissed = 0;
        for (int i = 0; i < nSamples; ++i) {
            // its.predAbsorption += sRec[i].throughput;

            if (!sRec[i].isValid) {
                nMissed++;
                if (m_visualize_invalid_samples) {
                    Spectrum tmp;
                    tmp.fromLinearRGB(100.0f, 0.0f, 0.0f);
                    result += tmp;
                }
                continue;
            }
            Spectrum throughput = bsdfWeight * m_eta * m_eta; // This eta multiplication accounts for outgoing location
            if (!m_disable_absorption)
                throughput *= sRec[i].throughput;

            if (m_use_ptracer_direction) {
                refractedD = sRec[i].outDir;
                RayDifferential indirectRay(sRec[i].p, refractedD, 0.0f);
                Intersection indirectRayIts;
                indirectRayIts.sampledColorChannel = sRec[i].sampledColorChannel;
                scene->rayIntersect(indirectRay, indirectRayIts);
                RadianceQueryRecord query(scene, sampler);
                query.newQuery(
                    RadianceQueryRecord::ERadiance | RadianceQueryRecord::EIntersection,
                    its.shape->getExteriorMedium()); //exiting the current shape
                query.depth = depth + 1;
                query.its = indirectRayIts;
                result += throughput * integrator->Li(indirectRay, query);
                continue;
            }

            if (m_visualize_absorption) {
                result += throughput;
                continue;
            }

            const Vector3f &normal = sRec[i].n;
            const Point3f &outPosition = sRec[i].p;
            { // Sample 'bsdf' for outgoing ray
                Frame frame(normal);
                Vector3f bsdfLocalDir = warp::squareToCosineHemisphere(sampler->next2D());
                Float bsdfPdf = warp::squareToCosineHemispherePdf(bsdfLocalDir);
                Vector3f bsdfRayDir = frame.toWorld(bsdfLocalDir);
                Intersection bsdfRayIts;
                RayDifferential bsdfSampleRay(outPosition, bsdfRayDir, its.time);

                if (m_disable_projection) {
                    // Escape the ray: Find new ray until the ray doesnt intersect the SSS object form inside anymore
                    std::tie(bsdfSampleRay, bsdfRayIts) = escapeObject(bsdfSampleRay, scene);
                }

                scene->rayIntersect(bsdfSampleRay, bsdfRayIts);
                bsdfRayIts.sampledColorChannel = sRec[i].sampledColorChannel;

                RadianceQueryRecord query(scene, sampler);
                query.newQuery(
                    RadianceQueryRecord::ERadiance | RadianceQueryRecord::EIntersection,
                    its.shape->getExteriorMedium()); //exiting the current shape
                query.depth = depth + 1;
                query.its = bsdfRayIts;
                query.extra |= RadianceQueryRecord::EIgnoreFirstBounceEmitted;

                // Evaluate illumination PDF in ray direction
                Spectrum emitted(0.0f);
                Float lumPdf = 0.0f;
                if (bsdfRayIts.isValid()) { // If emitter was hit, we can apply MIS
                    if (bsdfRayIts.isEmitter()) {
                        DirectSamplingRecord dRec;
                        dRec.ref = outPosition;
                        dRec.refN = normal;
                        emitted = bsdfRayIts.Le(-bsdfSampleRay.d);
                        dRec.setQuery(bsdfSampleRay, bsdfRayIts);
                        lumPdf = scene->pdfEmitterDirect(dRec);
                    }
                } else {
                    const Emitter *env = scene->getEnvironmentEmitter();
                    if (env) {
                        DirectSamplingRecord dRec;
                        dRec.refN = Vector(0.0f);
                        emitted = env->evalEnvironment(bsdfSampleRay);
                        if (env->fillDirectSamplingRecord(dRec, bsdfSampleRay))
                            lumPdf = scene->pdfEmitterDirect(dRec);
                    }
                }
                Spectrum indirect = throughput * integrator->Li(bsdfSampleRay, query);
                if (m_use_mis) {
                    Spectrum l = emitted * miWeight(bsdfPdf, lumPdf) * Sw(bsdfLocalDir);
                    result += indirect + throughput * l;
                    resultNoAbsorption += indirect + l;
                } else {
                    Spectrum t = indirect * Sw(bsdfLocalDir);
                    result += t;
                    resultNoAbsorption += t;
                }
            }
            {  // Peform next event estimation and MIS with the path traced result
                DirectSamplingRecord emitterRec(outPosition, its.time);
                Spectrum emitterSampleValue = scene->sampleEmitterDirect(emitterRec, sampler->next2D(), !m_disable_projection); // dont test visibility if we dont project
                if (m_disable_projection) {
                    Ray shadowRay(emitterRec.ref, emitterRec.d, Epsilon, emitterRec.dist * (1 - ShadowEpsilon), emitterRec.time);
                    if (isShadowedIgnoringSssObject(shadowRay, scene, emitterRec.dist)) {
                        emitterSampleValue = Spectrum(0.0f);
                    }
                }

                if (!emitterSampleValue.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(emitterRec.object);
                    const Float bsdfVal = INV_PI * std::max(dot(emitterRec.d, normal), 0.0f);

                    Frame local(normal);
                    if (bsdfVal > 0) {
                        Float bsdfPdf = (emitter->isOnSurface() && emitterRec.measure == ESolidAngle) ? bsdfVal : 0;
                        if (m_use_mis) {
                            Spectrum emitted = emitterSampleValue * bsdfVal * miWeight(emitterRec.pdf, bsdfPdf) * Sw(local.toLocal(emitterRec.d));
                            result += throughput * emitted;
                            resultNoAbsorption += emitted;
                        } else {
                            Spectrum emitted = emitterSampleValue * bsdfVal * Sw(local.toLocal(emitterRec.d));
                            result += throughput * emitted;
                            resultNoAbsorption += emitted;
                        }
                    }
                }
            }
        }
        if (m_use_ptracer) {
            its.predAbsorption = Spectrum(1.0 - (Float) nMissed / (Float) nSamples);
            its.missedProjection = 0.0f;
        } else {
            its.missedProjection = (Float) nMissed / (Float) nSamples;
            // its.predAbsorption /= nSamples;
        }
        its.filled = true;
        if (!m_use_ptracer)
            its.noAbsorption = resultNoAbsorption / nSamples;

        if (result.isNaN()) {
	    Log(EWarn, "VaeScatter encountered NaN value!");
            return Spectrum(0.0f);
        }
        return result / nSamples;
    }



    Spectrum Lo(const Scene *scene, Sampler *sampler,
            const Intersection &its, const Vector &d, int depth) const {

        if (dot(its.shFrame.n, d) < 0) // Discard if we somehow hit the object from inside
            return Spectrum(0.0f);
        return LoImpl(scene, sampler, its, d, depth, false);
    }


    void configure() {
    }

    void sampleOutgoingPosition(const Scene *scene, const Intersection &its, const Vector &d, Sampler *sampler,
        ScatterSamplingRecord *sRec, int nSamples) const {
        Spectrum albedo = m_albedoTexture->eval(its);

        if (nSamples == 3) {
            Vector inDir = -d;
            if (m_use_ptracer) {
                for (int i = 0; i < 3; ++i) {
                    sRec[i] = m_vaehelper->sample(scene, its.p, d, Vector(0.0f, 1.0f, 0.0f), m_sigmaT, albedo, m_g, m_eta, sampler, &its, true, i);
                    Spectrum tmp = sRec[i].throughput;
                    sRec[i].throughput = Spectrum(0.0f);
                    sRec[i].throughput[i] = tmp[i] * 3.0f;
                }
                return;
            }
            for (int i = 0; i < 3; ++i) {
                Vector inDir2 = inDir;
                Vector polyNormal = PolyUtils::adjustRayDirForPolynomialTracing(inDir2, its, 3, PolyUtils::getFitScaleFactor(m_medium, i), i);
                sRec[i] = m_vaehelper->sample(scene, its.p, inDir2, polyNormal, m_sigmaT, albedo, m_g, m_eta, sampler, &its, true, i);
                Spectrum tmp = sRec[i].throughput;
                sRec[i].throughput = Spectrum(0.0f);
                sRec[i].throughput[i] = tmp[i] * 3.0f;
            }
        } else {
            assert(nSamples == 1);
            if (m_use_rgb) {
                // Sample random color channel to sample

                bool randomSampleChannel = its.sampledColorChannel < 0;
                int channel = randomSampleChannel ? int(3 * sampler->next1D()) : its.sampledColorChannel;

                Vector inDir = -d;
                if (m_use_ptracer) {
                    sRec[0] = m_vaehelper->sample(scene, its.p, d, Vector(0.0f, 1.0f, 0.0f),
                                                  m_sigmaT, albedo, m_g, m_eta, sampler, &its, true, channel);
                } else {
                    Vector polyNormal = PolyUtils::adjustRayDirForPolynomialTracing(inDir, its, 3, PolyUtils::getFitScaleFactor(m_medium, channel), channel);
                    sRec[0] = m_vaehelper->sample(scene, its.p, inDir, polyNormal, m_sigmaT, albedo, m_g, m_eta, sampler, &its, true, channel);
                }

                Spectrum tmp = sRec[0].throughput * (randomSampleChannel ? 3.0f : 1.0f);
                sRec[0].throughput = Spectrum(0.0f);
                sRec[0].throughput[channel] = tmp[channel];
                sRec[0].sampledColorChannel = channel;
                return;
            } else {
                Vector inDir = -d;
                if (m_use_ptracer) {
                    sRec[0] = m_vaehelper->sample(scene, its.p, d, Vector(0.0f, 1.0f, 0.0f), m_sigmaT, albedo, m_g, m_eta, sampler, &its, true);
                    return;
                }
                Vector polyNormal = PolyUtils::adjustRayDirForPolynomialTracing(inDir, its, 3, PolyUtils::getFitScaleFactor(m_medium));
                sRec[0] = m_vaehelper->sample(scene, its.p, inDir, polyNormal, m_sigmaT, albedo, m_g, m_eta, sampler, &its, true);
            }
        }
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int cameraResID, int _samplerResID) {
        Log(EInfo, "Preprocessing SSS");
        ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("independent")));
        Log(EInfo, "n shapes %d", m_shapes.size());

        auto preprocStart = std::chrono::steady_clock::now();

        PolyUtils::PolyFitConfig pfConfig;
        pfConfig.regularization = m_polyRegularization;
        pfConfig.globalConstraintWeight = m_polyGlobalConstraintWeight;
        pfConfig.order = m_polyOrder;
        pfConfig.kernelEpsScale = m_kernelEpsScale;
        m_vaehelper->prepare(scene, m_shapes, m_sigmaT, m_albedo, m_g, m_eta, m_modelName,
            m_absModelName, m_angularModelName, m_outputDir, m_sssSamples, pfConfig);
        if (!m_use_ptracer) // if ML model is used, the config is well defined
            m_polyOrder = m_vaehelper->getConfig().polyOrder;


        auto preprocEnd = std::chrono::steady_clock::now();
        auto preprocDiff = preprocEnd - preprocStart;
        double totalSecondsPreproc = std::chrono::duration<double, std::milli> (preprocDiff).count() / 1000.0;
        Log(EInfo, "Preprocessing time: %fs", totalSecondsPreproc);
        return true;
    }

    MTS_DECLARE_CLASS()
private:
    Float m_eta;
    ref<Texture> m_albedoTexture;
    ref<BSDF> m_bsdf;
    // ref<Texture> m_sigmaT;
    Spectrum m_albedo, m_sigmaT;
    float m_g;
    float m_polyGlobalConstraintWeight, m_polyRegularization, m_kernelEpsScale;
    std::string m_scatter_model, m_modelName, m_absModelName, m_angularModelName, m_outputDir;
    int m_sssSamples, m_polyOrder;
    int m_maxSelfIntersections = 10;
    MediumParameters m_medium;
    ref<VaeHelper> m_vaehelper;
    bool m_use_ptracer, m_use_difftrans, m_use_mis, m_use_polynomials, m_disable_projection,
         m_disable_absorption, m_visualize_invalid_samples, m_visualize_absorption,
         m_use_ptracer_direction, m_use_rgb;

    mutable double totalScatterTime = 0.0;
    mutable double numScatterEvaluations = 0.0;

    mutable ThreadLocal<ScatterSamplingRecordArray> sRecSingleTls;
    mutable ThreadLocal<ScatterSamplingRecordArray> sRecBatchedTls;
    mutable ThreadLocal<ScatterSamplingRecordArray> sRecRgbTls;

};


MTS_IMPLEMENT_CLASS_S(VaeScatter, false, Subsurface)
MTS_EXPORT_PLUGIN(VaeScatter, "Vae SSS model");
MTS_NAMESPACE_END











