#!/usr/bin/env python3
import argparse
import datetime
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np

import exrpy
import freeze_model
import utils.experiments
import vae.global_config
from utils.printing import printr


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def create_cluster_render_job(args, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)

    if args.outputdir == vae.global_config.OUTPUT3D:
        args.outputdir = '/scratch/vicini/outputs/vae3d/'

    existing_dirs = os.listdir(tmpdir)
    job_id = int(max(existing_dirs)[-4:]) + 1 if len(existing_dirs) > 0 else 0

    job_folder = os.path.join(tmpdir, 'job{:04}'.format(job_id))
    os.makedirs(job_folder, exist_ok=True)
    print('job_folder: {}'.format(job_folder))

    # Create temporary python file to run render and post-process
    run_script = os.path.join(job_folder, 'render.run')
    print('run_script: {}'.format(run_script))

    if 'ref' in args.techniques:
        args.techniques.remove('ref')  # render other techniques separately

        if len(args.techniques) > 0:
            raise ValueError("Cannot schedule other techniques at the same time as reference rendering")
        max_spp_per_ref_job = 2048
        n_jobs = int(np.ceil(args.refspp // max_spp_per_ref_job))
        sbatch_ids = []
        for i in range(n_jobs):
            render_cmd = f'python3 render_results.py --scenes {" ".join(args.scenes)} ' + \
                f'--spp {args.spp} --refspp {max_spp_per_ref_job} --techniques ref{i:03d} --outputdir {args.outputdir} --reportfailure'

            run_script_i = os.path.join(job_folder, f'render{i}.run')
            with open(run_script_i, 'w') as f:
                f.write('#!/bin/bash -l\n\n')
                f.write('#SBATCH --workdir /home/vicini/code/mitsuba-ml/pysrc\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --ntasks 1\n')
                f.write('#SBATCH --cpus-per-task {}\n'.format(args.ncores))
                f.write('#SBATCH --mem {}\n'.format(2048 * args.ncores))
                f.write('#SBATCH --time 23:59:00\n')
                f.write('source ~/virtualenvs/tensorflow/bin/activate\n')
                f.write('srun {}\n'.format(render_cmd))

            output = subprocess.check_output('sbatch {}'.format(run_script_i), shell=True)
            submission_id = re.search('\d+$', output.strip().decode('utf-8')).group()
            sbatch_ids.append(submission_id)

        merge_script = os.path.join(job_folder, 'merge_renderings.run')
        with open(merge_script, 'w') as f:
            f.write('#!/bin/bash -l\n\n')
            f.write('#SBATCH --workdir /home/vicini/code/mitsuba-ml/pysrc\n')
            f.write('#SBATCH --nodes 1\n')
            f.write('#SBATCH --ntasks 1\n')
            f.write('#SBATCH --cpus-per-task 4\n')
            f.write('#SBATCH --mem {}\n'.format(2048 * 4))
            f.write('#SBATCH --time 00:30:00\n')
            f.write('source ~/virtualenvs/tensorflow/bin/activate\n')
            merge_cmd = f'python3 render_results.py --scenes {" ".join(args.scenes)} --outputdir {args.outputdir} --merge --reportfailure'
            f.write('srun {}\n'.format(merge_cmd))

        subprocess.call(f'sbatch -d {",".join(sbatch_ids)} {merge_script}', shell=True)
        return

    render_cmd = f'python3 render_results.py --scenes {" ".join(args.scenes)} ' + \
        f'--spp {args.spp} --refspp {args.refspp} --techniques {" ".join(args.techniques)} --outputdir {args.outputdir} --reportfailure'

    if args.noabs:
        render_cmd += ' --noabs'
    if args.subdir:
        render_cmd += f' --subdir {args.subdir}'
    if args.time > 0:
        render_cmd += f' --time {args.time}'

    render_cmd += f' --mtsargs "{args.mtsargs}"'
    with open(run_script, 'w') as f:
        f.write('#!/bin/bash -l\n\n')
        f.write('#SBATCH --workdir /home/vicini/code/mitsuba-ml/pysrc\n')
        f.write('#SBATCH --nodes 1\n')
        f.write('#SBATCH --ntasks 1\n')
        f.write('#SBATCH --cpus-per-task {}\n'.format(args.ncores))
        f.write('#SBATCH --mem {}\n'.format(2048 * args.ncores))
        f.write('#SBATCH --time 23:59:00\n')
        f.write('source ~/virtualenvs/tensorflow/bin/activate\n')
        f.write('srun {}\n'.format(render_cmd))

    subprocess.call('sbatch {}'.format(run_script), shell=True)


def get_scene_set(scene, scene_sets, scene_path):
    for ss in scene_sets:
        scene_folder = os.path.join(scene_path, ss, scene)
        if os.path.isdir(scene_folder):
            return ss


def is_float(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def render_timed(args, out_exr, timing_exr, render_cmd):
    probe_spp = 32
    target_secs = 60 * args.time
    spp_str = f'-Dspp={args.spp}'
    render_cmd_timing = render_cmd.replace(spp_str, f'-Dspp={probe_spp}')
    render_cmd_timing = render_cmd_timing.replace(f'-o {out_exr}', f'-o {timing_exr}')
    ret = subprocess.call(render_cmd_timing, shell=True)
    if ret != 0:
        raise Exception("Mitsuba non-zero exit code")
    duration_secs = vae.utils.extract_render_time_from_log(os.path.abspath(timing_exr))
    preproc_secs = vae.utils.extract_preprocess_time_from_log(os.path.abspath(timing_exr))
    sampling_secs = duration_secs - preproc_secs
    new_spp = int(np.ceil(probe_spp * (target_secs - preproc_secs) / sampling_secs))
    render_cmd_final = render_cmd.replace(spp_str, f'-Dspp={new_spp}')
    return subprocess.call(render_cmd_final, shell=True)


def run(args, cmd_string):
    """Main entry point of the training script"""

    techniques = {'ref': """
                <medium name="interior" type="homogeneous">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <phase type="hg">
                        <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    </phase>
                </medium>
                <bsdf type="$bsdf$$MEDIDX$$">
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <float name="alpha" value="$roughness$$MEDIDX$$"/>
                </bsdf>

                """,
                  'vae': """
                <subsurface type="vaescatter">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>
                """,
                  'ptracerref': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    <boolean name="useptracerdirection" value="true"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>
                """,

                  'ptracer': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>
                """,
                  'ptracerpoly': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>
                    <boolean name="usepolynomials" value="true"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>
                """,
                  'ptracerdifftrans': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>
                    <boolean name="difftrans" value="true"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>

                """,
                  'dipole': """
                <subsurface type="dipole">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                </subsurface>

                <bsdf type="plastic">
                    <spectrum name="diffuseReflectance" value="0.0"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                </bsdf>
                """,
                  'frisvad': """
                <subsurface type="dipole">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <boolean name="useFrisvad" value="true"/>
                    <integer name="irrSamples" value="1"/>
                    <float name="sampleMultiplier" value="1"/>
                </subsurface>

                <bsdf type="plastic">
                    <spectrum name="diffuseReflectance" value="0.0"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                </bsdf>
                """,
                  'dipolert': """
                <subsurface type="dipole_rt">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                     <float name="g" value="$meanCosine$$MEDIDX$$"/>
                     <float name="roughness" value="$roughness$$MEDIDX$$"/>
                </subsurface>
                """,
                  'fwddipole': """
                    <subsurface type="fwddip">
                        <string name="intIOR" value="air"/>
                        <string name="extIOR" value="air"/>
                        <string name="zvMode" value="diff"/>
                        <string name="tangentMode" value="Frisvad"/>
                        <boolean name="rejectInternalIncoming" value="true"/>
                        <boolean name="reciprocal" value="false"/>
                        <integer name="numSIR" value="1"/>
                        <boolean name="directSampling" value="true"/>
                        <boolean name="directSamplingMIS" value="true"/>
                        <boolean name="singleChannel" value="false"/>
                        <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                        $$ALBEDOBLOCK$$
                        <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    </subsurface>
                    <bsdf type="$bsdf$$MEDIDX$$">
                        <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                        <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                        <boolean name="noExternalReflection" value="false"/>
                        <float name="alpha" value="$roughness$$MEDIDX$$"/>
                    </bsdf>
                    """,
                  'dirpole': """
                    <subsurface type="dirpole">
                        <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                        <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                        <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                        $$ALBEDOBLOCK$$
                        <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    </subsurface>
                    <bsdf type="$bsdf$$MEDIDX$$">
                        <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                        <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                        <boolean name="noExternalReflection" value="false"/>
                        <float name="alpha" value="$roughness$$MEDIDX$$"/>
                    </bsdf>
                    """

                  }

    techniques_index_matched = {'ref': """
                <medium name="interior" type="homogeneous">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <phase type="hg">
                        <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    </phase>
                </medium>
                <bsdf type="$bsdf$$MEDIDX$$">
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                </bsdf>
                """,
                                'vae': """
                <subsurface type="vaescatter">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>
                """,
                                'ptracerref': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>

                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    <boolean name="useptracerdirection" value="true"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>
                """,
                                'ptracer': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    <boolean name="usemis" value="$useMis"/>
                    <integer name="sampleCount" value="$sssSplitFactor"/>
                    <string name="modelname" value="$modelName"/>
                    <string name="outputdir" value="$outputDir"/>

                </subsurface>
                """,
                                'ptracerpoly': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>
                    <boolean name="usepolynomials" value="true"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>

                    $$COMMONVAEBLOCK$$
                </subsurface>
                """,
                                'ptracerdifftrans': """
                <subsurface type="vaescatter">
                    <boolean name="bruteforce" value="true"/>
                    <boolean name="difftrans" value="true"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    $$COMMONVAEBLOCK$$
                </subsurface>

                """,
                                'dipole': """
                <subsurface type="dipole">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                </subsurface>

                """,
                                'frisvad': """
                <subsurface type="dipole">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                    <boolean name="useFrisvad" value="true"/>
                    <integer name="irrSamples" value="1"/>
                    <float name="sampleMultiplier" value="0.2"/>
                </subsurface>

                """,
                                'dipolert': """
                <subsurface type="dipole_rt">
                    <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                    $$ALBEDOBLOCK$$
                    <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                    <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                     <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    <float name="roughness" value="$roughness$$MEDIDX$$"/>
                </subsurface>""",
                                'fwddipole': """
                    <subsurface type="fwddip">
                        <string name="intIOR" value="air"/>
                        <string name="extIOR" value="air"/>
                        <string name="zvMode" value="diff"/>
                        <string name="tangentMode" value="Frisvad"/>
                        <boolean name="rejectInternalIncoming" value="true"/>
                        <boolean name="reciprocal" value="false"/>
                        <integer name="numSIR" value="1"/>
                        <boolean name="directSampling" value="true"/>
                        <boolean name="directSamplingMIS" value="true"/>
                        <boolean name="singleChannel" value="false"/>
                        <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                        $$ALBEDOBLOCK$$
                        <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    </subsurface>
                    <bsdf type="$bsdf$$MEDIDX$$">
                        <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                        <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                        <boolean name="noExternalReflection" value="false"/>
                        <float name="alpha" value="$roughness$$MEDIDX$$"/>
                    </bsdf>
                    """, 'dirpole': """
                    <subsurface type="dirpole">
                        <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                        <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                        <spectrum name="sigmaT" value="$sigmaT$$MEDIDX$$"/>
                        $$ALBEDOBLOCK$$
                        <float name="g" value="$meanCosine$$MEDIDX$$"/>
                    </subsurface>
                    <bsdf type="$bsdf$$MEDIDX$$">
                        <string name="intIOR" value="$intIOR$$MEDIDX$$"/>
                        <string name="extIOR" value="$extIOR$$MEDIDX$$"/>
                        <boolean name="noExternalReflection" value="false"/>
                        <float name="alpha" value="$roughness$$MEDIDX$$"/>
                    </bsdf>
                    """
                                }

    techniques['volpath'] = techniques['ref']
    techniques['ref'] = techniques['ptracerref']
    techniques_index_matched['volpath'] = techniques_index_matched['ref']
    techniques_index_matched['ref'] = techniques_index_matched['ptracerref']

    common_vae_block = """<boolean name="disableprojection" value="$disableProjection"/>
                    <boolean name="usemis" value="$useMis"/>
                    <integer name="sampleCount" value="$sssSplitFactor"/>
                    <string name="outputdir" value="$outputDir"/>
                    <float name="polyGlobalConstraintWeight" value="$polyGlobalConstraintWeight"/>
                    <float name="polyRegularization" value="$polyRegularization"/>
                    <float name="kernelEpsScale" value="$kernelEpsScale"/>
                    <boolean name="disableabsorption" value="$disableAbsorption"/>
                    <boolean name="showinvalidsamples" value="$showinvalidsamples"/>
                    <boolean name="visualizeabsorption" value="$visualizeabsorption"/>
                    <boolean name="ignoreavgconstraints" value="$ignoreavgconstraints"/>
                    <boolean name="lowkdtreethreshold" value="$lowkdtreethreshold"/>
                    <string name="modelname" value="$modelName"/>
                    <string name="absmodelname" value="$absModelName"/>
                    <string name="angularmodelname" value="$angularModelName"/>
                    """

    for t in techniques:
        techniques[t] = techniques[t].replace('$$COMMONVAEBLOCK$$', common_vae_block)
    for t in techniques_index_matched:
        techniques_index_matched[t] = techniques_index_matched[t].replace('$$COMMONVAEBLOCK$$', common_vae_block)

    preamble_string = """ <default name="modelName" value="None"/>
                          <default name="absModelName" value="None"/>
                          <default name="angularModelName" value="None"/>
                          <default name="outputDir" value="/hdd/code/mitsuba-ml/pysrc/outputs/vae3d/"/>
                          <default name="spp" value="32"/>
                          <default name="sssSplitFactor" value="1"/>
                          <default name="integrator" value="volpath"/>
                          <default name="rrDepth" value="1000"/>
                          <default name="disableAbsorption" value="false"/>
                          <default name="disableProjection" value="false"/>
                          <default name="ignoreavgconstraints" value="false"/>
                          <default name="lowkdtreethreshold" value="false"/>
                          <default name="useMis" value="true"/>
                          <default name="showinvalidsamples" value="false"/>
                          <default name="visualizeabsorption" value="false"/>
                          <default name="seed" value="13"/>
                          <default name="polyRegularization" value="-1.0"/>
                          <default name="polyGlobalConstraintWeight" value="-1.0"/>
                          <default name="kernelEpsScale" value="1.0"/>
    """

    integrator_string = """
        <integrator type="multichannel">
            <integrator type="$integrator">
                <integer name="rrDepth" value="$rrDepth"/>
            </integrator>
            <integrator type="field">
                <string name="field" value="noAbsorption"/>
            </integrator>
            <integrator type="field">
                <string name="field" value="onlyAbsorption"/>
            </integrator>
            <integrator type="field">
                <string name="field" value="missedProjection"/>
            </integrator>
        </integrator>
    """

    filmdetails_string = """
        <string name="pixelFormat" value="rgb, rgb, rgb, luminance"/>
        <string name="channelNames" value="color, noAbsorption, onlyAbsorption, missedProjection"/>
    """

    dipole_integrator_string = """
        <integrator type="$integrator">
            <integer name="rrDepth" value="$rrDepth"/>
            <integer name="maxDepth" value="20"/>
        </integrator>
    """

    dipole_filmdetails_string = """
        <string name="pixelFormat" value="rgb"/>
    """

    # scenes
    # techniques
    # SPP for each scene and its reference
    # scene path

    scene_sets = ['scenes', 'validation', 'generated']

    MAX_MEDIA = 10

    MITSUBA_FWD_DIPOLE_PATH = '../../mitsuba-fwd-dipole/'

    trained_models = sorted(glob.glob(os.path.join(args.outputdir, 'models', '*')))
    trained_models = [os.path.split(s)[-1] for s in trained_models]

    render_dir = os.path.join(args.outputdir, 'render')

    if len(args.scenes) == 0:
        args.scenes = ['spheres', 'cbox', 'dragon', 'lucy', 'simpleobjects2', 'simpleobjects2_hard', 'simpleobjects2_refract',
                       'simpleobjects', 'bunny', 'bunny_dense', 'flat_top_down', 'flat_uniform_lighting',
                       'botijo', 'carter', 'gargoyle', 'heads', 'lucy_uniform', 'thinning']

    if args.merge:

        def load_color(f):
            f = exrpy.InputFile(f)
            if 'color.R' in f.channels:
                return f.get('color')
            else:
                return f.get()

        for s in args.scenes:
            scene_set = get_scene_set(s, scene_sets, args.scenepath)
            ref_folder = os.path.join(render_dir, scene_set, s, 'ref')
            ref_files = glob.glob(ref_folder + '/*/ref*.exr')
            ref_str = ref_folder + '/*/ref*.exr'

            # Load and average all files
            for i, f in enumerate(ref_files):
                if i == 0:
                    mean_img = np.nan_to_num(load_color(f))
                else:
                    mean_img = mean_img * i / (i + 1) + np.nan_to_num(load_color(f)) / (i + 1)
            if len(ref_files) > 0:
                exrpy.write(os.path.join(render_dir, scene_set, s, 'ref', 'ref.exr'), mean_img)
            else:
                raise FileNotFoundError("Could not find any reference renderings!")

        return

    if args.cluster:
        create_cluster_render_job(args, os.path.join(args.outputdir, 'tmp'))
        return

    if args.spp < 8:
        printr('Using a SPP < Batch Size might cause problem with neural network rendering')

    if args.scenefile:
        args.scenes = ['dummy']

    for s in args.scenes:
        scene_set = get_scene_set(s, scene_sets, args.scenepath)
        if scene_set:
            scene_folder = os.path.join(args.scenepath, scene_set, s)
            base_xml = os.path.join(scene_folder, s + '.xml')
        elif args.scenefile:
            base_xml = args.scenefile
            s = os.path.split(os.path.dirname(base_xml))[-1]
            scene_folder = os.path.dirname(base_xml)
        else:
            raise Exception("Error finding scene")

        scene = ET.parse(base_xml).getroot()

        # Add reference to the voxel file
        for t in args.techniques:

            parts = t.split('/')
            t_model, t_abs_model, t_angular_model = None, None, None
            t_model = parts[0]
            if len(parts) > 1:
                t_abs_model = parts[1]
            if len(parts) > 2:
                t_angular_model = parts[2]
            t = '_'.join(parts)

            time_str = str(datetime.datetime.now().time()).replace(':', '_').replace('.', '_')
            cmd_hash = str(hashlib.md5(b'Hello World').hexdigest())
            render_xml = os.path.join(scene_folder, s + '_' + t + '_' + time_str + '_' + cmd_hash + '_tmp_scene.xml')

            model, model_abs, model_angular = None, None, None
            if t_model in trained_models:
                model = t_model
                model_abs = t_abs_model
                model_angular = t_angular_model
                # Check if this graph was already frozen and freeze it if it has not been
                if not os.path.isfile(os.path.join(args.outputdir, 'models', model, 'frozen_model.pb')):
                    freeze_model.simple_freeze(os.path.join(args.outputdir, 'models', model), ['scatter/out_pos_gen'])
                if model_abs and not os.path.isfile(os.path.join(args.outputdir, 'models_abs', model_abs, 'frozen_model.pb')):
                    freeze_model.simple_freeze(os.path.join(args.outputdir, 'models_abs',
                                                            model_abs), ['absorption/absorption'])
                if model_angular and not os.path.isfile(os.path.join(args.outputdir, 'models_angular', model_angular, 'frozen_model.pb')):
                    freeze_model.simple_freeze(os.path.join(args.outputdir, 'models_angular',
                                                            model_angular), ['angular/out_dir_gen'])

            # Detect if the medium is supposed to be index matched
            medium_index_matched = []
            medium_not_textured = []
            for i in range(MAX_MEDIA):
                int_ior = [v for v in scene.findall('default') if v.attrib['name'] == 'intIOR{}'.format(i)]
                ext_ior = [v for v in scene.findall('default') if v.attrib['name'] == 'extIOR{}'.format(i)]
                if len(int_ior) > 0 and len(ext_ior) > 0 and int_ior[0].attrib['value'] == ext_ior[0].attrib['value']:
                    medium_index_matched.append(True)
                else:
                    medium_index_matched.append(False)

                # Check if albedo is textured
                alb = [v for v in scene.findall('default') if v.attrib['name'] == f'albedo{i}']
                if len(alb) > 0:
                    numbers = alb[0].attrib['value'].split(', ')
                if len(alb) > 0 and (is_float(alb[0].attrib['value']) or (len(numbers) == 3 and is_float(numbers[0]))):
                    medium_not_textured.append(True)
                else:
                    medium_not_textured.append(False)

            # Load the XML and place the appropriate medium
            if model is None:
                if t.startswith('ref'):
                    tstr = techniques['ref']
                    tstr_index_matched = techniques_index_matched['ref']
                else:
                    tstr = techniques[t]
                    tstr_index_matched = techniques_index_matched[t]
            else:
                tstr = techniques['vae']
                tstr_index_matched = techniques_index_matched['vae']

            with open(base_xml, "r") as fin:
                with open(render_xml, "w") as fout:
                    for line in fin:
                        new_line = line
                        for i in range(MAX_MEDIA):
                            new_line = new_line.replace('$$PREAMBLE$$', preamble_string)
                            if t == 'dipole' or t == 'dipolert' or t == 'fwddipole' or t == 'frisvad' or t == 'dirpole':
                                new_line = new_line.replace('$$INTEGRATOR$$', dipole_integrator_string)
                                new_line = new_line.replace('$$FILMDETAILS$$', dipole_filmdetails_string)
                            else:
                                new_line = new_line.replace('$$INTEGRATOR$$', integrator_string)
                                new_line = new_line.replace('$$FILMDETAILS$$', filmdetails_string)
                            if medium_index_matched[i]:
                                technique_str = tstr_index_matched.replace('$$MEDIDX$$', str(i))
                            else:
                                technique_str = tstr.replace('$$MEDIDX$$', str(i))

                            if medium_not_textured[i]:
                                technique_str = technique_str.replace(
                                    '$$ALBEDOBLOCK$$', f'<spectrum name="albedo" value="$albedo{i}"/>')
                            else:
                                technique_str = technique_str.replace('$$ALBEDOBLOCK$$', f"""<texture name="albedo" type="bitmap">
                                            <string name="filename" value="$albedo{i}"/>
                                        </texture>""")

                            new_line = new_line.replace(f'$$MEDIUM{i}$$', technique_str)

                        fout.write(new_line)

            # Create path for outputfile

            if args.outfile:
                output_path = os.path.dirname(args.outfile)
                out_exr = args.outfile

            else:
                if args.subdir:
                    output_path = os.path.join(render_dir, scene_set, s, args.subdir)
                else:
                    output_path = os.path.join(render_dir, scene_set, s)
                if args.noabs:
                    output_path = os.path.join(output_path, t + '_noabs')
                    out_exr = os.path.join(output_path, t + '_noabs.exr')
                elif args.vizabs:
                    output_path = os.path.join(output_path, t + '_vizabs')
                    out_exr = os.path.join(output_path, t + '_vizabs.exr')
                elif args.showinvalidsamples:
                    output_path = os.path.join(output_path, t + '_invalidsamples')
                    out_exr = os.path.join(output_path, t + '_invalidsamples.exr')
                elif t != 'ref' and t.startswith('ref'):
                    job_id = get_trailing_number(t)
                    output_path = os.path.join(output_path, 'ref', f'{job_id:03d}')
                    out_exr = os.path.join(output_path, 'ref' + f'{job_id:03d}.exr')
                else:
                    output_path = os.path.join(output_path, t)
                    out_exr = os.path.join(output_path, t + '.exr')

            timing_exr = out_exr.replace('.exr', '_time.exr')

            os.makedirs(output_path, exist_ok=True)

            sss_split_factor = 1
            if args.spp < sss_split_factor:
                sss_split_factor = 1

            if t == 'ref' or t.startswith('ref'):
                # If we render reference, subdivide job into N jobs to be rendered separately
                seed = 123 if t == 'ref' else job_id + 42
                render_cmd = f'mitsuba {args.mtsargs} -Dintegrator=volpath -Dspp={args.refspp} -Dseed={seed} -o {out_exr} {render_xml}'
            elif t == 'dipole' or t == 'dipolert':
                render_cmd = f'mitsuba -p {args.ncores} {args.mtsargs} -DrrDepth=10 -Dintegrator=path -Dspp={args.spp} -o {out_exr} {render_xml}'
            elif t == 'fwddipole' or t == 'dirpole':
                render_cmd = f'mitsuba -p {args.ncores} {args.mtsargs} -DrrDepth=10 -Dintegrator=volpath -Dspp={args.spp} -o {out_exr} {render_xml}'
            else:
                render_cmd = f'mitsuba -p {args.ncores} {args.mtsargs} -DrrDepth=10 -Dintegrator=path -Dspp={args.spp // sss_split_factor} ' + \
                    f'-DsssSplitFactor={sss_split_factor} -DoutputDir={args.outputdir}'
                if model is not None:
                    render_cmd += f' -DmodelName={model}'
                else:
                    render_cmd += f' -DmodelName=None'

                if model_abs is not None:
                    render_cmd += f' -DabsModelName={model_abs}'
                else:
                    render_cmd += f' -DabsModelName=None'

                if model_angular is not None:
                    render_cmd += f' -DangularModelName={model_angular}'
                else:
                    render_cmd += f' -DangularModelName=None'

                if args.noabs:
                    render_cmd += ' -DdisableAbsorption=true'
                if args.vizabs:
                    render_cmd += ' -Dvisualizeabsorption=true'
                if args.showinvalidsamples:
                    render_cmd += ' -Dshowinvalidsamples=true'
                if args.noavgconstraints:
                    render_cmd += ' -Dignoreavgconstraints=true'
                if args.lowkdtreethreshold:
                    render_cmd += ' -Dlowkdtreethreshold=true'
                render_cmd += f' -o {out_exr}'
                render_cmd += ' ' + render_xml

            print('render_cmd: {}'.format(render_cmd))
            if t == 'fwddipole' or t == 'dirpole':
                if not os.path.isdir(MITSUBA_FWD_DIPOLE_PATH):
                    raise Exception(f"Couldnt find forward dipole model at {MITSUBA_FWD_DIPOLE_PATH}")

                render_cmd = './mitsuba-fwd.sh ' + render_cmd[7:]

            if args.time > 0:
                ret = render_timed(args, out_exr, timing_exr, render_cmd)
            else:
                ret = subprocess.call(render_cmd, shell=True)
            if ret != 0:
                raise Exception("Mitsuba non-zero exit code")
            # Move the used scene file to the output dir. Do not do this if we render a reference image (as this computation is split across nodes potentially)
            if not (t == 'ref' or t.startswith('ref')):
                shutil.move(os.path.abspath(render_xml), os.path.join(output_path, t + '.xml'))


def main(args):
    cmd_string = __file__ + ' ' + ' '.join(args)

    parser = argparse.ArgumentParser(description='''Generates training data and/or runs training of SSS predictor''')

    # parser.add_argument('--scenepath', default='/hdd/data/testscenes/nicescenes/')

    parser.add_argument('--scenefile', default=None)
    parser.add_argument('--subdir', default=None)
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--scenepath', default='../scenes/')
    parser.add_argument('--outputdir', default=vae.global_config.OUTPUT3D)
    parser.add_argument('--scenes', default=[], type=str, nargs='*')
    parser.add_argument('--techniques', default=['ref'], type=str, nargs='*')
    parser.add_argument('--spp', default=128, type=int)
    parser.add_argument('--time', default=-1, type=int)
    parser.add_argument('--ncores', default=12, type=int)
    parser.add_argument('--refspp', default=128, type=int)
    parser.add_argument('--cluster', help='If set, rendering will be done as SLURM cluster job', action='store_true')
    parser.add_argument('--merge', help='If set, will try to merge reference renderings', action='store_true')
    parser.add_argument(
        '--noabs', help='If set, absorption prediction will be disabled for debugging', action='store_true')
    parser.add_argument(
        '--vizabs', help='If set, absorption prediction will be visualized', action='store_true')
    parser.add_argument(
        '--showinvalidsamples', help='If set, failed samples contribute a debug color', action='store_true')
    parser.add_argument('--reportfailure', help='If set, exceptions are forwarded to slack (useful for cluster runs)',
                        action='store_true')
    parser.add_argument('--noavgconstraints', action='store_true')
    parser.add_argument('--lowkdtreethreshold', action='store_true')
    parser.add_argument('--mtsargs', default='')
    args = parser.parse_args(args)
    try:
        run(args, cmd_string)
    except:
        print(f"Command: {cmd_string}")
        print(f"Error: {traceback.format_exc()}")

if __name__ == "__main__":
    main(sys.argv[1:])
