import sys

import numpy as np
from tqdm import tqdm
from scipy.constants import c, m_e, m_p, c

from PyHEADTAIL.particles import slicing
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.elens.elens import ElectronLens
from LHC_parameters import *
from EL_parameters import *

from utils import *

N_SEGMENTS = 1
N_TURNS = int(1e5)
N_SLICES = 200
N_MACROPARTICLES = int(1e5)
N_TURNS_SLICEMONITOR = 2001


def get_damping_rate_and_phase(r, i):
    phase = 270+np.arctan2(r, i)/(2*np.pi)*360
    tune = np.sqrt(r**2+i**2)
    if tune == 0:
        dampingrate = 0
    else:
        dampingrate = 1/(2*np.pi*tune)
    return dampingrate, phase


def run(r, i, chromaticity, dQmax, folder, i_oct):
    np.random.seed(42)
    dQcoh_x, dQcoh_y = dQmax*MAX_TO_SHIFT_RATIO_PEL, dQmax*MAX_TO_SHIFT_RATIO_PEL
    long_map, trans_map, bunch = machine_setup(
        chromaticity, i_oct=i_oct, dQcoh_x=dQmax, dQcoh_y=dQmax)
    bunch_monitor = get_bunch_monitor(
        folder, r, i, N_TURNS)
    slicer = slicing.UniformBinSlicer(n_slices=N_SLICES, n_sigma_z=4)
    slice_monitor = get_slice_monitor(
        folder, r, i, slicer, N_TURNS_SLICEMONITOR)
    particle_monitor = get_particle_monitor(
        folder, dQmax*1e3, N_TURNS_SLICEMONITOR, stride=int(5e2))

    trans_one_turn = [m for m in trans_map]
    Q_X, Q_Y = 62.31, 60.32
    (Q_X, Q_Y) = (Q_X-MAX_TO_SHIFT_RATIO_PEL *
                  dQmax, Q_Y-MAX_TO_SHIFT_RATIO_PEL*dQmax)

    (beta_x_inj, beta_y_inj) = (C/(2*np.pi)/Q_X, C/(2*np.pi)/Q_Y)
    dampingrate, phase = get_damping_rate_and_phase(r, i)
    antidamper_x = TransverseDamper(dampingrate_x=dampingrate, dampingrate_y=None,
                                    phase=phase, local_beta_function=beta_x_inj, verbose=False)
    antidamper_y = TransverseDamper(dampingrate_x=None, dampingrate_y=dampingrate,
                                    phase=phase, local_beta_function=beta_y_inj, verbose=False)
    L_e = 2
    sigma_e = 4.0*bunch.sigma_x()
    Ue = 10e3
    gamma_e = 1 + Ue * e / (m_e * c**2)
    beta_e = np.sqrt(1 - gamma_e**-2)
    I_a = 17e3
    I_max = A/Z*dQmax*I_a*(sigma_e/bunch.sigma_x())**2*(8*np.pi *
                                                        bunch.epsn_x())/L_e*m_p/m_e*beta_e*bunch.beta/(1+bunch.beta*beta_e)
    z = np.linspace(-4*bunch.sigma_z(), 4*bunch.sigma_z(), N_SLICES)
    sigma_z = bunch.sigma_z()
    I_e = I_max*np.exp(-z**2/(2*sigma_z**2))
    pelens = ElectronLens(L_e, I_e/N_SEGMENTS, sigma_e,
                          sigma_e, beta_e, dist='KV')
    trans_one_turn = []
    for m in trans_map:
        trans_one_turn.append(m)
        trans_one_turn.append(pelens)
    map_ = trans_one_turn + [long_map, antidamper_y]
    track_slices = False
    slice_turn = 0
    for turn in tqdm(range(N_TURNS)):
        for m_ in map_:
            m_.track(bunch)
        bunch_monitor.dump(bunch)
        if not track_slices:
            if bunch.mean_x() > 1e-2 or bunch.mean_y() > 1e-2 or (turn >= N_TURNS - N_TURNS_SLICEMONITOR):
                track_slices = True
        else:
            if slice_turn < N_TURNS_SLICEMONITOR:
                slice_monitor.dump(bunch)
                slice_turn += 1
        if not all(c < 1e20 for c in [bunch.epsn_x(), bunch.epsn_y(), bunch.epsn_z(), bunch.mean_x(), bunch.mean_y(), bunch.mean_z()]):
            print('*** STOPPING SIMULATION: non-finite bunch stats!')
            break
        if r == 0 and i == 0:
            particle_monitor.dump(bunch)


if __name__ == '__main__':
    slurm_array_id = int(sys.argv[1])
    chromaticity = float(sys.argv[2])
    dQmax = float(sys.argv[3])
    folder = sys.argv[4]
    i_oct = float(sys.argv[5])
    points = []
    dQi = dQmax*np.linspace(0, 0.3, 31)
    dQr = dQmax*np.linspace(-.8, .8, 21)
    np.save(folder+'dQi.npy', dQi)
    np.save(folder+'dQr.npy', dQr)
    for i in dQi:
        for r in dQr:
            points.append((r, i))
    run(points[slurm_array_id][0], points[slurm_array_id]
        [1], chromaticity, dQmax, folder, i_oct)
    sys.exit(0)
