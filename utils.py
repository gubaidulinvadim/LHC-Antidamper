import numpy as np
from scipy.constants import c, e, m_p
from PyHEADTAIL.general.printers import SilentPrinter
from PyHEADTAIL.trackers.detuners import Chromaticity
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.trackers.longitudinal_tracking import RFSystems
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import AmplitudeDetuning
from LHC_parameters import *


def get_bunch_monitor(folder, dQr, dQi, n_turns, parameters_dict=None):
    filename = folder+'BM(dQre={0:.3f},dQim={1:.3f})'.format(dQr*1e3, dQi*1e3)
    bunch_monitor = BunchMonitor(
        filename=filename, n_steps=n_turns,
        parameters_dict=parameters_dict, write_buffer_every=1000
    )
    return bunch_monitor


def get_slice_monitor(folder, dQr, dQi, slicer, n_turns_slices=1000):
    filename = folder+'SLM(dQre={0:.3f},dQim={1:.3f})'.format(dQr*1e3, dQi*1e3)
    slice_monitor = SliceMonitor(
        filename=filename, n_steps=n_turns_slices, slicer=slicer, write_buffer_every=1000
    )
    return slice_monitor


def get_particle_monitor(folder, param, n_turns, stride=50, parameters_dict=None):
    filename = folder+'PM(param={0:.3f})'.format(param)
    particle_monitor = ParticleMonitor(
        filename=filename, n_steps=n_turns,
        parameters_dict=parameters_dict, write_buffer_every=1000, stride=stride
    )
    return particle_monitor


N_MACROPARTICLES = int(5.e5)
N_SEGMENTS = 1


def machine_setup(chroma, i_oct=0, dQcoh_x=0, dQcoh_y=0):
    np.random.seed(42)
    chromaticity = chroma
    Q_X, Q_Y = 62.31, 60.32
    (Q_X, Q_Y) = (Q_X-dQcoh_x, Q_Y-dQcoh_y)
    ################################################################
    ###                 LONGITUDINAL MAP SET UP                  ###
    ################################################################
    PHI1 = 0 if (GAMMA**-2-GAMMA_T**-2) < 0 else pi
    PHI2 = pi+PHI1
    long_map = RFSystems(C, [H_RF1, H_RF2, ], [V_RF, V_RF2, ], [PHI1, PHI2, ],
                         ALPHA_0, GAMMA, mass=A*m_p, charge=Z*e)
    ################################################################
    ###             TRANSVERSE MAP SET UP                        ###
    ################################################################
    s = np.arange(0, N_SEGMENTS + 1) * C / N_SEGMENTS
    alpha_x, alpha_y = ALPHA_X_INJ * \
        np.ones(N_SEGMENTS), ALPHA_Y_INJ * np.ones(N_SEGMENTS)
    beta_x, beta_y = BETA_X_INJ * \
        np.ones(N_SEGMENTS), BETA_Y_INJ * np.ones(N_SEGMENTS)
    D_x, D_y = np.zeros(N_SEGMENTS),  np.zeros(N_SEGMENTS)

    ampl_det = AmplitudeDetuning.from_octupole_currents_LHC(
        i_focusing=i_oct, i_defocusing=-i_oct)
    chroma = Chromaticity(Qp_x=[chromaticity*Q_X], Qp_y=[0*Q_Y])
    trans_map = TransverseMap(s, alpha_x, beta_x, D_x,
                              alpha_y, beta_y, D_y, Q_X, Q_Y, [chroma, ampl_det])
    #################################################################
    ###                     BUNCH SET UP                          ###
    #################################################################
    bunch = generators.ParticleGenerator(macroparticlenumber=N_MACROPARTICLES, intensity=INTENSITY, charge=Z*e,
                                         gamma=GAMMA, mass=A*m_p, circumference=C,
                                         distribution_x=generators.gaussian2D(EGEOX), alpha_x=ALPHA_X_INJ, beta_x=BETA_X_INJ,
                                         distribution_y=generators.gaussian2D(EGEOY), alpha_y=ALPHA_Y_INJ, beta_y=BETA_Y_INJ,
                                         limit_n_rms_x=3., limit_n_rms_y=3.,
                                         distribution_z=generators.RF_bucket_distribution(long_map.get_bucket(gamma=GAMMA), sigma_z=SIGMA_Z,
                                                                                          warningprinter=SilentPrinter(), printer=SilentPrinter())
                                         ).generate()

    return long_map, trans_map, bunch
