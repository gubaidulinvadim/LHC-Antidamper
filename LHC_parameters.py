from scipy.constants import e, m_p, c, pi
from numpy import sqrt

E_KIN = 6.5e12
GAMMA = 1 + E_KIN * e / (m_p * c**2)
BETA = sqrt(1 - GAMMA**-2)
C = 26658.883
R = C / (2.*pi)
A, Z = 1, 1
INTENSITY = 1.2e11
ALPHA_0 = [53.83**-2]
GAMMA_T = 1. / sqrt(ALPHA_0)
Q_S = 1.74e-3
OMEGA_REV = 2*pi*c/C
OMEGA_S = Q_S*OMEGA_REV
SIGMA_Z = 0.06
H_RF1, H_RF2 = 35640, 71280
V_RF, V_RF2 = 10e6, 0
EPSN_X, EPSN_Y = 2.5e-6, 2.5e-6
EGEOX = EPSN_X / (BETA * GAMMA)
EGEOY = EPSN_Y / (BETA * GAMMA)
ALPHA_X_INJ, ALPHA_Y_INJ = 0., 0.
BETA_X_INJ, BETA_Y_INJ = 92.7, 93.2
Q_X = 62.28
Q_Y = 60.31
