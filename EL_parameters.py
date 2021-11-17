from scipy.constants import e, m_e, c, pi
from numpy import sqrt
U_E = 10e3
GAMMA_E = 1 + U_E * e / (m_e * c**2)
BETA_E = sqrt(1 - GAMMA_E**-2)
I_ALFVEN = 17e3
L_E = 2.
