import os
import numpy as np
import tensorflow as tf
"""
Tensorflow implementation of Grimme's D3 method (only Becke-Johnson damping is implemented)
Grimme, Stefan, et al. "A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu." The Journal of chemical physics 132.15 (2010): 154104.
"""


# relative filepath to package folder
package_directory = os.path.dirname(os.path.abspath(__file__))

# conversion factors used in grimme d3 code
d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
d3_autoev = 27.21138505  # for converting a.u. to eV

# global parameters (the values here are the standard for HF)
d3_s6 = 1.0000
d3_s8 = 0.9171
d3_a1 = 0.3385
d3_a2 = 2.8830
d3_k1 = 16.000
d3_k2 = 4/3
d3_k3 = -4.000

# tables with reference values
d3_c6ab = np.load(os.path.join(package_directory, "tables", "c6ab.npy"))
d3_r0ab = np.load(os.path.join(package_directory, "tables", "r0ab.npy"))
d3_rcov = np.load(os.path.join(package_directory, "tables", "rcov.npy"))
d3_r2r4 = np.load(os.path.join(package_directory, "tables", "r2r4.npy"))
d3_maxc = 5  # maximum number of coordination complexes


def _smootherstep(r, cutoff):
    '''
    computes a smooth step from 1 to 0 starting at 1 bohr
    before the cutoff
    '''
    cuton = cutoff-1
    x = (cutoff-r)/(cutoff-cuton)
    x2 = x**2
    x3 = x2*x
    x4 = x3*x
    x5 = x4*x
    return tf.where(r <= cuton, tf.ones_like(x), tf.where(r >= cutoff, tf.zeros_like(x), 6*x5-15*x4+10*x3))


def _ncoord(Zi, Zj, r, idx_i, cutoff=None, k1=d3_k1, rcov=d3_rcov):
    '''
    compute coordination numbers by adding an inverse damping function
    '''
    rco = tf.gather(rcov, Zi) + tf.gather(rcov, Zj)
    rr = tf.cast(rco, r.dtype)/r
    damp = 1.0/(1.0+tf.exp(-k1*(rr-1.0)))
    if cutoff is not None:
        damp *= _smootherstep(r, cutoff)
    return tf.segment_sum(damp, idx_i)


def _getc6(ZiZj, nci, ncj, c6ab=d3_c6ab, k3=d3_k3):
    '''
    interpolate c6
    '''
    # gather the relevant entries from the table
    c6ab_ = tf.cast(tf.gather_nd(c6ab, ZiZj), nci.dtype)
    # calculate c6 coefficients
    c6mem = -1.0e99*tf.ones_like(nci)
    r_save = 1.0e99*tf.ones_like(nci)
    rsum = tf.zeros_like(nci)
    csum = tf.zeros_like(nci)
    for i in range(d3_maxc):
        for j in range(d3_maxc):
            cn0 = c6ab_[:, i, j, 0]
            cn1 = c6ab_[:, i, j, 1]
            cn2 = c6ab_[:, i, j, 2]
            r = (cn1-nci)**2 + (cn2-ncj)**2
            r_save = tf.where(r < r_save, r, r_save)
            c6mem = tf.where(r < r_save, cn0, c6mem)
            tmp1 = tf.exp(k3*r)
            rsum += tf.where(cn0 > 0.0, tmp1,     tf.zeros_like(tmp1))
            csum += tf.where(cn0 > 0.0, tmp1*cn0, tf.zeros_like(tmp1))
    c6 = tf.where(rsum > 0.0, csum/rsum, c6mem)
    return c6


def edisp(Z, r, idx_i, idx_j, cutoff=None, r2=None,
          r6=None, r8=None, s6=d3_s6, s8=d3_s8, a1=d3_a1, a2=d3_a2, k1=d3_k1, k2=d3_k2,
          k3=d3_k3, c6ab=d3_c6ab, r0ab=d3_r0ab, rcov=d3_rcov, r2r4=d3_r2r4):
    '''
    compute d3 dispersion energy in Hartree
    r: distance in bohr!
    '''
    # compute all necessary quantities
    Zi = tf.gather(Z, idx_i)
    Zj = tf.gather(Z, idx_j)
    ZiZj = tf.stack([Zi, Zj], axis=1)  # necessary for gatherin
    nc = _ncoord(Zi, Zj, r, idx_i, cutoff=cutoff,
                 rcov=rcov)  # coordination numbers
    nci = tf.gather(nc, idx_i)
    ncj = tf.gather(nc, idx_j)
    c6 = _getc6(ZiZj, nci, ncj, c6ab=c6ab, k3=k3)  # c6 coefficients
    c8 = 3*c6*tf.cast(tf.gather(r2r4, Zi), c6.dtype) * \
        tf.cast(tf.gather(r2r4, Zj), c6.dtype)  # c8 coefficient

    # compute all necessary powers of the distance
    if r2 is None:
        r2 = r**2  # square of distances
    if r6 is None:
        r6 = r2**3
    if r8 is None:
        r8 = r6*r2

    # Becke-Johnson damping, zero-damping introduces spurious repulsion
    # and is therefore not supported/implemented
    tmp = a1*tf.sqrt(c8/c6) + a2
    tmp2 = tmp**2
    tmp6 = tmp2**3
    tmp8 = tmp6*tmp2
    if cutoff is None:
        e6 = 1/(r6+tmp6)
        e8 = 1/(r8+tmp8)
    else:  # apply cutoff
        cut2 = cutoff**2
        cut6 = cut2**3
        cut8 = cut6*cut2
        cut6tmp6 = cut6 + tmp6
        cut8tmp8 = cut8 + tmp8
        e6 = 1/(r6+tmp6) - 1/cut6tmp6 + 6*cut6/cut6tmp6**2 * (r/cutoff-1)
        e8 = 1/(r8+tmp8) - 1/cut8tmp8 + 8*cut8/cut8tmp8**2 * (r/cutoff-1)
        e6 = tf.where(r < cutoff, e6, tf.zeros_like(e6))
        e8 = tf.where(r < cutoff, e8, tf.zeros_like(e8))
    e6 = -0.5*s6*c6*e6
    e8 = -0.5*s8*c8*e8
    return tf.segment_sum(e6+e8, idx_i)
