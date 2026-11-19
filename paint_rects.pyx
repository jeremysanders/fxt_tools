# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

# paint rectangles (for exposure map)
#
# Each input quad (a warped input pixel) is distributed over the output pixels
# it overlaps, weighted by overlap area.
#
# Speedups vs. the generic Sutherland-Hodgman version:
#  * output pixels are axis-aligned, so clipping is done with four axis-aligned
#    half-plane clips (no slopes, no division by (x2-x1), no cross products)
#  * the quad is clipped to a column slab once per output column, then each
#    row only needs a clip of that (small) slab polygon
#  * row range is tightened per column to the slab polygon's y-extent
#  * tighter bounding-box loop limits (no zero-area border pixels)
#  * no struct-by-value copying; everything on stack arrays, runs without the GIL
#  * fast path when a quad lies entirely inside one output pixel
#  * quads with value 0 are skipped

import numpy as np
from libc.math cimport floor, fabs

DEF MAXV = 16

# Clip polygon (ia, ib) of n points against the half-plane
#   sgn * (a - bound) >= 0
# where a is the "clipping" coordinate array and b the other one.
# Output goes to (oa, ob); returns the new vertex count.
cdef inline int clip_axis(const double* ia, const double* ib, int n,
                          double* oa, double* ob,
                          double bound, double sgn) noexcept nogil:
    cdef int i, m = 0
    cdef double pa, pb, ca, cb, dp, dc, t
    pa = ia[n-1]
    pb = ib[n-1]
    dp = sgn * (pa - bound)
    for i in range(n):
        ca = ia[i]
        cb = ib[i]
        dc = sgn * (ca - bound)
        if (dc >= 0) != (dp >= 0):
            t = dp / (dp - dc)
            oa[m] = bound
            ob[m] = pb + t * (cb - pb)
            m += 1
        if dc >= 0:
            oa[m] = ca
            ob[m] = cb
            m += 1
        pa = ca
        pb = cb
        dp = dc
    return m

# absolute polygon area (shoelace)
cdef inline double poly_area(const double* x, const double* y, int n) noexcept nogil:
    cdef int i, j = n - 1
    cdef double a = 0.
    for i in range(n):
        a += (x[j] + x[i]) * (y[j] - y[i])
        j = i
    return fabs(0.5 * a)


def paint_rects(double[:,::1] qye, double[:,::1] qxe, double[:,::1] qva, double[:,::1] oimg):
    """qye = yedges, qxe = xedges, qva = values, oimg = output image."""

    cdef int oyw = oimg.shape[0]
    cdef int oxw = oimg.shape[1]
    cdef int qyw = qva.shape[0]
    cdef int qxw = qva.shape[1]

    assert qye.shape[0] == qyw+1 and qye.shape[1] == qxw+1
    assert qxe.shape[0] == qyw+1 and qxe.shape[1] == qxw+1

    cdef int y, x, oy, ox, n, n1, n2, n3, n4
    cdef int xlo, xhi, ylo, yhi, oylo, oyhi
    cdef double v, minx, maxx, miny, maxy, sminy, smaxy, fx, fy

    # polygon buffers: quad, and ping-pong buffers for x/y coordinates
    cdef double qx[MAXV]
    cdef double qy[MAXV]
    cdef double sx[MAXV]   # slab polygon (after x clipping)
    cdef double sy[MAXV]
    cdef double tx[MAXV]
    cdef double ty[MAXV]
    cdef double ux[MAXV]
    cdef double uy[MAXV]

    with nogil:
        for y in range(qyw):
            for x in range(qxw):
                v = qva[y, x]
                if v == 0:
                    continue

                qx[0] = qxe[y,   x  ]
                qy[0] = qye[y,   x  ]
                qx[1] = qxe[y+1, x  ]
                qy[1] = qye[y+1, x  ]
                qx[2] = qxe[y+1, x+1]
                qy[2] = qye[y+1, x+1]
                qx[3] = qxe[y,   x+1]
                qy[3] = qye[y,   x+1]

                minx = min(min(qx[0], qx[1]), min(qx[2], qx[3]))
                maxx = max(max(qx[0], qx[1]), max(qx[2], qx[3]))
                miny = min(min(qy[0], qy[1]), min(qy[2], qy[3]))
                maxy = max(max(qy[0], qy[1]), max(qy[2], qy[3]))

                # output pixel i covers [i-0.5, i+0.5]; skip quads fully outside
                # (also rejects NaN, since comparisons with NaN are false)
                if not (maxx >= -0.5 and minx <= oxw - 0.5 and
                        maxy >= -0.5 and miny <= oyw - 0.5):
                    continue

                # range of output pixels touched (as doubles first: avoids int overflow)
                fx = floor(minx + 0.5)
                xlo = <int>(fx if fx > 0 else 0)
                fx = floor(maxx + 0.5)
                xhi = <int>(fx if fx < oxw - 1 else oxw - 1)
                fy = floor(miny + 0.5)
                ylo = <int>(fy if fy > 0 else 0)
                fy = floor(maxy + 0.5)
                yhi = <int>(fy if fy < oyw - 1 else oyw - 1)

                if xlo > xhi or ylo > yhi:
                    continue

                # fast path: quad entirely inside a single output pixel
                if (xlo == xhi and ylo == yhi and
                        minx >= xlo - 0.5 and maxx <= xlo + 0.5 and
                        miny >= ylo - 0.5 and maxy <= ylo + 0.5):
                    oimg[ylo, xlo] += poly_area(qx, qy, 4) * v
                    continue

                for ox in range(xlo, xhi + 1):
                    # clip to column slab ox-0.5 <= x <= ox+0.5
                    n1 = clip_axis(qx, qy, 4, tx, ty, ox - 0.5, 1.0)
                    if n1 < 3:
                        continue
                    n = clip_axis(tx, ty, n1, sx, sy, ox + 0.5, -1.0)
                    if n < 3:
                        continue

                    # y-extent of the slab polygon limits the rows to visit
                    sminy = sy[0]
                    smaxy = sy[0]
                    for n2 in range(1, n):
                        if sy[n2] < sminy:
                            sminy = sy[n2]
                        if sy[n2] > smaxy:
                            smaxy = sy[n2]
                    fy = floor(sminy + 0.5)
                    oylo = <int>(fy if fy > ylo else ylo)
                    fy = floor(smaxy + 0.5)
                    oyhi = <int>(fy if fy < yhi else yhi)

                    for oy in range(oylo, oyhi + 1):
                        # clip to row oy-0.5 <= y <= oy+0.5 (clipping coord is y)
                        n3 = clip_axis(sy, sx, n, ty, tx, oy - 0.5, 1.0)
                        if n3 < 3:
                            continue
                        n4 = clip_axis(ty, tx, n3, uy, ux, oy + 0.5, -1.0)
                        if n4 < 3:
                            continue
                        oimg[oy, ox] += poly_area(ux, uy, n4) * v
