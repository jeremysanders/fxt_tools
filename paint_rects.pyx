# cython: language_level=3

# paint rectangles (for exposure map)

import cython
import numpy as np
from libc.math cimport floor, ceil, fabs

cdef struct Poly:
    int num
    double[8][2] pts

# clip double
cdef double clip_d(double v, double minv, double maxv) noexcept:
    return min(max(v, minv), maxv)

# clip integer
cdef int clip_i(int v, int minv, int maxv) noexcept:
    return min(max(v, minv), maxv)

# whether inside or outside shape
cdef int is_inside(double[2] p1, double[2] p2, double[2] q) noexcept:
    cdef double R = (p2[0]-p1[0])*(q[1]-p1[1]) - (p2[1]-p1[1])*(q[0]-p1[0])
    if R <= 0:
        return 1
    return 0

@cython.cdivision(True)
cdef void compute_intersection(double[2] p1, double[2] p2,
                               double[2] p3, double[2] p4,
                               double[2] o):
    cdef double x, m1, b1, y, m2, b2
    if p2[0]-p1[0] == 0:
        x = p1[0]
        m2 = (p4[1]-p3[1])/(p4[0]-p3[0])
        b2 = p3[1] - m2*p3[0]
        y = m2*x + b2
    elif p4[0]-p3[0] == 0:
        x = p3[0]
        m1 = (p2[1]-p1[1])/(p2[0]-p1[0])
        b1 = p1[1] - m1*p1[0]
        y = m1*x + b1
    else:
        m1 = (p2[1]-p1[1]) / (p2[0]-p1[0])
        b1 = p1[1] - m1*p1[0]
        m2 = (p4[1]-p3[1]) / (p4[0]-p3[0])
        b2 = p3[1] - m2*p3[0]
        x = (b2-b1) / (m1-m2)
        y = m1*x + b1
    o[0] = x
    o[1] = y

# cdef void doprint(Poly poly):
#     out = []
#     for i in range(poly.num):
#         out.append('[%.4f %.4f]' % (poly.pts[i][0], poly.pts[i][1]))
#     print(' '.join(out))

# Sutherland–Hodgman Algorithm
# see https://github.com/mhdadk/sutherland-hodgman/blob/main/SH.py
# points need to be in correct order (anticlockwise?) or overlap is zero
cdef Poly poly_clip(Poly spoly, Poly cpoly) noexcept:

    cdef Poly npoly, opoly
    opoly = spoly

    cdef double[2] cedge1, cedge2, sedge1, sedge2, inter

    cdef int i, j
    for i in range(cpoly.num):
        npoly = opoly
        opoly.num = 0
       
        if i==0:
            cedge1 = cpoly.pts[cpoly.num-1]
        else:
            cedge1 = cpoly.pts[i-1]
        cedge2 = cpoly.pts[i]

        for j in range(npoly.num):
            if j==0:
                sedge1 = npoly.pts[npoly.num-1]
            else:
                sedge1 = npoly.pts[j-1]
            sedge2 = npoly.pts[j]
            if is_inside(cedge1, cedge2, sedge2):
                if not is_inside(cedge1, cedge2, sedge1):
                    compute_intersection(sedge1, sedge2, cedge1, cedge2, inter)
                    opoly.pts[opoly.num] = inter
                    opoly.num += 1
                opoly.pts[opoly.num] = sedge2
                opoly.num += 1
            elif is_inside(cedge1, cedge2, sedge1):
                compute_intersection(sedge1, sedge2, cedge1, cedge2, inter)
                opoly.pts[opoly.num] = inter
                opoly.num += 1

    return opoly

# polygon area - maybe issue with sign
cdef double poly_area(Poly p) noexcept:
    cdef int i, j
    cdef double a

    a = 0.
    j = p.num-1
    for i in range(p.num):
        a += (p.pts[j][0]+p.pts[i][0]) * (p.pts[j][1]-p.pts[i][1])
        j = i
    return 0.5*a

# python test function for clipping
def test_clip(double[:,:] spoly, double[:,:] cpoly):
    cdef Poly _spoly, _cpoly
    cdef int i, onum

    assert spoly.shape[1] == 2
    assert cpoly.shape[1] == 2

    _spoly.num = spoly.shape[0]
    for i in range(spoly.shape[0]):
        _spoly.pts[i][0] = spoly[i,0]
        _spoly.pts[i][1] = spoly[i,1]
    _cpoly.num = cpoly.shape[0]
    for i in range(cpoly.shape[0]):
        _cpoly.pts[i][0] = cpoly[i,0]
        _cpoly.pts[i][1] = cpoly[i,1]

    #onum = clip(_spoly, _cpoly, _opoly)
    #onum = test_foo(_opoly)

    cdef Poly outp
    outp = poly_clip(_spoly, _cpoly)

    out = np.zeros((outp.num,2))
    for i in range(outp.num):
        out[i,0] = outp.pts[i][0]
        out[i,1] = outp.pts[i][1]
    return out

@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def paint_rects(double[:,::1] qye, double[:,::1] qxe, double[:,::1] qva, double[:,::1] oimg):
    """qye = yedges, qxe = xedges, qva = values, oimg = output image."""

    cdef int oyw = oimg.shape[0]
    cdef int oxw = oimg.shape[1]

    cdef int qyw = qva.shape[0]
    cdef int qxw = qva.shape[1]

    assert qye.shape[0]==qyw+1 and qye.shape[1]==qxw+1
    assert qxe.shape[0]==qyw+1 and qxe.shape[1]==qxw+1

    cdef int y, x, oy, ox
    cdef double v, a
    # coordinates of input pixel
    cdef double x1, x2, x3, x4, y1, y2, y3, y4
    # coordinates of output pixel
    cdef double xp1, xp2, xp3, xp4, yp1, yp2, yp3, yp4
    # range to iterate over in output
    cdef int xmin, xmax, ymin, ymax

    cdef Poly polyin, polypix, polyclip
    cdef double aclip

    polyin.num = 4
    polypix.num = 4

    for y in range(qyw):
        for x in range(qxw):
            x1 = qxe[y,   x  ]
            y1 = qye[y,   x  ]
            x2 = qxe[y+1, x  ]
            y2 = qye[y+1, x  ]
            x3 = qxe[y+1, x+1]
            y3 = qye[y+1, x+1]
            x4 = qxe[y,   x+1]
            y4 = qye[y,   x+1]
            v  = qva[y,   x  ]

            xmin = int(clip_d(floor(min(x1, x2, x3, x4)), -1, oxw))
            ymin = int(clip_d(floor(min(y1, y2, y3, y4)), -1, oyw))
            xmax = int(clip_d( ceil(max(x1, x2, x3, x4)), -1, oxw))
            ymax = int(clip_d( ceil(max(y1, y2, y3, y4)), -1, oyw))

            # out of bounds
            if ( (xmin==-1 and xmax==-1) or (xmin==oxw and xmax==oxw) or
                 (ymin==-1 and ymax==-1) or (ymin==oyw and ymax==oyw)):
                continue

            polyin.pts[0][0] = x1
            polyin.pts[0][1] = y1
            polyin.pts[1][0] = x2
            polyin.pts[1][1] = y2
            polyin.pts[2][0] = x3
            polyin.pts[2][1] = y3
            polyin.pts[3][0] = x4
            polyin.pts[3][1] = y4

            xmin = clip_i(xmin, 0, oxw-1)
            ymin = clip_i(ymin, 0, oyw-1)
            xmax = clip_i(xmax, 0, oxw-1)
            ymax = clip_i(ymax, 0, oyw-1)

            # iterate over overlapping output pixels with input pixel
            for oy in range(ymin, ymax+1):
                for ox in range(xmin, xmax+1):
                    polypix.pts[0][0] = ox - 0.5
                    polypix.pts[0][1] = oy - 0.5
                    polypix.pts[1][0] = ox - 0.5
                    polypix.pts[1][1] = oy + 0.5
                    polypix.pts[2][0] = ox + 0.5
                    polypix.pts[2][1] = oy + 0.5
                    polypix.pts[3][0] = ox + 0.5
                    polypix.pts[3][1] = oy - 0.5

                    polyclip = poly_clip(polyin, polypix)
                    aclip = fabs(poly_area(polyclip))  # would be 1 for complete overlap
                    oimg[oy, ox] += aclip * v
