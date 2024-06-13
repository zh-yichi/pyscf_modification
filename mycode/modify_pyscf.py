###### modifications to pyscf ######

import sys

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

######### complex ghf-scf stability improvement, implemented by Wang Xubo #########
#############        overwrite pyscf/soscf/ciah.py/davidson_cc        #############
def davidson_cc(h_op, g_op, precond, x0, tol=1e-10, xs=[], ax=[],
                max_cycle=30, lindep=1e-14, dot=numpy.dot, verbose=logger.WARN):

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    toloose = numpy.sqrt(tol)
    # the first trial vector is (1,0,0,...), which is not included in xs
    xs = list(xs)
    ax = list(ax)
    nx = len(xs)

    problem_size = x0.size
    max_cycle = min(max_cycle, problem_size)
    heff = numpy.zeros((max_cycle+nx+1,max_cycle+nx+1), dtype=x0.dtype)
    ovlp = numpy.eye(max_cycle+nx+1, dtype=x0.dtype)
    if nx == 0:
        xs.append(x0)
        ax.append(h_op(x0))
    else:
        for i in range(1, nx+1):
            for j in range(1, i+1):
                heff[i,j] = dot(xs[i-1].conj(), ax[j-1])
                ovlp[i,j] = dot(xs[i-1].conj(), xs[j-1])
            heff[1:i,i] = heff[i,1:i].conj()
            ovlp[1:i,i] = ovlp[i,1:i].conj()

    w_t = 0
    for istep in range(max_cycle):
        g = g_op()
        nx = len(xs)
        for i in range(nx):
            heff[i+1,0] = dot(xs[i].conj(), g)
            heff[nx,i+1] = dot(xs[nx-1].conj(), ax[i])
            ovlp[nx,i+1] = dot(xs[nx-1].conj(), xs[i])
        heff[0,:nx+1] = heff[:nx+1,0].conj()
        heff[1:nx,nx] = heff[nx,1:nx].conj()
        ovlp[1:nx,nx] = ovlp[nx,1:nx].conj()
        ####################################
        heff = heff.real #suggested by Xubo#
        ovlp = ovlp.real #suggested by Xubo#
        ####################################
        nvec = nx + 1
        #s0 = scipy.linalg.eigh(ovlp[:nvec,:nvec])[0][0]
        #if s0 < lindep:
        #    yield True, istep, w_t, xtrial, hx, dx, s0
        #    break
        wlast = w_t
        xtrial, w_t, v_t, index, seig = \
                _regular_step(heff[:nvec,:nvec], ovlp[:nvec,:nvec], xs,
                              lindep, log)
        s0 = seig[0]
        hx = _dgemv(v_t[1:], ax)
        # note g*v_t[0], as the first trial vector is (1,0,0,...)
        dx = hx + g*v_t[0] - w_t * v_t[0]*xtrial
        norm_dx = numpy.linalg.norm(dx)
        log.debug1('... AH step %d  index= %d  |dx|= %.5g  eig= %.5g  v[0]= %.5g  lindep= %.5g',
                   istep+1, index, norm_dx, w_t, v_t[0].real, s0)
        hx *= 1/v_t[0] # == h_op(xtrial)
        if ((abs(w_t-wlast) < tol and norm_dx < toloose) or
            s0 < lindep or
            istep+1 == problem_size):
            # Avoid adding more trial vectors if hessian converged
            yield True, istep+1, w_t, xtrial, hx, dx, s0
            if s0 < lindep or norm_dx < lindep:# or numpy.linalg.norm(xtrial) < lindep:
                # stop the iteration because eigenvectors would be barely updated
                break
        else:
            yield False, istep+1, w_t, xtrial, hx, dx, s0
            x0 = precond(dx, w_t)
            xs.append(x0)
            ax.append(h_op(x0))
######################################################################################

################## generate complex ghf-orbital density .cube file####################
##################          add to pyscf/tools/cubegen.py         ####################
def ghf_orbital(mol, outfile, coeff, nx=80, ny=80, nz=80, resolution=RESOLUTION,
            margin=BOX_MARGIN):

    from pyscf.pbc.gto import Cell
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval'
    if isinstance(mol, Cell):
        GTOval = 'PBC' + GTOval

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.empty(ngrids)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        nao = ao.shape[1]
        orb_on_grid_alpha = abs(numpy.dot(ao, coeff[:nao]))
        orb_on_grid_belta = abs(numpy.dot(ao, coeff[nao:]))
        orb_on_grid[ip0:ip1] = numpy.sqrt(orb_on_grid_alpha**2 + orb_on_grid_belta**2)
    orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)

    # Write out orbital to the .cube file
    cc.write(orb_on_grid, outfile, comment='Orbital value in real space (1/Bohr^3)')
    return orb_on_grid
########################################################################################