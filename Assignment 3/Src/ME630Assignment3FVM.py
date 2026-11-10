# =========================================================================
#  2D UNSTEADY incompressible laminar flow solver for teaching
#  Finite Volume Method | Staggered grid | SIMPLE algorithm
#  Convection: first-order upwind | Time: implicit (backward) Euler
#
#  ONE code, several problems. Only Section 1 (CASE SETUP) changes between
#  problems: geometry, fluid properties, boundary conditions, time step.
#  The discretisation and the SIMPLE loop (Sections 3-5) stay the same.
#
#  Cases provided (set caseName in Section 1):
#    'channel'  developing flow between two fixed plates, started from rest
#               (inlet / outlet in x, no-slip walls)
#    'couette'  start-up of Couette flow: upper plate starts moving
#               (periodic in x, fixed lower wall, moving upper wall);
#               add a pressure gradient for Couette-Poiseuille flow
#    'stokes1'  Stokes first problem: plate impulsively started
#               (periodic in x, moving lower wall, far-field upper boundary)
#    'stokes2'  Stokes second problem: plate oscillating in its own plane
#               (periodic in x, oscillating lower wall, far field above)
#
#  Boundary-condition options available in Section 1:
#    x-direction:   xBC = 'inlet-outlet'  (velocity inlet at x = 0,
#                                          zero-gradient outlet at x = L)
#                   xBC = 'periodic'      (infinite in x, flow repeats)
#    y-boundaries:  southBC / northBC = 'wall'  no-slip wall moving
#                                               tangentially with uSouth(t)
#                                               or uNorth(t)
#                                    = 'slip'  zero shear (symmetry plane
#                                               or far field)
#    Mean pressure gradient dpdx (periodic cases): drives the flow as a
#    uniform source -dpdx in the x-momentum equation.
#
#  Governing equations (constant rho, mu):
#     d(rho u)/dx + d(rho v)/dy = 0
#     d(rho u)/dt + div(rho u U) = -dp/dx + div(mu grad u) - dpdx
#     d(rho v)/dt + div(rho v U) = -dp/dy + div(mu grad v)
#
#  Discrete momentum equation for each velocity (Versteeg & Malalasekera,
#  Ch. 6 and 8), implicit Euler in time:
#     aP*phi_P = sum(a_nb*phi_nb) + b
#     a_nb (upwind):  aE = De + max(-Fe,0)   aW = Dw + max(Fw,0)
#                     aN = Dn + max(-Fn,0)   aS = Ds + max(Fs,0)
#     aP = sum(a_nb) + aP0,      aP0 = rho*dx*dy/dt
#     b  = (pressure difference)*area + aP0*phi_P^old + other sources
#
#  Staggered grid:
#     p(i,j) at cell centres     i = 0..Nx-1, j = 0..Ny-1
#     u(i,j) on west faces       i = 0..Nx,   j = 0..Ny-1
#     v(i,j) on south faces      i = 0..Nx-1, j = 0..Ny
#  In periodic cases face Nx is the same face as face 0; the code keeps
#  u(Nx,:) = u(0,:) at all times.
# =========================================================================

import os
import sys
import traceback
import time as clock
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import erfc


class ConsoleLogger:
    # Display messages in the terminal and save them when the folder is known.
    def __init__(self,terminal):
        self.terminal=terminal
        self.pending=[]
        self.logFile=None

    def write(self,message):
        self.terminal.write(message)
        if self.logFile is None:
            self.pending.append(message)
        else:
            self.logFile.write(message)
            self.logFile.flush()

    def flush(self):
        self.terminal.flush()
        if self.logFile is not None:
            self.logFile.flush()

    def close(self):
        if self.logFile is not None:
            self.logFile.close()
            self.logFile=None


def StartConsoleLog(outputFolder):
    # Save earlier menu output too, then log all further console prints.
    if not isinstance(sys.stdout,ConsoleLogger):
        sys.stdout=ConsoleLogger(sys.stdout)
    if not isinstance(sys.stderr,ConsoleLogger):
        sys.stderr=ConsoleLogger(sys.stderr)
    logFile=open(os.path.join(outputFolder,'console_output.txt'),'w',encoding='utf-8')
    for logger in (sys.stdout,sys.stderr):
        logFile.write(''.join(logger.pending))
        logger.pending.clear()
        logger.logFile=logFile
    logFile.flush()


def WallConductance(bcType,mu,dx,dy):
    # Diffusion conductance between the first u-node and a y-boundary.
    # A wall is dy/2 from the node: D = mu*dx/(dy/2).
    # A slip (zero-shear) boundary transmits no shear: D = 0.
    if bcType=='wall':
        return 2*mu*dx/dy
    if bcType=='slip':
        return 0.0
    raise ValueError(f'Unknown y-boundary type "{bcType}"')


def ApplyOutlet(u,rho,dy):
    # Zero-gradient outflow, scaled so that outflow = inflow.
    mdotIn=rho*np.sum(u[0,:])*dy
    u[-1,:]=u[-2,:]
    mdotOut=rho*np.sum(u[-1,:])*dy
    if abs(mdotOut)>0:
        u[-1,:]*=mdotIn/mdotOut
    return u


def ExactProfile(caseName,y,t,parameters):
    # Exact u(y,t) for each case (for verification).
    U0=parameters['U0']
    H=parameters['H']
    nu=parameters['nu']
    if caseName=='channel':
        # Fully developed Poiseuille profile
        return 6*U0*(y/H)*(1-y/H)
    if caseName=='couette':
        # Start-up from rest, by separation of variables.
        # Walls moving at Ubot (y = 0) and Utop (y = H) for t > 0,
        # G = -dpdx/(2 mu). Steady part:
        #   u_s = Ubot + (Utop - Ubot)*y/H + G*y*(H - y)
        # transient: sum b_n exp(-n^2 pi^2 nu t/H^2) sin(n pi y/H),
        # with b_n = -(2/H)*integral(u_s sin(n pi y/H)) so u = 0 at t = 0.
        Ubot=parameters['Ubot']
        Utop=parameters['Utop']
        G=-parameters['dpdx']/(2*parameters['mu'])
        exactVelocity=Ubot+(Utop-Ubot)*y/H+G*y*(H-y)
        for n in range(1,201):
            s=(-1)**n
            bn=-2*(Ubot*(1-s)/(n*np.pi)-(Utop-Ubot)*s/(n*np.pi)+2*G*H**2*(1-s)/(n*np.pi)**3)
            exactVelocity+=bn*np.exp(-(n*np.pi)**2*nu*t/H**2)*np.sin(n*np.pi*y/H)
        return exactVelocity
    if caseName=='stokes1':
        # Impulsively started plate (speed Ubot)
        return parameters['Ubot']*erfc(y/(2*np.sqrt(nu*t)))
    if caseName=='stokes2':
        # Oscillating plate, periodic state
        omega=parameters['omega']
        k=np.sqrt(omega/(2*nu))
        return U0*np.exp(-k*y)*np.cos(omega*t-k*y)
    raise ValueError(f'Unknown caseName "{caseName}"')


def Solve5(aP,aE,aW,aN,aS,b,periodicX):
    # Solve the 5-point system on an nI x nJ grid:
    #   aP*phi_P = aE*phi_E + aW*phi_W + aN*phi_N + aS*phi_S + b
    # Assemble a sparse matrix and use SciPy's direct sparse solver.
    # periodicX: east of i = nI-1 wraps to i = 0; west of i = 0 wraps to nI-1.
    nI,nJ=aP.shape
    numberOfUnknowns=nI*nJ
    indexGrid=np.arange(numberOfUnknowns).reshape(nI,nJ)
    rows=[indexGrid.ravel()]
    columns=[indexGrid.ravel()]
    values=[aP.ravel()]

    if periodicX:
        eastIndices=np.roll(indexGrid,-1,axis=0)
        westIndices=np.roll(indexGrid,1,axis=0)
        rows.extend([indexGrid.ravel(),indexGrid.ravel()])
        columns.extend([eastIndices.ravel(),westIndices.ravel()])
        values.extend([-aE.ravel(),-aW.ravel()])
    else:
        rows.extend([indexGrid[:-1,:].ravel(),indexGrid[1:,:].ravel()])
        columns.extend([indexGrid[1:,:].ravel(),indexGrid[:-1,:].ravel()])
        values.extend([-aE[:-1,:].ravel(),-aW[1:,:].ravel()])

    rows.extend([indexGrid[:,:-1].ravel(),indexGrid[:,1:].ravel()])
    columns.extend([indexGrid[:,1:].ravel(),indexGrid[:,:-1].ravel()])
    values.extend([-aN[:,:-1].ravel(),-aS[:,1:].ravel()])

    systemMatrix=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=(numberOfUnknowns,numberOfUnknowns)).tocsr()
    solution=spsolve(systemMatrix,b.ravel())
    if not np.all(np.isfinite(solution)):
        raise RuntimeError('Five-point linear system produced non-finite values')
    return solution.reshape(nI,nJ)


def main(case, algorithm='SIMPLE', convectionScheme = 'UDS'):
    # ====================== 1. CASE SETUP (students edit here) ==============
    if(case== 1):
        caseName='channel'
    elif(case== 2):
        caseName='couette'
    elif(case== 3):
        caseName='stokes1'
    elif(case== 4):
        caseName='stokes2'
    elif(case== 5):
        caseName='bfs'
    else:
        raise ValueError(f'Unknown case number "{case}"')
    
    omega=0.0

    if caseName=='channel':
        '''
        =======
        ANSWER A1.3 - Obtained by changing Nx, Ny and dt values here.
        =======
        '''
        # Developing channel flow, started from rest
        L=10; H=1                   # Domain length and height [m]
        Nx=100; Ny=40               # Number of pressure cells
        rho=1; mu=0.01              # Re = rho*U0*H/mu = 100
        U0=1                        # Inlet velocity [m/s]
        xBC='inlet-outlet'
        uInlet=lambda y,t: U0*np.ones_like(y)  # Uniform inflow
        southBC='wall'; uSouth=lambda t: 0     # Fixed lower wall
        northBC='wall'; uNorth=lambda t: 0     # Fixed upper wall
        dpdx=0                      # Not used with inlet-outlet
        '''
        =======
        ANSWER A1.4
        =====
        '''
        dt= 1.0; tEnd=60            # Time step and end time [s]
        tOut=np.array([1,2,5,10,20,60],dtype=float)
        entryLengthAnalytical=0.05*(rho*U0*H/mu)*H
        entryLengthNumerical = None
        stopWhenSteady=True
        Re=rho*U0*H/mu
        caseFolderName=f'{caseName}_{Re:g}_{Nx}_{Ny}_{dt:g}'

    elif caseName=='couette':
        # Start-up of Couette flow
        L=0.1; H=1
        Nx=4; Ny=40                 # No x-variation: a few cells suffice
        rho=1; mu=0.1               # Viscous time H^2/nu = 10 s
        U0=1                        # Speed of the upper plate
        xBC='periodic'
        southBC='wall'; uSouth=lambda t: 0
        northBC='wall'; uNorth=lambda t: U0  # Upper plate starts at t = 0
        dpdx=0                      # Try -0.4 for Couette-Poiseuille flow,
                                    # or uNorth = 0, dpdx < 0 for start-up
                                    # of plane Poiseuille flow
        dt=0.02; tEnd=10
        tOut=np.array([0.1,0.5,1,2,5,10],dtype=float)
        stopWhenSteady=False
        Re=rho*U0*H/mu
        caseFolderName=f'{caseName}_{Re:g}_{Nx}_{Ny}_{dt:g}'

    elif caseName=='stokes1':
        # Stokes first problem (Rayleigh problem)
        L=0.02; H=1                 # H must be >> penetration depth
        Nx=4; Ny=200
        rho=1; mu=0.01
        U0=1
        xBC='periodic'
        southBC='wall'; uSouth=lambda t: U0  # Plate impulsively started
        northBC='slip'; uNorth=lambda t: 0   # Far field: zero shear
        dpdx=0
        dt=0.001; tEnd=1
        tOut=np.array([0.05,0.1,0.25,0.5,1],dtype=float)
        stopWhenSteady=False
        Re=rho*U0*H/mu
        caseFolderName=f'{caseName}_{Re:g}_{Nx}_{Ny}_{dt:g}'

    elif caseName=='stokes2':
        # Stokes second problem (oscillating plate)
        L=0.02; H=1
        Nx=4; Ny=200
        rho=1; mu=0.01
        U0=1; omega=2*np.pi         # Amplitude and angular frequency
        T=2*np.pi/omega            # Period [s]
        xBC='periodic'
        southBC='wall'; uSouth=lambda t: U0*np.cos(omega*t)
        northBC='slip'; uNorth=lambda t: 0
        dpdx=0
        dt=T/200; tEnd=4*T          # Four periods; start-up transient decays fast
        tOut=tEnd-T+np.arange(8)*T/8  # Eight phases of the last period
        stopWhenSteady=False
        Re=rho*U0*H/mu
        caseFolderName=f'{caseName}_{Re:g}_{Nx}_{Ny}_{dt:g}'

    elif caseName=='bfs':
        # Backward-facing step (Geometry from Armaly)
        S=1; h=17/16; H=S+h
        L=40*S
        Ny=33; Nx=320
        rho=1; U0=1
        Re=400
        mu=rho*U0*(2*h)/Re
        xBC='inlet-outlet'
        # Parabolic inlet profile above the step
        uInlet=lambda y,t: np.where(y>S,6*U0*((y-S)/h)*(1-(y-S)/h),0.0)
        southBC='wall'; uSouth=lambda t: 0
        northBC='wall'; uNorth=lambda t: 0
        dpdx=0
        dt= 0.01 ; tEnd=100
        tOut=np.array([1,5,10,20,50,100],dtype=float)
        stopWhenSteady=True
        caseFolderName=f'{caseName}_{Re:g}_{Nx}_{Ny}_{dt:g}'

    else:
        raise ValueError(f'Unknown caseName "{caseName}"')

    # ---------------------- SIMPLE controls ---------------------------------
    alphaU=0.8                     # Momentum under-relaxation (within a time step)
    alphaP=0.3                     # Pressure under-relaxation
    maxInner=200                   # Max SIMPLE iterations per time step
    tolInner=1e-6                  # Convergence of each time step
    tolSteady=1e-7                 # |u^(n+1)-u^n|/U0 below this -> steady

    # Keep separate results for different algorithms, schemes and relaxation.
    caseFolderName+=f'_{algorithm}_{convectionScheme}_aU{alphaU:g}'
    outputFolder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Results',caseFolderName)
    os.makedirs(outputFolder,exist_ok=True)
    StartConsoleLog(outputFolder)
    print(f'Output folder: {outputFolder}')
    print(f'Algorithm: {algorithm} | Scheme: {convectionScheme} | Re: {Re:g} | alphaU: {alphaU:g} | alphaP: {alphaP:g}')

    # ====================== 2. Grid and initial fields ======================
    dx=L/Nx; dy=H/Ny
    xc=(np.arange(Nx)+0.5)*dx
    yc=(np.arange(Ny)+0.5)*dy
    xu=np.arange(Nx+1)*dx
    yv=np.arange(Ny+1)*dy
    nu=mu/rho
    isPer=xBC=='periodic'

    # Initial condition: fluid at rest
    u=np.zeros((Nx+1,Ny))
    v=np.zeros((Nx,Ny+1))
    p=np.zeros((Nx,Ny))
    if not isPer:
        u[0,:]=uInlet(yc,0)
    dU=np.zeros((Nx+1,Ny))  # d-coefficients; zero on inlet/outlet faces
    dV=np.zeros((Nx,Ny+1))  # Those faces are never corrected

    # Index sets: which u-faces are unknowns, and their neighbours
    if isPer:
        Iu=np.arange(Nx)          # All faces (face Nx = face 0)
        IuE=np.roll(Iu,-1)        # East neighbour face (wraps round)
        IuW=np.roll(Iu,1)         # West neighbour face
        cW=np.roll(Iu,1)          # Pressure cell west of each face
        cE=Iu.copy()              # Pressure cell east of each face
    else:
        Iu=np.arange(1,Nx)       # Interior faces (inlet 0, outlet Nx known)
        IuE=Iu+1
        IuW=Iu-1
        cW=Iu-1
        cE=Iu.copy()

    aP0=rho*dx*dy/dt             # Transient coefficient rho*V/dt (u and v CVs)
    fRef=rho*U0*H               # Reference mass flux for the residual
    nSteps=round(tEnd/dt)
    nOut=np.rint(tOut/dt).astype(int)
    uSnap=[None]*len(tOut)
    tSnap=np.full(len(tOut),np.nan)
    nInner=np.zeros(nSteps,dtype=int)
    
    # Velocity history at the penetration depth
    if caseName=='stokes2':
        delta=np.sqrt(2*nu/omega)
        timeHistory=[]
        velocityHistory=[]

    print(f'Case "{caseName}": {Nx} x {Ny} cells, dt = {dt:g} s, {nSteps} steps')
    print(f'Courant number U0*dt/dx = {U0*dt/dx:.2f}, diffusion number nu*dt/dy^2 = {nu*dt/dy**2:.2f}')
    print('(implicit scheme: no stability limit, but accuracy needs small dt)')

    # ====================== 3. Time loop ====================================
    algorithmStartTime= clock.process_time()
    for n in range(1,nSteps+1):
        t=n*dt                       # New time level t^(n+1)
        uOld=u.copy(); vOld=v.copy() # Solution at the old time level t^n

        # Boundary values at the new time level
        uS=uSouth(t); uN=uNorth(t)
        if not isPer:
            u[0,:]=uInlet(yc,t)

        
        # ================= 4. SIMPLE/SIMPEC/SIMPLER iterations within the time step ========
        for it in range(1,maxInner+1):
            uIt=u.copy(); vIt=v.copy()  # Previous iterate (for relaxation/check)

            # --------------- 4a. u-momentum ---------------------------------
            # Mass fluxes through the four faces of each u control volume
            Fe=rho*0.5*(u[Iu,:]+u[IuE,:])*dy
            Fw=rho*0.5*(u[IuW,:]+u[Iu,:])*dy
            Fn=rho*0.5*(v[cW,1:Ny+1]+v[cE,1:Ny+1])*dx
            Fs=rho*0.5*(v[cW,:Ny]+v[cE,:Ny])*dx

            # Diffusion conductances D = mu*area/distance
            nU=len(Iu)
            De=np.full((nU,Ny),mu*dy/dx); Dw=De.copy()
            Dn=np.full((nU,Ny),mu*dx/dy); Ds=Dn.copy()
            Ds[:,0]=WallConductance(southBC,mu,dx,dy)  # y = 0 boundary
            Dn[:,-1]=WallConductance(northBC,mu,dx,dy) # y = H boundary

            # Convection discretisation: UDS, CDS, or CDS deferred correction
            if convectionScheme=='UDS' or convectionScheme=='CDS-DC':
                # First-order upwind neighbour coefficients
                aE=De+np.maximum(-Fe,0)
                aW=Dw+np.maximum(Fw,0)
                aN=Dn+np.maximum(-Fn,0)
                aS=Ds+np.maximum(Fs,0)
            elif convectionScheme=='CDS':
                aE=De-Fe/2
                aW=Dw+Fw/2
                aN=Dn-Fn/2
                aS=Ds+Fs/2
            else:
                raise ValueError(f'Unknown convectionScheme "{convectionScheme}"')

            # Central coefficient and source (implicit Euler: aP0 terms)
            aP=aE+aW+aN+aS+aP0
            b=(p[cW,:]-p[cE,:])*dy-dpdx*dx*dy+aP0*uOld[Iu,:]

            if convectionScheme=='CDS-DC':
                # Explicit CDS-UDS flux difference from previous inner iterate
                uEastValues=uIt[IuE,:]
                uWestValues=uIt[IuW,:]
                uNorthValues=np.concatenate((uIt[Iu,1:Ny],np.full((nU,1),uN)),axis=1)
                uSouthValues=np.concatenate((np.full((nU,1),uS),uIt[Iu,:Ny-1]),axis=1)
                b-=0.5*(np.abs(Fe)*(uEastValues-uIt[Iu,:])
                        -np.abs(Fw)*(uIt[Iu,:]-uWestValues)
                        +np.abs(Fn)*(uNorthValues-uIt[Iu,:])
                        -np.abs(Fs)*(uIt[Iu,:]-uSouthValues))

            # Boundary neighbours: known values moved to the source
            b[:,0]+=aS[:,0]*uS; aS[:,0]=0    # Wall at y = 0
            b[:,-1]+=aN[:,-1]*uN; aN[:,-1]=0 # Wall at y = H
            # ('slip' gives Ds or Dn = 0, so those lines add nothing)
            if not isPer:
                b[0,:]+=aW[0,:]*u[0,:]; aW[0,:]=0     # Inlet
                b[-1,:]+=aE[-1,:]*u[Nx,:]; aE[-1,:]=0 # Outlet

            # Implicit under-relaxation (does not change the converged answer)
            aP=aP/alphaU
            b=b+(1-alphaU)*aP*u[Iu,:]
            if algorithm=='SIMPLEC':
                dU[Iu,:]=dy/(aP-aE-aW-aN-aS)
            else:
                dU[Iu,:]=dy/aP     # SIMPLE: u' = dU*(p'_W - p'_E)
            u[Iu,:]=Solve5(aP,aE,aW,aN,aS,b,isPer)
            if isPer:
                u[Nx,:]=u[0,:] # Face Nx is face 0
                dU[Nx,:]=dU[0,:]
            if algorithm=='SIMPLER':
                uMomentumAP=aP.copy(); uMomentumAE=aE.copy(); uMomentumAW=aW.copy()
                uMomentumAN=aN.copy(); uMomentumAS=aS.copy(); uMomentumB=b.copy()

            # --------------- 4b. v-momentum ---------------------------------
            # Unknowns: v[:,1:Ny]; v = 0 on both y-boundaries (wall or slip)
            Fe=rho*0.5*(u[1:Nx+1,:Ny-1]+u[1:Nx+1,1:Ny])*dy
            Fw=rho*0.5*(u[:Nx,:Ny-1]+u[:Nx,1:Ny])*dy
            Fn=rho*0.5*(v[:,1:Ny]+v[:,2:Ny+1])*dx
            Fs=rho*0.5*(v[:,:Ny-1]+v[:,1:Ny])*dx

            De=np.full((Nx,Ny-1),mu*dy/dx); Dw=De.copy()
            Dn=np.full((Nx,Ny-1),mu*dx/dy); Ds=Dn.copy()
            if not isPer:
                Dw[0,:]=2*mu*dy/dx  # Inlet (v = 0) is dx/2 away
                De[-1,:]=0         # Outlet: zero gradient

            if convectionScheme=='UDS' or convectionScheme=='CDS-DC':
                aE=De+np.maximum(-Fe,0)
                aW=Dw+np.maximum(Fw,0)
                aN=Dn+np.maximum(-Fn,0)
                aS=Ds+np.maximum(Fs,0)
            elif convectionScheme=='CDS':
                aE=De-Fe/2
                aW=Dw+Fw/2
                aN=Dn-Fn/2
                aS=Ds+Fs/2
            else:
                raise ValueError(f'Unknown convectionScheme "{convectionScheme}"')

            aP=aE+aW+aN+aS+aP0
            b=(p[:,:Ny-1]-p[:,1:Ny])*dx+aP0*vOld[:,1:Ny]

            if convectionScheme=='CDS-DC':
                # Explicit CDS-UDS flux difference from previous inner iterate
                if isPer:
                    vEastValues=np.roll(vIt[:,1:Ny],-1,axis=0)
                    vWestValues=np.roll(vIt[:,1:Ny],1,axis=0)
                else:
                    vEastValues=np.concatenate((vIt[1:,1:Ny],vIt[-1:,1:Ny]),axis=0)
                    vWestValues=np.concatenate((np.zeros((1,Ny-1)),vIt[:Nx-1,1:Ny]),axis=0)
                b-=0.5*(np.abs(Fe)*(vEastValues-vIt[:,1:Ny])
                        -np.abs(Fw)*(vIt[:,1:Ny]-vWestValues)
                        +np.abs(Fn)*(vIt[:,2:Ny+1]-vIt[:,1:Ny])
                        -np.abs(Fs)*(vIt[:,1:Ny]-vIt[:,:Ny-1]))

            if not isPer:
                aW[0,:]=0   # Inlet: v = 0 (adds nothing)
                aE[-1,:]=0  # Outlet: v_E = v_P
            aS[:,0]=0; aN[:,-1]=0 # v = 0 at y = 0 and y = H

            aP=aP/alphaU
            b=b+(1-alphaU)*aP*v[:,1:Ny]
            if algorithm=='SIMPLEC':
                dV[:,1:Ny]=dx/(aP-aE-aW-aN-aS)
            else:
                dV[:,1:Ny]=dx/aP
            v[:,1:Ny]=Solve5(aP,aE,aW,aN,aS,b,isPer)
            if algorithm=='SIMPLER':
                vMomentumAP=aP.copy(); vMomentumAE=aE.copy(); vMomentumAW=aW.copy()
                vMomentumAN=aN.copy(); vMomentumAS=aS.copy(); vMomentumB=b.copy()

            # --------------- 4c. outlet (inlet-outlet cases only) -----------

            if not isPer:
                u=ApplyOutlet(u,rho,dy)

            # --------------- 4c(i). SIMPLER pressure equation ----------------
            if algorithm=='SIMPLER':
                # Form pseudo-velocities from the momentum equations without pressure
                uHat=uIt.copy(); vHat=vIt.copy()
                uNorthNeighbor=np.concatenate((uIt[Iu,1:Ny],np.zeros((len(Iu),1))),axis=1)
                uSouthNeighbor=np.concatenate((np.zeros((len(Iu),1)),uIt[Iu,:Ny-1]),axis=1)
                uHat[Iu,:]=(uMomentumAE*uIt[IuE,:]+uMomentumAW*uIt[IuW,:]
                            +uMomentumAN*uNorthNeighbor+uMomentumAS*uSouthNeighbor
                            +uMomentumB-(p[cW,:]-p[cE,:])*dy)/uMomentumAP

                if isPer:
                    uHat[Nx,:]=uHat[0,:]
                    vEastNeighbor=np.roll(vIt[:,1:Ny],-1,axis=0)
                    vWestNeighbor=np.roll(vIt[:,1:Ny],1,axis=0)
                else:
                    uHat=ApplyOutlet(uHat,rho,dy)
                    vEastNeighbor=np.concatenate((vIt[1:,1:Ny],np.zeros((1,Ny-1))),axis=0)
                    vWestNeighbor=np.concatenate((np.zeros((1,Ny-1)),vIt[:Nx-1,1:Ny]),axis=0)

                vHat[:,1:Ny]=(vMomentumAE*vEastNeighbor+vMomentumAW*vWestNeighbor
                              +vMomentumAN*vIt[:,2:Ny+1]+vMomentumAS*vIt[:,:Ny-1]
                              +vMomentumB-(p[:,:Ny-1]-p[:,1:Ny])*dx)/vMomentumAP

                # Solve the pressure equation using the pseudo-velocity fluxes
                aE=rho*dU[1:Nx+1,:]*dy
                aW=rho*dU[:Nx,:]*dy
                aN=np.zeros((Nx,Ny)); aS=np.zeros((Nx,Ny))
                aN[:,:Ny-1]=rho*dV[:,1:Ny]*dx
                aS[:,1:Ny]=rho*dV[:,1:Ny]*dx
                aP=aE+aW+aN+aS
                b=rho*(uHat[:Nx,:]-uHat[1:Nx+1,:])*dy+rho*(vHat[:,:Ny]-vHat[:,1:Ny+1])*dx

                iR=Nx-1; jR=0
                aP[iR,jR]=1; aE[iR,jR]=0; aW[iR,jR]=0; aN[iR,jR]=0; aS[iR,jR]=0
                b[iR,jR]=0
                pressureBeforeSIMPLER=p.copy()
                p=Solve5(aP,aE,aW,aN,aS,b,isPer)

                # Re-solve the unchanged momentum systems using the new pressure
                b=uMomentumB+((p[cW,:]-p[cE,:])
                              -(pressureBeforeSIMPLER[cW,:]-pressureBeforeSIMPLER[cE,:]))*dy
                u[Iu,:]=Solve5(uMomentumAP,uMomentumAE,uMomentumAW,uMomentumAN,uMomentumAS,b,isPer)
                b=vMomentumB+((p[:,:Ny-1]-p[:,1:Ny])
                              -(pressureBeforeSIMPLER[:,:Ny-1]-pressureBeforeSIMPLER[:,1:Ny]))*dx
                v[:,1:Ny]=Solve5(vMomentumAP,vMomentumAE,vMomentumAW,vMomentumAN,vMomentumAS,b,isPer)
                if isPer:
                    u[Nx,:]=u[0,:]
                else:
                    u=ApplyOutlet(u,rho,dy)

            # --------------- 4d. pressure-correction equation ---------------
            # a_nb = rho*d*A on faces that are corrected; dU = 0 on inlet and
            # outlet faces, so those neighbours drop out automatically.
            aE=rho*dU[1:Nx+1,:]*dy
            aW=rho*dU[:Nx,:]*dy
            aN=np.zeros((Nx,Ny)); aS=np.zeros((Nx,Ny))
            aN[:,:Ny-1]=rho*dV[:,1:Ny]*dx
            aS[:,1:Ny]=rho*dV[:,1:Ny]*dx
            aP=aE+aW+aN+aS
            b=rho*(u[:Nx,:]-u[1:Nx+1,:])*dy+rho*(v[:,:Ny]-v[:,1:Ny+1])*dx
            massRes=np.sum(np.abs(b))/fRef

            # p' is fixed by the boundaries only up to a constant: set p'=0
            # in one reference cell
            iR=Nx-1; jR=0
            aP[iR,jR]=1; aE[iR,jR]=0; aW[iR,jR]=0; aN[iR,jR]=0; aS[iR,jR]=0
            b[iR,jR]=0
            pp=Solve5(aP,aE,aW,aN,aS,b,isPer)

            # --------------- 4e. correct p, u, v ----------------------------
            if algorithm=='SIMPLE' or algorithm=='SIMPLEC':
                p=p+alphaP*pp
            elif algorithm=='SIMPLER':
                pass
            else:
                raise ValueError(f'Unknown algorithm "{algorithm}"')

            u[Iu,:]+=dU[Iu,:]*(pp[cW,:]-pp[cE,:])
            v[:,1:Ny]+=dV[:,1:Ny]*(pp[:,:Ny-1]-pp[:,1:Ny])
            if isPer:
                u[Nx,:]=u[0,:]
            elif algorithm!='SIMPLER':
                u=ApplyOutlet(u,rho,dy)

            # --------------- 4f. convergence of this time step --------------
            if algorithm=='SIMPLER':
                # Check continuity after velocity correction
                b=rho*(u[:Nx,:]-u[1:Nx+1,:])*dy+rho*(v[:,:Ny]-v[:,1:Ny+1])*dx
                massRes=np.sum(np.abs(b))/fRef

            change=max(np.max(np.abs(u-uIt)),np.max(np.abs(v-vIt)))/U0
            if massRes<tolInner and change<tolInner:
                break

        nInner[n-1]=it
        # Store velocity at y=delta over the last period
        if caseName=='stokes2' and t>=tEnd-T-1e-10:
            uMean=np.mean(u[:Nx,:],axis=0)
            uDelta=np.interp(delta,yc,uMean)
            timeHistory.append(t)
            velocityHistory.append(uDelta/U0)
        if it==maxInner:
            print(f'Warning: step {n}: {algorithm} not converged (mass {massRes:.1e}, change {change:.1e})')

        # ================= 5. Store results, check steady state ==============
        for k in np.flatnonzero(nOut==n):
            uSnap[k]=u.copy()
            tSnap[k]=t

        stepChange=np.max(np.abs(u-uOld))/U0
        if n%max(1,round(nSteps/10))==0:
            print(f't = {t:7.3f} s | {algorithm} iterations {it:3d} | max|u^(n+1)-u^n|/U0 = {stepChange:.2e}')
        if stopWhenSteady and stepChange<tolSteady:
            print(f'Steady state reached at t = {t:.2f} s (step {n})')
            uSnap[-1]=u.copy(); tSnap[-1]=t # Final state as last snapshot
            break


    algorithmCPUTime= clock.process_time()-algorithmStartTime
    print(f'{algorithm} total CPU time: {algorithmCPUTime:.3f} s')

    nInner=nInner[:n]
    averageIterations=np.mean(nInner)
    print(f'{algorithm} iterations per time step: min {np.min(nInner)}, average {averageIterations:.2f}, max {np.max(nInner)}')

    # Save iteration history for plotting in LabPlot or other software.
    iterationData=np.column_stack((np.arange(1,len(nInner)+1),np.arange(1,len(nInner)+1)*dt,nInner))
    iterationFile=os.path.join(outputFolder,'iteration_history.csv')
    np.savetxt(iterationFile,iterationData,delimiter=',',header='TimeStep,Time_s,Iterations',comments='',fmt=['%d','%.12g','%d'])
    print(f'Saved iteration history: {iterationFile}')

    # ====================== 6. Comparison with exact solutions ==============
    # Wall speeds for the exact solutions: the plates are taken to move at
    # constant speed for t > 0 (true for the couette and stokes1 cases)
    if caseName=='bfs':
        cellCenteredUVelocity=0.5*(u[:-1,:]+u[1:,:])
        cellCenteredVVelocity=0.5*(v[:,:-1]+v[:,1:])
        normalisedUVelocity=cellCenteredUVelocity/U0
        normalisedVVelocity=cellCenteredVVelocity/U0
        velocityMagnitude=np.sqrt(normalisedUVelocity**2+normalisedVVelocity**2)
        pressureGridXCoordinates=xc/S
        pressureGridYCoordinates=yc/S
        # ============================================================
        # 4a. Locate wall separation and reattachment points
        # ============================================================

        def findVelocitySignChanges(velocityProfile,xCoordinates):
            crossingLocations=[]

            for i in range(len(velocityProfile)-1):
                velocityLeft=velocityProfile[i]
                velocityRight=velocityProfile[i+1]

                if velocityLeft*velocityRight<0:
                    xLeft=xCoordinates[i]
                    xRight=xCoordinates[i+1]

                    zeroCrossing=xLeft-velocityLeft*(xRight-xLeft)/(velocityRight-velocityLeft)
                    crossingLocations.append(zeroCrossing)

            return crossingLocations

        bottomWallCrossings=findVelocitySignChanges(
            normalisedUVelocity[:,0],pressureGridXCoordinates
        )

        if len(bottomWallCrossings)==2:
            bottomWallCrossings=bottomWallCrossings[1:]

        topWallCrossings=findVelocitySignChanges(
            normalisedUVelocity[:,-1],pressureGridXCoordinates
        )

        print(f'Bottom wall zero crossings (x/S): {bottomWallCrossings}')
        print(f'Top wall zero crossings (x/S): {topWallCrossings}')

        # ============================================================
        # 1. Compute velocity field at pressure cell centres
        # ============================================================

        cellCenteredUVelocity=0.5*(u[:-1,:]+u[1:,:])
        cellCenteredVVelocity=0.5*(v[:,:-1]+v[:,1:])

        normalisedUVelocity=cellCenteredUVelocity/U0
        normalisedVVelocity=cellCenteredVVelocity/U0

        velocityMagnitude=np.sqrt(normalisedUVelocity**2+normalisedVVelocity**2)

        # ============================================================
        # 2. Prepare pressure grid coordinates and velocity fields
        # ============================================================

        pressureGridXCoordinates=xc/S
        pressureGridYCoordinates=yc/S

        # Matplotlib expects arrays of shape (Ny,Nx)
        streamplotUVelocity=normalisedUVelocity.T
        streamplotVVelocity=normalisedVVelocity.T
        streamplotVelocityMagnitude=velocityMagnitude.T

        xPlotMesh,yPlotMesh=np.meshgrid(
            pressureGridXCoordinates,
            pressureGridYCoordinates
        )

        # ============================================================
        # 3. Prepare downstream velocity profiles
        # ============================================================

        downstreamStations=[1,2,4,6,8,10,15,20,30,40]
        downstreamProfileData=[]

        for station in downstreamStations:
            nearestStreamwiseIndex=np.argmin(np.abs(xc-station*S))
            downstreamProfileData.append((
                station,
                normalisedUVelocity[nearestStreamwiseIndex,:]
            ))

        # ============================================================
        # 4. Prepare velocity vector sampling
        # ============================================================

        vectorSamplingStepX=5
        vectorSamplingStepY=2

        quiverXCoordinates=xPlotMesh[::vectorSamplingStepY,::vectorSamplingStepX]
        quiverYCoordinates=yPlotMesh[::vectorSamplingStepY,::vectorSamplingStepX]

        quiverUVelocity=streamplotUVelocity[::vectorSamplingStepY,::vectorSamplingStepX]
        quiverVVelocity=streamplotVVelocity[::vectorSamplingStepY,::vectorSamplingStepX]
        quiverVelocityMagnitude=streamplotVelocityMagnitude[::vectorSamplingStepY,::vectorSamplingStepX]

        # ============================================================
        # 5. Plot results
        # ============================================================

        figure,axes=plt.subplots(3,1,num='Backward Facing Step',figsize=(16,15),
                                 gridspec_kw={'height_ratios':[1,1,1.25]})
        
        # ------------------------------------------------------------
        # Bottom subplot: downstream velocity profiles
        # ------------------------------------------------------------

        profileAxis=axes[2]

        for station,velocityProfile in downstreamProfileData:
            profileAxis.plot(
                velocityProfile,
                pressureGridYCoordinates,
                linewidth=2,
                label=f'x/S = {station}'
            )

        profileAxis.axvline(0,color='black',linewidth=0.8)
        profileAxis.set_xlabel(r'$u/U_0$')
        profileAxis.set_ylabel(r'$y/S$')
        profileAxis.set_title('Backward-Facing Step\nDownstream Velocity Profiles')
        profileAxis.grid(True,alpha=0.3)
        profileAxis.legend(loc='upper right',fontsize=9)

        # ------------------------------------------------------------
        # Top subplot: velocity vectors and streamlines
        # ------------------------------------------------------------

        flowFieldAxis=axes[0]

        # Streamlines coloured using solid color
        streamlinePlot=flowFieldAxis.streamplot(
            pressureGridXCoordinates,
            pressureGridYCoordinates,
            streamplotUVelocity,
            streamplotVVelocity,
            color='lightgrey',
            density= 2.50,
            linewidth= 1.5,
            arrowsize=1.0
        )

        # Velocity vectors coloured using inferno
        quiverPlot=flowFieldAxis.quiver(
            quiverXCoordinates,
            quiverYCoordinates,
            quiverUVelocity,
            quiverVVelocity,
            quiverVelocityMagnitude,
            headwidth=3,
            headlength=4,
            headaxislength=3,
            cmap='inferno',
            angles='xy',
            scale_units='xy',
            alpha=0.95
        )
        figure.canvas.draw()
        quiverPlot.scale*=2


        # ------------------------------------------------------------
        # Mark wall separation and reattachment locations
        # ------------------------------------------------------------

        for i,location in enumerate(bottomWallCrossings):
            flowFieldAxis.axvline(
                location,
                color='darkorange',
                linestyle='--',
                linewidth=2,
                label=f'Bottom wall ({i+1}): x/S = {location:.2f}'
            )

        if len(topWallCrossings)>0:
            flowFieldAxis.axvline(
                topWallCrossings[0],
                color='darkgreen',
                linestyle='--',
                linewidth=2,
                label=f'Top wall (1): x/S = {topWallCrossings[0]:.2f}'
            )

        if len(topWallCrossings)>1:
            flowFieldAxis.axvline(
                topWallCrossings[1],
                color='darkviolet',
                linestyle='--',
                linewidth=2,
                label=f'Top wall (2): x/S = {topWallCrossings[1]:.2f}'
            )

        flowFieldAxis.legend(loc='upper right',fontsize=12)

        # Step face
        flowFieldAxis.plot([0,0],[0,1],'k-',linewidth=4)

        flowFieldAxis.set_xlabel(r'$x/S$')
        flowFieldAxis.set_ylabel(r'$y/S$')
        flowFieldAxis.set_title('Backward-Facing Step\nVelocity Vectors and Streamlines')
        flowFieldAxis.set_xlim(0,40)
        flowFieldAxis.set_ylim(0,H/S)

        # ------------------------------------------------------------
        # Middle subplot: cell Peclet number heatmap
        # ------------------------------------------------------------

        # Directional face Peclet numbers F/D for the pressure cells.
        pecletEast=rho*np.abs(u[1:,:])*dx/mu
        pecletWest=rho*np.abs(u[:-1,:])*dx/mu
        pecletNorth=rho*np.abs(v[:,1:])*dy/mu
        pecletSouth=rho*np.abs(v[:,:-1])*dy/mu
        cellPecletNumber=np.maximum.reduce((pecletEast,pecletWest,pecletNorth,pecletSouth))

        pecletAxis=axes[1]
        pecletPlot=pecletAxis.pcolormesh(
            np.arange(Nx+1)*dx/S,
            np.arange(Ny+1)*dy/S,
            cellPecletNumber.T,
            cmap='inferno',
            shading='flat'
        )
        pecletAxis.set_xlabel(r'$x/S$')
        pecletAxis.set_ylabel(r'$y/S$')
        pecletAxis.set_title('Backward-Facing Step\nMaximum Cell Péclet Number (F/D)')
        pecletAxis.set_xlim(0,L/S)
        pecletAxis.set_ylim(0,H/S)
        pecletColorbar=figure.colorbar(pecletPlot,ax=pecletAxis,fraction=0.046,pad=0.12)
        pecletColorbar.set_label(r'Maximum directional cell $Pe = |F|/D$')

        print(f'Cell Peclet numbers: max {np.max(cellPecletNumber):.3f}, '
              f'cells above 2: {np.count_nonzero(cellPecletNumber>2)}/{cellPecletNumber.size}')

        # ------------------------------------------------------------
        # 6. Add colour bars
        # ------------------------------------------------------------
        vectorColorbar=figure.colorbar(
            quiverPlot,
            ax=flowFieldAxis,
            fraction=0.046,
            pad=0.12
        )
        vectorColorbar.set_label(r'Vector colour: $|\mathbf{V}|/U_0$')

        figure.subplots_adjust(
            left=0.08,
            right=0.88,
            top=0.96,
            bottom=0.06,
            hspace=0.48
        )

    elif(caseName != 'bfs'):
        parameters={'U0':U0,'H':H,'nu':nu,'mu':mu,'dpdx':dpdx,'omega':omega,'Ubot':uSouth(tEnd),'Utop':uNorth(tEnd)}
        keep=np.flatnonzero(~np.isnan(tSnap)) # Snapshots actually reached
        yy=np.linspace(0,H,400)
        print('\n   t [s]   max|u - u_exact| / U0')
        for k in keep:
            if isPer:
                uNum=np.mean(uSnap[k][:Nx,:],axis=0) # No x-variation: average over x
            else:
                uNum=uSnap[k][Nx,:]                 # Channel: outlet profile
            error=np.max(np.abs(uNum-ExactProfile(caseName,yc,tSnap[k],parameters)))/U0
            print(f'  {tSnap[k]:7.3f}   {error:.2e}')
        if caseName=='channel':
            print('(channel: compared with the fully developed Poiseuille profile at the outlet)')


        # ---- Stokes second problem: amplitude and phase ----
        if caseName=='stokes2':
            timeHistory=np.array(timeHistory)
            velocityHistory=np.array(velocityHistory)

            fitMatrix=np.column_stack((
                np.cos(omega*timeHistory),
                np.sin(omega*timeHistory),
                np.ones_like(timeHistory)
            ))

            a,b,c=np.linalg.lstsq(
                fitMatrix,velocityHistory,rcond=None
            )[0]

            amplitude=np.sqrt(a**2+b**2)
            phaseLag=np.arctan2(b,a)

            exactAmplitude=np.exp(-1)
            exactPhaseLag=1.0

            stepsPerPeriod=round(T/dt)
            cellsPerDelta=delta/dy

            amplitudeError=100*abs(amplitude-exactAmplitude)/exactAmplitude
            phaseError=abs(phaseLag-exactPhaseLag)

            print('\n--- Stokes II: Convergence results ---')
            print(f'Steps per period   : {stepsPerPeriod}')
            print(f'Ny                 : {Ny}')
            print(f'Cells per delta    : {cellsPerDelta:.4f}')
            print(f'Amplitude          : {amplitude:.8f}')
            print(f'Exact amplitude    : {exactAmplitude:.8f}')
            print(f'Amplitude error [%]: {amplitudeError:.4f}')
            print(f'Phase lag [rad]    : {phaseLag:.8f}')
            print(f'Exact lag [rad]    : {exactPhaseLag:.8f}')
            print(f'Phase error [rad]  : {phaseError:.8e}')

            if amplitudeError<=1.0:
                print('Amplitude accuracy : PASS (within 1%)')
            else:
                print('Amplitude accuracy : FAIL (above 1%)')

    # ====================== 7. Figures ======================================


    # ====================== 7. Figures ======================================
    plt.rcParams.update({'font.size':12,'axes.titlesize':17,'axes.titleweight':'semibold',
                         'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,
                         'legend.fontsize':11,'figure.facecolor':'white',
                         'savefig.dpi':200})
    colors=plt.get_cmap('tab10')(np.linspace(0,1,len(tOut)))

    # ---- Figure 1: velocity profiles vs exact solution ----

    # ====================== ADDED CODE START ======================
    if caseName=='stokes1':
        fig,(ax,axError)=plt.subplots(1,2,num='Velocity profiles',figsize=(15,7))
        maximumAbsoluteError=np.zeros(len(keep))
    elif (caseName != 'bfs'):
        fig,ax=plt.subplots(num='Velocity profiles',figsize=(11,8))
    # ====================== ADDED CODE END ========================

    if(caseName != 'bfs'):
        for k in keep:
            uNum=np.mean(uSnap[k][:Nx,:],axis=0) if isPer else uSnap[k][Nx,:]
            ax.plot(ExactProfile(caseName,yy,tSnap[k],parameters)/U0,yy/H,'-',color=colors[k],linewidth=2,label='_nolegend_')
            step=max(1,round(Ny/25))  # Thin out the symbols
            if caseName=='stokes2':
                phase=round((omega*tSnap[k]%(2*np.pi))*180/np.pi)
                label=rf'$\omega t = {phase}^\circ$'
            else:
                label=f't = {tSnap[k]:g} s'
            ax.plot(uNum[::step]/U0,yc[::step]/H,'o',color=colors[k],markersize=5,label=label)

            # ====================== ADDED CODE START ======================
            if caseName=='stokes1':
                analyticalVelocity=ExactProfile(caseName,yc,tSnap[k],parameters)
                maximumAbsoluteError[np.where(keep==k)[0][0]]=np.max(np.abs(uNum-analyticalVelocity))/U0
            # ====================== ADDED CODE END ========================
            ax.set_xlabel(r'Normalised velocity, $u/U_0$',labelpad=10)
            ax.set_ylabel(r'Normalised height, $y/H$',labelpad=10)
            if caseName=='channel':
                ax.set_title('Developing Channel Flow\nFVM vs Fully Developed Poiseuille Profile',pad=18)
            elif caseName=='couette':
                ax.set_title('Start-up Couette Flow\nFVM vs Analytical Series Solution',pad=18)
            elif caseName=='stokes1':
                ax.set_title('Stokes First Problem\nFVM vs Analytical erfc Solution',pad=18)
                ax.set_ylim(0,0.5)
            elif caseName=='stokes2':
                ax.set_title('Stokes Second Problem\nFVM vs Analytical Periodic Solution',pad=18)
                ax.set_ylim(0,0.35)
            ax.grid(True,alpha=0.3)
            ax.tick_params(axis='both',which='major',length=5)
            if len(keep)>0:
                ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.16),ncol=min(4,len(keep)),
                        frameon=False,title='Time / Phase',title_fontsize=12)

    # ====================== ADDED CODE START ======================
    if caseName=='stokes1':
        axError.plot(tSnap[keep],maximumAbsoluteError,'o-',color='tab:red',linewidth=2,markersize=6)
        axError.set_xlabel('Time [s]',labelpad=10)
        axError.set_ylabel('Max Absolute Error [m/s]',labelpad=10)
        axError.set_title('Max Absolute Error vs Time',pad=18)
        axError.grid(True,alpha=0.3)
        axError.tick_params(axis='both',which='major',length=5)
        fig.subplots_adjust(left=0.08,right=0.96,top=0.85,bottom=0.29,wspace=0.3)
    else:
        if(caseName != 'bfs'):
            fig.subplots_adjust(left=0.12,right=0.96,top=0.85,bottom=0.29)

    # ====================== ADDED CODE END ========================

    # ====================== Stokes 1: Similarity Solution ======================
    if caseName=='stokes1':
        fig,ax=plt.subplots(num='Stokes 1 Similarity Solution',figsize=(10,7))

        for k in keep:
            t=tSnap[k]
            eta=yc/(2*np.sqrt(nu*t))
            uNum=np.mean(uSnap[k][:Nx,:],axis=0)

            ax.plot(eta,uNum/U0,'o',color=colors[k],markersize=3,
                    label=f't = {t:g} s')

        etaExact=np.linspace(0,3,400)
        ax.plot(etaExact,erfc(etaExact),'k-',linewidth=2,
                label='Analytical Solution')

        ax.set_xlabel(r'$\eta=y/(2\sqrt{\nu t})$',labelpad=10)
        ax.set_ylabel(r'$u/U_0$',labelpad=10)
        ax.set_title('Stokes First Problem\nSimilarity Profile',pad=18)
        ax.set_xlim(0,3)
        ax.set_ylim(0,1.05)
        ax.grid(True,alpha=0.3)
        ax.legend(loc='upper right',frameon=False)

        fig.tight_layout()
    # ====================== End Similarity Solution ===========================


    # ====================== Stokes 1: Wall Shear Stress ======================
    if caseName=='stokes1':
        timeValues=tSnap[keep]
        numericalShear=np.zeros(len(keep))
        analyticalShear=-mu*U0/np.sqrt(np.pi*nu*timeValues)

        for i,k in enumerate(keep):
            uNum=np.mean(uSnap[k][:Nx,:],axis=0)
            numericalShear[i]=mu*(uNum[0]-U0)/(dy/2)

        fig,ax=plt.subplots(num='Stokes 1 Wall Shear Stress',figsize=(10,7))

        ax.loglog(timeValues,-numericalShear,'o-',linewidth=2,markersize=6,
                  label='FVM Solution')
        ax.loglog(timeValues,-analyticalShear,'k--',linewidth=2,
                  label='Analytical Solution')

        ax.set_xlabel('Time [s]',labelpad=10)
        ax.set_ylabel(r'$|\tau_w|$ [Pa]',labelpad=10)
        ax.set_title('Stokes First Problem\nWall Shear Stress',pad=18)
        ax.grid(True,which='both',alpha=0.3)
        ax.legend(loc='upper right',frameon=False)

        fig.tight_layout()
    # ====================== End Wall Shear Stress ===========================


    # ---- Figure 2 (channel only): flow field and centreline development ----
    if caseName=='channel' and len(keep)>0:
        fig,axes=plt.subplots(2,1,num='Channel flow field',figsize=(12,10))
        fig.suptitle('Developing Channel Flow: Velocity Field and Centreline Evolution',
                     fontsize=18,fontweight='semibold',y=0.97)
        xGrid,yGrid=np.meshgrid(xu,np.concatenate(([0],yc,[H])))
        uLast=uSnap[keep[-1]]
        velocityField=np.column_stack((np.zeros(Nx+1),uLast,np.zeros(Nx+1))).T
        contour=axes[0].contourf(xGrid,yGrid,velocityField,levels=30)
        colorbar=fig.colorbar(contour,ax=axes[0],pad=0.02)
        colorbar.set_label(r'Velocity, $u$ [m/s]',fontsize=13)
        colorbar.ax.tick_params(labelsize=11)
        axes[0].set_xlabel('Streamwise distance, x [m]')
        axes[0].set_ylabel('Height, y [m]')
        axes[0].set_title(f'Streamwise Velocity at t = {tSnap[keep[-1]]:.1f} s',pad=12)
        for k in keep:
            uc=uSnap[k]
            axes[1].plot(xu,0.5*(uc[:,Ny//2-1]+uc[:,Ny//2])/U0,
                         color=colors[k],linewidth=2,label=f't = {tSnap[k]:g} s')
        axes[1].plot([0,L],[1.5,1.5],'k--',linewidth=1.8,label='Fully developed')
        axes[1].set_xlabel('Streamwise distance, x [m]')
        axes[1].set_ylabel(r'Centreline velocity, $u_c/U_0$')
        axes[1].grid(True,alpha=0.3)
        axes[1].set_title('Centreline Velocity Development',pad=12)
        axes[1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.3),
                       ncol=4,frameon=False)
        fig.subplots_adjust(left=0.12,right=0.91,top=0.88,bottom=0.2,hspace=0.64)
        '''
        =======
        ANSWER A1.2
        =======
        '''
        # Centreline velocity from the final solution
        centrelineVelocity=0.5*(uLast[:,Ny//2-1]+uLast[:,Ny//2])
        targetVelocity=0.99*1.5*U0
        # First x-location where centreline velocity reaches 99%
        indices=np.where(centrelineVelocity>=targetVelocity)[0]
        if len(indices)>0:
            i99=indices[0]
            entryLengthNumerical=xu[i99]
            axes[0].axvline(entryLengthNumerical,color='purple',linestyle='--',linewidth=2.5,label=f'Numerical: {entryLengthNumerical:.3f} m')
        # Analytical entrance length
        axes[0].axvline(entryLengthAnalytical,color='red',linestyle='--',linewidth=2.5,label=f'Analytical: {entryLengthAnalytical:.3f} m')
        # Entrance length legend
        axes[0].legend(loc='upper right',fontsize=10,frameon=True,facecolor='white',framealpha=0.9)
        # Print entrance lengths
        print('\nHydrodynamic Entrance Lengths')
        print(f'Analytical : {entryLengthAnalytical:.6f} m')
        if entryLengthNumerical is not None:
            print(f'Numerical  : {entryLengthNumerical:.6f} m')
            print(f'Difference : {abs(entryLengthNumerical-entryLengthAnalytical):.6f} m')
        else:
            print('Numerical  : 99% threshold not reached within the domain')
        '''
        =======
        '''

    '''
    =======
    ANSWER A3.3
    =======
    '''
    if caseName=='stokes2':
        k=np.sqrt(omega/(2*nu))
        time=np.linspace(tEnd-T,tEnd,500)
        phase=omega*time

        plateVelocity=np.cos(phase)
        wallShear=-np.cos(phase+np.pi/4)

        fig,ax=plt.subplots(num='Stokes 2 Wall Shear Stress',figsize=(11,6))

        ax.plot(time,plateVelocity,'b-',linewidth=2,
                label=r'Plate velocity, $U_w/U_0$')

        ax.plot(time,wallShear,'r--',linewidth=2,
                label=r'Wall shear, $\tau_w/(\sqrt{2}\mu kU_0)$')

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Normalised value')
        ax.set_title('Stokes Second Problem: Wall Shear Stress')
        ax.grid(True,alpha=0.3)
        ax.legend()

        fig.tight_layout()


    # ---- Figure 3: SIMPLE iterations needed per time step ----
    fig,ax=plt.subplots(num=f'{algorithm} iterations',figsize=(11,6.5))
    ax.plot(np.arange(1,len(nInner)+1)*dt,nInner,'b-',linewidth=2,
            label=f'{algorithm} iterations per time step')
    ax.axhline(averageIterations,color='red',linestyle=':',linewidth=2,
               label=f'Average = {averageIterations:.2f} iterations')
    ax.grid(True,alpha=0.3)
    ax.set_xlabel('Simulation time, t [s]',labelpad=10)
    ax.set_ylabel(f'Number of {algorithm} iterations',labelpad=10)
    ax.set_title(f'{algorithm} Algorithm: Convergence Iteration History',pad=18)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.19),frameon=False)
    fig.subplots_adjust(left=0.12,right=0.96,top=0.86,bottom=0.26)

    # Save all figures before showing them (including case-specific plots).
    for figureNumber in plt.get_fignums():
        currentFigure=plt.figure(figureNumber)
        figureName=currentFigure.get_label() or 'Flow Field'
        figureName=figureName.replace(' ','_')
        figurePath=os.path.join(outputFolder,f'{figureNumber:02d}_{figureName}.png')
        currentFigure.savefig(figurePath,dpi=200,bbox_inches='tight')
        print(f'Saved figure: {figurePath}')

    plt.show()

if __name__=='__main__':
    originalStdout=sys.stdout
    originalStderr=sys.stderr
    sys.stdout=ConsoleLogger(originalStdout)
    sys.stderr=ConsoleLogger(originalStderr)
    print('='*90, '\n')
    print('ME630 Assignment 3 | Meghaj Nandkumar Barhate | 261050047')
    print('\n'+'='*90)

    print('\nAvailable cases:')
    print('1.  channel  - Developing flow between two fixed plates, started from rest')
    print('2.  couette  - Start-up of Couette flow: upper plate starts moving')
    print('3.  stokes1  - Stokes first problem: plate impulsively started')
    print('4.  stokes2  - Stokes second problem: plate oscillating in its own plane')
    print('5.  bfs      - Backward-facing step flow (Armaly geometry)')
    case = int(input('Enter the case number : '))
    print('\n'+'='*90)

    print('Available algorithms:')
    print('1.  SIMPLE   - Standard SIMPLE algorithm')
    print('2.  SIMPLEC  - SIMPLEC algorithm')
    print('3.  SIMPLER  - SIMPLER algorithm')
    algoSelect = input('Enter the algorithm (SIMPLE, SIMPLEC, SIMPLER) : ')
    if algoSelect == '1':
        algorithm = 'SIMPLE'
    elif algoSelect == '2':
        algorithm = 'SIMPLEC'
    elif algoSelect == '3':
        algorithm = 'SIMPLER'
    else:
        print('Invalid algorithm selection. Defaulting to SIMPLE.')
        algorithm = 'SIMPLE'
    print('\n'+'='*90)

    print('Available convection schemes:')
    print('1.  Upwind   - First-order upwind scheme')
    print('2.  Central  - Second-order central differencing scheme')
    print('3.  Central Difference - Deferred correction scheme')
    schemeSelect = input('Enter the convection scheme (UDS, CDS, CDS-DC) : ')
    if schemeSelect == '1':
        convectionScheme = 'UDS'
    elif schemeSelect == '2':
        convectionScheme = 'CDS'
    elif schemeSelect == '3':
        convectionScheme = 'CDS-DC'
    else:
        print('Invalid convection scheme selection. Defaulting to Upwind.')
        convectionScheme = 'UDS'

    print('\n'+'='*90)

    try:
        main(case, algorithm, convectionScheme)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        consoleLogger=sys.stdout
        sys.stdout=originalStdout
        sys.stderr=originalStderr
        if isinstance(consoleLogger,ConsoleLogger):
            consoleLogger.close()
