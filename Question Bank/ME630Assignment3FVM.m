%% =========================================================================
%  unsteady_SIMPLE_FVM.m
%
%  2D UNSTEADY incompressible laminar flow solver for teaching
%  Finite Volume Method | Staggered grid | SIMPLE algorithm
%  Convection: first-order upwind | Time: implicit (backward) Euler
%
%  ONE code, several problems. Only Section 1 (CASE SETUP) changes between
%  problems: geometry, fluid properties, boundary conditions, time step.
%  The discretisation and the SIMPLE loop (Sections 3-5) stay the same.
%
%  Cases provided (set caseName in Section 1):
%    'channel'  developing flow between two fixed plates, started from rest
%               (inlet / outlet in x, no-slip walls)
%    'couette'  start-up of Couette flow: upper plate starts moving
%               (periodic in x, fixed lower wall, moving upper wall);
%               add a pressure gradient for Couette-Poiseuille flow
%    'stokes1'  Stokes first problem: plate impulsively started
%               (periodic in x, moving lower wall, far-field upper boundary)
%    'stokes2'  Stokes second problem: plate oscillating in its own plane
%               (periodic in x, oscillating lower wall, far field above)
%
%  Boundary-condition options available in Section 1:
%    x-direction:   xBC = 'inlet-outlet'  (velocity inlet at x = 0,
%                                          zero-gradient outlet at x = L)
%                   xBC = 'periodic'      (infinite in x, flow repeats)
%    y-boundaries:  southBC / northBC = 'wall'  no-slip wall moving
%                                               tangentially with uSouth(t)
%                                               or uNorth(t)
%                                    = 'slip'  zero shear (symmetry plane
%                                               or far field)
%    Mean pressure gradient dpdx (periodic cases): drives the flow as a
%    uniform source -dpdx in the x-momentum equation.
%
%  Governing equations (constant rho, mu):
%     d(rho u)/dx + d(rho v)/dy = 0
%     d(rho u)/dt + div(rho u U) = -dp/dx + div(mu grad u) - dpdx
%     d(rho v)/dt + div(rho v U) = -dp/dy + div(mu grad v)
%
%  Discrete momentum equation for each velocity (Versteeg & Malalasekera,
%  Ch. 6 and 8), implicit Euler in time:
%     aP*phi_P = sum(a_nb*phi_nb) + b
%     a_nb (upwind):  aE = De + max(-Fe,0)   aW = Dw + max(Fw,0)
%                     aN = Dn + max(-Fn,0)   aS = Ds + max(Fs,0)
%     aP = sum(a_nb) + aP0,      aP0 = rho*dx*dy/dt
%     b  = (pressure difference)*area + aP0*phi_P^old + other sources
%
%  Staggered grid (as in channel_flow_FVM.m):
%     p(i,j) at cell centres     i = 1..Nx,   j = 1..Ny
%     u(i,j) on west faces       i = 1..Nx+1, j = 1..Ny
%     v(i,j) on south faces      i = 1..Nx,   j = 1..Ny+1
%  In periodic cases face Nx+1 is the same face as face 1; the code keeps
%  u(Nx+1,:) = u(1,:) at all times.
%
%  Requires MATLAB R2016b or newer (local functions at the end of a script).
%% =========================================================================

clear; clc; close all;

%% ====================== 1. CASE SETUP (students edit here) ==============
caseName = 'stokes2';      % 'channel' | 'couette' | 'stokes1' | 'stokes2'

switch caseName

    case 'channel'          % --- developing channel flow, started from rest
        L = 10;   H = 1;            % domain length and height [m]
        Nx = 100; Ny = 20;          % number of pressure cells
        rho = 1;  mu = 0.01;        % Re = rho*U0*H/mu = 100
        U0  = 1;                    % inlet velocity [m/s]
        xBC = 'inlet-outlet';
        uInlet  = @(y,t) U0*ones(size(y));   % uniform inflow
        southBC = 'wall';  uSouth = @(t) 0;  % fixed lower wall
        northBC = 'wall';  uNorth = @(t) 0;  % fixed upper wall
        dpdx = 0;                   % (not used with inlet-outlet)
        dt   = 0.25;  tEnd = 60;    % time step and end time [s]
        tOut = [1 2 5 10 20 60];    % times at which results are stored
        stopWhenSteady = true;      % end the run once the flow is steady

    case 'couette'          % --- start-up of Couette flow
        L = 0.1;  H = 1;
        Nx = 4;   Ny = 40;          % no x-variation: a few cells suffice
        rho = 1;  mu = 0.1;         % viscous time H^2/nu = 10 s
        U0  = 1;                    % speed of the upper plate
        xBC = 'periodic';
        southBC = 'wall';  uSouth = @(t) 0;
        northBC = 'wall';  uNorth = @(t) U0;   % upper plate starts at t = 0
        dpdx = 0;                   % try -0.4 for Couette-Poiseuille flow,
                                    % or uNorth = 0, dpdx < 0 for start-up
                                    % of plane Poiseuille flow
        dt   = 0.02;  tEnd = 10;
        tOut = [0.1 0.5 1 2 5 10];
        stopWhenSteady = false;

    case 'stokes1'          % --- Stokes first problem (Rayleigh problem)
        L = 0.02; H = 1;            % H must be >> penetration depth
        Nx = 4;   Ny = 200;
        rho = 1;  mu = 0.01;
        U0  = 1;
        xBC = 'periodic';
        southBC = 'wall';  uSouth = @(t) U0;   % plate impulsively started
        northBC = 'slip';  uNorth = @(t) 0;    % far field: zero shear
        dpdx = 0;
        dt   = 0.002;  tEnd = 1;
        tOut = [0.05 0.1 0.25 0.5 1];
        stopWhenSteady = false;

    case 'stokes2'          % --- Stokes second problem (oscillating plate)
        L = 0.02; H = 1;
        Nx = 4;   Ny = 200;
        rho = 1;  mu = 0.01;
        U0  = 1;  omega = 2*pi;     % amplitude and angular frequency
        T   = 2*pi/omega;           % period [s]
        xBC = 'periodic';
        southBC = 'wall';  uSouth = @(t) U0*cos(omega*t);
        northBC = 'slip';  uNorth = @(t) 0;
        dpdx = 0;
        dt   = T/400;  tEnd = 4*T;  % 4 periods; start-up transient decays fast
        tOut = tEnd - T + (0:7)*T/8;            % 8 phases of the last period
        stopWhenSteady = false;

    otherwise
        error('Unknown caseName "%s"', caseName);
end
if ~exist('omega', 'var'), omega = 0; end

%% ---------------------- SIMPLE controls ---------------------------------
alpha_u  = 0.8;            % momentum under-relaxation (within a time step)
alpha_p  = 0.3;            % pressure under-relaxation
maxInner = 200;            % max SIMPLE iterations per time step
tolInner = 1e-6;           % convergence of each time step
tolSteady = 1e-7;          % |u^{n+1} - u^n|/U0 below this -> steady

%% ====================== 2. Grid and initial fields ======================
dx = L/Nx;   dy = H/Ny;
xc = ((1:Nx) - 0.5)*dx;   yc = ((1:Ny) - 0.5)*dy;
xu = (0:Nx)*dx;           yv = (0:Ny)*dy;
nu = mu/rho;
isPer = strcmp(xBC, 'periodic');

% Initial condition: fluid at rest
u = zeros(Nx+1, Ny);
v = zeros(Nx,   Ny+1);
p = zeros(Nx,   Ny);
if ~isPer
    u(1,:) = uInlet(yc, 0);                 % inlet face
end
dU = zeros(Nx+1, Ny);      % d-coefficients; stay zero on inlet/outlet faces
dV = zeros(Nx,   Ny+1);    % (those faces are never corrected)

% Index sets: which u-faces are unknowns, and their neighbours
if isPer
    Iu  = 1:Nx;                    % all faces (face Nx+1 = face 1)
    IuE = [2:Nx, 1];               % east neighbour face (wraps round)
    IuW = [Nx, 1:Nx-1];            % west neighbour face
    cW  = [Nx, 1:Nx-1];            % pressure cell west of each face
    cE  = 1:Nx;                    % pressure cell east of each face
else
    Iu  = 2:Nx;                    % interior faces (inlet 1, outlet Nx+1 known)
    IuE = 3:Nx+1;
    IuW = 1:Nx-1;
    cW  = 1:Nx-1;
    cE  = 2:Nx;
end

aP0 = rho*dx*dy/dt;        % transient coefficient rho*V/dt (u and v CVs)
fRef = rho*U0*H;           % reference mass flux for the residual

nSteps = round(tEnd/dt);
nOut   = round(tOut/dt);   % time-step numbers at which fields are stored
uSnap  = cell(1, numel(tOut));
tSnap  = NaN(1, numel(tOut));
nInner = zeros(1, nSteps); % SIMPLE iterations used per time step

fprintf('Case "%s": %d x %d cells, dt = %g s, %d steps\n', caseName, Nx, Ny, dt, nSteps);
fprintf('Courant number U0*dt/dx = %.2f, diffusion number nu*dt/dy^2 = %.2f\n', ...
        U0*dt/dx, nu*dt/dy^2);
fprintf('(implicit scheme: no stability limit, but accuracy needs small dt)\n');

%% ====================== 3. Time loop ====================================
for n = 1:nSteps
    t = n*dt;                       % new time level t^{n+1}
    uOld = u;  vOld = v;            % solution at the old time level t^n

    % Boundary values at the new time level
    uS = uSouth(t);   uN = uNorth(t);
    if ~isPer, u(1,:) = uInlet(yc, t); end

    %% ================= 4. SIMPLE iterations within the time step ========
    for it = 1:maxInner
        u_it = u;  v_it = v;        % previous iterate (for relaxation/check)

        %% --------------- 4a. u-momentum ---------------------------------
        % Mass fluxes through the four faces of each u control volume
        Fe = rho*0.5*(u(Iu,:)  + u(IuE,:))*dy;
        Fw = rho*0.5*(u(IuW,:) + u(Iu,:) )*dy;
        Fn = rho*0.5*(v(cW,2:Ny+1) + v(cE,2:Ny+1))*dx;
        Fs = rho*0.5*(v(cW,1:Ny)   + v(cE,1:Ny)  )*dx;

        % Diffusion conductances  D = mu*area/distance
        nU = numel(Iu);
        De = mu*dy/dx*ones(nU,Ny);  Dw = De;
        Dn = mu*dx/dy*ones(nU,Ny);  Ds = Dn;
        Ds(:,1)  = wallConductance(southBC, mu, dx, dy);   % y = 0 boundary
        Dn(:,Ny) = wallConductance(northBC, mu, dx, dy);   % y = H boundary

        % First-order upwind neighbour coefficients
        aE = De + max(-Fe, 0);
        aW = Dw + max( Fw, 0);
        aN = Dn + max(-Fn, 0);
        aS = Ds + max( Fs, 0);

        % Central coefficient and source (implicit Euler: aP0 terms)
        aP = aE + aW + aN + aS + aP0;
        b  = (p(cW,:) - p(cE,:))*dy ...        % pressure force (p_W - p_E)*A
           - dpdx*dx*dy ...                    % imposed mean pressure gradient
           + aP0*uOld(Iu,:);                   % old time level

        % ---- boundary neighbours: known values moved to the source ----
        b(:,1)  = b(:,1)  + aS(:,1)*uS;   aS(:,1)  = 0;   % wall at y = 0
        b(:,Ny) = b(:,Ny) + aN(:,Ny)*uN;  aN(:,Ny) = 0;   % wall at y = H
        % ('slip' gives Ds or Dn = 0, so those lines add nothing)
        if ~isPer
            b(1,:)   = b(1,:)   + aW(1,:).*u(1,:);      aW(1,:)   = 0;  % inlet
            b(end,:) = b(end,:) + aE(end,:).*u(Nx+1,:); aE(end,:) = 0;  % outlet
        end

        % Implicit under-relaxation (does not change the converged answer)
        aP = aP/alpha_u;
        b  = b + (1 - alpha_u)*aP.*u(Iu,:);

        dU(Iu,:) = dy./aP;                    % SIMPLE: u' = dU*(p'_W - p'_E)
        u(Iu,:)  = solve5(aP, aE, aW, aN, aS, b, isPer);
        if isPer
            u(Nx+1,:)  = u(1,:);              % face Nx+1 is face 1
            dU(Nx+1,:) = dU(1,:);
        end

        %% --------------- 4b. v-momentum ---------------------------------
        % Unknowns: v(:, 2:Ny); v = 0 on both y-boundaries (wall or slip)
        Fe = rho*0.5*(u(2:Nx+1,1:Ny-1) + u(2:Nx+1,2:Ny))*dy;
        Fw = rho*0.5*(u(1:Nx,  1:Ny-1) + u(1:Nx,  2:Ny))*dy;
        Fn = rho*0.5*(v(:,2:Ny)   + v(:,3:Ny+1))*dx;
        Fs = rho*0.5*(v(:,1:Ny-1) + v(:,2:Ny)  )*dx;

        De = mu*dy/dx*ones(Nx,Ny-1);  Dw = De;
        Dn = mu*dx/dy*ones(Nx,Ny-1);  Ds = Dn;
        if ~isPer
            Dw(1,:)  = 2*mu*dy/dx;            % inlet (v = 0) is dx/2 away
            De(Nx,:) = 0;                     % outlet: zero gradient
        end

        aE = De + max(-Fe, 0);
        aW = Dw + max( Fw, 0);
        aN = Dn + max(-Fn, 0);
        aS = Ds + max( Fs, 0);
        aP = aE + aW + aN + aS + aP0;
        b  = (p(:,1:Ny-1) - p(:,2:Ny))*dx + aP0*vOld(:,2:Ny);

        if ~isPer
            aW(1,:)  = 0;                     % inlet: v = 0 (adds nothing)
            aE(Nx,:) = 0;                     % outlet: v_E = v_P
        end
        aS(:,1) = 0;  aN(:,end) = 0;          % v = 0 at y = 0 and y = H

        aP = aP/alpha_u;
        b  = b + (1 - alpha_u)*aP.*v(:,2:Ny);
        dV(:,2:Ny) = dx./aP;
        v(:,2:Ny)  = solve5(aP, aE, aW, aN, aS, b, isPer);

        %% --------------- 4c. outlet (inlet-outlet cases only) -----------
        if ~isPer, u = applyOutlet(u, rho, dy); end

        %% --------------- 4d. pressure-correction equation ---------------
        % a_nb = rho*d*A on faces that are corrected; dU = 0 on inlet and
        % outlet faces, so those neighbours drop out automatically.
        aE = rho*dU(2:Nx+1,:)*dy;
        aW = rho*dU(1:Nx,:)*dy;
        aN = zeros(Nx,Ny);  aS = aN;
        aN(:,1:Ny-1) = rho*dV(:,2:Ny)*dx;
        aS(:,2:Ny)   = rho*dV(:,2:Ny)*dx;
        aP = aE + aW + aN + aS;
        b  = rho*(u(1:Nx,:) - u(2:Nx+1,:))*dy + rho*(v(:,1:Ny) - v(:,2:Ny+1))*dx;
        massRes = sum(abs(b(:)))/fRef;

        % p' is fixed by the boundaries only up to a constant: set p' = 0
        % in one reference cell
        iR = Nx; jR = 1;
        aP(iR,jR) = 1; aE(iR,jR) = 0; aW(iR,jR) = 0; aN(iR,jR) = 0; aS(iR,jR) = 0;
        b(iR,jR)  = 0;
        pp = solve5(aP, aE, aW, aN, aS, b, isPer);

        %% --------------- 4e. correct p, u, v ----------------------------
        p = p + alpha_p*pp;
        u(Iu,:)   = u(Iu,:)   + dU(Iu,:).*(pp(cW,:) - pp(cE,:));
        v(:,2:Ny) = v(:,2:Ny) + dV(:,2:Ny).*(pp(:,1:Ny-1) - pp(:,2:Ny));
        if isPer
            u(Nx+1,:) = u(1,:);
        else
            u = applyOutlet(u, rho, dy);
        end

        %% --------------- 4f. convergence of this time step --------------
        change = max([abs(u(:) - u_it(:)); abs(v(:) - v_it(:))])/U0;
        if massRes < tolInner && change < tolInner, break; end
    end
    nInner(n) = it;
    if it == maxInner
        warning('Step %d: SIMPLE not converged (mass %.1e, change %.1e)', n, massRes, change);
    end

    %% ================= 5. Store results, check steady state ==============
    k = find(nOut == n);
    if ~isempty(k), uSnap{k} = u;  tSnap(k) = t; end

    stepChange = max(abs(u(:) - uOld(:)))/U0;
    if mod(n, max(1, round(nSteps/10))) == 0
        fprintf('t = %7.3f s | SIMPLE iterations %3d | max|u^{n+1}-u^n|/U0 = %.2e\n', ...
                t, it, stepChange);
    end
    if stopWhenSteady && stepChange < tolSteady
        fprintf('Steady state reached at t = %.2f s (step %d)\n', t, n);
        uSnap{end} = u;  tSnap(end) = t;   % final state as last snapshot
        break
    end
end
nInner = nInner(1:n);
fprintf('SIMPLE iterations per time step: min %d, mean %.1f, max %d\n', ...
        min(nInner), mean(nInner), max(nInner));

%% ====================== 6. Comparison with exact solutions ==============
% Wall speeds for the exact solutions: the plates are taken to move at
% constant speed for t > 0 (true for the couette and stokes1 cases)
prm = struct('U0',U0,'H',H,'nu',nu,'mu',mu,'dpdx',dpdx,'omega',omega, ...
             'Ubot',uSouth(tEnd),'Utop',uNorth(tEnd));
keep = find(~isnan(tSnap));                 % snapshots actually reached
yy   = linspace(0, H, 400);
fprintf('\n   t [s]   max|u - u_exact| / U0\n');
for k = keep
    if isPer
        uNum = mean(uSnap{k}(1:Nx,:), 1);   % no x-variation: average over x
    else
        uNum = uSnap{k}(Nx+1,:);            % channel: outlet profile
    end
    err = max(abs(uNum - exactProfile(caseName, yc, tSnap(k), prm)))/U0;
    fprintf('  %7.3f   %.2e\n', tSnap(k), err);
end
if strcmp(caseName, 'channel')
    fprintf('(channel: compared with the fully developed Poiseuille profile at the outlet)\n');
end

%% ====================== 7. Figures ======================================
cols = lines(numel(tOut));

% ---- Figure 1: velocity profiles vs exact solution ----
figure('Name', 'Velocity profiles', 'Color', 'w'); hold on; box on; grid on;
for k = keep
    if isPer
        uNum = mean(uSnap{k}(1:Nx,:), 1);
    else
        uNum = uSnap{k}(Nx+1,:);
    end
    plot(exactProfile(caseName, yy, tSnap(k), prm)/U0, yy/H, '-', ...
         'Color', cols(k,:), 'LineWidth', 1.3, 'HandleVisibility', 'off');
    step = max(1, round(Ny/25));            % thin out the symbols
    if strcmp(caseName, 'stokes2')
        lab = sprintf('\\omega t = %d^o', round(mod(omega*tSnap(k), 2*pi)*180/pi));
    else
        lab = sprintf('t = %g s', tSnap(k));
    end
    plot(uNum(1:step:end)/U0, yc(1:step:end)/H, 'o', 'Color', cols(k,:), ...
         'MarkerSize', 4, 'DisplayName', lab);
end
xlabel('u / U_0'); ylabel('y / H');
switch caseName
    case 'channel'
        title('Outlet profile (symbols) vs fully developed Poiseuille (lines)');
    case 'couette'
        title('Couette start-up: FVM (symbols) vs series solution (lines)');
    case 'stokes1'
        title('Stokes first problem: FVM (symbols) vs erfc solution (lines)');
        ylim([0 0.5]);
    case 'stokes2'
        title('Stokes second problem: FVM (symbols) vs exact (lines)');
        ylim([0 0.35]);
end
legend('Location', 'eastoutside');

% ---- Figure 2 (channel only): flow field and centreline development ----
if strcmp(caseName, 'channel')
    figure('Name', 'Channel flow field', 'Color', 'w', 'Position', [100 100 900 500]);
    subplot(2,1,1);
    [Xg, Yg] = meshgrid(xu, [0, yc, H]);
    uLast = uSnap{keep(end)};
    contourf(Xg, Yg, [zeros(Nx+1,1), uLast, zeros(Nx+1,1)]', 30, 'LineStyle', 'none');
    cb = colorbar;  ylabel(cb, 'u [m/s]');
    xlabel('x [m]'); ylabel('y [m]');
    title(sprintf('u at t = %.1f s', tSnap(keep(end))));
    subplot(2,1,2); hold on; box on; grid on
    for k = keep
        uc = uSnap{k};
        plot(xu, 0.5*(uc(:,Ny/2) + uc(:,Ny/2+1))/U0, 'Color', cols(k,:), ...
             'LineWidth', 1.3, 'DisplayName', sprintf('t = %g s', tSnap(k)));
    end
    plot([0 L], [1.5 1.5], 'k--', 'DisplayName', 'fully developed');
    xlabel('x [m]'); ylabel('u_{centreline} / U_0');
    legend('Location', 'southeast'); title('Centreline velocity');
end

% ---- Figure 3: SIMPLE iterations needed per time step ----
figure('Name', 'SIMPLE iterations', 'Color', 'w');
plot((1:numel(nInner))*dt, nInner, 'b-', 'LineWidth', 1.2); grid on;
xlabel('t [s]'); ylabel('SIMPLE iterations');
title('Iterations needed to converge each time step');

%% ====================== 8. Local functions ==============================

function D = wallConductance(bcType, mu, dx, dy)
% WALLCONDUCTANCE  Diffusion conductance between the first u-node and a
%   y-boundary. A wall is dy/2 from the node: D = mu*dx/(dy/2).
%   A slip (zero-shear) boundary transmits no shear: D = 0.
    switch bcType
        case 'wall', D = 2*mu*dx/dy;
        case 'slip', D = 0;
        otherwise,   error('Unknown y-boundary type "%s"', bcType);
    end
end

function u = applyOutlet(u, rho, dy)
% APPLYOUTLET  Zero-gradient outflow, scaled so that outflow = inflow.
    mdotIn   = rho*sum(u(1,:))*dy;
    u(end,:) = u(end-1,:);
    mdotOut  = rho*sum(u(end,:))*dy;
    if abs(mdotOut) > 0
        u(end,:) = u(end,:)*(mdotIn/mdotOut);
    end
end

function ue = exactProfile(caseName, y, t, P)
% EXACTPROFILE  Exact u(y,t) for each case (for verification).
    switch caseName
        case 'channel'      % fully developed Poiseuille profile
            ue = 6*P.U0*(y/P.H).*(1 - y/P.H);
        case 'couette'      % start-up from rest, by separation of variables
            % walls moving at Ubot (y = 0) and Utop (y = H) for t > 0,
            % G = -dpdx/(2 mu). Steady part:
            %   u_s = Ubot + (Utop - Ubot)*y/H + G*y*(H - y)
            % transient: sum b_n exp(-n^2 pi^2 nu t/H^2) sin(n pi y/H),
            % with b_n = -(2/H)*integral(u_s sin(n pi y/H)) so that u = 0 at t = 0
            G  = -P.dpdx/(2*P.mu);
            ue = P.Ubot + (P.Utop - P.Ubot)*y/P.H + G*y.*(P.H - y);
            for n = 1:200
                s  = (-1)^n;
                bn = -2*( P.Ubot*(1 - s)/(n*pi) ...
                        - (P.Utop - P.Ubot)*s/(n*pi) ...
                        + 2*G*P.H^2*(1 - s)/(n*pi)^3 );
                ue = ue + bn*exp(-(n*pi)^2*P.nu*t/P.H^2)*sin(n*pi*y/P.H);
            end
        case 'stokes1'      % impulsively started plate (speed Ubot)
            ue = P.Ubot*erfc(y./(2*sqrt(P.nu*t)));
        case 'stokes2'      % oscillating plate, periodic state
            k  = sqrt(P.omega/(2*P.nu));
            ue = P.U0*exp(-k*y).*cos(P.omega*t - k*y);
        otherwise
            ue = NaN(size(y));
    end
end

function phi = solve5(aP, aE, aW, aN, aS, b, periodicX)
% SOLVE5  Solve the 5-point system on an nI x nJ grid
%     aP*phi_P = aE*phi_E + aW*phi_W + aN*phi_N + aS*phi_S + b
%   by assembling a sparse matrix and using MATLAB's direct solver.
%   periodicX = true: the east neighbour of i = nI is i = 1 (and the west
%   neighbour of i = 1 is i = nI). Otherwise those links are absent.=++===
    [nI, nJ] = size(aP);
    N   = nI*nJ;
    idx = reshape(1:N, nI, nJ);            % global index k = i + (j-1)*nI
    r = idx(:);  c = idx(:);  val = aP(:);
    if periodicX
        iE = idx([2:nI, 1], :);            % wrap-around neighbours
        iW = idx([nI, 1:nI-1], :);
        r = [r; idx(:); idx(:)];
        c = [c; iE(:);  iW(:)];
        val = [val; -aE(:); -aW(:)];
    else
        P = idx(1:end-1,:); E = idx(2:end,:);
        r = [r; P(:)]; c = [c; E(:)]; val = [val; -reshape(aE(1:end-1,:),[],1)];
        P = idx(2:end,:);   W = idx(1:end-1,:);
        r = [r; P(:)]; c = [c; W(:)]; val = [val; -reshape(aW(2:end,:),[],1)];
    end
    P = idx(:,1:end-1);  Nn = idx(:,2:end);      % north links (j < nJ)
    r = [r; P(:)]; c = [c; Nn(:)]; val = [val; -reshape(aN(:,1:end-1),[],1)];
    P = idx(:,2:end);    S = idx(:,1:end-1);     % south links (j > 1)
    r = [r; P(:)]; c = [c; S(:)];  val = [val; -reshape(aS(:,2:end),[],1)];
    A   = sparse(r, c, val, N, N);               % repeated entries are summed
    phi = reshape(A\b(:), nI, nJ);
end
