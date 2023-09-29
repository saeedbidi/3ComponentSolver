#ifndef EMM_HYDRO_K_H_
#define EMM_HYDRO_K_H_

#include "EMM_index_macros.H"
#include "EMM_parm.H"
#include "EMM_eos.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_DEVICE
inline
void
EMM_ctoprim (int i, int j, int k,
             amrex::Array4<amrex::Real> const& u,
             amrex::Array4<amrex::Real> const& q,
             amrex::GeometryData const& geomdata,
             Parm const& parm) noexcept
{
    using namespace amrex::literals;

    amrex::Real alpha1 = u(i,j,k,GALPHA1);
    amrex::Real alpha2 = u(i,j,k,GALPHA2);
    amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;

    int ll = 0;
    amrex::Real d = 1E2_rt;
    if(alpha1 > 1.0_rt-2.0_rt*parm.alpha_min/d){alpha1 = 1.0_rt-2.0_rt*parm.alpha_min/d; ll = 1;}
    if(alpha2 > 1.0_rt-2.0_rt*parm.alpha_min/d){alpha2 = 1.0_rt-2.0_rt*parm.alpha_min/d; ll = 1;}
    if(alpha1 < parm.alpha_min/d){alpha1 = parm.alpha_min/d; ll = 1;}
    if(alpha2 < parm.alpha_min/d){alpha2 = parm.alpha_min/d; ll = 1;}
    if(ll == 1){
         alpha3 = 1.0_rt - alpha1 - alpha2;
         alpha1 = alpha1/(alpha1+alpha2+alpha3);
         alpha2 = alpha2/(alpha1+alpha2+alpha3);
    }

    amrex::Real mrho = u(i,j,k,UARHO1) + u(i,j,k,UARHO2) + u(i,j,k,UARHO3);
    amrex::Real rho1 = u(i,j,k,UARHO1)/alpha1;
    amrex::Real rho2 = u(i,j,k,UARHO2)/alpha2;
    amrex::Real rho3 = u(i,j,k,UARHO3)/alpha3;
    amrex::Real ux = u(i,j,k,UMX)/mrho;
    amrex::Real uy = u(i,j,k,UMY)/mrho;
    amrex::Real uz = u(i,j,k,UMZ)/mrho;
    amrex::Real e = u(i,j,k,URHOE)/mrho - 0.5_rt*(ux*ux + uy*uy + uz*uz);
    amrex::Real V11 = u(i,j,k,GV11); amrex::Real V12 = u(i,j,k,GV12); amrex::Real V13 = u(i,j,k,GV13);
    amrex::Real V21 = u(i,j,k,GV21); amrex::Real V22 = u(i,j,k,GV22); amrex::Real V23 = u(i,j,k,GV23);
    amrex::Real V31 = u(i,j,k,GV31); amrex::Real V32 = u(i,j,k,GV32); amrex::Real V33 = u(i,j,k,GV33);

    amrex::Real i1 = compute_i1(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    amrex::Real G1 = parm.eos_G01;
    amrex::Real G2 = parm.eos_G02;
    amrex::Real G3 = parm.eos_G03;
    amrex::Real es1 = (G1/(2.0_rt*parm.eos_rho01))*(i1 - 3.0_rt);
    amrex::Real es2 = (G2/(2.0_rt*parm.eos_rho02))*(i1 - 3.0_rt);
    amrex::Real es3 = (G3/(2.0_rt*parm.eos_rho03))*(i1 - 3.0_rt);

    if (alpha1 < 0.0_rt) {Print() << "C2P negative VOF1:     " << alpha1 << "\n";}
    if (alpha2 < 0.0_rt) {Print() << "C2P negative VOF2:     " << alpha2 << "\n";}
    if (alpha3 < 0.0_rt) {Print() << "C2P negative VOF3:     " << alpha3 << "\n";}
    if (rho1 < 0.0_rt)   {Print() << "C2P negative density1: " << rho1 << "\n";}
    if (rho2 < 0.0_rt)   {Print() << "C2P negative density2: " << rho2 << "\n";}
    if (rho3 < 0.0_rt)   {Print() << "C2P negative density3: " << rho3 << "\n";}

    // amrex::Real p = p_finder(alpha1, alpha2, alpha3, rho1, rho2, rho3, u(i,j,k,GPRESS), u(i,j,k,GT1), u(i,j,k,GT2), u(i,j,k,GT3), mrho*e);
    amrex::Real p = compute_pressure(mrho, e, es1, es2, es3, alpha1, alpha2, alpha3, rho1, rho2, rho3, u(i,j,k,GPRESS), u(i,j,k,GT1), u(i,j,k,GT2), u(i,j,k,GT3));
    // amrex::Real p = compute_pressure(mrho, e, alpha1, alpha2, alpha3, rho1, rho2, rho3, 1E5, 300.0, 300.0, 300.0);
    // if(p < 2300.0_rt) { p = 2300.0_rt; }

    amrex::Real T1 = 0.0;
    amrex::Real T2 = 0.0;
    amrex::Real T3 = 0.0;
    T1 = compute_temperature(alpha1, rho1, p, u(i,j,k,GT1), 1);
    T2 = compute_temperature(alpha2, rho2, p, u(i,j,k,GT2), 2);
    T3 = compute_temperature(alpha3, rho3, p, u(i,j,k,GT3), 3);
    // T1 = compute_temperature(alpha1, rho1, p, 300.0, 1);
    // T2 = compute_temperature(alpha2, rho2, p, 300.0, 2);
    // T3 = compute_temperature(alpha3, rho3, p, 300.0, 3);
    // Print() << "temperature1: " << T1 << "\n";
    // Print() << "temperature2: " << T2 << "\n";
    // Print() << "temperature3: " << T3 << "\n";

    q(i,j,k,QALPHA1) = alpha1; // u(i,j,k,GALPHA1)
    q(i,j,k,QALPHA2) = alpha2; // u(i,j,k,GALPHA2)
    q(i,j,k,QRHO1) = u(i,j,k,UARHO1)/alpha1;
    q(i,j,k,QRHO2) = u(i,j,k,UARHO2)/alpha2;
    q(i,j,k,QRHO3) = u(i,j,k,UARHO3)/alpha3;
    q(i,j,k,QU) = ux;
    q(i,j,k,QV) = uy;
    q(i,j,k,QW) = uz;
    q(i,j,k,QPRES) = p;
    q(i,j,k,QPI) = u(i,j,k,UARP)/u(i,j,k,UARHO1);
    q(i,j,k,QV11) = u(i,j,k,GV11);
    q(i,j,k,QV21) = u(i,j,k,GV21);
    q(i,j,k,QV31) = u(i,j,k,GV31);
    q(i,j,k,QV12) = u(i,j,k,GV12);
    q(i,j,k,QV22) = u(i,j,k,GV22);
    q(i,j,k,QV32) = u(i,j,k,GV32);
    q(i,j,k,QV13) = u(i,j,k,GV13);
    q(i,j,k,QV23) = u(i,j,k,GV23);
    q(i,j,k,QV33) = u(i,j,k,GV33);

    u(i,j,k,GPRESS) = p;
    u(i,j,k,GT1) = T1;
    u(i,j,k,GT2) = T2;
    u(i,j,k,GT3) = T3;
}

AMREX_GPU_DEVICE
inline
void
EMM_thermo (int i, int j, int k,
             amrex::Array4<amrex::Real> const& u_new,
             amrex::Array4<amrex::Real> const& u_old,
             Parm const& parm) noexcept
{
    BL_PROFILE("EMM_thermo()");
    using namespace amrex::literals;

    amrex::Real alpha1 = u_new(i,j,k,GALPHA1);
    amrex::Real alpha2 = u_new(i,j,k,GALPHA2);
    amrex::Real alpha3 = 1.0_rt - alpha1 - alpha3;

    int ll = 0;
    amrex::Real d = 1E2;
    if(alpha1 > 1.0_rt-2.0_rt*parm.alpha_min/d){alpha1 = 1.0_rt-2.0_rt*parm.alpha_min/d; ll = 1;}
    if(alpha2 > 1.0_rt-2.0_rt*parm.alpha_min/d){alpha2 = 1.0_rt-2.0_rt*parm.alpha_min/d; ll = 1;}
    if(alpha1 < parm.alpha_min/d){alpha1 = parm.alpha_min/d; ll = 1;}
    if(alpha2 < parm.alpha_min/d){alpha2 = parm.alpha_min/d; ll = 1;}
    if(ll == 1){
         alpha3 = 1.0_rt - alpha1 - alpha2;
         alpha1 = alpha1/(alpha1+alpha2+alpha3);
         alpha2 = alpha2/(alpha1+alpha2+alpha3);
    }

    amrex::Real mrho = u_new(i,j,k,UARHO1) + u_new(i,j,k,UARHO2) + u_new(i,j,k,UARHO3);
    amrex::Real rho1 = u_new(i,j,k,UARHO1)/alpha1;
    amrex::Real rho2 = u_new(i,j,k,UARHO2)/alpha2;
    amrex::Real rho3 = u_new(i,j,k,UARHO3)/alpha3;
    amrex::Real ux = u_new(i,j,k,UMX)/mrho;
    amrex::Real uy = u_new(i,j,k,UMY)/mrho;
    amrex::Real uz = u_new(i,j,k,UMZ)/mrho;
    amrex::Real e = u_new(i,j,k,URHOE)/mrho - 0.5_rt*(ux*ux + uy*uy + uz*uz);
    amrex::Real V11 = u_new(i,j,k,GV11); amrex::Real V12 = u_new(i,j,k,GV12); amrex::Real V13 = u_new(i,j,k,GV13);
    amrex::Real V21 = u_new(i,j,k,GV21); amrex::Real V22 = u_new(i,j,k,GV22); amrex::Real V23 = u_new(i,j,k,GV23);
    amrex::Real V31 = u_new(i,j,k,GV31); amrex::Real V32 = u_new(i,j,k,GV32); amrex::Real V33 = u_new(i,j,k,GV33);

    amrex::Real i1 = compute_i1(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    amrex::Real G1 = parm.eos_G01;
    amrex::Real G2 = parm.eos_G02;
    amrex::Real G3 = parm.eos_G03;
    amrex::Real es1 = (G1/(2.0_rt*parm.eos_rho01))*(i1 - 3.0_rt);
    amrex::Real es2 = (G2/(2.0_rt*parm.eos_rho02))*(i1 - 3.0_rt);
    amrex::Real es3 = (G3/(2.0_rt*parm.eos_rho03))*(i1 - 3.0_rt);

    amrex::Real p = compute_pressure(mrho, e, es1, es2, es3, alpha1, alpha2, alpha3, rho1, rho2, rho3, u_old(i,j,k,GPRESS), u_old(i,j,k,GT1), u_old(i,j,k,GT2), u_old(i,j,k,GT3));

    amrex::Real T1 = 0.0;
    amrex::Real T2 = 0.0;
    amrex::Real T3 = 0.0;
    T1 = compute_temperature(alpha1, rho1, p, u_old(i,j,k,GT1), 1);
    T2 = compute_temperature(alpha2, rho2, p, u_old(i,j,k,GT2), 2);
    T3 = compute_temperature(alpha3, rho3, p, u_old(i,j,k,GT3), 3);

    u_new(i,j,k,GPRESS) = p;
    u_new(i,j,k,GT1) = T1;
    u_new(i,j,k,GT2) = T2;
    u_new(i,j,k,GT3) = T3;
}

AMREX_GPU_DEVICE
inline
void
EMM_flux_to_dudt (int i, int j, int k, int n,
                  amrex::Array4<amrex::Real> const& dudt,
                  amrex::Array4<amrex::Real> const& HCC,
                  amrex::Array4<amrex::Real> const& UCC,
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fx,
                  amrex::Array4<amrex::Real const> const& fy,
                  amrex::Array4<amrex::Real const> const& fz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& USx,
                  amrex::Array4<amrex::Real const> const& USy,
                  amrex::Array4<amrex::Real const> const& USz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& VSx,
                  amrex::Array4<amrex::Real const> const& VSy,
                  amrex::Array4<amrex::Real const> const& VSz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& WSx,
                  amrex::Array4<amrex::Real const> const& WSy,
                  amrex::Array4<amrex::Real const> const& WSz),
                  AMREX_D_DECL(amrex::Array4<amrex::Real const> const& VFx,
                  amrex::Array4<amrex::Real const> const& VFy,
                  amrex::Array4<amrex::Real const> const& VFz),
                  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    // const amrex::Real* prob_lo = geomdata.ProbLo();
    // const amrex::Real* dx      = geomdata.CellSize();
    if(n < NCONS){
        dudt(i,j,k,n) = dxinv[0]*(fx(i,j,k,n) - fx(i+1,j,k,n))
            +           dxinv[1]*(fy(i,j,k,n) - fy(i,j+1,k,n))
#if (AMREX_SPACEDIM == 3)
            +           dxinv[2]*(fz(i,j,k,n) - fz(i,j,k+1,n))
#endif
        ;
    }else{
        dudt(i,j,k,n) = dxinv[0]*( fx(i,j,k,n) - fx(i+1,j,k,n)
            +           HCC(i,j,k,n-NCONS)*( USx(i,j,k) - USx(i+1,j,k) )
            +           UCC(i,j,k,n-NCONS)*( VFx(i,j,k,n-NCONS) - VFx(i+1,j,k,n-NCONS)) )

            +           dxinv[1]*( fy(i,j,k,n) - fy(i,j+1,k,n)
            +           HCC(i,j,k,n-NCONS)*( VSy(i,j,k) - VSy(i,j+1,k) )
            +           UCC(i,j,k,n-NCONS)*( VFy(i,j,k,n-NCONS) - VFy(i,j+1,k,n-NCONS)) )
#if (AMREX_SPACEDIM == 3)
            +           dxinv[2]*( fz(i,j,k,n) - fz(i,j,k+1,n)
            +           HCC(i,j,k,n-NCONS)*( WSz(i,j,k) - WSz(i,j,k+1) )
            +           UCC(i,j,k,n-NCONS)*( VFz(i,j,k,n-NCONS) - VFz(i,j,k+1,n-NCONS)) )
#endif
        ;
    }
}

AMREX_GPU_DEVICE
inline
void
EMM_axisymmetricAdd (int i, int j, int k, amrex::Real dt, int iteration,
                    amrex::Array4<amrex::Real> const& dudt,
                    amrex::Array4<amrex::Real const> const& q,
                    amrex::Array4<amrex::Real const> const& u,
                    amrex::GeometryData const& geomdata,
                    Parm const& parm,
                    amrex::Real time) noexcept
{
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* dx      = geomdata.CellSize();
    amrex::Real x = prob_lo[0] + (i+0.5_rt)*dx[0];
    amrex::Real y = prob_lo[1] + (j+0.5_rt)*dx[1];
    amrex::Real r = x;
    amrex::Real cc = 1.0_rt;

    amrex::Real V11 = q(i,j,k,QV11); amrex::Real V12 = q(i,j,k,QV12); amrex::Real V13 = q(i,j,k,QV13);
    amrex::Real V21 = q(i,j,k,QV21); amrex::Real V22 = q(i,j,k,QV22); amrex::Real V23 = q(i,j,k,QV23);
    amrex::Real V31 = q(i,j,k,QV31); amrex::Real V32 = q(i,j,k,QV32); amrex::Real V33 = q(i,j,k,QV33);

    amrex::Real alpha1 = q(i,j,k,QALPHA1);
    amrex::Real alpha2 = q(i,j,k,QALPHA2);
    amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;

    amrex::Real rho1 = q(i,j,k,QRHO1); amrex::Real rho2 = q(i,j,k,QRHO2); amrex::Real rho3 = q(i,j,k,QRHO3);
    amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
    amrex::Real mrho_0 = alpha1*parm.eos_rho01+alpha2*parm.eos_rho02+alpha3*parm.eos_rho03;

    amrex::Real i1 = compute_i1(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    amrex::Real es1 = (parm.eos_G01/(1.0_rt*parm.eos_rho01))*(i1 - 3.0);
    amrex::Real es2 = (parm.eos_G02/(1.0_rt*parm.eos_rho02))*(i1 - 3.0);
    amrex::Real es3 = (parm.eos_G03/(1.0_rt*parm.eos_rho03))*(i1 - 3.0);

    amrex::Real T1 = compute_temperature(alpha1, rho1, q(i, j, k, QPRES), u(i,j,k,GT1), 1);
    amrex::Real T2 = compute_temperature(alpha2, rho2, q(i, j, k, QPRES), u(i,j,k,GT2), 2);
    amrex::Real T3 = compute_temperature(alpha3, rho3, q(i, j, k, QPRES), u(i,j,k,GT3), 3);
    // amrex::Real T1 = compute_temperature(alpha1, rho1, q(i, j, k, QPRES), 300.0, 1);
    // amrex::Real T2 = compute_temperature(alpha2, rho2, q(i, j, k, QPRES), 300.0, 2);
    // amrex::Real T3 = compute_temperature(alpha3, rho3, q(i, j, k, QPRES), 300.0, 3);
    amrex::Real eint1 = compute_internal_energy(q(i, j, k, QPRES), T1, rho1, 1) + es1;
    amrex::Real eint2 = compute_internal_energy(q(i, j, k, QPRES), T2, rho2, 2) + es2;
    amrex::Real eint3 = compute_internal_energy(q(i, j, k, QPRES), T3, rho3, 3) + es3;

    amrex::Real Y1 = alpha1*rho1/mrho;
    amrex::Real Y2 = alpha2*rho2/mrho;
    amrex::Real Y3 = alpha3*rho3/mrho;
    amrex::Real E = Y1*eint1 + Y2*eint2 + Y3*eint3 + 0.5_rt*q(i, j, k, QU)*q(i, j, k, QU) + 0.5_rt*q(i, j, k, QV)*q(i, j, k, QV);
    amrex::Real G1 = parm.eos_G01;
    amrex::Real G2 = parm.eos_G02;
    amrex::Real G3 = parm.eos_G03;
    amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
    amrex::Real Sigma11 = - q(i, j, k, QPRES) + G*compute_devB11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    amrex::Real Sigma22 = - q(i, j, k, QPRES) + G*compute_devB22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    amrex::Real Sigma33 = - q(i, j, k, QPRES) + G*compute_devB33(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    amrex::Real Sigma12 = G*compute_devB12(V11,V12,V13,V21,V22,V23,V31,V32,V33);

    amrex::Real c1sq = amrex::max(compute_SoS(q(i,j,k,QPRES), T1, q(i,j,k,QRHO1), 1), parm.smallr);
    amrex::Real c2sq = amrex::max(compute_SoS(q(i,j,k,QPRES), T2, q(i,j,k,QRHO2), 2), parm.smallr);
    amrex::Real c3sq = amrex::max(compute_SoS(q(i,j,k,QPRES), T3, q(i,j,k,QRHO3), 3), parm.smallr);
    // amrex::Real c1sq = parm.eos_gamma1*(q(i,j,k,QPRES)+parm.eos_pinf1)/q(i,j,k,QRHO1) + (4.0/3.0)*(parm.eos_G01/parm.eos_rho01);
    // amrex::Real c2sq = parm.eos_gamma2*(q(i,j,k,QPRES)+parm.eos_pinf2)/q(i,j,k,QRHO2);
    // amrex::Real c3sq = parm.eos_gamma3*(q(i,j,k,QPRES)+parm.eos_pinf3)/q(i,j,k,QRHO3);

    amrex::Real cpsq = 1.0_rt/(mrho*(alpha1/(rho1*c1sq) + alpha2/(rho2*c2sq) + alpha3/(rho3*c3sq)));
    amrex::Real K1 = 0.0_rt;
    amrex::Real K2 = 0.0_rt;
    // if(iteration >= parm.K_activation){
    if(time >= parm.K_time){
        K1 = alpha1*( (mrho*cpsq)/(rho1*c1sq) - 1.0_rt);
        K2 = alpha2*( (mrho*cpsq)/(rho2*c2sq) - 1.0_rt);
    }

    dudt(i, j, k, UARHO1)     -= (cc/r)*( alpha1*rho1*q(i, j, k, QU) );
    dudt(i, j, k, UARHO2)     -= (cc/r)*( alpha2*rho2*q(i, j, k, QU) );
    dudt(i, j, k, UARHO3)     -= (cc/r)*( alpha3*rho3*q(i, j, k, QU) );
    dudt(i, j, k, UMX)        -= (cc/r)*( mrho*q(i, j, k, QU)*q(i, j, k, QU) - Sigma11 + Sigma33 );
    dudt(i, j, k, UMY)        -= (cc/r)*( mrho*q(i, j, k, QU)*q(i, j, k, QV) - Sigma12 );
    dudt(i, j, k, URHOE)      -= (cc/r)*( q(i, j, k, QU)*(mrho*E - Sigma11) - q(i, j, k, QV)*Sigma12 );
    dudt(i, j, k, GALPHA1)    -= (cc/r)*( -K1*q(i, j, k, QU) );
    dudt(i, j, k, GALPHA2)    -= (cc/r)*( -K2*q(i, j, k, QU) );
    dudt(i, j, k, GV11)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV11)*q(i, j, k, QU) );
    dudt(i, j, k, GV21)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV21)*q(i, j, k, QU) );
    dudt(i, j, k, GV31)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV31)*q(i, j, k, QU) - q(i, j, k, QV31)*q(i, j, k, QU) );
    dudt(i, j, k, GV12)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV12)*q(i, j, k, QU) );
    dudt(i, j, k, GV22)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV22)*q(i, j, k, QU) );
    dudt(i, j, k, GV32)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV32)*q(i, j, k, QU) - q(i, j, k, QV32)*q(i, j, k, QU) );
    dudt(i, j, k, GV13)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV13)*q(i, j, k, QU) );
    dudt(i, j, k, GV23)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV23)*q(i, j, k, QU) );
    dudt(i, j, k, GV33)       -= (cc/r)*( (1.0_rt/3.0)*q(i, j, k, QV33)*q(i, j, k, QU) - q(i, j, k, QV33)*q(i, j, k, QU) );
}

namespace {

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real xlimiter (int i, int j, int k, amrex::Array4<amrex::Real const> const& q,
                int comp) noexcept
{
    using namespace amrex::literals;

    amrex::Real slopex;
    amrex::Real qp1 = q(i+1,j,k,comp)-q(i,j,k,comp);
    amrex::Real qm1 = q(i,j,k,comp) - q(i-1,j,k,comp);
    if (std::fabs(qp1) == 0.0_rt){
        slopex = 0.0_rt;
    }else{
        amrex::Real r = qm1/qp1;
        amrex::Real lr;
        lr = amrex::max(0.0_rt,amrex::min(1.0_rt,r)); // Minmod
        // lr = amrex::max(0.0_rt,amrex::max(amrex::min(2.0_rt*r,1.0_rt),amrex::min(r,2.0_rt))); //Superbee
        // lr = amrex::max(0.0_rt,amrex::min(amrex::min(2.0_rt*r,0.5_rt*(1.0_rt+r)),2.0_rt)); //MC
        // lr = (r + std::fabs(r))/(1 + std::fabs(r));
        slopex = lr*qp1;
    }

    return slopex;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real ylimiter (int i, int j, int k, amrex::Array4<amrex::Real const> const& q,
                int comp) noexcept
{
    using namespace amrex::literals;

    amrex::Real slopey;
    amrex::Real qp1 = q(i,j+1,k,comp)-q(i,j,k,comp);
    amrex::Real qm1 = q(i,j,k,comp) - q(i,j-1,k,comp);
    if (std::fabs(qp1) == 0.0_rt){
        slopey = 0.0_rt;
    }else{
        amrex::Real r = qm1/qp1;
        amrex::Real lr;
        lr = amrex::max(0.0_rt,amrex::min(1.0_rt,r)); // Minmod
        // lr = amrex::max(0.0_rt,amrex::max(amrex::min(2.0_rt*r,1.0_rt),amrex::min(r,2.0_rt)));
        // lr = amrex::max(0.0_rt,amrex::min(amrex::min(2.0_rt*r,0.5_rt*(1.0_rt+r)),2.0_rt)); //MC
        // lr = (r + std::fabs(r))/(1 + std::fabs(r));
        slopey = lr*qp1;
    }

    return slopey;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real zlimiter (int i, int j, int k, amrex::Array4<amrex::Real const> const& q,
                int comp) noexcept
{
    using namespace amrex::literals;

    amrex::Real slopez;
    amrex::Real qp1 = q(i,j,k+1,comp)-q(i,j,k,comp);
    amrex::Real qm1 = q(i,j,k,comp) - q(i,j,k-1,comp);
    if (std::fabs(qp1) == 0.0_rt){
        slopez = 0.0_rt;
    }else{
        amrex::Real r = qm1/qp1;
        amrex::Real lr;
        lr = amrex::max(0.0_rt,amrex::min(1.0_rt,r)); // Minmod
        // lr = amrex::max(0.0_rt,amrex::max(amrex::min(2.0_rt*r,1.0_rt),amrex::min(r,2.0_rt))); // Superbee
        // lr = amrex::max(0.0_rt,amrex::min(amrex::min(2.0_rt*r,0.5_rt*(1.0_rt+r)),2.0_rt)); // MC
        // lr = (r + std::fabs(r))/(1 + std::fabs(r));
        slopez = lr*qp1;
    }

    return slopez;
}
}

AMREX_GPU_DEVICE
inline
void
EMM_PMUSCL_reconstruct_x (int i, int j, int k, // Loop running on the cell centers, However qL,qR are face centered
             amrex::Array4<amrex::Real> const& qL,
             amrex::Array4<amrex::Real> const& qR,
             amrex::Array4<amrex::Real const> const& q, amrex::Real dt,
             amrex::GeometryData const& geomdata,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using namespace amrex::literals;
    
    // int al = 0;
    // if(q(i,j,k,QALPHA1) <= parm.alpha_min){al = 1;}
    // if(q(i,j,k,QALPHA2) <= parm.alpha_min){al = 1;}
    // if(q(i,j,k,QALPHA1) >= 1.0_rt-1.0_rt*parm.alpha_min){al = 1;}
    // if(q(i,j,k,QALPHA2) >= 1.0_rt-1.0_rt*parm.alpha_min){al = 1;}
    // if(1.0_rt - q(i,j,k,QALPHA1) - q(i,j,k,QALPHA2) < parm.alpha_min){al = 1;}

    qL(i,j,k,QALPHA1) = q(i,j,k,QALPHA1) - 0.5_rt*xlimiter(i, j, k, q, QALPHA1);
    qL(i,j,k,QALPHA2) = q(i,j,k,QALPHA2) - 0.5_rt*xlimiter(i, j, k, q, QALPHA2);
    qL(i,j,k,QRHO1) = q(i,j,k,QRHO1) - 0.5_rt*xlimiter(i, j, k, q,QRHO1);
    qL(i,j,k,QRHO2) = q(i,j,k,QRHO2) - 0.5_rt*xlimiter(i, j, k, q,QRHO2);
    qL(i,j,k,QRHO3) = q(i,j,k,QRHO3) - 0.5_rt*xlimiter(i, j, k, q,QRHO3);
    qL(i,j,k,QU) = q(i,j,k,QU) - 0.5_rt*xlimiter(i, j, k, q,QU);
    qL(i,j,k,QV) = q(i,j,k,QV) - 0.5_rt*xlimiter(i, j, k, q,QV);
    qL(i,j,k,QW) = q(i,j,k,QW) - 0.5_rt*xlimiter(i, j, k, q,QW);
    qL(i,j,k,QPRES) = q(i,j,k,QPRES) - 0.5_rt*xlimiter(i, j, k, q,QPRES);
    qL(i,j,k,QPI) = q(i,j,k,QPI) - 0.5_rt*xlimiter(i, j, k, q,QPI);
    qL(i,j,k,QV11) = q(i,j,k,QV11) - 0.5*xlimiter(i, j, k, q,QV11);
    qL(i,j,k,QV21) = q(i,j,k,QV21) - 0.5*xlimiter(i, j, k, q,QV21);
    qL(i,j,k,QV31) = q(i,j,k,QV31) - 0.5*xlimiter(i, j, k, q,QV31);
    qL(i,j,k,QV12) = q(i,j,k,QV12) - 0.5*xlimiter(i, j, k, q,QV12);
    qL(i,j,k,QV22) = q(i,j,k,QV22) - 0.5*xlimiter(i, j, k, q,QV22);
    qL(i,j,k,QV32) = q(i,j,k,QV32) - 0.5*xlimiter(i, j, k, q,QV32);
    qL(i,j,k,QV13) = q(i,j,k,QV13) - 0.5*xlimiter(i, j, k, q,QV13);
    qL(i,j,k,QV23) = q(i,j,k,QV23) - 0.5*xlimiter(i, j, k, q,QV23);
    qL(i,j,k,QV33) = q(i,j,k,QV33) - 0.5*xlimiter(i, j, k, q,QV33);
 
    // qR(i+1,j,k)
    qR(i+1,j,k,QALPHA1) = q(i,j,k,QALPHA1) + 0.5_rt*xlimiter(i, j, k, q,QALPHA1);
    qR(i+1,j,k,QALPHA2) = q(i,j,k,QALPHA2) + 0.5_rt*xlimiter(i, j, k, q,QALPHA2);
    qR(i+1,j,k,QRHO1) = q(i,j,k,QRHO1) + 0.5_rt*xlimiter(i, j, k, q,QRHO1);
    qR(i+1,j,k,QRHO2) = q(i,j,k,QRHO2) + 0.5_rt*xlimiter(i, j, k, q,QRHO2);
    qR(i+1,j,k,QRHO3) = q(i,j,k,QRHO3) + 0.5_rt*xlimiter(i, j, k, q,QRHO3);
    qR(i+1,j,k,QU) = q(i,j,k,QU) + 0.5_rt*xlimiter(i, j, k, q,QU);
    qR(i+1,j,k,QV) = q(i,j,k,QV) + 0.5_rt*xlimiter(i, j, k, q,QV);
    qR(i+1,j,k,QW) = q(i,j,k,QW) + 0.5_rt*xlimiter(i, j, k, q,QW);
    qR(i+1,j,k,QPRES) = q(i,j,k,QPRES) + 0.5_rt*xlimiter(i, j, k, q,QPRES);
    qR(i+1,j,k,QPI) = q(i,j,k,QPI) + 0.5_rt*xlimiter(i, j, k, q,QPI);
    qR(i+1,j,k,QV11) = q(i,j,k,QV11) + 0.5*xlimiter(i, j, k, q,QV11);
    qR(i+1,j,k,QV21) = q(i,j,k,QV21) + 0.5*xlimiter(i, j, k, q,QV21);
    qR(i+1,j,k,QV31) = q(i,j,k,QV31) + 0.5*xlimiter(i, j, k, q,QV31);
    qR(i+1,j,k,QV12) = q(i,j,k,QV12) + 0.5*xlimiter(i, j, k, q,QV12);
    qR(i+1,j,k,QV22) = q(i,j,k,QV22) + 0.5*xlimiter(i, j, k, q,QV22);
    qR(i+1,j,k,QV32) = q(i,j,k,QV32) + 0.5*xlimiter(i, j, k, q,QV32);
    qR(i+1,j,k,QV13) = q(i,j,k,QV13) + 0.5*xlimiter(i, j, k, q,QV13);
    qR(i+1,j,k,QV23) = q(i,j,k,QV23) + 0.5*xlimiter(i, j, k, q,QV23);
    qR(i+1,j,k,QV33) = q(i,j,k,QV33) + 0.5*xlimiter(i, j, k, q,QV33);

    // if(al == 1){qL(i,j,k,QALPHA1) = q(i,j,k,QALPHA1); qR(i+1,j,k,QALPHA1) = q(i,j,k,QALPHA1);}
    // if(al == 1){qL(i,j,k,QALPHA2) = q(i,j,k,QALPHA2); qR(i+1,j,k,QALPHA2) = q(i,j,k,QALPHA2);}

    // qL(i,j,k,QALPHA1) = q(i,j,k,QALPHA1);
    // qL(i,j,k,QALPHA2) = q(i,j,k,QALPHA2);
    // qL(i,j,k,QRHO1) = q(i,j,k,QRHO1);
    // qL(i,j,k,QRHO2) = q(i,j,k,QRHO2);
    // qL(i,j,k,QRHO3) = q(i,j,k,QRHO3);
    // qL(i,j,k,QU) = q(i,j,k,QU);
    // qL(i,j,k,QV) = q(i,j,k,QV);
    // qL(i,j,k,QW) = q(i,j,k,QW);
    // qL(i,j,k,QPRES) = q(i,j,k,QPRES);
    // qL(i,j,k,QPI) = q(i,j,k,QPI);
    // qL(i,j,k,QV11) = q(i,j,k,QV11);
    // qL(i,j,k,QV21) = q(i,j,k,QV21);
    // qL(i,j,k,QV31) = q(i,j,k,QV31);
    // qL(i,j,k,QV12) = q(i,j,k,QV12);
    // qL(i,j,k,QV22) = q(i,j,k,QV22);
    // qL(i,j,k,QV32) = q(i,j,k,QV32);
    // qL(i,j,k,QV13) = q(i,j,k,QV13);
    // qL(i,j,k,QV23) = q(i,j,k,QV23);
    // qL(i,j,k,QV33) = q(i,j,k,QV33);

    // // qR(i+1,j,k)
    // qR(i+1,j,k,QALPHA1) = q(i,j,k,QALPHA1);
    // qR(i+1,j,k,QALPHA2) = q(i,j,k,QALPHA2);
    // qR(i+1,j,k,QRHO1) = q(i,j,k,QRHO1);
    // qR(i+1,j,k,QRHO2) = q(i,j,k,QRHO2);
    // qR(i+1,j,k,QRHO3) = q(i,j,k,QRHO3);
    // qR(i+1,j,k,QU) = q(i,j,k,QU);
    // qR(i+1,j,k,QV) = q(i,j,k,QV);
    // qR(i+1,j,k,QW) = q(i,j,k,QW);
    // qR(i+1,j,k,QPRES) = q(i,j,k,QPRES);
    // qR(i+1,j,k,QPI) = q(i,j,k,QPI);
    // qR(i+1,j,k,QV11) = q(i,j,k,QV11);
    // qR(i+1,j,k,QV21) = q(i,j,k,QV21);
    // qR(i+1,j,k,QV31) = q(i,j,k,QV31);
    // qR(i+1,j,k,QV12) = q(i,j,k,QV12);
    // qR(i+1,j,k,QV22) = q(i,j,k,QV22);
    // qR(i+1,j,k,QV32) = q(i,j,k,QV32);
    // qR(i+1,j,k,QV13) = q(i,j,k,QV13);
    // qR(i+1,j,k,QV23) = q(i,j,k,QV23);
    // qR(i+1,j,k,QV33) = q(i,j,k,QV33);
}

AMREX_GPU_DEVICE
inline
void
EMM_PMUSCL_reconstruct_y (int i, int j, int k,
             amrex::Array4<amrex::Real> const& qL,
             amrex::Array4<amrex::Real> const& qR,
             amrex::Array4<amrex::Real const> const& q, amrex::Real dt,
             amrex::GeometryData const& geomdata,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using namespace amrex::literals;

    // int al = 0;
    // if(q(i,j,k,QALPHA1) <= parm.alpha_min){al = 1;}
    // if(q(i,j,k,QALPHA2) <= parm.alpha_min){al = 1;}
    // if(q(i,j,k,QALPHA1) >= 1.0_rt-1.0_rt*parm.alpha_min){al = 1;}
    // if(q(i,j,k,QALPHA2) >= 1.0_rt-1.0_rt*parm.alpha_min){al = 1;}
    // if(1.0_rt - q(i,j,k,QALPHA1) - q(i,j,k,QALPHA2) < parm.alpha_min){al = 1;}

    qL(i,j,k,QALPHA1) = q(i,j,k,QALPHA1) - 0.5_rt*ylimiter(i, j, k, q, QALPHA1);
    qL(i,j,k,QALPHA2) = q(i,j,k,QALPHA2) - 0.5_rt*ylimiter(i, j, k, q, QALPHA2);
    qL(i,j,k,QRHO1) = q(i,j,k,QRHO1) - 0.5_rt*ylimiter(i, j, k, q,QRHO1);
    qL(i,j,k,QRHO2) = q(i,j,k,QRHO2) - 0.5_rt*ylimiter(i, j, k, q,QRHO2);
    qL(i,j,k,QRHO3) = q(i,j,k,QRHO3) - 0.5_rt*ylimiter(i, j, k, q,QRHO3);
    qL(i,j,k,QU) = q(i,j,k,QU) - 0.5_rt*ylimiter(i, j, k, q,QU);
    qL(i,j,k,QV) = q(i,j,k,QV) - 0.5_rt*ylimiter(i, j, k, q,QV);
    qL(i,j,k,QW) = q(i,j,k,QW) - 0.5_rt*ylimiter(i, j, k, q,QW);
    qL(i,j,k,QPRES) = q(i,j,k,QPRES) - 0.5_rt*ylimiter(i, j, k, q,QPRES);
    qL(i,j,k,QPI) = q(i,j,k,QPI) - 0.5_rt*ylimiter(i, j, k, q,QPI);
    qL(i,j,k,QV11) = q(i,j,k,QV11) - 0.5*ylimiter(i, j, k, q,QV11);
    qL(i,j,k,QV21) = q(i,j,k,QV21) - 0.5*ylimiter(i, j, k, q,QV21);
    qL(i,j,k,QV31) = q(i,j,k,QV31) - 0.5*ylimiter(i, j, k, q,QV31);
    qL(i,j,k,QV12) = q(i,j,k,QV12) - 0.5*ylimiter(i, j, k, q,QV12);
    qL(i,j,k,QV22) = q(i,j,k,QV22) - 0.5*ylimiter(i, j, k, q,QV22);
    qL(i,j,k,QV32) = q(i,j,k,QV32) - 0.5*ylimiter(i, j, k, q,QV32);
    qL(i,j,k,QV13) = q(i,j,k,QV13) - 0.5*ylimiter(i, j, k, q,QV13);
    qL(i,j,k,QV23) = q(i,j,k,QV23) - 0.5*ylimiter(i, j, k, q,QV23);
    qL(i,j,k,QV33) = q(i,j,k,QV33) - 0.5*ylimiter(i, j, k, q,QV33);
    
    // qR(i+1,j,k
    qR(i,j+1,k,QALPHA1) = q(i,j,k,QALPHA1) + 0.5_rt*ylimiter(i, j, k, q,QALPHA1);
    qR(i,j+1,k,QALPHA2) = q(i,j,k,QALPHA2) + 0.5_rt*ylimiter(i, j, k, q,QALPHA2);
    qR(i,j+1,k,QRHO1) = q(i,j,k,QRHO1) + 0.5_rt*ylimiter(i, j, k, q,QRHO1);
    qR(i,j+1,k,QRHO2) = q(i,j,k,QRHO2) + 0.5_rt*ylimiter(i, j, k, q,QRHO2);
    qR(i,j+1,k,QRHO3) = q(i,j,k,QRHO3) + 0.5_rt*ylimiter(i, j, k, q,QRHO3);
    qR(i,j+1,k,QU) = q(i,j,k,QU) + 0.5_rt*ylimiter(i, j, k, q,QU);
    qR(i,j+1,k,QV) = q(i,j,k,QV) + 0.5_rt*ylimiter(i, j, k, q,QV);
    qR(i,j+1,k,QW) = q(i,j,k,QW) + 0.5_rt*ylimiter(i, j, k, q,QW);
    qR(i,j+1,k,QPRES) = q(i,j,k,QPRES) + 0.5_rt*ylimiter(i, j, k, q,QPRES);
    qR(i,j+1,k,QPI) = q(i,j,k,QPI) + 0.5_rt*ylimiter(i, j, k, q,QPI);
    qR(i,j+1,k,QV11) = q(i,j,k,QV11) + 0.5*ylimiter(i, j, k, q,QV11);
    qR(i,j+1,k,QV21) = q(i,j,k,QV21) + 0.5*ylimiter(i, j, k, q,QV21);
    qR(i,j+1,k,QV31) = q(i,j,k,QV31) + 0.5*ylimiter(i, j, k, q,QV31);
    qR(i,j+1,k,QV12) = q(i,j,k,QV12) + 0.5*ylimiter(i, j, k, q,QV12);
    qR(i,j+1,k,QV22) = q(i,j,k,QV22) + 0.5*ylimiter(i, j, k, q,QV22);
    qR(i,j+1,k,QV32) = q(i,j,k,QV32) + 0.5*ylimiter(i, j, k, q,QV32);
    qR(i,j+1,k,QV13) = q(i,j,k,QV13) + 0.5*ylimiter(i, j, k, q,QV13);
    qR(i,j+1,k,QV23) = q(i,j,k,QV23) + 0.5*ylimiter(i, j, k, q,QV23);
    qR(i,j+1,k,QV33) = q(i,j,k,QV33) + 0.5*ylimiter(i, j, k, q,QV33);

    // if(al == 1){qL(i,j,k,QALPHA1) = q(i,j,k,QALPHA1); qR(i,j+1,k,QALPHA1) = q(i,j,k,QALPHA1);}
    // if(al == 1){qL(i,j,k,QALPHA2) = q(i,j,k,QALPHA2); qR(i,j+1,k,QALPHA2) = q(i,j,k,QALPHA2);}

    // qL(i,j,k,QALPHA1) = q(i,j,k,QALPHA1);
    // qL(i,j,k,QALPHA2) = q(i,j,k,QALPHA2);
    // qL(i,j,k,QRHO1) = q(i,j,k,QRHO1);
    // qL(i,j,k,QRHO2) = q(i,j,k,QRHO2);
    // qL(i,j,k,QRHO3) = q(i,j,k,QRHO3);
    // qL(i,j,k,QU) = q(i,j,k,QU);
    // qL(i,j,k,QV) = q(i,j,k,QV);
    // qL(i,j,k,QW) = q(i,j,k,QW);
    // qL(i,j,k,QPRES) = q(i,j,k,QPRES);
    // qL(i,j,k,QPI) = q(i,j,k,QPI);
    // qL(i,j,k,QV11) = q(i,j,k,QV11);
    // qL(i,j,k,QV21) = q(i,j,k,QV21);
    // qL(i,j,k,QV31) = q(i,j,k,QV31);
    // qL(i,j,k,QV12) = q(i,j,k,QV12);
    // qL(i,j,k,QV22) = q(i,j,k,QV22);
    // qL(i,j,k,QV32) = q(i,j,k,QV32);
    // qL(i,j,k,QV13) = q(i,j,k,QV13);
    // qL(i,j,k,QV23) = q(i,j,k,QV23);
    // qL(i,j,k,QV33) = q(i,j,k,QV33);

    // // qR(i+1,j,k)
    // qR(i,j+1,k,QALPHA1) = q(i,j,k,QALPHA1);
    // qR(i,j+1,k,QALPHA2) = q(i,j,k,QALPHA2);
    // qR(i,j+1,k,QRHO1) = q(i,j,k,QRHO1);
    // qR(i,j+1,k,QRHO2) = q(i,j,k,QRHO2);
    // qR(i,j+1,k,QRHO3) = q(i,j,k,QRHO3);
    // qR(i,j+1,k,QU) = q(i,j,k,QU);
    // qR(i,j+1,k,QV) = q(i,j,k,QV);
    // qR(i,j+1,k,QW) = q(i,j,k,QW);
    // qR(i,j+1,k,QPRES) = q(i,j,k,QPRES);
    // qR(i,j+1,k,QPI) = q(i,j,k,QPI);
    // qR(i,j+1,k,QV11) = q(i,j,k,QV11);
    // qR(i,j+1,k,QV21) = q(i,j,k,QV21);
    // qR(i,j+1,k,QV31) = q(i,j,k,QV31);
    // qR(i,j+1,k,QV12) = q(i,j,k,QV12);
    // qR(i,j+1,k,QV22) = q(i,j,k,QV22);
    // qR(i,j+1,k,QV32) = q(i,j,k,QV32);
    // qR(i,j+1,k,QV13) = q(i,j,k,QV13);
    // qR(i,j+1,k,QV23) = q(i,j,k,QV23);
    // qR(i,j+1,k,QV33) = q(i,j,k,QV33);
}

AMREX_GPU_DEVICE
inline
void
EMM_PMUSCL_reconstruct_z (int i, int j, int k,
             amrex::Array4<amrex::Real> const& qL,
             amrex::Array4<amrex::Real> const& qR,
             amrex::Array4<amrex::Real const> const& q, amrex::Real dt,
             amrex::GeometryData const& geomdata,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using namespace amrex::literals;

    qL(i,j,k,QALPHA1) = q(i,j,k,QALPHA1) - 0.5_rt*zlimiter(i, j, k, q, QALPHA1);
    qL(i,j,k,QALPHA2) = q(i,j,k,QALPHA2) - 0.5_rt*zlimiter(i, j, k, q, QALPHA2);
    qL(i,j,k,QRHO1) = q(i,j,k,QRHO1) - 0.5_rt*zlimiter(i, j, k, q,QRHO1);
    qL(i,j,k,QRHO2) = q(i,j,k,QRHO2) - 0.5_rt*zlimiter(i, j, k, q,QRHO2);
    qL(i,j,k,QRHO3) = q(i,j,k,QRHO3) - 0.5_rt*zlimiter(i, j, k, q,QRHO3);
    qL(i,j,k,QU) = q(i,j,k,QU) - 0.5_rt*zlimiter(i, j, k, q,QU);
    qL(i,j,k,QV) = q(i,j,k,QV) - 0.5_rt*zlimiter(i, j, k, q,QV);
    qL(i,j,k,QW) = q(i,j,k,QW) - 0.5_rt*zlimiter(i, j, k, q,QW);
    qL(i,j,k,QPRES) = q(i,j,k,QPRES) - 0.5_rt*zlimiter(i, j, k, q,QPRES);
    qL(i,j,k,QPI) = q(i,j,k,QPI) - 0.5_rt*zlimiter(i, j, k, q,QPI);
    qL(i,j,k,QV11) = q(i,j,k,QV11) - 0.5*zlimiter(i, j, k, q,QV11);
    qL(i,j,k,QV21) = q(i,j,k,QV21) - 0.5*zlimiter(i, j, k, q,QV21);
    qL(i,j,k,QV31) = q(i,j,k,QV31) - 0.5*zlimiter(i, j, k, q,QV31);
    qL(i,j,k,QV12) = q(i,j,k,QV12) - 0.5*zlimiter(i, j, k, q,QV12);
    qL(i,j,k,QV22) = q(i,j,k,QV22) - 0.5*zlimiter(i, j, k, q,QV22);
    qL(i,j,k,QV32) = q(i,j,k,QV32) - 0.5*zlimiter(i, j, k, q,QV32);
    qL(i,j,k,QV13) = q(i,j,k,QV13) - 0.5*zlimiter(i, j, k, q,QV13);
    qL(i,j,k,QV23) = q(i,j,k,QV23) - 0.5*zlimiter(i, j, k, q,QV23);
    qL(i,j,k,QV33) = q(i,j,k,QV33) - 0.5*zlimiter(i, j, k, q,QV33);

    // qR(i+1,j,k
    qR(i,j,k+1,QALPHA1) = q(i,j,k,QALPHA1) + 0.5_rt*zlimiter(i, j, k, q,QALPHA1);
    qR(i,j,k+1,QALPHA2) = q(i,j,k,QALPHA2) + 0.5_rt*zlimiter(i, j, k, q,QALPHA2);
    qR(i,j,k+1,QRHO1) = q(i,j,k,QRHO1) + 0.5_rt*zlimiter(i, j, k, q,QRHO1);
    qR(i,j,k+1,QRHO2) = q(i,j,k,QRHO2) + 0.5_rt*zlimiter(i, j, k, q,QRHO2);
    qR(i,j,k+1,QRHO3) = q(i,j,k,QRHO3) + 0.5_rt*zlimiter(i, j, k, q,QRHO3);
    qR(i,j,k+1,QU) = q(i,j,k,QU) + 0.5_rt*zlimiter(i, j, k, q,QU);
    qR(i,j,k+1,QV) = q(i,j,k,QV) + 0.5_rt*zlimiter(i, j, k, q,QV);
    qR(i,j,k+1,QW) = q(i,j,k,QW) + 0.5_rt*zlimiter(i, j, k, q,QW);
    qR(i,j,k+1,QPRES) = q(i,j,k,QPRES) + 0.5_rt*zlimiter(i, j, k, q,QPRES);
    qR(i,j,k+1,QPI) = q(i,j,k,QPI) + 0.5_rt*zlimiter(i, j, k, q,QPI);
    qR(i,j,k+1,QV11) = q(i,j,k,QV11) + 0.5*zlimiter(i, j, k, q,QV11);
    qR(i,j,k+1,QV21) = q(i,j,k,QV21) + 0.5*zlimiter(i, j, k, q,QV21);
    qR(i,j,k+1,QV31) = q(i,j,k,QV31) + 0.5*zlimiter(i, j, k, q,QV31);
    qR(i,j,k+1,QV12) = q(i,j,k,QV12) + 0.5*zlimiter(i, j, k, q,QV12);
    qR(i,j,k+1,QV22) = q(i,j,k,QV22) + 0.5*zlimiter(i, j, k, q,QV22);
    qR(i,j,k+1,QV32) = q(i,j,k,QV32) + 0.5*zlimiter(i, j, k, q,QV32);
    qR(i,j,k+1,QV13) = q(i,j,k,QV13) + 0.5*zlimiter(i, j, k, q,QV13);
    qR(i,j,k+1,QV23) = q(i,j,k,QV23) + 0.5*zlimiter(i, j, k, q,QV23);
    qR(i,j,k+1,QV33) = q(i,j,k,QV33) + 0.5*zlimiter(i, j, k, q,QV33);
}

namespace {

AMREX_GPU_DEVICE
inline
void
riemann (Parm const& parm,
         const amrex::Real rho1L, const amrex::Real rho2L, const amrex::Real rho3L, const amrex::Real alpha1L, const amrex::Real alpha2L, const amrex::Real alpha3L,
         const amrex::Real mrhoL, const amrex::Real uL, const amrex::Real ut1L, const amrex::Real ut2L,
         const amrex::Real EL, const amrex::Real piL, const amrex::Real cL,
         const amrex::Real V11L, const amrex::Real V12L, const amrex::Real V13L,
         const amrex::Real V21L, const amrex::Real V22L, const amrex::Real V23L,
         const amrex::Real V31L, const amrex::Real V32L, const amrex::Real V33L,
         const amrex::Real Sigma11L, const amrex::Real Sigma21L, const amrex::Real Sigma31L,
         const amrex::Real rho1R, const amrex::Real rho2R, const amrex::Real rho3R, const amrex::Real alpha1R, const amrex::Real alpha2R, const amrex::Real alpha3R,
         const amrex::Real mrhoR, const amrex::Real uR, const amrex::Real ut1R, const amrex::Real ut2R,
         const amrex::Real ER, const amrex::Real piR, const amrex::Real cR,
         const amrex::Real V11R, const amrex::Real V12R, const amrex::Real V13R,
         const amrex::Real V21R, const amrex::Real V22R, const amrex::Real V23R,
         const amrex::Real V31R, const amrex::Real V32R, const amrex::Real V33R,
         const amrex::Real Sigma11R, const amrex::Real Sigma21R, const amrex::Real Sigma31R,
         amrex::Real& flxarho1, amrex::Real& flxarho2, amrex::Real& flxarho3, amrex::Real& flxrhou,
         amrex::Real& flxrhout, amrex::Real& flxrhoutt, amrex::Real& flxrhoE, amrex::Real& flxarp,
         amrex::Real& flxalpha1, amrex::Real& flxalpha2,
         amrex::Real& flxV11, amrex::Real& flxV12, amrex::Real& flxV13,
         amrex::Real& flxV21, amrex::Real& flxV22, amrex::Real& flxV23,
         amrex::Real& flxV31, amrex::Real& flxV32, amrex::Real& flxV33,
         amrex::Real& US, amrex::Real& VS, amrex::Real& WS,
         amrex::Real& VF11, amrex::Real& VF12, amrex::Real& VF13,
         amrex::Real& VF21, amrex::Real& VF22, amrex::Real& VF23,
         amrex::Real& VF31, amrex::Real& VF32, amrex::Real& VF33,int cdir) noexcept
{
    using namespace amrex::literals;

    amrex::Real ub = 0.5*(uL+uR);
    amrex::Real cb = 0.5*(cL+cR);

    amrex::Real SL = amrex::min(ub-cb,uL-cL); //o
    amrex::Real SR = amrex::max(ub+cb,uR+cR); //o
    // amrex::Real SL = amrex::min(uL-cL,uR-cR); //o
    // amrex::Real SR = amrex::max(uL+cL,uR+cR); //o

    // Contact wave-speed
    amrex::Real uStar = (mrhoL*uL*(SL-uL) - mrhoR*uR*(SR-uR) - Sigma11R + Sigma11L)/(mrhoL*(SL-uL)-mrhoR*(SR-uR));

    // Star variables
    amrex::Real alpha1S_L = alpha1L; amrex::Real alpha1S_R = alpha1R; //o
    amrex::Real alpha2S_L = alpha2L; amrex::Real alpha2S_R = alpha2R; //o
    amrex::Real alpha3S_L = alpha3L; amrex::Real alpha3S_R = alpha3R; //o

    amrex::Real al1rho1S_L = alpha1S_L*rho1L*(SL-uL)/(SL-uStar); amrex::Real al1rho1S_R = alpha1S_R*rho1R*(SR-uR)/(SR-uStar);
    amrex::Real al2rho2S_L = alpha2S_L*rho2L*(SL-uL)/(SL-uStar); amrex::Real al2rho2S_R = alpha2S_R*rho2R*(SR-uR)/(SR-uStar);
    amrex::Real al3rho3S_L = alpha3S_L*rho3L*(SL-uL)/(SL-uStar); amrex::Real al3rho3S_R = alpha3S_R*rho3R*(SR-uR)/(SR-uStar);

    amrex::Real rho1S_L = rho1L*(SL-uL)/(SL-uStar); amrex::Real rho1S_R = rho1R*(SR-uR)/(SR-uStar);
    amrex::Real rho2S_L = rho2L*(SL-uL)/(SL-uStar); amrex::Real rho2S_R = rho2R*(SR-uR)/(SR-uStar);
    amrex::Real rho3S_L = rho3L*(SL-uL)/(SL-uStar); amrex::Real rho3S_R = rho3R*(SR-uR)/(SR-uStar);

    amrex::Real mrhoS_L = al1rho1S_L + al2rho2S_L + al3rho3S_L;
    amrex::Real mrhoS_R = al1rho1S_R + al2rho2S_R + al3rho3S_R;

    amrex::Real G1L = parm.eos_G01; amrex::Real G1R = parm.eos_G01;
    amrex::Real G2L = parm.eos_G02; amrex::Real G2R = parm.eos_G02;
    amrex::Real G3L = parm.eos_G03; amrex::Real G3R = parm.eos_G03;
    
    amrex::Real GL = alpha1S_L*G1L + alpha2S_L*G2L + alpha3S_L*G3L;
    amrex::Real GR = alpha1S_R*G1R + alpha2S_R*G2R + alpha3S_R*G3R;
    // Mixture Shear Modulus
    amrex::Real bL = std::sqrt(GL/mrhoS_L);
    amrex::Real bR = std::sqrt(GR/mrhoS_R);

    // Mixture shear wave speed
    amrex::Real StL = uStar - bL;
    amrex::Real StR = uStar + bR;

    // amrex::Real ES_R = mrhoR*((SR-uR)/(SR-uStar))*(ER/mrhoR) + (uStar-uR)*(uStar-Sigma11R/(mrhoR*(SR-uR)));
    // amrex::Real ES_L = mrhoL*((SL-uL)/(SL-uStar))*(EL/mrhoL) + (uStar-uL)*(uStar-Sigma11L/(mrhoL*(SL-uL))); 

    // amrex::Real rhoES_R = ((SR-uR)/(SR-uStar))*mrhoR*ER + (uStar-uR)*(mrhoR*uStar-Sigma11R/(SR-uR));
    // amrex::Real rhoES_L = ((SL-uL)/(SL-uStar))*mrhoL*EL + (uStar-uL)*(mrhoL*uStar-Sigma11L/(SL-uL));
    // amrex::Real ES_R = rhoES_R/mrhoS_R;
    // amrex::Real ES_L = rhoES_L/mrhoS_L;
    amrex::Real ES_R = ER + (uStar-uR)*(uStar-Sigma11R/(mrhoR*(SR-uR)));
    amrex::Real ES_L = EL + (uStar-uL)*(uStar-Sigma11L/(mrhoL*(SL-uL)));

    amrex::Real V21S_L = V21L*(SL-uL)/(SL-uStar); amrex::Real V31S_L = V31L*(SL-uL)/(SL-uStar); 
    amrex::Real V22S_L = V22L*(SL-uL)/(SL-uStar); amrex::Real V32S_L = V32L*(SL-uL)/(SL-uStar); 
    amrex::Real V23S_L = V23L*(SL-uL)/(SL-uStar); amrex::Real V33S_L = V33L*(SL-uL)/(SL-uStar);

    amrex::Real V21S_R = V21R*(SR-uR)/(SR-uStar); amrex::Real V31S_R = V31R*(SR-uR)/(SR-uStar);
    amrex::Real V22S_R = V22R*(SR-uR)/(SR-uStar); amrex::Real V32S_R = V32R*(SR-uR)/(SR-uStar);
    amrex::Real V23S_R = V23R*(SR-uR)/(SR-uStar); amrex::Real V33S_R = V33R*(SR-uR)/(SR-uStar);

    // Variables for the Star Star State
    amrex::Real Sigma21SS = 0.0_rt; amrex::Real Sigma31SS = 0.0_rt;
    amrex::Real ut1SS_L   = ut1L; amrex::Real ut1SS_R   = ut1R;
    amrex::Real ut2SS_L   = ut2L; amrex::Real ut2SS_R   = ut2R;
    amrex::Real V21SS_L = V21S_L; amrex::Real V31SS_L = V31S_L; amrex::Real V21SS_R = V21S_R; amrex::Real V31SS_R = V31S_R;
    amrex::Real V22SS_L = V22S_L; amrex::Real V32SS_L = V32S_L; amrex::Real V22SS_R = V22S_R; amrex::Real V32SS_R = V32S_R;
    amrex::Real V23SS_L = V23S_L; amrex::Real V33SS_L = V33S_L; amrex::Real V23SS_R = V23S_R; amrex::Real V33SS_R = V33S_R;
    amrex::Real ESS_L = ES_L; amrex::Real ESS_R = ES_R;

    if(StL != uStar || StR != uStar){
        Sigma21SS = (mrhoS_L*(uStar-StL)*mrhoS_R*(uStar-StR)*(ut1L-ut1R) + mrhoS_L*(uStar-StL)*Sigma21R - mrhoS_R*(uStar-StR)*Sigma21L)/(mrhoS_L*(uStar-StL) - mrhoS_R*(uStar-StR));
        Sigma31SS = (mrhoS_L*(uStar-StL)*mrhoS_R*(uStar-StR)*(ut2L-ut2R) + mrhoS_L*(uStar-StL)*Sigma31R - mrhoS_R*(uStar-StR)*Sigma31L)/(mrhoS_L*(uStar-StL) - mrhoS_R*(uStar-StR));
    }
    if(StL != uStar){
        ut1SS_L = ut1L + (1.0_rt/(mrhoS_L*(uStar-StL)))*(Sigma21SS-Sigma21L);
        ut2SS_L = ut2L + (1.0_rt/(mrhoS_L*(uStar-StL)))*(Sigma31SS-Sigma31L);
        ESS_L   = ES_L + (1.0_rt/(mrhoS_L*(uStar-StL)))*(ut1SS_L*Sigma21SS-ut1L*Sigma21L + ut2SS_L*Sigma31SS-ut2L*Sigma31L);

        V21SS_L = V21S_L + (1.0_rt/(uStar-StL))*(V11L*(ut1SS_L - ut1L)); V31SS_L = V31S_L + (1.0_rt/(uStar-StL))*(V11L*(ut2SS_L - ut2L));
        V22SS_L = V22S_L + (1.0_rt/(uStar-StL))*(V12L*(ut1SS_L - ut1L)); V32SS_L = V32S_L + (1.0_rt/(uStar-StL))*(V12L*(ut2SS_L - ut2L));
        V23SS_L = V23S_L + (1.0_rt/(uStar-StL))*(V13L*(ut1SS_L - ut1L)); V33SS_L = V33S_L + (1.0_rt/(uStar-StL))*(V13L*(ut2SS_L - ut2L));
    }
    if(StR != uStar){
        ut1SS_R = ut1R + (1.0_rt/(mrhoS_R*(uStar-StR)))*(Sigma21SS-Sigma21R);
        ut2SS_R = ut2R + (1.0_rt/(mrhoS_R*(uStar-StR)))*(Sigma31SS-Sigma31R);
        ESS_R   = ES_R + (1.0_rt/(mrhoS_R*(uStar-StR)))*(ut1SS_R*Sigma21SS-ut1R*Sigma21R + ut2SS_R*Sigma31SS-ut2R*Sigma31R);

        V21SS_R = V21S_R + (1.0_rt/(uStar-StR))*(V11R*(ut1SS_R - ut1R)); V31SS_R = V31S_R + (1.0_rt/(uStar-StR))*(V11R*(ut2SS_R - ut2R));
        V22SS_R = V22S_R + (1.0_rt/(uStar-StR))*(V12R*(ut1SS_R - ut1R)); V32SS_R = V32S_R + (1.0_rt/(uStar-StR))*(V12R*(ut2SS_R - ut2R));
        V23SS_R = V23S_R + (1.0_rt/(uStar-StR))*(V13R*(ut1SS_R - ut1R)); V33SS_R = V33S_R + (1.0_rt/(uStar-StR))*(V13R*(ut2SS_R - ut2R));
    }

    if (SL >= 0.0_rt){
        // Left State
        // Print() << "Left State" << "\n";
        flxarho1 = alpha1L*rho1L*uL;
        flxarho2 = alpha2L*rho2L*uL;
        flxarho3 = alpha3L*rho3L*uL;
        flxrhou = mrhoL*uL*uL - Sigma11L;
        flxrhout = mrhoL*uL*ut1L - Sigma21L;
        flxrhoutt = mrhoL*uL*ut2L - Sigma31L;
        flxrhoE = uL*(mrhoL*EL-Sigma11L) - ut1L*Sigma21L - ut2L*Sigma31L;
        flxarp = alpha1L*rho1L*piL*uL;
        flxalpha1 = alpha1L*uL;
        flxalpha2 = alpha2L*uL;
        flxV11 = 0.0_rt;
        flxV21 = uL*V21L-ut1L*V11L;
        flxV31 = uL*V31L-ut2L*V11L;
        flxV12 = 0.0_rt;
        flxV22 = uL*V22L-ut1L*V12L;
        flxV32 = uL*V32L-ut2L*V12L;
        flxV13 = 0.0_rt;
        flxV23 = uL*V23L-ut1L*V13L;
        flxV33 = uL*V33L-ut2L*V13L;
        // Face velocities for non-conservative update
        US = uL;
        VS = ut1L;
        WS = ut2L;
        VF11 = V11L; VF12 = V12L; VF13 = V13L;
        VF21 = V11L; VF22 = V12L; VF23 = V13L;
        VF31 = V11L; VF32 = V12L; VF33 = V13L;
    }
    if (SL < 0.0_rt && 0.0_rt <= StL ){
        // Left Star State
        // Print() << "Left Star State" << "\n";
        flxarho1 = alpha1L*rho1L*uL + SL*(al1rho1S_L - alpha1L*rho1L);
        flxarho2 = alpha2L*rho2L*uL + SL*(al2rho2S_L - alpha2L*rho2L);
        flxarho3 = alpha3L*rho3L*uL + SL*(al3rho3S_L - alpha3L*rho3L);
        flxrhou = mrhoL*uL*uL - Sigma11L + SL*(mrhoS_L*uStar - mrhoL*uL);
        flxrhout = mrhoL*uL*ut1L - Sigma21L + SL*(mrhoS_L*ut1L - mrhoL*ut1L);
        flxrhoutt = mrhoL*uL*ut2L - Sigma31L + SL*(mrhoS_L*ut2L - mrhoL*ut2L);
        flxrhoE = uL*(mrhoL*EL-Sigma11L) - ut1L*Sigma21L - ut2L*Sigma31L + SL*(mrhoS_L*ES_L - mrhoL*EL);
        flxarp = alpha1L*rho1L*piL*uL + SL*(al1rho1S_L*piL - alpha1L*rho1L*piL);
        flxalpha1 = alpha1S_L*uStar;
        flxalpha2 = alpha2S_L*uStar;
        // flxalpha = alpha1L*uL + SL*(alpha1S_L - alpha1L);
        flxV11 = 0.0_rt;
        flxV21 = uL*V21L-ut1L*V11L + SL*(V21S_L - V21L);
        flxV31 = uL*V31L-ut2L*V11L + SL*(V31S_L - V31L);
        flxV12 = 0.0_rt;
        flxV22 = uL*V22L-ut1L*V12L + SL*(V22S_L - V22L);
        flxV32 = uL*V32L-ut2L*V12L + SL*(V32S_L - V32L);
        flxV13 = 0.0_rt;
        flxV23 = uL*V23L-ut1L*V13L + SL*(V23S_L - V23L);
        flxV33 = uL*V33L-ut2L*V13L + SL*(V33S_L - V33L);
        // flxV11 = 0.0_rt;
        // flxV21 = uStar*V21S_L-ut1L*V11L;
        // flxV31 = uStar*V31S_L-ut2L*V11L;
        // flxV12 = 0.0_rt;
        // flxV22 = uStar*V22S_L-ut1L*V12L;
        // flxV32 = uStar*V32S_L-ut2L*V12L;
        // flxV13 = 0.0_rt;
        // flxV23 = uStar*V23S_L-ut1L*V13L;
        // flxV33 = uStar*V33S_L-ut2L*V13L;
        // Face velocities for non-conservative update
        US = uStar; //VF11 = V11L;
        VS = ut1L; //VF12 = V12L;
        WS = ut2L; //VF13 = V13L;
        VF11 = V11L; VF12 = V12L; VF13 = V13L;
        VF21 = V11L; VF22 = V12L; VF23 = V13L;
        VF31 = V11L; VF32 = V12L; VF33 = V13L;
    }
    if (StL < 0.0_rt && 0.0_rt <= uStar ){
        // Left Star Star State
        // Print() << "Left Star Star State" << "\n";
        flxarho1 = alpha1L*rho1L*uL + SL*(al1rho1S_L - alpha1L*rho1L);
        flxarho2 = alpha2L*rho2L*uL + SL*(al2rho2S_L - alpha2L*rho2L);
        flxarho3 = alpha3L*rho3L*uL + SL*(al3rho3S_L - alpha3L*rho3L);
        flxrhou = mrhoL*uL*uL - Sigma11L + SL*(mrhoS_L*uStar - mrhoL*uL);
        flxrhout = mrhoL*uL*ut1L - Sigma21L + SL*(mrhoS_L*ut1L - mrhoL*ut1L) + StL*(mrhoS_L*ut1SS_L - mrhoS_L*ut1L);
        flxrhoutt = mrhoL*uL*ut2L - Sigma31L + SL*(mrhoS_L*ut2L - mrhoL*ut2L) + StL*(mrhoS_L*ut2SS_L - mrhoS_L*ut2L);
        flxrhoE = uL*(mrhoL*EL-Sigma11L) - ut1L*Sigma21L - ut2L*Sigma31L + SL*(mrhoS_L*ES_L - mrhoL*EL) + StL*(mrhoS_L*ESS_L - mrhoS_L*ES_L);
        flxarp = alpha1L*rho1L*piL*uL + SL*(al1rho1S_L*piL - alpha1L*rho1L*piL);
        flxalpha1 = alpha1S_L*uStar;
        flxalpha2 = alpha2S_L*uStar;
        // flxalpha = alpha1L*uL + SL*(alpha1S_L - alpha1L);
        flxV11 = 0.0_rt;
        flxV21 = uL*V21L-ut1L*V11L + SL*(V21S_L - V21L) + StL*(V21SS_L - V21S_L);
        flxV31 = uL*V31L-ut2L*V11L + SL*(V31S_L - V31L) + StL*(V31SS_L - V31S_L);
        flxV12 = 0.0_rt;
        flxV22 = uL*V22L-ut1L*V12L + SL*(V22S_L - V22L) + StL*(V22SS_L - V22S_L);
        flxV32 = uL*V32L-ut2L*V12L + SL*(V32S_L - V32L) + StL*(V32SS_L - V32S_L);
        flxV13 = 0.0_rt;
        flxV23 = uL*V23L-ut1L*V13L + SL*(V23S_L - V23L) + StL*(V23SS_L - V23S_L);
        flxV33 = uL*V33L-ut2L*V13L + SL*(V33S_L - V33L) + StL*(V33SS_L - V33S_L);
        // flxV11 = 0.0_rt;
        // flxV21 = uStar*V21SS_L-ut1SS_L*V11L;
        // flxV31 = uStar*V31SS_L-ut2SS_L*V11L;
        // flxV12 = 0.0_rt;
        // flxV22 = uStar*V22SS_L-ut1SS_L*V12L;
        // flxV32 = uStar*V32SS_L-ut2SS_L*V12L;
        // flxV13 = 0.0_rt;
        // flxV23 = uStar*V23SS_L-ut1SS_L*V13L;
        // flxV33 = uStar*V33SS_L-ut2SS_L*V13L;
        // Face velocities for non-conservative update
        US = uStar; //VF11 = V11L;
        VS = ut1SS_L; //VF12 = V12L;
        WS = ut2SS_L; //VF13 = V13L;
        VF11 = V11L; VF12 = V12L; VF13 = V13L;
        VF21 = V11L; VF22 = V12L; VF23 = V13L;
        VF31 = V11L; VF32 = V12L; VF33 = V13L;
    }
    if (uStar < 0.0_rt && 0.0_rt < StR ){
        // Right Star Star State
        // Print() << "Right Star Star State" << "\n";
        flxarho1 = alpha1R*rho1R*uR + SR*(al1rho1S_R - alpha1R*rho1R);
        flxarho2 = alpha2R*rho2R*uR + SR*(al2rho2S_R - alpha2R*rho2R);
        flxarho3 = alpha3R*rho3R*uR + SR*(al3rho3S_R - alpha3R*rho3R);
        flxrhou = mrhoR*uR*uR - Sigma11R + SR*(mrhoS_R*uStar - mrhoR*uR);
        flxrhout = mrhoR*uR*ut1R - Sigma21R + SR*(mrhoS_R*ut1R - mrhoR*ut1R) + StR*(mrhoS_R*ut1SS_R - mrhoS_R*ut1R);
        flxrhoutt = mrhoR*uR*ut2R - Sigma31R + SR*(mrhoS_R*ut2R - mrhoR*ut2R) + StR*(mrhoS_R*ut2SS_R - mrhoS_R*ut2R);
        flxrhoE = uR*(mrhoR*ER-Sigma11R) - ut1R*Sigma21R - ut2R*Sigma31R + SR*(mrhoS_R*ES_R - mrhoR*ER) + StR*(mrhoS_R*ESS_R - mrhoS_R*ES_R);
        flxarp = alpha1R*rho1R*piR*uR + SR*(al1rho1S_R*piR - alpha1R*rho1R*piR);
        flxalpha1 = alpha1S_R*uStar;
        flxalpha2 = alpha2S_R*uStar;
        // flxalpha = alpha1R*uR + SR*(alpha1S_R - alpha1R);
        flxV11 = 0.0_rt;
        flxV21 = uR*V21R-ut1R*V11R + SR*(V21S_R - V21R) + StR*(V21SS_R - V21S_R);
        flxV31 = uR*V31R-ut2R*V11R + SR*(V31S_R - V31R) + StR*(V31SS_R - V31S_R);
        flxV12 = 0.0_rt;
        flxV22 = uR*V22R-ut1R*V12R + SR*(V22S_R - V22R) + StR*(V22SS_R - V22S_R);
        flxV32 = uR*V32R-ut2R*V12R + SR*(V32S_R - V32R) + StR*(V32SS_R - V32S_R);
        flxV13 = 0.0_rt;
        flxV23 = uR*V23R-ut1R*V13R + SR*(V23S_R - V23R) + StR*(V23SS_R - V23S_R);
        flxV33 = uR*V33R-ut2R*V13R + SR*(V33S_R - V33R) + StR*(V33SS_R - V33S_R);
        // flxV11 = 0.0_rt;
        // flxV21 = uStar*V21SS_R-ut1SS_R*V11R;
        // flxV31 = uStar*V31SS_R-ut2SS_R*V11R;
        // flxV12 = 0.0_rt;
        // flxV22 = uStar*V22SS_R-ut1SS_R*V12R;
        // flxV32 = uStar*V32SS_R-ut2SS_R*V12R;
        // flxV13 = 0.0_rt;
        // flxV23 = uStar*V23SS_R-ut1SS_R*V13R;
        // flxV33 = uStar*V33SS_R-ut2SS_R*V13R;
        // Face velocities for non-conservative update
        US = uStar; //VF11 = V11R;
        VS = ut1SS_R; //VF12 = V12R;
        WS = ut2SS_R; //VF13 = V13R;
        VF11 = V11R; VF12 = V12R; VF13 = V13R;
        VF21 = V11R; VF22 = V12R; VF23 = V13R;
        VF31 = V11R; VF32 = V12R; VF33 = V13R;
    }
    if (StR <= 0.0_rt && 0.0_rt < SR ){
        // Right Star State
        // Print() << "Right Star State" << "\n";
        flxarho1 = alpha1R*rho1R*uR + SR*(al1rho1S_R - alpha1R*rho1R);
        flxarho2 = alpha2R*rho2R*uR + SR*(al2rho2S_R - alpha2R*rho2R);
        flxarho3 = alpha3R*rho3R*uR + SR*(al3rho3S_R - alpha3R*rho3R);
        flxrhou = mrhoR*uR*uR - Sigma11R + SR*(mrhoS_R*uStar - mrhoR*uR);
        flxrhout = mrhoR*uR*ut1R - Sigma21R + SR*(mrhoS_R*ut1R - mrhoR*ut1R);
        flxrhoutt = mrhoR*uR*ut2R - Sigma31R + SR*(mrhoS_R*ut2R - mrhoR*ut2R);
        flxrhoE = uR*(mrhoR*ER-Sigma11R) - ut1R*Sigma21R - ut2R*Sigma31R + SR*(mrhoS_R*ES_R - mrhoR*ER);
        flxarp = alpha1R*rho1R*piR*uR + SR*(al1rho1S_R*piR - alpha1R*rho1R*piR);
        flxalpha1 = alpha1S_R*uStar;
        flxalpha2 = alpha2S_R*uStar;
        // flxalpha = alpha1R*uR + SR*(alpha1S_R - alpha1R);
        flxV11 = 0.0_rt;
        flxV21 = uR*V21R-ut1R*V11R + SR*(V21S_R - V21R);
        flxV31 = uR*V31R-ut2R*V11R + SR*(V31S_R - V31R);
        flxV12 = 0.0_rt;
        flxV22 = uR*V22R-ut1R*V12R + SR*(V22S_R - V22R);
        flxV32 = uR*V32R-ut2R*V12R + SR*(V32S_R - V32R);
        flxV13 = 0.0_rt;
        flxV23 = uR*V23R-ut1R*V13R + SR*(V23S_R - V23R);
        flxV33 = uR*V33R-ut2R*V13R + SR*(V33S_R - V33R);
        // flxV11 = 0.0_rt;
        // flxV21 = uStar*V21S_R-ut1R*V11R;
        // flxV31 = uStar*V31S_R-ut2R*V11R;
        // flxV12 = 0.0_rt;
        // flxV22 = uStar*V22S_R-ut1R*V12R;
        // flxV32 = uStar*V32S_R-ut2R*V12R;
        // flxV13 = 0.0_rt;
        // flxV23 = uStar*V23S_R-ut1R*V13R;
        // flxV33 = uStar*V33S_R-ut2R*V13R;
        // Face velocities for non-conservative update
        US = uStar; //VF11 = V11R;
        VS = ut1R; //VF12 = V12R;
        WS = ut2R; //VF13 = V13R;
        VF11 = V11R; VF12 = V12R; VF13 = V13R;
        VF21 = V11R; VF22 = V12R; VF23 = V13R;
        VF31 = V11R; VF32 = V12R; VF33 = V13R;
    }

    if ( 0.0_rt >= SR ){
        // Right State
        // Print() << "Right State" << "\n";
        flxarho1 = alpha1R*rho1R*uR;
        flxarho2 = alpha2R*rho2R*uR;
        flxarho3 = alpha3R*rho3R*uR;
        flxrhou = mrhoR*uR*uR - Sigma11R;
        flxrhout = mrhoR*uR*ut1R - Sigma21R;
        flxrhoutt = mrhoR*uR*ut2R - Sigma31R;
        flxrhoE = uR*(mrhoR*ER-Sigma11R) - ut1R*Sigma21R - ut2R*Sigma31R;
        flxarp = alpha1R*rho1R*piR*uR;
        flxalpha1 = alpha1R*uR;
        flxalpha2 = alpha2R*uR;
        flxV11 = 0.0_rt;
        flxV21 = uR*V21R-ut1R*V11R;
        flxV31 = uR*V31R-ut2R*V11R;
        flxV12 = 0.0_rt;
        flxV22 = uR*V22R-ut1R*V12R;
        flxV32 = uR*V32R-ut2R*V12R;
        flxV13 = 0.0_rt;
        flxV23 = uR*V23R-ut1R*V13R;
        flxV33 = uR*V33R-ut2R*V13R;
        // Face velocities for non-conservative update
        US = uR; //VF11 = V11R;
        VS = ut1R; //VF12 = V12R;
        WS = ut2R; //VF13 = V13R;
        VF11 = V11R; VF12 = V12R; VF13 = V13R;
        VF21 = V11R; VF22 = V12R; VF23 = V13R;
        VF31 = V11R; VF32 = V12R; VF33 = V13R;
    }
}
}

AMREX_GPU_DEVICE
inline
void
EMM_riemann_x (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fx,
               amrex::Array4<amrex::Real> const& USx,
               amrex::Array4<amrex::Real> const& VSx,
               amrex::Array4<amrex::Real> const& WSx,
               amrex::Array4<amrex::Real> const& VFx,
               amrex::Array4<amrex::Real> const& qL,
               amrex::Array4<amrex::Real> const& qR,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& u,
               Parm const& parm) noexcept
{
    using namespace amrex::literals;
    // Print() << "riemann x---------" << "\n";
    // Right
    amrex::Real rho1R = qL(i,j,k,QRHO1);
    amrex::Real rho2R = qL(i,j,k,QRHO2);
    amrex::Real rho3R = qL(i,j,k,QRHO3);
    amrex::Real alpha1R = qL(i,j,k,QALPHA1);
    amrex::Real alpha2R = qL(i,j,k,QALPHA2);
    amrex::Real alpha3R = 1.0_rt - alpha1R - alpha2R;
    amrex::Real uR = qL(i,j,k,QU);
    amrex::Real ut1R = qL(i,j,k,QV);
    amrex::Real ut2R = qL(i,j,k,QW);
    amrex::Real pR = qL(i,j,k,QPRES);
    amrex::Real piR = qL(i,j,k,QPI);
    amrex::Real V11R = qL(i,j,k,QV11); amrex::Real V12R = qL(i,j,k,QV12); amrex::Real V13R = qL(i,j,k,QV13);
    amrex::Real V21R = qL(i,j,k,QV21); amrex::Real V22R = qL(i,j,k,QV22); amrex::Real V23R = qL(i,j,k,QV23);
    amrex::Real V31R = qL(i,j,k,QV31); amrex::Real V32R = qL(i,j,k,QV32); amrex::Real V33R = qL(i,j,k,QV33);

    // int ll = 0;
    // amrex::Real d = 1E2;
    // if(alpha1R > 1.0_rt-1.0_rt*parm.alpha_min/d){alpha1R = 1.0_rt-1.0_rt*parm.alpha_min/d; ll = 1;}
    // if(alpha2R > 1.0_rt-1.0_rt*parm.alpha_min/d){alpha2R = 1.0_rt-1.0_rt*parm.alpha_min/d; ll = 1;}
    // if(alpha1R < parm.alpha_min/d){alpha1R = parm.alpha_min/d; ll = 1;}
    // if(alpha2R < parm.alpha_min/d){alpha2R = parm.alpha_min/d; ll = 1;}
    // if(ll == 1){
    //      alpha3R = 1.0_rt - alpha1R - alpha2R;
    //      alpha1R = alpha1R/(alpha1R+alpha2R+alpha3R);
    //      alpha2R = alpha2R/(alpha1R+alpha2R+alpha3R);
    // }

    amrex::Real mrhoR = alpha1R*rho1R + alpha2R*rho2R + alpha3R*rho3R;
    amrex::Real mrho_0R = alpha1R*parm.eos_rho01+alpha2R*parm.eos_rho02+alpha3R*parm.eos_rho03;
    amrex::Real i1R = compute_i1(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);

    amrex::Real G1R = parm.eos_G01;
    amrex::Real G2R = parm.eos_G02;
    amrex::Real G3R = parm.eos_G03;
    amrex::Real es1R = (G1R/(1.0_rt*parm.eos_rho01))*(i1R - 3.0);
    amrex::Real es2R = (G2R/(1.0_rt*parm.eos_rho02))*(i1R - 3.0);
    amrex::Real es3R = (G3R/(1.0_rt*parm.eos_rho03))*(i1R - 3.0);
    amrex::Real T1R = compute_temperature(alpha1R, rho1R, pR, u(i+1,j,k,GT1), 1);
    amrex::Real T2R = compute_temperature(alpha2R, rho2R, pR, u(i+1,j,k,GT2), 2);
    amrex::Real T3R = compute_temperature(alpha3R, rho3R, pR, u(i+1,j,k,GT3), 3);
    // amrex::Real T1R = compute_temperature(alpha1R, rho1R, pR, 300.0, 1);
    // amrex::Real T2R = compute_temperature(alpha2R, rho2R, pR, 300.0, 2);
    // amrex::Real T3R = compute_temperature(alpha3R, rho3R, pR, 300.0, 3);
    amrex::Real eint1R = compute_internal_energy(pR, T1R, rho1R, 1) + es1R;
    amrex::Real eint2R = compute_internal_energy(pR, T2R, rho2R, 2) + es2R;
    amrex::Real eint3R = compute_internal_energy(pR, T3R, rho3R, 3) + es3R;
    // amrex::Real eint1R = compute_NASG_ie(pR, rho1R, 1) + es1R;
    // amrex::Real eint2R = compute_NASG_ie(pR, rho2R, 2);
    // amrex::Real eint3R = compute_NASG_ie(pR, rho3R, 3);

    amrex::Real Y1R = alpha1R*rho1R/mrhoR;
    amrex::Real Y2R = alpha2R*rho2R/mrhoR;
    amrex::Real Y3R = alpha3R*rho3R/mrhoR;
    amrex::Real ER = Y1R*eint1R + Y2R*eint2R + Y3R*eint3R + 0.5_rt*uR*uR + 0.5_rt*ut1R*ut1R + 0.5_rt*ut2R*ut2R;

    // Compute the speed of sound
    amrex::Real c1sqR = amrex::max(compute_SoS(pR, T1R, rho1R, 1) + (4.0/3.0)*parm.eos_G01/parm.eos_rho01, parm.smallr);
    amrex::Real c2sqR = amrex::max(compute_SoS(pR, T2R, rho2R, 2) + (4.0/3.0)*parm.eos_G02/parm.eos_rho02, parm.smallr);
    amrex::Real c3sqR = amrex::max(compute_SoS(pR, T3R, rho3R, 3) + (4.0/3.0)*parm.eos_G03/parm.eos_rho03, parm.smallr);
    amrex::Real cR = std::sqrt(Y1R*c1sqR + Y2R*c2sqR + Y3R*c3sqR);
    amrex::Real GR = alpha1R*G1R + alpha2R*G2R + alpha3R*G3R;

    amrex::Real Sigma11R = - pR + GR*compute_devB11(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);
    amrex::Real Sigma21R = GR*compute_devB21(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);
    amrex::Real Sigma31R = GR*compute_devB31(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);

    // Left
    amrex::Real rho1L = qR(i,j,k,QRHO1);
    amrex::Real rho2L = qR(i,j,k,QRHO2);
    amrex::Real rho3L = qR(i,j,k,QRHO3);
    amrex::Real alpha1L = qR(i,j,k,QALPHA1);
    amrex::Real alpha2L = qR(i,j,k,QALPHA2);
    amrex::Real alpha3L = 1.0_rt - alpha1L - alpha2L;
    amrex::Real uL = qR(i,j,k,QU);
    amrex::Real ut1L = qR(i,j,k,QV);
    amrex::Real ut2L = qR(i,j,k,QW);
    amrex::Real pL = qR(i,j,k,QPRES);
    amrex::Real piL = qR(i,j,k,QPI);
    amrex::Real V11L = qR(i,j,k,QV11); amrex::Real V12L = qR(i,j,k,QV12); amrex::Real V13L = qR(i,j,k,QV13);
    amrex::Real V21L = qR(i,j,k,QV21); amrex::Real V22L = qR(i,j,k,QV22); amrex::Real V23L = qR(i,j,k,QV23);
    amrex::Real V31L = qR(i,j,k,QV31); amrex::Real V32L = qR(i,j,k,QV32); amrex::Real V33L = qR(i,j,k,QV33);

    // ll = 0;
    // if(alpha1L > 1.0_rt-1.0_rt*parm.alpha_min/d){alpha1L = 1.0_rt-1.0_rt*parm.alpha_min/d; ll = 1;}
    // if(alpha2L > 1.0_rt-1.0_rt*parm.alpha_min/d){alpha2L = 1.0_rt-1.0_rt*parm.alpha_min/d; ll = 1;}
    // if(alpha1L < parm.alpha_min/d){alpha1L = parm.alpha_min/d; ll = 1;}
    // if(alpha2L < parm.alpha_min/d){alpha2L = parm.alpha_min/d; ll = 1;}
    // if(ll == 1){
    //      alpha3L = 1.0_rt - alpha1L - alpha2L;
    //      alpha1L = alpha1L/(alpha1L+alpha2L+alpha3L);
    //      alpha2L = alpha2L/(alpha1L+alpha2L+alpha3L);
    // }

    amrex::Real mrhoL = alpha1L*rho1L + alpha2L*rho2L + alpha3L*rho3L;
    amrex::Real mrho_0L = alpha1L*parm.eos_rho01+alpha2L*parm.eos_rho02+alpha3L*parm.eos_rho03;
    amrex::Real i1L = compute_i1(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);

    amrex::Real G1L = parm.eos_G01;
    amrex::Real G2L = parm.eos_G02;
    amrex::Real G3L = parm.eos_G03;
    amrex::Real es1L = (G1L/(1.0_rt*parm.eos_rho01))*(i1L - 3.0);
    amrex::Real es2L = (G2L/(1.0_rt*parm.eos_rho02))*(i1L - 3.0);
    amrex::Real es3L = (G3L/(1.0_rt*parm.eos_rho03))*(i1L - 3.0);
    amrex::Real T1L = compute_temperature(alpha1L, rho1L, pL, u(i,j,k,GT1), 1);
    amrex::Real T2L = compute_temperature(alpha2L, rho2L, pL, u(i,j,k,GT2), 2);
    amrex::Real T3L = compute_temperature(alpha3L, rho3L, pL, u(i,j,k,GT3), 3);
    // amrex::Real T1L = compute_temperature(alpha1L, rho1L, pL, 300.0, 1);
    // amrex::Real T2L = compute_temperature(alpha2L, rho2L, pL, 300.0, 2);
    // amrex::Real T3L = compute_temperature(alpha3L, rho3L, pL, 300.0, 3);
    amrex::Real eint1L = compute_internal_energy(pL, T1L, rho1L, 1) + es1L;
    amrex::Real eint2L = compute_internal_energy(pL, T2L, rho2L, 2) + es2L;
    amrex::Real eint3L = compute_internal_energy(pL, T3L, rho3L, 3) + es3L;
    // amrex::Real eint1L = compute_NASG_ie(pL, rho1L, 1) + es1L;
    // amrex::Real eint2L = compute_NASG_ie(pL, rho2L, 2);
    // amrex::Real eint3L = compute_NASG_ie(pL, rho3L, 3);
    amrex::Real Y1L = alpha1L*rho1L/mrhoL;
    amrex::Real Y2L = alpha2L*rho2L/mrhoL;
    amrex::Real Y3L = alpha3L*rho3L/mrhoL;
    amrex::Real EL = Y1L*eint1L + Y2L*eint2L + Y3L*eint3L + 0.5_rt*uL*uL + 0.5_rt*ut1L*ut1L + 0.5_rt*ut2L*ut2L;

    amrex::Real c1sqL = amrex::max(compute_SoS(pL, T1L, rho1L, 1) + (4.0/3.0)*parm.eos_G01/parm.eos_rho01, parm.smallr);
    amrex::Real c2sqL = amrex::max(compute_SoS(pL, T2L, rho2L, 2) + (4.0/3.0)*parm.eos_G02/parm.eos_rho02, parm.smallr);
    amrex::Real c3sqL = amrex::max(compute_SoS(pL, T3L, rho3L, 3) + (4.0/3.0)*parm.eos_G03/parm.eos_rho03, parm.smallr);

    amrex::Real cL = std::sqrt(Y1L*c1sqL + Y2L*c2sqL + Y3L*c3sqL);
    amrex::Real GL = alpha1L*G1L + alpha2L*G2L + alpha3L*G3L;

    amrex::Real Sigma11L = - pL + GL*compute_devB11(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);
    amrex::Real Sigma21L = GL*compute_devB21(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);
    amrex::Real Sigma31L = GL*compute_devB31(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);

    int cdir = 0;
    riemann(parm,
            rho1L, rho2L, rho3L, alpha1L, alpha2L, alpha3L, mrhoL, uL, ut1L, ut2L, EL, piL, cL, V11L, V12L, V13L, V21L, V22L, V23L, V31L, V32L, V33L,
            Sigma11L, Sigma21L, Sigma31L,
            rho1R, rho2R, rho3R, alpha1R, alpha2R, alpha3R, mrhoR, uR, ut1R, ut2R, ER, piR, cR, V11R, V12R, V13R, V21R, V22R, V23R, V31R, V32R, V33R,
            Sigma11R, Sigma21R, Sigma31R,
            fx(i,j,k,UARHO1), fx(i,j,k,UARHO2), fx(i,j,k,UARHO3), fx(i,j,k,UMX), fx(i,j,k,UMY), fx(i,j,k,UMZ), fx(i,j,k,URHOE), fx(i,j,k,UARP), fx(i,j,k,GALPHA1), fx(i,j,k,GALPHA2),
            fx(i,j,k,GV11), fx(i,j,k,GV12), fx(i,j,k,GV13),
            fx(i,j,k,GV21), fx(i,j,k,GV22), fx(i,j,k,GV23),
            fx(i,j,k,GV31), fx(i,j,k,GV32), fx(i,j,k,GV33),
            // Non-conservative Face Velocities given by Riemann Solver
            USx(i,j,k), VSx(i,j,k), WSx(i,j,k),
            VFx(i,j,k,0), VFx(i,j,k,3), VFx(i,j,k,6),
            VFx(i,j,k,1), VFx(i,j,k,4), VFx(i,j,k,7),
            VFx(i,j,k,2), VFx(i,j,k,5), VFx(i,j,k,8), cdir);
    VFx(i,j,k,9) = 0.0_rt; VFx(i,j,k,10) = 0.0_rt;
}

AMREX_GPU_DEVICE
inline
void
EMM_riemann_y (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fy,
               amrex::Array4<amrex::Real> const& USy,
               amrex::Array4<amrex::Real> const& VSy,
               amrex::Array4<amrex::Real> const& WSy,
               amrex::Array4<amrex::Real> const& VFy,
               amrex::Array4<amrex::Real> const& qL,
               amrex::Array4<amrex::Real> const& qR,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& u,
               Parm const& parm) noexcept
{
    using namespace amrex::literals;

    // Right
    amrex::Real rho1R = qL(i,j,k,QRHO1);
    amrex::Real rho2R = qL(i,j,k,QRHO2);
    amrex::Real rho3R = qL(i,j,k,QRHO3);
    amrex::Real alpha1R = qL(i,j,k,QALPHA1);
    amrex::Real alpha2R = qL(i,j,k,QALPHA2);
    amrex::Real alpha3R = 1.0_rt - alpha1R - alpha2R;
    amrex::Real uR = qL(i,j,k,QV);
    amrex::Real ut1R = qL(i,j,k,QU);
    amrex::Real ut2R = qL(i,j,k,QW);
    amrex::Real pR = qL(i,j,k,QPRES);
    amrex::Real piR = qL(i,j,k,QPI);
    amrex::Real V11R = qL(i,j,k,QV11); amrex::Real V12R = qL(i,j,k,QV12); amrex::Real V13R = qL(i,j,k,QV13);
    amrex::Real V21R = qL(i,j,k,QV21); amrex::Real V22R = qL(i,j,k,QV22); amrex::Real V23R = qL(i,j,k,QV23);
    amrex::Real V31R = qL(i,j,k,QV31); amrex::Real V32R = qL(i,j,k,QV32); amrex::Real V33R = qL(i,j,k,QV33);

    amrex::Real mrhoR = alpha1R*rho1R + alpha2R*rho2R + alpha3R*rho3R;
    amrex::Real mrho_0R = alpha1R*parm.eos_rho01+alpha2R*parm.eos_rho02+alpha3R*parm.eos_rho03;
    amrex::Real i1R = compute_i1(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);

    amrex::Real G1R = parm.eos_G01;
    amrex::Real G2R = parm.eos_G02;
    amrex::Real G3R = parm.eos_G03;
    amrex::Real es1R = (G1R/(1.0_rt*parm.eos_rho01))*(i1R - 3.0);
    amrex::Real es2R = (G2R/(1.0_rt*parm.eos_rho02))*(i1R - 3.0);
    amrex::Real es3R = (G3R/(1.0_rt*parm.eos_rho03))*(i1R - 3.0);
    amrex::Real T1R = compute_temperature(alpha1R, rho1R, pR, u(i,j+1,k,GT1), 1);
    amrex::Real T2R = compute_temperature(alpha2R, rho2R, pR, u(i,j+1,k,GT2), 2);
    amrex::Real T3R = compute_temperature(alpha3R, rho3R, pR, u(i,j+1,k,GT3), 3);
    // amrex::Real T1R = compute_temperature(alpha1R, rho1R, pR, 300.0, 1);
    // amrex::Real T2R = compute_temperature(alpha2R, rho2R, pR, 300.0, 2);
    // amrex::Real T3R = compute_temperature(alpha3R, rho3R, pR, 300.0, 3);
    amrex::Real eint1R = compute_internal_energy(pR, T1R, rho1R, 1) + es1R;
    amrex::Real eint2R = compute_internal_energy(pR, T2R, rho2R, 2) + es2R;
    amrex::Real eint3R = compute_internal_energy(pR, T3R, rho3R, 3) + es3R;
    // amrex::Real eint1R = compute_NASG_ie(pR, rho1R, 1) + es1R;
    // amrex::Real eint2R = compute_NASG_ie(pR, rho2R, 2);
    // amrex::Real eint3R = compute_NASG_ie(pR, rho3R, 3);

    amrex::Real Y1R = alpha1R*rho1R/mrhoR;
    amrex::Real Y2R = alpha2R*rho2R/mrhoR;
    amrex::Real Y3R = alpha3R*rho3R/mrhoR;
    amrex::Real ER = Y1R*eint1R + Y2R*eint2R + Y3R*eint3R + 0.5_rt*uR*uR + 0.5_rt*ut1R*ut1R + 0.5_rt*ut2R*ut2R;
    // Compute the speed of sound
    amrex::Real c1sqR = amrex::max(compute_SoS(pR, T1R, rho1R, 1) + (4.0/3.0)*parm.eos_G01/parm.eos_rho01, parm.smallr);
    amrex::Real c2sqR = amrex::max(compute_SoS(pR, T2R, rho2R, 2) + (4.0/3.0)*parm.eos_G02/parm.eos_rho02, parm.smallr);
    amrex::Real c3sqR = amrex::max(compute_SoS(pR, T3R, rho3R, 3) + (4.0/3.0)*parm.eos_G03/parm.eos_rho03, parm.smallr);

    amrex::Real cR = std::sqrt(Y1R*c1sqR + Y2R*c2sqR + Y3R*c3sqR);
    amrex::Real GR = alpha1R*G1R + alpha2R*G2R + alpha3R*G3R;

    amrex::Real Sigma12R = GR*compute_devB12(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);
    amrex::Real Sigma22R = - pR + GR*compute_devB22(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);
    amrex::Real Sigma32R = GR*compute_devB32(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);

    // Left
    amrex::Real rho1L = qR(i,j,k,QRHO1);
    amrex::Real rho2L = qR(i,j,k,QRHO2);
    amrex::Real rho3L = qR(i,j,k,QRHO3);
    amrex::Real alpha1L = qR(i,j,k,QALPHA1);
    amrex::Real alpha2L = qR(i,j,k,QALPHA2);
    amrex::Real alpha3L = 1.0_rt - alpha1L - alpha2L;
    amrex::Real uL = qR(i,j,k,QV);
    amrex::Real ut1L = qR(i,j,k,QU);
    amrex::Real ut2L = qR(i,j,k,QW);
    amrex::Real pL = qR(i,j,k,QPRES);
    amrex::Real piL = qR(i,j,k,QPI);
    amrex::Real V11L = qR(i,j,k,QV11); amrex::Real V12L = qR(i,j,k,QV12); amrex::Real V13L = qR(i,j,k,QV13);
    amrex::Real V21L = qR(i,j,k,QV21); amrex::Real V22L = qR(i,j,k,QV22); amrex::Real V23L = qR(i,j,k,QV23);
    amrex::Real V31L = qR(i,j,k,QV31); amrex::Real V32L = qR(i,j,k,QV32); amrex::Real V33L = qR(i,j,k,QV33);
    
    amrex::Real mrhoL = alpha1L*rho1L + alpha2L*rho2L + alpha3L*rho3L;
    amrex::Real mrho_0L = alpha1L*parm.eos_rho01+alpha2L*parm.eos_rho02+alpha3L*parm.eos_rho03;
    amrex::Real i1L = compute_i1(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);

    amrex::Real G1L = parm.eos_G01;
    amrex::Real G2L = parm.eos_G02;
    amrex::Real G3L = parm.eos_G03;
    amrex::Real es1L = (G1L/(1.0_rt*parm.eos_rho01))*(i1L - 3.0);
    amrex::Real es2L = (G2L/(1.0_rt*parm.eos_rho02))*(i1L - 3.0);
    amrex::Real es3L = (G3L/(1.0_rt*parm.eos_rho03))*(i1L - 3.0);
    amrex::Real T1L = compute_temperature(alpha1L, rho1L, pL, u(i,j,k,GT1), 1);
    amrex::Real T2L = compute_temperature(alpha2L, rho2L, pL, u(i,j,k,GT2), 2);
    amrex::Real T3L = compute_temperature(alpha3L, rho3L, pL, u(i,j,k,GT3), 3);
    // amrex::Real T1L = compute_temperature(alpha1L, rho1L, pL, 300.0, 1);
    // amrex::Real T2L = compute_temperature(alpha2L, rho2L, pL, 300.0, 2);
    // amrex::Real T3L = compute_temperature(alpha3L, rho3L, pL, 300.0, 3);
    amrex::Real eint1L = compute_internal_energy(pL, T1L, rho1L, 1) + es1L;
    amrex::Real eint2L = compute_internal_energy(pL, T2L, rho2L, 2) + es2L;
    amrex::Real eint3L = compute_internal_energy(pL, T3L, rho3L, 3) + es3L;
    // amrex::Real eint1L = compute_NASG_ie(pL, rho1L, 1) + es1L;
    // amrex::Real eint2L = compute_NASG_ie(pL, rho2L, 2);
    // amrex::Real eint3L = compute_NASG_ie(pL, rho3L, 3);

    amrex::Real Y1L = alpha1L*rho1L/mrhoL;
    amrex::Real Y2L = alpha2L*rho2L/mrhoL;
    amrex::Real Y3L = alpha3L*rho3L/mrhoL;
    amrex::Real EL = Y1L*eint1L + Y2L*eint2L + Y3L*eint3L + 0.5_rt*uL*uL + 0.5_rt*ut1L*ut1L + 0.5_rt*ut2L*ut2L;
    // Compute the speed of sound
    amrex::Real c1sqL = amrex::max(compute_SoS(pL, T1L, rho1L, 1) + (4.0/3.0)*parm.eos_G01/parm.eos_rho01, parm.smallr);
    amrex::Real c2sqL = amrex::max(compute_SoS(pL, T2L, rho2L, 2) + (4.0/3.0)*parm.eos_G02/parm.eos_rho02, parm.smallr);
    amrex::Real c3sqL = amrex::max(compute_SoS(pL, T3L, rho3L, 3) + (4.0/3.0)*parm.eos_G03/parm.eos_rho03, parm.smallr);
    // Longitunal Wave Speed
    amrex::Real cL = std::sqrt(Y1L*c1sqL + Y2L*c2sqL + Y3L*c3sqL);
    amrex::Real GL = alpha1L*G1L + alpha2L*G2L + alpha3L*G3L;

    amrex::Real Sigma12L = GL*compute_devB12(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);
    amrex::Real Sigma22L = - pL + GL*compute_devB22(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);
    amrex::Real Sigma32L = GL*compute_devB32(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);

    int cdir = 1;
    riemann(parm,
            rho1L, rho2L, rho3L, alpha1L, alpha2L, alpha3L, mrhoL, uL, ut1L, ut2L, EL, piL, cL, V21L, V22L, V23L, V11L, V12L, V13L, V31L, V32L, V33L,
            Sigma22L, Sigma12L, Sigma32L,
            rho1R, rho2R, rho3R, alpha1R, alpha2R, alpha3R, mrhoR, uR, ut1R, ut2R, ER, piR, cR, V21R, V22R, V23R, V11R, V12R, V13R, V31R, V32R, V33R,
            Sigma22R, Sigma12R, Sigma32R,
            fy(i,j,k,UARHO1), fy(i,j,k,UARHO2), fy(i,j,k,UARHO3), fy(i,j,k,UMY), fy(i,j,k,UMX), fy(i,j,k,UMZ), fy(i,j,k,URHOE), fy(i,j,k,UARP), fy(i,j,k,GALPHA1), fy(i,j,k,GALPHA2),
            fy(i,j,k,GV21), fy(i,j,k,GV22), fy(i,j,k,GV23),
            fy(i,j,k,GV11), fy(i,j,k,GV12), fy(i,j,k,GV13),
            fy(i,j,k,GV31), fy(i,j,k,GV32), fy(i,j,k,GV33),
            // Non-conservative Face Velocities given by Riemann Solver
            VSy(i,j,k), USy(i,j,k), WSy(i,j,k),
            VFy(i,j,k,0), VFy(i,j,k,3), VFy(i,j,k,6),
            VFy(i,j,k,1), VFy(i,j,k,4), VFy(i,j,k,7),
            VFy(i,j,k,2), VFy(i,j,k,5), VFy(i,j,k,8), cdir);
    VFy(i,j,k,9) = 0.0_rt; VFy(i,j,k,10) = 0.0_rt;
}

AMREX_GPU_DEVICE
inline
void
EMM_riemann_z (int i, int j, int k,
               amrex::Array4<amrex::Real> const& fz,
               amrex::Array4<amrex::Real> const& USz,
               amrex::Array4<amrex::Real> const& VSz,
               amrex::Array4<amrex::Real> const& WSz,
               amrex::Array4<amrex::Real> const& VFz,
               amrex::Array4<amrex::Real> const& qL,
               amrex::Array4<amrex::Real> const& qR,
               amrex::Array4<amrex::Real const> const& q,
               amrex::Array4<amrex::Real const> const& u,
               Parm const& parm) noexcept
{
    using namespace amrex::literals;

    // Right
    amrex::Real rho1R = qL(i,j,k,QRHO1);
    amrex::Real rho2R = qL(i,j,k,QRHO2);
    amrex::Real rho3R = qL(i,j,k,QRHO3);
    amrex::Real alpha1R = qL(i,j,k,QALPHA1);
    amrex::Real alpha2R = qL(i,j,k,QALPHA2);
    amrex::Real alpha3R = 1.0_rt - alpha1R - alpha2R;
    amrex::Real uR = qL(i,j,k,QW);
    amrex::Real ut1R = qL(i,j,k,QU);
    amrex::Real ut2R = qL(i,j,k,QV);
    amrex::Real pR = qL(i,j,k,QPRES);
    amrex::Real piR = qL(i,j,k,QPI);
    amrex::Real V11R = qL(i,j,k,QV11); amrex::Real V12R = qL(i,j,k,QV12); amrex::Real V13R = qL(i,j,k,QV13);
    amrex::Real V21R = qL(i,j,k,QV21); amrex::Real V22R = qL(i,j,k,QV22); amrex::Real V23R = qL(i,j,k,QV23);
    amrex::Real V31R = qL(i,j,k,QV31); amrex::Real V32R = qL(i,j,k,QV32); amrex::Real V33R = qL(i,j,k,QV33);

    amrex::Real mrhoR = alpha1R*rho1R + alpha2R*rho2R + alpha3R*rho3R;
    amrex::Real mrho_0R = alpha1R*parm.eos_rho01+alpha2R*parm.eos_rho02+alpha3R*parm.eos_rho03;
    amrex::Real i1R = compute_i1(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);

    amrex::Real G1R = parm.eos_G01;
    amrex::Real G2R = parm.eos_G02;
    amrex::Real G3R = parm.eos_G03;
    amrex::Real es1R = (G1R/(1.0_rt*parm.eos_rho01))*(i1R - 3.0);
    amrex::Real es2R = (G2R/(1.0_rt*parm.eos_rho02))*(i1R - 3.0);
    amrex::Real es3R = (G3R/(1.0_rt*parm.eos_rho03))*(i1R - 3.0);
    // amrex::Real T1R = compute_temperature(alpha1R, rho1R, pR, u(i,j,k+1,GT1), 1);
    // amrex::Real T2R = compute_temperature(alpha2R, rho2R, pR, u(i,j,k+1,GT2), 2);
    // amrex::Real T3R = compute_temperature(alpha3R, rho3R, pR, u(i,j,k+1,GT3), 3);
    amrex::Real T1R = compute_temperature(alpha1R, rho1R, pR, 300.0, 1);
    amrex::Real T2R = compute_temperature(alpha2R, rho2R, pR, 300.0, 2);
    amrex::Real T3R = compute_temperature(alpha3R, rho3R, pR, 300.0, 3);
    amrex::Real eint1R = compute_internal_energy(pR, T1R, rho1R, 1) + es1R;
    amrex::Real eint2R = compute_internal_energy(pR, T2R, rho2R, 2) + es2R;
    amrex::Real eint3R = compute_internal_energy(pR, T3R, rho3R, 3) + es3R;
    // amrex::Real eint1R = compute_NASG_ie(pR, rho1R, 1) + es1R;
    // amrex::Real eint2R = compute_NASG_ie(pR, rho2R, 2);
    // amrex::Real eint3R = compute_NASG_ie(pR, rho3R, 3);

    amrex::Real Y1R = alpha1R*rho1R/mrhoR;
    amrex::Real Y2R = alpha2R*rho2R/mrhoR;
    amrex::Real Y3R = alpha3R*rho3R/mrhoR;
    amrex::Real ER = Y1R*eint1R + Y2R*eint2R + Y3R*eint3R + 0.5_rt*uR*uR + 0.5_rt*ut1R*ut1R + 0.5_rt*ut2R*ut2R;

    amrex::Real c1sqR = compute_SoS(pR, T1R, rho1R, 1) + (4.0/3.0)*parm.eos_G01/parm.eos_rho01;
    amrex::Real c2sqR = compute_SoS(pR, T2R, rho2R, 2) + (4.0/3.0)*parm.eos_G02/parm.eos_rho02;
    amrex::Real c3sqR = compute_SoS(pR, T3R, rho3R, 3) + (4.0/3.0)*parm.eos_G03/parm.eos_rho03;

    amrex::Real cR = std::sqrt(Y1R*c1sqR + Y2R*c2sqR + Y3R*c3sqR);
    amrex::Real GR = alpha1R*G1R + alpha2R*G2R + alpha3R*G3R;

    amrex::Real Sigma13R = GR*compute_devB13(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);
    amrex::Real Sigma23R = GR*compute_devB23(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);
    amrex::Real Sigma33R = - pR + GR*compute_devB33(V11R,V12R,V13R,V21R,V22R,V23R,V31R,V32R,V33R);
    
    // Left
    amrex::Real rho1L = qR(i,j,k,QRHO1);
    amrex::Real rho2L = qR(i,j,k,QRHO2);
    amrex::Real rho3L = qR(i,j,k,QRHO3);
    amrex::Real alpha1L = qR(i,j,k,QALPHA1);
    amrex::Real alpha2L = qR(i,j,k,QALPHA2);
    amrex::Real alpha3L = 1.0_rt - alpha1L - alpha2L;
    amrex::Real uL = qR(i,j,k,QW);
    amrex::Real ut1L = qR(i,j,k,QU);
    amrex::Real ut2L = qR(i,j,k,QV);
    amrex::Real pL = qR(i,j,k,QPRES);
    amrex::Real piL = qR(i,j,k,QPI);
    amrex::Real V11L = qR(i,j,k,QV11); amrex::Real V12L = qR(i,j,k,QV12); amrex::Real V13L = qR(i,j,k,QV13);
    amrex::Real V21L = qR(i,j,k,QV21); amrex::Real V22L = qR(i,j,k,QV22); amrex::Real V23L = qR(i,j,k,QV23);
    amrex::Real V31L = qR(i,j,k,QV31); amrex::Real V32L = qR(i,j,k,QV32); amrex::Real V33L = qR(i,j,k,QV33);
    
    amrex::Real mrhoL = alpha1L*rho1L + alpha2L*rho2L + alpha3L*rho3L;
    amrex::Real mrho_0L = alpha1L*parm.eos_rho01+alpha2L*parm.eos_rho02+alpha3L*parm.eos_rho03;
    amrex::Real i1L = compute_i1(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);

    amrex::Real G1L = parm.eos_G01;
    amrex::Real G2L = parm.eos_G02;
    amrex::Real G3L = parm.eos_G03;
    amrex::Real es1L = (G1L/(1.0_rt*parm.eos_rho01))*(i1L - 3.0);
    amrex::Real es2L = (G2L/(1.0_rt*parm.eos_rho02))*(i1L - 3.0);
    amrex::Real es3L = (G3L/(1.0_rt*parm.eos_rho03))*(i1L - 3.0);
    // amrex::Real T1L = compute_temperature(alpha1L, rho1L, pL, u(i,j,k,GT1), 1);
    // amrex::Real T2L = compute_temperature(alpha2L, rho2L, pL, u(i,j,k,GT2), 2);
    // amrex::Real T3L = compute_temperature(alpha3L, rho3L, pL, u(i,j,k,GT3), 3);
    amrex::Real T1L = compute_temperature(alpha1L, rho1L, pL, 300.0, 1);
    amrex::Real T2L = compute_temperature(alpha2L, rho2L, pL, 300.0, 2);
    amrex::Real T3L = compute_temperature(alpha3L, rho3L, pL, 300.0, 3);
    amrex::Real eint1L = compute_internal_energy(pL, T1L, rho1L, 1) + es1L;
    amrex::Real eint2L = compute_internal_energy(pL, T2L, rho2L, 2) + es2L;
    amrex::Real eint3L = compute_internal_energy(pL, T3L, rho3L, 3) + es3L;
    // amrex::Real eint1L = compute_NASG_ie(pL, rho1L, 1) + es1L;
    // amrex::Real eint2L = compute_NASG_ie(pL, rho2L, 2);
    // amrex::Real eint3L = compute_NASG_ie(pL, rho3L, 3);

    amrex::Real Y1L = alpha1L*rho1L/mrhoL;
    amrex::Real Y2L = alpha2L*rho2L/mrhoL;
    amrex::Real Y3L = alpha3L*rho3L/mrhoL;
    amrex::Real EL = Y1L*eint1L + Y2L*eint2L + Y3L*eint3L + 0.5_rt*uL*uL + 0.5_rt*ut1L*ut1L + 0.5_rt*ut2L*ut2L;
    amrex::Real c1sqL = compute_SoS(pL, T1L, rho1L, 1) + (4.0/3.0)*parm.eos_G01/parm.eos_rho01;
    amrex::Real c2sqL = compute_SoS(pL, T2L, rho2L, 2) + (4.0/3.0)*parm.eos_G02/parm.eos_rho02;
    amrex::Real c3sqL = compute_SoS(pL, T3L, rho3L, 3) + (4.0/3.0)*parm.eos_G03/parm.eos_rho03;

    amrex::Real cL = std::sqrt(Y1L*c1sqL + Y2L*c2sqL + Y3L*c3sqL);
    amrex::Real GL = alpha1L*G1L + alpha2L*G2L + alpha3L*G3L;

    amrex::Real Sigma13L = GL*compute_devB13(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);
    amrex::Real Sigma23L = GL*compute_devB23(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);
    amrex::Real Sigma33L = - pL + GL*compute_devB33(V11L,V12L,V13L,V21L,V22L,V23L,V31L,V32L,V33L);

    int cdir = 2;
    riemann(parm,
            rho1L, rho2L, rho3L, alpha1L, alpha2L, alpha3L, mrhoL, uL, ut1L, ut2L, EL, piL, cL, V31L, V32L, V33L, V11L, V12L, V13L, V21L, V22L, V23L,
            Sigma33L, Sigma13L, Sigma23L,
            rho1R, rho2R, rho3R, alpha1R, alpha2R, alpha3R, mrhoR, uR, ut1R, ut2R, ER, piR, cR, V31R, V32R, V33R, V11R, V12R, V13R, V21R, V22R, V23R,
            Sigma33R, Sigma13R, Sigma23R,
            fz(i,j,k,UARHO1), fz(i,j,k,UARHO2), fz(i,j,k,UARHO3), fz(i,j,k,UMZ), fz(i,j,k,UMX), fz(i,j,k,UMY), fz(i,j,k,URHOE), fz(i,j,k,UARP), fz(i,j,k,GALPHA1), fz(i,j,k,GALPHA2),
            fz(i,j,k,GV31), fz(i,j,k,GV32), fz(i,j,k,GV33),
            fz(i,j,k,GV11), fz(i,j,k,GV12), fz(i,j,k,GV13),
            fz(i,j,k,GV21), fz(i,j,k,GV22), fz(i,j,k,GV23),
            // Non-conservative Face Velocities given by Riemann Solver
            WSz(i,j,k), USz(i,j,k), VSz(i,j,k),
            VFz(i,j,k,0), VFz(i,j,k,3), VFz(i,j,k,6),
            VFz(i,j,k,1), VFz(i,j,k,4), VFz(i,j,k,7),
            VFz(i,j,k,2), VFz(i,j,k,5), VFz(i,j,k,8), cdir);
            // VFz(i,j,k,2), VFz(i,j,k,5), VFz(i,j,k,8),
            // VFz(i,j,k,0), VFz(i,j,k,3), VFz(i,j,k,6),
            // VFz(i,j,k,1), VFz(i,j,k,4), VFz(i,j,k,7), cdir);
    VFz(i,j,k,9) = 0.0_rt; VFz(i,j,k,10) = 0.0_rt;
}

// AMREX_GPU_DEVICE
// inline
// void
// EMM_riemann_z (int i, int j, int k,
//                amrex::Array4<amrex::Real> const& fz,
//                amrex::Array4<amrex::Real> const& USz,
//                amrex::Array4<amrex::Real> const& VSz,
//                amrex::Array4<amrex::Real> const& WSz,
//                amrex::Array4<amrex::Real> const& qL,
//                amrex::Array4<amrex::Real> const& qR,
//                amrex::Array4<amrex::Real const> const& q,
//                Parm const& parm) noexcept
// {
//     using namespace amrex::literals;

//     // Left
//     amrex::Real rho1R = qL(i,j,k,QRHO1);
//     amrex::Real rho2R = qL(i,j,k,QRHO2);
//     amrex::Real alpha1R = qL(i,j,k,QALPHA);
//     amrex::Real alpha2R = 1.0_rt - qL(i,j,k,QALPHA);
//     amrex::Real uR = qL(i,j,k,QU);
//     amrex::Real ut1R = qL(i,j,k,QV);
//     amrex::Real ut2R = qL(i,j,k,QW);
//     amrex::Real pR = qL(i,j,k,QPRES);

//     amrex::Real mrhoR = alpha1R*rho1R + alpha2R*rho2R;
//     amrex::Real eint1R = (pR+parm.eos_gamma1*parm.eos_pinf1)/(rho1R*(parm.eos_gamma1-1.0_rt));
//     amrex::Real eint2R = (pR+parm.eos_gamma2*parm.eos_pinf2)/(rho2R*(parm.eos_gamma2-1.0_rt));
//     amrex::Real Y1R = alpha1R*rho1R/mrhoR;
//     amrex::Real Y2R = alpha2R*rho2R/mrhoR;
//     amrex::Real ER = Y1R*eint1R + Y2R*eint2R + 0.5_rt*uR*uR + 0.5_rt*ut1R*ut1R + 0.5_rt*ut2R*ut2R;
//     //amrex::Real c1sqR = parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R;
//     //amrex::Real c2sqR = parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R;
//     //amrex::Real rhocsqR = 1.0_rt/(alpha1R/(rho1R*c1sqR)+alpha2R/(rho2R*c2sqR));
//     //amrex::Real cR = std::sqrt(rhocsqR/mrhoR);
//     amrex::Real cR = std::sqrt(Y1R*parm.eos_gamma1*(pR+parm.eos_pinf1)/rho1R
//                     + Y2R*parm.eos_gamma2*(pR+parm.eos_pinf2)/rho2R);

//     // Right
//     amrex::Real rho1L = qR(i,j,k,QRHO1);
//     amrex::Real rho2L = qR(i,j,k,QRHO2);
//     amrex::Real alpha1L = qR(i,j,k,QALPHA);
//     amrex::Real alpha2L = 1.0_rt - qR(i,j,k,QALPHA);
//     amrex::Real uL = qR(i,j,k,QU);
//     amrex::Real ut1L = qR(i,j,k,QV);
//     amrex::Real ut2L = qR(i,j,k,QW);
//     amrex::Real pL = qR(i,j,k,QPRES);

//     amrex::Real mrhoL = alpha1L*rho1L + alpha2L*rho2L;
//     amrex::Real eint1L = (pL+parm.eos_gamma1*parm.eos_pinf1)/(rho1L*(parm.eos_gamma1-1.0_rt));
//     amrex::Real eint2L = (pL+parm.eos_gamma2*parm.eos_pinf2)/(rho2L*(parm.eos_gamma2-1.0_rt));
//     amrex::Real Y1L = alpha1L*rho1L/mrhoL;
//     amrex::Real Y2L = alpha2L*rho2L/mrhoL;
//     amrex::Real EL = Y1L*eint1L + Y2L*eint2L + 0.5_rt*uL*uL + 0.5_rt*ut1L*ut1L + 0.5_rt*ut2L*ut2L;
//     //amrex::Real c1sqL = parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L;
//     //amrex::Real c2sqL = parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L;
//     //amrex::Real rhocsqL = 1.0_rt/(alpha1L/(rho1L*c1sqL)+alpha2L/(rho2L*c2sqL));
//     //amrex::Real cL = std::sqrt(rhocsqL/mrhoL);
//     amrex::Real cL = std::sqrt(Y1L*parm.eos_gamma1*(pL+parm.eos_pinf1)/rho1L
//                     + Y2L*parm.eos_gamma2*(pL+parm.eos_pinf2)/rho2L);

//     int cdir = 1;
//     riemann(parm.eos_gamma1, parm.eos_pinf1, parm.eos_gamma2, parm.eos_pinf2,
//             rho1L, rho2L, alpha1L, alpha2L, mrhoL, uL, ut1L, ut2L, pL, eint1L, eint2L, Y1L, Y2L, EL, cL,
//             rho1R, rho2R, alpha1R, alpha2R, mrhoR, uR, ut1R, ut2R, pR, eint1R, eint2R, Y1R, Y2R, ER, cR,
//             fz(i,j,k,UARHO1), fz(i,j,k,UARHO2), fz(i,j,k,UMZ), fz(i,j,k,UMX), fz(i,j,k,UMY), fz(i,j,k,URHOE),
//             fz(i,j,k,GALPHA),
//             //Non-conservative Face Velocities given by Riemann Solver
//             WSz(i,j,k), USz(i,j,k), VSz(i,j,k), cdir);
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_THINC_reconstruct_x (int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qL,
//              amrex::Array4<amrex::Real> const& qR,
//              amrex::Array4<amrex::Real const> const& q,  const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;

//     for(int comp=0; comp<nprim; comp++){
//         //Coeficients
//         amrex::Real qmin = amrex::min(q(i-1,j,k,comp),q(i+1,j,k,comp));
//         amrex::Real qmax = amrex::max(q(i-1,j,k,comp),q(i+1,j,k,comp))-qmin;
//         amrex::Real theta = 0.0_rt;
//         if(q(i+1,j,k,comp)-q(i-1,j,k,comp) < 0.0_rt){
//             theta = -1.0_rt;
//         }else if (q(i+1,j,k,comp)-q(i-1,j,k,comp) > 0.0_rt){
//             theta = 1.0_rt;
//         }

//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
//         amrex::Real B = exp(Beta*theta*(2.0_rt*C-1.0_rt));
//         amrex::Real A = (B/cosh(Beta)-1.0_rt)/tanh(Beta);

//         //Print() << "A--- : " << A << "\n";
//         //Print() << "B--- : " << B << "\n";
//         //Print() << "C--- : " << C << "\n";
//         //Print() << "1.0_rt+A*tanh(Beta) : " <<  1.0_rt+A*tanh(Beta) << "\n";
//         // qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//         // qR(i+1,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//         qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//         qR(i+1,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_THINC_reconstruct_y (int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qL,
//              amrex::Array4<amrex::Real> const& qR,
//              amrex::Array4<amrex::Real const> const& q,  const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;

//     for(int comp=0; comp<nprim; comp++){
//         //Coeficients
//         amrex::Real qmin = amrex::min(q(i,j-1,k,comp),q(i,j+1,k,comp));
//         amrex::Real qmax = amrex::max(q(i,j-1,k,comp),q(i,j+1,k,comp))-qmin;
//         amrex::Real theta;
//         if(q(i,j+1,k,comp)-q(i,j-1,k,comp) < 0.0_rt){
//             theta =-1.0_rt;
//         }else if (q(i,j+1,k,comp)-q(i,j-1,k,comp) > 0.0_rt){
//             theta = 1.0_rt;
//         }else{
//             theta = 0.0_rt;
//         }
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
//         amrex::Real B = exp(Beta*theta*(2.0_rt*C-1.0_rt));
//         amrex::Real A = (B/cosh(Beta)-1.0_rt)/tanh(Beta);

//         qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//         qR(i,j+1,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_THINC_reconstruct_z (int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qL,
//              amrex::Array4<amrex::Real> const& qR,
//              amrex::Array4<amrex::Real const> const& q,  const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;

//     for(int comp=0; comp<nprim; comp++){
//         //Coeficients
//         amrex::Real qmin = amrex::min(q(i,j,k-1,comp),q(i,j,k+1,comp));
//         amrex::Real qmax = amrex::max(q(i,j,k-1,comp),q(i,j,k+1,comp))-qmin;
//         amrex::Real theta;
//         if(q(i,j,k+1,comp)-q(i,j,k-1,comp) < 0.0_rt){
//             theta =-1.0_rt;
//         }else if (q(i,j,k+1,comp)-q(i,j,k-1,comp) > 0.0_rt){
//             theta = 1.0_rt;
//         }else{
//             theta = 0.0_rt;
//         }
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
//         amrex::Real B = exp(Beta*theta*(2.0_rt*C-1.0_rt));
//         amrex::Real A = (B/cosh(Beta)-1.0_rt)/tanh(Beta);

//         qL(i,j,k,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*A);
//         qR(i,j,k+1,comp) = qmin + 0.5_rt*qmax*(1.0_rt+theta*(tanh(Beta)+A)/(1.0_rt+A*tanh(Beta)));
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_TBV_x(int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qR_M, 
//              amrex::Array4<amrex::Real> const& qL_M, 
//              amrex::Array4<amrex::Real> const& qL_T,
//              amrex::Array4<amrex::Real> const& qR_T,
//              amrex::Array4<amrex::Real> const& q, const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;
//     amrex::Real TBV_T;
//     amrex::Real TBV_M;

//     for(int comp=0; comp<nprim; comp++){
//         amrex::Real qmin = amrex::min(q(i-1,j,k,comp),q(i+1,j,k,comp));
//         amrex::Real qmax = amrex::max(q(i-1,j,k,comp),q(i+1,j,k,comp))-qmin;
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);

//         //Compute total boundary variations TBV for each cell
//         TBV_M =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_M(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_T(i+1,j,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_T(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i+1,j,k,comp)-qL_M(i+1,j,k,comp))));

//         TBV_T =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_M(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_T(i+1,j,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_T(i+1,j,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i+1,j,k,comp)-qL_M(i+1,j,k,comp))));

//         if(delta < C < 1.0_rt-delta && (q(i+1,j,k,comp)-q(i,j,k,comp))*(q(i,j,k,comp)-q(i-1,j,k,comp))>0.0_rt 
//                 && TBV_T < TBV_M){
//             qL_M(i,j,k,comp) = qL_T(i,j,k,comp);
//             qR_M(i+1,j,k,comp) = qR_T(i+1,j,k,comp);
//         }
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_TBV_y(int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qR_M, 
//              amrex::Array4<amrex::Real> const& qL_M, 
//              amrex::Array4<amrex::Real> const& qL_T,
//              amrex::Array4<amrex::Real> const& qR_T,
//              amrex::Array4<amrex::Real> const& q, const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;
//     amrex::Real TBV_T;
//     amrex::Real TBV_M;
//     for(int comp=0; comp<nprim; comp++){
//         amrex::Real qmin = amrex::min(q(i,j-1,k,comp),q(i,j+1,k,comp));
//         amrex::Real qmax = amrex::max(q(i,j-1,k,comp),q(i,j+1,k,comp))-qmin;
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
        
//         //Compute total boundary variations TBV for each cell
//         TBV_M =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_M(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_T(i,j+1,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_T(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j+1,k,comp)-qL_M(i,j+1,k,comp))));
        
//         TBV_T =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_M(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_T(i,j+1,k,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_T(i,j+1,k,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j+1,k,comp)-qL_M(i,j+1,k,comp))));
//         if(delta < C < 1.0_rt-delta && (q(i,j+1,k,comp)-q(i,j,k,comp))*(q(i,j,k,comp)-q(i,j-1,k,comp))>0.0_rt 
//                 && TBV_T < TBV_M){
//             qL_M(i,j,k,comp) = qL_T(i,j,k,comp);
//             qR_M(i,j+1,k,comp) = qR_T(i,j+1,k,comp);
//         }
//     }
// }

// AMREX_GPU_DEVICE
// inline
// void
// EMM_TBV_z(int i, int j, int k,
//              amrex::Array4<amrex::Real> const& qR_M, 
//              amrex::Array4<amrex::Real> const& qL_M, 
//              amrex::Array4<amrex::Real> const& qL_T,
//              amrex::Array4<amrex::Real> const& qR_T,
//              amrex::Array4<amrex::Real> const& q, const int nprim) noexcept
// {
//     amrex::Real Beta=1.6_rt;
//     amrex::Real epsilon=1E-8_rt;
//     amrex::Real delta=1E-4_rt;
//     amrex::Real TBV_T;
//     amrex::Real TBV_M;
//     for(int comp=0; comp<nprim; comp++){
//         amrex::Real qmin = amrex::min(q(i,j,k-1,comp),q(i,j,k+1,comp));
//         amrex::Real qmax = amrex::max(q(i,j,k-1,comp),q(i,j,k+1,comp))-qmin;
//         amrex::Real C = (q(i,j,k,comp)-qmin+epsilon)/(qmax+epsilon);
        
//         //Compute total boundary variations TBV for each cell
//         TBV_M =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_M(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_T(i,j,k+1,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_T(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_M(i,j,k,comp)) + std::abs(qR_M(i,j,k+1,comp)-qL_M(i,j,k+1,comp))));
        
//         TBV_T =amrex::min(amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_M(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_T(i,j,k+1,comp))),
//                           amrex::min(std::abs(qR_M(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_T(i,j,k+1,comp)),
//                                      std::abs(qR_T(i,j,k,comp)-qL_T(i,j,k,comp)) + std::abs(qR_T(i,j,k+1,comp)-qL_M(i,j,k+1,comp))));
//         if(delta < C < 1.0_rt-delta && (q(i,j,k+1,comp)-q(i,j,k,comp))*(q(i,j,k,comp)-q(i,j,k-1,comp))>0.0_rt 
//                 && TBV_T < TBV_M){
//             qL_M(i,j,k,comp) = qL_T(i,j,k,comp);
//             qR_M(i,j,k+1,comp) = qR_T(i,j,k+1,comp);
//         }
//     }
// }
#endif
