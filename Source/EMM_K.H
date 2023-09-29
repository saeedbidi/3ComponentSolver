#ifndef EMM_K_H_
#define EMM_K_H_

#include "EMM_index_macros.H"
#include <AMReX_FArrayBox.H>
#include <limits>
#include <cmath>

#include "emm_prob.H"
#include "EMM_parm.H"
#include "EMM_eos.H"

AMREX_GPU_HOST_DEVICE
inline
amrex::Real
EMM_estdt (amrex::Box const& bx, amrex::Array4<Real const> const& state,
           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
           Parm const& parm) noexcept
{
    using namespace amrex::literals;
    // Print() << "CFL---------" << "\n";
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || (__CUDACC_VER_MINOR__ != 2)
    amrex::Real dt = std::numeric_limits<amrex::Real>::max();
#else
    amrex::Real dt = 1.e37_rt;
#endif

    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                amrex::Real alpha1 = state(i,j,k,GALPHA1);
                amrex::Real alpha2 = state(i,j,k,GALPHA2);
                amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
                int ll = 0;
                if(alpha1 > 1.0-2.0*parm.alpha_min){alpha1 = 1.0-2.0*parm.alpha_min; ll = 1;}
                if(alpha2 > 1.0-2.0*parm.alpha_min){alpha2 = 1.0-2.0*parm.alpha_min; ll = 1;}
                if(alpha1 < parm.alpha_min){alpha1 = parm.alpha_min; ll = 1;}
                if(alpha2 < parm.alpha_min){alpha2 = parm.alpha_min; ll = 1;}
                if(ll == 1){
                    alpha3 = 1.0_rt - alpha1 - alpha2;
                    alpha1 = alpha1/(alpha1+alpha2+alpha3);
                    alpha2 = alpha2/(alpha1+alpha2+alpha3);
                }
                amrex::Real arho1 = state(i,j,k,UARHO1);
                amrex::Real arho2 = state(i,j,k,UARHO2);
                amrex::Real arho3 = state(i,j,k,UARHO3);
                amrex::Real mrho = arho1 + arho2 + arho3;
                amrex::Real rho1 = state(i,j,k,UARHO1)/alpha1;
                amrex::Real rho2 = state(i,j,k,UARHO2)/alpha2;
                amrex::Real rho3 = state(i,j,k,UARHO3)/alpha3;
                amrex::Real mx  = state(i,j,k,UMX);
                amrex::Real my  = state(i,j,k,UMY);
                amrex::Real mz  = state(i,j,k,UMZ);
                amrex::Real mrhoinv = 1.0_rt/mrho;
                amrex::Real rhoinv1 = 1.0_rt/rho1;
                amrex::Real rhoinv2 = 1.0_rt/rho2;
                amrex::Real rhoinv3 = 1.0_rt/rho3;
                amrex::Real vx = mx*mrhoinv;
                amrex::Real vy = my*mrhoinv;
                amrex::Real vz = mz*mrhoinv;
                amrex::Real ei = state(i,j,k,URHOE)/mrho - 0.5_rt*(vx*vx + vy*vy + vz*vz);
                amrex::Real V11 = state(i,j,k,GV11); amrex::Real V12 = state(i,j,k,GV12); amrex::Real V13 = state(i,j,k,GV13);
                amrex::Real V21 = state(i,j,k,GV21); amrex::Real V22 = state(i,j,k,GV22); amrex::Real V23 = state(i,j,k,GV23);
                amrex::Real V31 = state(i,j,k,GV31); amrex::Real V32 = state(i,j,k,GV32); amrex::Real V33 = state(i,j,k,GV33);
                amrex::Real i1 = compute_i1(V11,V12,V13,V21,V22,V23,V31,V32,V33);
                amrex::Real es1 = (parm.eos_G01/(2.0*parm.eos_rho01))*(i1 - 3.0);
                amrex::Real es2 = (parm.eos_G02/(2.0*parm.eos_rho02))*(i1 - 3.0);
                amrex::Real es3 = (parm.eos_G03/(2.0*parm.eos_rho03))*(i1 - 3.0);

                amrex::Real p = compute_pressure(mrho, ei, es1, es2, es3, alpha1, alpha2, alpha3, rho1, rho2, rho3, state(i,j,k,GPRESS), state(i,j,k,GT1), state(i,j,k,GT2), state(i,j,k,GT3));
                // amrex::Real p = compute_pressure(mrho, ei, alpha1, alpha2, alpha3, rho1, rho2, rho3, 1E5, 300.0_rt, 300.0_rt, 300.0_rt);
                // Compute Speed of sound
                amrex::Real T1 = compute_temperature(alpha1, rho1, p, state(i,j,k,GT1), 1);
                amrex::Real T2 = compute_temperature(alpha2, rho2, p, state(i,j,k,GT2), 2);
                amrex::Real T3 = compute_temperature(alpha3, rho3, p, state(i,j,k,GT3), 3);
                // amrex::Real T1 = compute_temperature(alpha1, rho1, p, 300.0, 1);
                // amrex::Real T2 = compute_temperature(alpha2, rho2, p, 300.0, 2);
                // amrex::Real T3 = compute_temperature(alpha3, rho3, p, 300.0, 3);
                amrex::Real c1sq = amrex::max(compute_SoS(p, T1, rho1, 1)+ (4.0/3.0)*parm.eos_G01/parm.eos_rho01, parm.smallr);
                amrex::Real c2sq = amrex::max(compute_SoS(p, T2, rho2, 2)+ (4.0/3.0)*parm.eos_G02/parm.eos_rho02, parm.smallr);
                amrex::Real c3sq = amrex::max(compute_SoS(p, T3, rho3, 3)+ (4.0/3.0)*parm.eos_G03/parm.eos_rho03, parm.smallr);

                amrex::Real Y1 = amrex::max(alpha1*rho1/mrho,parm.smallr);
                amrex::Real Y2 = amrex::max(alpha2*rho2/mrho,parm.smallr);
                amrex::Real Y3 = amrex::max(alpha3*rho3/mrho,parm.smallr);

                amrex::Real mcs = std::sqrt(Y1*c1sq + Y2*c2sq + Y3*c3sq);
                amrex::Real dtx = dx[0]/(std::abs(vx)+mcs);
                amrex::Real dty = dx[1]/(std::abs(vy)+mcs);
#if (AMREX_SPACEDIM == 2)
                dt = amrex::min(dt,amrex::min(dtx,dty));
                // Print() <<  "dt " << dt << "\n";
#endif
#if (AMREX_SPACEDIM == 3)
                amrex::Real dtz = dx[2]/(std::abs(vz)+mcs);
                dt = amrex::min(dt,amrex::min(dtx,amrex::min(dty,dtz)));
#endif
            }
        }
    }
    // Print() << "------------" << "\n";
    return dt;
}

#endif
