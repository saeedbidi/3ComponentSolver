#include "EMM_derive.H"
#include "EMM.H"
#include "EMM_parm.H"
#include "EMM_eos.H"

using namespace amrex;

void EMM_derpres (const Box& bx, FArrayBox& pfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       p    = pfab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,7); amrex::Real alpha2 = dat(i,j,k,8); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real mrho = dat(i,j,k,1) + dat(i,j,k,2) + dat(i,j,k,3);
        amrex::Real rho1 = dat(i,j,k,1)/alpha1;
        amrex::Real rho2 = dat(i,j,k,2)/alpha2;
        amrex::Real rho3 = dat(i,j,k,3)/alpha3;
        amrex::Real vx = dat(i,j,k,4)/mrho;
        amrex::Real vy = dat(i,j,k,5)/mrho;
        amrex::Real vz = dat(i,j,k,6)/mrho;
        amrex::Real ei = dat(i,j,k,0)/mrho - 0.5_rt*(vx*vx + vy*vy + vz*vz);

        amrex::Real V11 = dat(i,j,k,9);
        amrex::Real V21 = dat(i,j,k,10);
        amrex::Real V31 = dat(i,j,k,11);
        amrex::Real V12 = dat(i,j,k,12);
        amrex::Real V22 = dat(i,j,k,13);
        amrex::Real V32 = dat(i,j,k,14);
        amrex::Real V13 = dat(i,j,k,15);
        amrex::Real V23 = dat(i,j,k,16);
        amrex::Real V33 = dat(i,j,k,17);

        amrex::Real i1 = compute_i1(V11,V12,V13,V21,V22,V23,V31,V32,V33);

        amrex::Real es1 = (parm->eos_G01/(2.0*parm->eos_rho01))*(i1 - 3.0);
        amrex::Real es2 = (parm->eos_G02/(2.0*parm->eos_rho02))*(i1 - 3.0);
        amrex::Real es3 = (parm->eos_G03/(2.0*parm->eos_rho03))*(i1 - 3.0);

        // amrex::Real peos = ( mrho*ei - alpha1*rho1*es1 - ( alpha1*parm->eos_gamma1*parm->eos_pinf1/(parm->eos_gamma1-1.0_rt) 
        //     + alpha2*parm->eos_gamma2*parm->eos_pinf2/(parm->eos_gamma2-1.0_rt) + alpha3*parm->eos_gamma3*parm->eos_pinf3/(parm->eos_gamma3-1.0_rt)) )/( alpha1/(parm->eos_gamma1-1.0_rt)+alpha2/(parm->eos_gamma2-1.0_rt)+alpha3/(parm->eos_gamma3-1.0_rt));
        // amrex::Real peos = compute_pressure(mrho, ei, alpha1, alpha2, alpha3, rho1, rho2, rho3, dat(i,j,k,18), dat(i,j,k,19), dat(i,j,k,20), dat(i,j,k,21));
        amrex::Real peos = compute_pressure(mrho, ei, es1, es2, es3, alpha1, alpha2, alpha3, rho1, rho2, rho3, dat(i,j,k,18), dat(i,j,k,19), dat(i,j,k,20), dat(i,j,k,21));
        p(i,j,k,dcomp) = peos;
    });
}

// void EMM_derpc (const Box& bx, FArrayBox& pcfab, int dcomp, int /*ncomp*/,
//                   const FArrayBox& datfab, const Geometry& /*geomdata*/,
//                   Real /*time*/, const int* /*bcrec*/, int /*level*/)
// {
//     auto const dat = datfab.array();
//     auto       pc    = pcfab.array();
//     Parm const* parm = EMM::parm.get();
//     amrex::ParallelFor(bx,
//     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//     {
//         amrex::Real alpha1 = dat(i,j,k,0); amrex::Real alpha2 = 1.0_rt - dat(i,j,k,0);
//         amrex::Real mrho = dat(i,j,k,1) + dat(i,j,k,2);
//         amrex::Real rho1 = dat(i,j,k,1)/alpha1;
//         amrex::Real rho2 = dat(i,j,k,2)/alpha2;

//         amrex::Real pc1 = (parm->eos_K01/parm->eos_alpha1)*std::pow(rho1/parm->eos_rho01,parm->eos_alpha1+1.0_rt)*(std::pow(rho1/parm->eos_rho01,parm->eos_alpha1)-1.0_rt);
//         pc(i,j,k,dcomp) = alpha1*pc1;
//     });
// }

// void EMM_derps (const Box& bx, FArrayBox& psfab, int dcomp, int /*ncomp*/,
//                   const FArrayBox& datfab, const Geometry& /*geomdata*/,
//                   Real /*time*/, const int* /*bcrec*/, int /*level*/)
// {
//     auto const dat = datfab.array();
//     auto       ps    = psfab.array();
//     Parm const* parm = EMM::parm.get();
//     amrex::ParallelFor(bx,
//     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//     {
//         amrex::Real alpha1 = dat(i,j,k,0); amrex::Real alpha2 = 1.0_rt - dat(i,j,k,0);
//         amrex::Real mrho = dat(i,j,k,1) + dat(i,j,k,2);
//         amrex::Real rho1 = dat(i,j,k,1)/alpha1;
//         amrex::Real rho2 = dat(i,j,k,2)/alpha2;
//         amrex::Real V11 = dat(i,j,k,3);
//         amrex::Real V21 = dat(i,j,k,4);
//         amrex::Real V31 = dat(i,j,k,5);
//         amrex::Real V12 = dat(i,j,k,6);
//         amrex::Real V22 = dat(i,j,k,7);
//         amrex::Real V32 = dat(i,j,k,8);
//         amrex::Real V13 = dat(i,j,k,9);
//         amrex::Real V23 = dat(i,j,k,10);
//         amrex::Real V33 = dat(i,j,k,11);
//         amrex::Real I2 = compute_I2(V11,V12,V13,V21,V22,V23,V31,V32,V33);

//         amrex::Real ps1 = parm->eos_Beta1*parm->eos_G01*I2*std::pow(rho1/parm->eos_rho01,1.0_rt + parm->eos_Beta1);
//         ps(i,j,k,dcomp) = alpha1*ps1;
//     });
// }

// void EMM_derpt (const Box& bx, FArrayBox& ptfab, int dcomp, int /*ncomp*/,
//                   const FArrayBox& datfab, const Geometry& /*geomdata*/,
//                   Real /*time*/, const int* /*bcrec*/, int /*level*/)
// {
//     auto const dat = datfab.array();
//     auto       pt    = ptfab.array();
//     Parm const* parm = EMM::parm.get();
//     amrex::ParallelFor(bx,
//     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//     {
//         amrex::Real alpha1 = dat(i,j,k,6); amrex::Real alpha2 = 1.0_rt - dat(i,j,k,6);
//         amrex::Real mrho = dat(i,j,k,1) + dat(i,j,k,2);
//         amrex::Real rho1 = dat(i,j,k,1)/alpha1;
//         amrex::Real rho2 = dat(i,j,k,2)/alpha2;
//         amrex::Real vx = dat(i,j,k,3)/mrho;
//         amrex::Real vy = dat(i,j,k,4)/mrho;
//         amrex::Real vz = dat(i,j,k,5)/mrho;
//         amrex::Real ei = dat(i,j,k,0)/mrho - 0.5_rt*(vx*vx + vy*vy + vz*vz);

//         amrex::Real V11 = dat(i,j,k,7);
//         amrex::Real V21 = dat(i,j,k,8);
//         amrex::Real V31 = dat(i,j,k,9);
//         amrex::Real V12 = dat(i,j,k,10);
//         amrex::Real V22 = dat(i,j,k,11);
//         amrex::Real V32 = dat(i,j,k,12);
//         amrex::Real V13 = dat(i,j,k,13);
//         amrex::Real V23 = dat(i,j,k,14);
//         amrex::Real V33 = dat(i,j,k,15);

//         amrex::Real I2 = compute_I2(V11,V12,V13,V21,V22,V23,V31,V32,V33);

//         amrex::Real ec1 = (parm->eos_K01/(2.0*parm->eos_rho01*parm->eos_alpha1*parm->eos_alpha1))*std::pow(std::pow(rho1/parm->eos_rho01,parm->eos_alpha1)-1.0_rt,2.0);
//         amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
//         amrex::Real es1 = (G1/rho1)*I2;

//         // GR & SG EOS
//         amrex::Real pteos = rho1*parm->eos_Gama1*(ei - ec1 - es1);

//         pt(i,j,k,dcomp) = alpha1*pteos;
//     });
// }

void EMM_derelS11 (const Box& bx, FArrayBox& elS11fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       elS11    = elS11fab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        // amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
        // amrex::Real G2 = parm->eos_G02*std::pow(rho2/parm->eos_rho02,parm->eos_Beta2+1.0_rt);
        amrex::Real G1 = parm->eos_G01;
        amrex::Real G2 = parm->eos_G02;
        amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
        // elS11(i,j,k,dcomp) = 2.0*G*compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // elS11(i,j,k,dcomp) = G*(mrho/mrho_0)*compute_devB11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        elS11(i,j,k,dcomp) = G*compute_devB11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    });
}

void EMM_derelS22 (const Box& bx, FArrayBox& elS22fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       elS22    = elS22fab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        // amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
        // amrex::Real G2 = parm->eos_G02*std::pow(rho2/parm->eos_rho02,parm->eos_Beta2+1.0_rt);
        amrex::Real G1 = parm->eos_G01;
        amrex::Real G2 = parm->eos_G02;
        amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
        // elS22(i,j,k,dcomp) = 2.0*G*compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // elS22(i,j,k,dcomp) = G*(mrho/mrho_0)*compute_devB22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        elS22(i,j,k,dcomp) = G*compute_devB22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    });
}

void EMM_derelS33 (const Box& bx, FArrayBox& elS33fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       elS33    = elS33fab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        // amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
        // amrex::Real G2 = parm->eos_G02*std::pow(rho2/parm->eos_rho02,parm->eos_Beta2+1.0_rt);
        amrex::Real G1 = parm->eos_G01;
        amrex::Real G2 = parm->eos_G02;
        amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
        // elS33(i,j,k,dcomp) = 2.0*G*compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // elS33(i,j,k,dcomp) = G*(mrho/mrho_0)*compute_devB33(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        elS33(i,j,k,dcomp) = G*compute_devB33(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    });
}

void EMM_derelS21 (const Box& bx, FArrayBox& elS21fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       elS21    = elS21fab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        // amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
        // amrex::Real G2 = parm->eos_G02*std::pow(rho2/parm->eos_rho02,parm->eos_Beta2+1.0_rt);
        amrex::Real G1 = parm->eos_G01;
        amrex::Real G2 = parm->eos_G02;
        amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
        // elS21(i,j,k,dcomp) = 2.0*G*compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // elS21(i,j,k,dcomp) = G*(mrho/mrho_0)*compute_devB21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        elS21(i,j,k,dcomp) = G*compute_devB21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    });
}

void EMM_derelS23 (const Box& bx, FArrayBox& elS23fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       elS23    = elS23fab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        // amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
        // amrex::Real G2 = parm->eos_G02*std::pow(rho2/parm->eos_rho02,parm->eos_Beta2+1.0_rt);
        amrex::Real G1 = parm->eos_G01;
        amrex::Real G2 = parm->eos_G02;
        amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
        // elS23(i,j,k,dcomp) = 2.0*G*compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // elS23(i,j,k,dcomp) = G*(mrho/mrho_0)*compute_devB23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        elS23(i,j,k,dcomp) = G*compute_devB23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    });
}

void EMM_derelS31 (const Box& bx, FArrayBox& elS31fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       elS31    = elS31fab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        // amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
        // amrex::Real G2 = parm->eos_G02*std::pow(rho2/parm->eos_rho02,parm->eos_Beta2+1.0_rt);
        amrex::Real G1 = parm->eos_G01;
        amrex::Real G2 = parm->eos_G02;
        amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
        // elS31(i,j,k,dcomp) = 2.0*G*compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // elS31(i,j,k,dcomp) = G*(mrho/mrho_0)*compute_devB31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        elS31(i,j,k,dcomp) = G*compute_devB31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    });
}

void EMM_derMaxPrincipalStress (const Box& bx, FArrayBox& maxPrincipStressfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       maxPrincipStress    = maxPrincipStressfab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        amrex::Real G1 = parm->eos_G01; amrex::Real G2 = parm->eos_G02; amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;
        // amrex::Real elS11 = 2.0*G*compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // amrex::Real elS22 = 2.0*G*compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // amrex::Real elS21 = 2.0*G*compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);

        amrex::Real elS11 = G*compute_devB11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        amrex::Real elS22 = G*compute_devB22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        amrex::Real elS21 = G*compute_devB21(V11,V12,V13,V21,V22,V23,V31,V32,V33);

        maxPrincipStress(i,j,k,dcomp) = 0.5*(elS11+elS22) + std::sqrt(std::pow(0.5*(elS11-elS22),2.0)+elS21*elS21);
    });
}

void EMM_derMinPrincipalStress (const Box& bx, FArrayBox& minPrincipStressfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       minPrincipStress    = minPrincipStressfab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        amrex::Real G1 = parm->eos_G01; amrex::Real G2 = parm->eos_G02; amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;

        // amrex::Real elS11 = 2.0*G*compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // amrex::Real elS22 = 2.0*G*compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // amrex::Real elS21 = 2.0*G*compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);

        amrex::Real elS11 = G*compute_devB11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        amrex::Real elS22 = G*compute_devB22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        amrex::Real elS21 = G*compute_devB21(V11,V12,V13,V21,V22,V23,V31,V32,V33);

        minPrincipStress(i,j,k,dcomp) = 0.5*(elS11+elS22) - std::sqrt(std::pow(0.5*(elS11-elS22),2.0)+elS21*elS21);
    });
}

void EMM_derMaxShearStress (const Box& bx, FArrayBox& maxShearStressfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       maxShearStress    = maxShearStressfab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,3); amrex::Real alpha2 = dat(i,j,k,4); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        amrex::Real rho1 = dat(i,j,k,0)/alpha1;
        amrex::Real rho2 = dat(i,j,k,1)/alpha2;
        amrex::Real rho3 = dat(i,j,k,2)/alpha3;
        amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
        amrex::Real mrho_0 = alpha1*parm->eos_rho01+alpha2*parm->eos_rho02+alpha3*parm->eos_rho03;

        amrex::Real V11 = dat(i,j,k,5);
        amrex::Real V21 = dat(i,j,k,6);
        amrex::Real V31 = dat(i,j,k,7);
        amrex::Real V12 = dat(i,j,k,8);
        amrex::Real V22 = dat(i,j,k,9);
        amrex::Real V32 = dat(i,j,k,10);
        amrex::Real V13 = dat(i,j,k,11);
        amrex::Real V23 = dat(i,j,k,12);
        amrex::Real V33 = dat(i,j,k,13);

        amrex::Real G1 = parm->eos_G01; amrex::Real G2 = parm->eos_G02; amrex::Real G3 = parm->eos_G03;
        // amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2 + alpha3*G3/parm->eos_Gama3)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2 + alpha3/parm->eos_Gama3);
        amrex::Real G = alpha1*G1 + alpha2*G2 + alpha3*G3;

        // amrex::Real elS11 = 2.0*G*compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // amrex::Real elS22 = 2.0*G*compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        // amrex::Real elS21 = 2.0*G*compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);

        amrex::Real elS11 = G*compute_devB11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        amrex::Real elS22 = G*compute_devB22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        amrex::Real elS21 = G*compute_devB21(V11,V12,V13,V21,V22,V23,V31,V32,V33);

        maxShearStress(i,j,k,dcomp) = std::sqrt(std::pow(0.5*(elS11-elS22),2.0)+elS21*elS21);
    });
}

void EMM_derStrainEnergy (const Box& bx, FArrayBox& strainEnergyfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       strainEnergy    = strainEnergyfab.array();
    Parm const* parm = EMM::parm.get();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real V11 = dat(i,j,k,0);
        amrex::Real V21 = dat(i,j,k,1);
        amrex::Real V31 = dat(i,j,k,2);
        amrex::Real V12 = dat(i,j,k,3);
        amrex::Real V22 = dat(i,j,k,4);
        amrex::Real V32 = dat(i,j,k,5);
        amrex::Real V13 = dat(i,j,k,6);
        amrex::Real V23 = dat(i,j,k,7);
        amrex::Real V33 = dat(i,j,k,8);

        amrex::Real G1 = parm->eos_G01; amrex::Real G2 = parm->eos_G02; amrex::Real G3 = parm->eos_G03;
        amrex::Real i1 = compute_i1(V11,V12,V13,V21,V22,V23,V31,V32,V33);
        strainEnergy(i,j,k,dcomp) = (G1/(2.0*parm->eos_rho01))*(i1 - 3.0);
    });
}

void EMM_dervel (const Box& bx, FArrayBox& velfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       vel = velfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        vel(i,j,k,dcomp) = dat(i,j,k,0)/(dat(i,j,k,1)+dat(i,j,k,2)+dat(i,j,k,3));
    });
}

void EMM_derden (const Box& bx, FArrayBox& denfab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       den = denfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        den(i,j,k,dcomp) = dat(i,j,k,0) + dat(i,j,k,1) + dat(i,j,k,2);
    });
}

void EMM_deralpha3 (const Box& bx, FArrayBox& alphafab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       alpha = alphafab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        alpha(i,j,k,dcomp) = 1.0_rt - dat(i,j,k,0) - dat(i,j,k,1);
    });
}

void EMM_derT1 (const Box& bx, FArrayBox& T1fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       T1 = T1fab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        T1(i,j,k,dcomp) = compute_temperature(dat(i,j,k,0), dat(i,j,k,1)/dat(i,j,k,0), dat(i,j,k,2), dat(i,j,k,3), 1);
    });
}

void EMM_derT2 (const Box& bx, FArrayBox& T2fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       T2 = T2fab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        T2(i,j,k,dcomp) = compute_temperature(dat(i,j,k,0), dat(i,j,k,1)/dat(i,j,k,0), dat(i,j,k,2), dat(i,j,k,3), 2);
    });
}

void EMM_derT3 (const Box& bx, FArrayBox& T3fab, int dcomp, int /*ncomp*/,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    auto const dat = datfab.array();
    auto       T3 = T3fab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real alpha1 = dat(i,j,k,0);
        amrex::Real alpha2 = dat(i,j,k,1);
        amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
        T3(i,j,k,dcomp) = compute_temperature(alpha3, dat(i,j,k,2)/alpha3, dat(i,j,k,3), dat(i,j,k,4), 3);
    });
}