#ifndef EMM_PARM_H_
#define EMM_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct Parm
    : public amrex::Gpu::Managed
{
    // Godunov Romenski EOS:
    // amrex::Real eos_Gama1 = 1.0;
    // amrex::Real eos_Gama2 = 1.0;
    // amrex::Real eos_Gama3 = 1.0;
    // amrex::Real eos_K01 = 0.0;
    // amrex::Real eos_K02 = 0.0;
    amrex::Real eos_rho01 = 1.0;
    amrex::Real eos_rho02 = 1.0;
    amrex::Real eos_rho03 = 1.0;
    amrex::Real eos_G01 = 0.0;
    amrex::Real eos_G02 = 0.0;
    amrex::Real eos_G03 = 0.0;
    // amrex::Real eos_alpha1 = 1.0;
    // amrex::Real eos_alpha2 = 1.0;
    // amrex::Real eos_Beta1 = 1.0;
    // amrex::Real eos_Beta2 = 1.0;
    // amrex::Real eos_Cv1 = 9E2;
    // amrex::Real eos_Cv2 = 3.9E2;
    // amrex::Real eos_cv1 = 9E2;
    // amrex::Real eos_cv2 = 3.9E2;
    // amrex::Real eos_T01 = 298.0;
    // amrex::Real eos_T02 = 298.0;
    
    //Plasticity parameters
    // amrex::Real eos_c1 = 0.0;
    // amrex::Real eos_c2 = 0.0;
    // amrex::Real eos_c3 = 0.0;
    // amrex::Real eos_n = 0.0;
    // amrex::Real eos_X0 = 0.0;

    // NASG:
    amrex::Real eos_gamma1 = 0.0;
    amrex::Real eos_gamma2 = 0.0;
    amrex::Real eos_gamma3 = 0.0;
    amrex::Real eos_pinf1 = 0.0;
    amrex::Real eos_pinf2 = 0.0;
    amrex::Real eos_pinf3 = 0.0;
    amrex::Real eos_b1 = 0.0;
    amrex::Real eos_b2 = 0.0;
    amrex::Real eos_b3 = 0.0;
    amrex::Real eos_q1 = 0.0;
    amrex::Real eos_q2 = 0.0;
    amrex::Real eos_q3 = 0.0;
    amrex::Real eos_cv1 = 0.0;
    amrex::Real eos_cv2 = 0.0;
    amrex::Real eos_cv3 = 0.0;
    
    // Minimum Volume Fraction
    amrex::Real alpha_min = 1.0e-6_rt;
    int K_activation = 1;
    amrex::Real K_time = 0.0;
    
    amrex::Real smallr = 1.e-19_rt;
    amrex::Real smallp = 1.e-10_rt;

    // Complex Thermodynamics Phase 1
    amrex::Real TMINT1 = 0.0;
    amrex::Real PMINT1 = 0.0;
    amrex::Real TMAXT1 = 0.0;
    amrex::Real PMAXT1 = 0.0;
    amrex::Real DT1 = 0.0;
    amrex::Real DP1 = 0.0;

    int mmTP1 = 121;
    int nnTP1 = 376;
    int tableRows1 = 45496;
    int tableColumns1 = 10;
    int tabulated = 0;
    int tabulated1 = 0;
    int tabulated2 = 0;
    int tabulated3 = 0;

    int coord_type = 0;
    void Initialize ();
};

#endif
