#ifndef EMM_PROB_H_
#define EMM_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "EMM_index_macros.H"
#include "EMM_parm.H"
#include "emm_prob_parm.H"
#include "EMM_eos.H"
#include "EMM.H"
#include "Python.h"

AMREX_GPU_DEVICE
inline
void
emm_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, Parm const& parm)//, ProbParm const& prob_parm)
{
    using namespace amrex::literals;

    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* dx      = geomdata.CellSize();

    amrex::Real x = prob_lo[0] + (i+0.5_rt)*dx[0];
    amrex::Real y = prob_lo[1] + (j+0.5_rt)*dx[1];
    amrex::Real alpha1, alpha2, alpha3, rho1_0, rho1, rho2, rho3, alpha_min, p, p0, alrhoeint1, alrhoeint2, alrhoeint3, eints, rad;
    amrex::Real V11, V12, V13, V21, V22, V23, V31, V32, V33, Ls, a, b, c, v;
    amrex::Real eint1, eint2, eint3, T1, T2, T3;

    alpha_min = parm.alpha_min;
    rad = 0.05E-3_rt;

    amrex::Real tissuePos = 0.0_rt; // 10E-4_rt - 2.0_rt*rad;
    amrex::Real bubpos = tissuePos + 2.0_rt*rad;

    //Init All the Domain with:
    alpha1 = alpha_min;                 //Solid
    alpha2 = alpha_min;                 //Air
    alpha3 = 1.0_rt - alpha1 - alpha2;  //Water

    V11 = 1.0_rt; V21 = 0.0_rt; V31 = 0.0_rt;
    V12 = 0.0_rt; V22 = 1.0_rt; V32 = 0.0_rt;
    V13 = 0.0_rt; V23 = 0.0_rt; V33 = 1.0_rt;
    //amrex::Real I2 = compute_I2(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    amrex::Real i1 = compute_i1(V11,V12,V13,V21,V22,V23,V31,V32,V33);
    // rho1 = parm.eos_rho01;
    // rho2 = parm.eos_rho02;
    // rho3 = parm.eos_rho03;

    p = 1.0E5_rt;
    T1 = 300.0_rt; T2 = 300.0_rt; T3 = 300.0_rt;
    rho1 = 1.225_rt; rho2 = 1050.0_rt; rho3 = 1000.0_rt;

    amrex::Real r = std::sqrt(x*x + (y-bubpos)*(y-bubpos));
    if(( (y-bubpos)*(y-bubpos))+(((x-0.0_rt)*(x-0.0_rt))) < std::pow(rad,2.0_rt) && y >= tissuePos){
        //Bubble Region
        alpha3 = alpha_min;
        alpha2 = alpha_min;
        alpha1 = 1.0_rt - alpha2 - alpha3;
    }
    // Liver Region:
    // if (y <= tissuePos) {
    //     alpha1 = alpha_min;
    //     alpha3 = alpha_min;
    //     alpha2 = 1.0_rt - alpha1 - alpha3;
    // }
    // amrex::Real G1 = parm.eos_G01;
    // amrex::Real es1 = (G1/parm.eos_rho01)*I2;
    // amrex::Real es1 = (G1/(2.0*parm.eos_rho01))*(i1 - 3.0);

    if (parm.tabulated1 == 1) {
      // rho1 = 1.225_rt;
      // T1 = T_finder(alpha1, rho1, p, 300.0, 1);
      rho1 = TPF(ORHO, T1, std::log10(p), 1);
      eint1 = TPF(OE, T1, std::log10(p), 1);
      alrhoeint1 = alpha1*rho1*eint1;
    } else {
      rho1 = 1.0_rt/((parm.eos_gamma1-1.0_rt)*parm.eos_cv1*T1/(p+parm.eos_pinf1) + parm.eos_b1);
      alrhoeint1 = alpha1*rho1*((p+parm.eos_gamma1*parm.eos_pinf1)/(parm.eos_gamma1-1.0_rt)*(1.0_rt/rho1 - parm.eos_b1) + parm.eos_q1);
    }
    if (parm.tabulated2 == 1) {
      rho2 = TPF(ORHO, T2, std::log10(p), 2);
      eint2 = TPF(OE, T2, std::log10(p), 2);
      alrhoeint2 = alpha2*rho2*eint2;
    } else {
      rho2 = 1.0_rt/((parm.eos_gamma2-1.0_rt)*parm.eos_cv2*T2/(p+parm.eos_pinf2) + parm.eos_b2);
      alrhoeint2 = alpha2*rho2*((p+parm.eos_gamma2*parm.eos_pinf2)/(parm.eos_gamma2-1.0_rt)*(1.0_rt/rho2 - parm.eos_b2) + parm.eos_q2);
    }
    if (parm.tabulated3 == 1) {
      rho3 = TPF(ORHO, T3, std::log10(p), 3);
      eint3 = TPF(OE, T3, std::log10(p), 3);
      alrhoeint3 = alpha3*rho3*eint3;
    } else {
      rho3 = 1.0_rt/((parm.eos_gamma3-1.0_rt)*parm.eos_cv3*T3/(p+parm.eos_pinf3) + parm.eos_b3);
      alrhoeint3 = alpha3*rho3*((p+parm.eos_gamma3*parm.eos_pinf3)/(parm.eos_gamma3-1.0_rt)*(1.0_rt/rho3 - parm.eos_b3) + parm.eos_q3);
    }

    // T1 = compute_temperature(alpha1, rho1, p, 300.0, 1);
    // T2 = compute_temperature(alpha2, rho2, p, 300.0, 2);
    // T3 = compute_temperature(alpha3, rho3, p, 300.0, 3);

    state(i,j,k,UARHO1 ) = alpha1*rho1;
    state(i,j,k,UARHO2 ) = alpha2*rho2;
    state(i,j,k,UARHO3 ) = alpha3*rho3;
    state(i,j,k,UMX  ) = 0.0_rt;
    state(i,j,k,UMY  ) = 0.0_rt;
    state(i,j,k,UMZ  ) = 0.0_rt;
    state(i,j,k,URHOE) = alrhoeint1 + alrhoeint2 + alrhoeint3;
    state(i,j,k,UARP) = 0.0_rt;
    state(i,j,k,GALPHA1) =  alpha1;
    state(i,j,k,GALPHA2) =  alpha2;
    state(i,j,k,GV11) =  V11;
    state(i,j,k,GV21) =  V21;
    state(i,j,k,GV31) =  V31;
    state(i,j,k,GV12) =  V12;
    state(i,j,k,GV22) =  V22;
    state(i,j,k,GV32) =  V32;
    state(i,j,k,GV13) =  V13;
    state(i,j,k,GV23) =  V23;
    state(i,j,k,GV33) =  V33;
    state(i,j,k,GPRESS) =  p;
    state(i,j,k,GT1) =  T1;
    state(i,j,k,GT2) =  T2;
    state(i,j,k,GT3) =  T3;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const amrex::Real s_int[NV],
  amrex::Real s_ext[NV],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata)
{
    amrex::Real p0, b, a, p, rho1, rho2, rho3, alpha1, alpha2, alpha3, c, v, alrhoeint1, alrhoeint2, alrhoeint3, alpha_min;
    amrex::Real V11, V12, V13, V21, V22, V23, V31, V32, V33;
    amrex::Real eint1, eint2, eint3, T1, T2, T3;
    Parm const* lparm = EMM::parm.get();

    alpha_min = EMM::parm->alpha_min;

    p0 = 1E5;

    alpha1 = alpha_min;                 //Solid
    alpha2 = alpha_min;                 //Air
    alpha3 = 1.0_rt - alpha1 - alpha2;  //Water

    rho1 = EMM::parm->eos_rho01;
    rho2 = EMM::parm->eos_rho02;
    rho3 = EMM::parm->eos_rho03;

    V11 = 1.0_rt; V21 = 0.0_rt; V31 = 0.0_rt;
    V12 = 0.0_rt; V22 = 1.0_rt; V32 = 0.0_rt;
    V13 = 0.0_rt; V23 = 0.0_rt; V33 = 1.0_rt;

    // p = 1E5 + 2.0*30E6*std::exp(-9.1E5*time)*std::cos(2.0*M_PI*83.3E3*time+M_PI/3.0);
    // if(p < 2300){p = 2300;}
    // Create some Python objects that will later be assigned values.
    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *tValue;
    // Convert the file name to a Python string.
    PyObject * sys = PyImport_ImportModule("sys");
    PyObject * path = PyObject_GetAttrString(sys, "path");
    PyList_Append(path, PyUnicode_FromString("."));
    PyObject * ModuleString = PyUnicode_FromString((char*) "TimePressureBC");
    //pName = PyUnicode_FromString("Sample");
    //if(pName == NULL) printf("Calling the pName failed.\n");
    // Import the file as a Python module.
    pModule = PyImport_Import(ModuleString);
    if(pModule == NULL) PyErr_Print(); //printf("Calling the pModule failed.\n");
    // Create a dictionary for the contents of the module.
    pDict = PyModule_GetDict(pModule);
    // Get the add method from the dictionary.
    pFunc = PyDict_GetItemString(pDict, "interpP");
    // Create a Python tuple to hold the arguments to the method.
    pArgs = PyTuple_New(1);
    // Convert 2 to a Python integer.
    tValue = PyFloat_FromDouble(time);
    // Set the Python int as the first and second arguments to the method.
    PyTuple_SetItem(pArgs, 0, tValue);
    // Call the function with the arguments.
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    // Print a message if calling the method failed.
    if(pResult == NULL) {PyErr_Print(); printf("Calling the interpP method failed.\n");}
    // Convert the result to a long from a Python object.
    double pR = PyFloat_AsDouble(pResult);
    //printf("The result is %e.\n", p);
    p = pR;

    T1 = 300.0_rt; T2 = 300.0_rt; T3 = 300.0_rt;
    rho2 = 1050.0_rt; rho1 = 1.225_rt; rho3 = 1000.0_rt;

    if (EMM::parm->tabulated1 == 1) {
      // rho1 = 1.225_rt;
      // T1 = T_finder(alpha1, rho1, p, 300.0, 1);
      // eint1 = TPF(OE, T1, std::log10(p), 1);
      // alrhoeint1 = alpha1*rho1*eint1;
      rho1 = TPF(ORHO, T1, std::log10(p), 1);
      eint1 = TPF(OE, T1, std::log10(p), 1);
      alrhoeint1 = alpha1*rho1*eint1;
    } else {
      rho1 = 1.0_rt/((EMM::parm->eos_gamma1-1.0_rt)*EMM::parm->eos_cv1*T1/(p+EMM::parm->eos_pinf1) + EMM::parm->eos_b1);
      alrhoeint1 = alpha1*rho1*((p+EMM::parm->eos_gamma1*EMM::parm->eos_pinf1)/(EMM::parm->eos_gamma1-1.0_rt)*(1.0_rt/rho1 - EMM::parm->eos_b1) + EMM::parm->eos_q1);
    }
    if (EMM::parm->tabulated2 == 1) {
      rho2 = TPF(ORHO, T2, std::log10(p), 2);
      eint2 = TPF(OE, T2, std::log10(p), 2);
      alrhoeint2 = alpha2*rho2*eint2;
    } else {
      rho2 = 1.0_rt/((EMM::parm->eos_gamma2-1.0_rt)*EMM::parm->eos_cv2*T2/(p+EMM::parm->eos_pinf2) + EMM::parm->eos_b2);
      alrhoeint2 = alpha2*rho2*((p+EMM::parm->eos_gamma2*EMM::parm->eos_pinf2)/(EMM::parm->eos_gamma2-1.0_rt)*(1.0_rt/rho2 - EMM::parm->eos_b2) + EMM::parm->eos_q2);
    }
    if (EMM::parm->tabulated3 == 1) {
      rho3 = TPF(ORHO, T3, std::log10(p), 3);
      eint3 = TPF(OE, T3, std::log10(p), 3);
      alrhoeint3 = alpha3*rho3*eint3;
    } else {
      rho3 = 1.0_rt/((EMM::parm->eos_gamma3-1.0_rt)*EMM::parm->eos_cv3*T3/(p+EMM::parm->eos_pinf3) + EMM::parm->eos_b3);
      alrhoeint3 = alpha3*rho3*((p+EMM::parm->eos_gamma3*EMM::parm->eos_pinf3)/(EMM::parm->eos_gamma3-1.0_rt)*(1.0_rt/rho3 - EMM::parm->eos_b3) + EMM::parm->eos_q3);
    }

    // Print() << "rho1:    " << rho1 << "\n";
    // Print() << "rho2:    " << rho2 << "\n";
    // Print() << "rho3:    " << rho3 << "\n";
    // T1 = compute_temperature(alpha1, rho1, p, 300.0, 1);
    // T2 = compute_temperature(alpha2, rho2, p, 300.0, 2);
    // T3 = compute_temperature(alpha3, rho3, p, 300.0, 3);

    s_ext[UARHO1] = alpha1*rho1;
    s_ext[UARHO2] = alpha2*rho2;
    s_ext[UARHO3] = alpha3*rho3;
    s_ext[UMX] = 0.0_rt;
    s_ext[UMY] = s_int[UMY];
    s_ext[UMZ] = 0.0_rt;
    s_ext[URHOE] = alrhoeint1 + alrhoeint2 + alrhoeint3 + 0.5*s_int[UMY]*s_int[UMY]/(alpha1*rho1+alpha2*rho2+alpha3*rho3);
    s_ext[UARP] = 0.0_rt;
    s_ext[GALPHA1] = alpha1;
    s_ext[GALPHA2] = alpha2;
    s_ext[GV11] =  V11;
    s_ext[GV21] =  V21;
    s_ext[GV31] =  V31;
    s_ext[GV12] =  V12;
    s_ext[GV22] =  V22;
    s_ext[GV32] =  V32;
    s_ext[GV13] =  V13;
    s_ext[GV23] =  V23;
    s_ext[GV33] =  V33;
    s_ext[GPRESS] =  p;
    s_ext[GT1] =  T1;
    s_ext[GT2] =  T2;
    s_ext[GT3] =  T3;
}


#endif
