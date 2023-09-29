#ifndef EMM_EOS_H_
#define EMM_EOS_H_

#include <cmath>
#include "EMM_parm.H"
#include "EMM.H"
#include "EMM_index_macros.H"

namespace {
// -------------------------------------------------------------------------------------------------
// Complex Thermodynamics Model:
// -------------------------------------------------------------------------------------------------

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_NASG_pressure(
   amrex::Real mrho,
   amrex::Real e,
   amrex::Real alpha1,
   amrex::Real alpha2,
   amrex::Real alpha3,
   amrex::Real rho1,
   amrex::Real rho2,
   amrex::Real rho3) noexcept
{
   // - alpha1*rho1*es1
   amrex::Real p = ( mrho*e - ( 
        (alpha1*(1.0_rt-rho1*EMM::parm->eos_b1)*EMM::parm->eos_gamma1*EMM::parm->eos_pinf1)/(EMM::parm->eos_gamma1-1.0_rt)
      + (alpha2*(1.0_rt-rho2*EMM::parm->eos_b2)*EMM::parm->eos_gamma2*EMM::parm->eos_pinf2)/(EMM::parm->eos_gamma2-1.0_rt)
      + (alpha3*(1.0_rt-rho3*EMM::parm->eos_b3)*EMM::parm->eos_gamma3*EMM::parm->eos_pinf3)/(EMM::parm->eos_gamma3-1.0_rt))
      - alpha1*rho1*EMM::parm->eos_q1 - alpha2*rho2*EMM::parm->eos_q2 - alpha3*rho3*EMM::parm->eos_q3)
      / (alpha1*(1.0_rt-rho1*EMM::parm->eos_b1)/(EMM::parm->eos_gamma1-1.0_rt)
      +  alpha2*(1.0_rt-rho2*EMM::parm->eos_b2)/(EMM::parm->eos_gamma2-1.0_rt)
      +  alpha3*(1.0_rt-rho3*EMM::parm->eos_b3)/(EMM::parm->eos_gamma3-1.0_rt));
   return p;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_NASG_temperature(
   amrex::Real rho,
   amrex::Real p,
   int phaseNum
) noexcept
{
   amrex::Real T = 0.0;
   if (phaseNum == 1){
      T = (1.0_rt/rho-EMM::parm->eos_b1)*(p+EMM::parm->eos_pinf1)/EMM::parm->eos_cv1/(EMM::parm->eos_gamma1 - 1.0_rt);
   } else if (phaseNum == 2) {
      T = (1.0_rt/rho-EMM::parm->eos_b2)*(p+EMM::parm->eos_pinf2)/EMM::parm->eos_cv2/(EMM::parm->eos_gamma2 - 1.0_rt);
   } else if (phaseNum == 3) {
      T = (1.0_rt/rho-EMM::parm->eos_b3)*(p+EMM::parm->eos_pinf3)/EMM::parm->eos_cv3/(EMM::parm->eos_gamma3 - 1.0_rt);
   }
   return T;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_NASG_ie(
   amrex::Real p,
   amrex::Real rho,
   int phaseNum
) noexcept
{
   amrex::Real eint = 0.0_rt;
   if (phaseNum == 1){
      eint = (p+EMM::parm->eos_gamma1*EMM::parm->eos_pinf1)/(EMM::parm->eos_gamma1-1.0_rt)*(1.0_rt/rho - EMM::parm->eos_b1) + EMM::parm->eos_q1;
   } else if (phaseNum == 2) {
      eint = (p+EMM::parm->eos_gamma2*EMM::parm->eos_pinf2)/(EMM::parm->eos_gamma2-1.0_rt)*(1.0_rt/rho - EMM::parm->eos_b2) + EMM::parm->eos_q2;
   } else if (phaseNum == 3) {
      eint = (p+EMM::parm->eos_gamma3*EMM::parm->eos_pinf3)/(EMM::parm->eos_gamma3-1.0_rt)*(1.0_rt/rho - EMM::parm->eos_b3) + EMM::parm->eos_q3;
   }
   return eint;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_NASG_SoS(
   amrex::Real p,
   amrex::Real rho,
   int phaseNum
) noexcept
{
   amrex::Real csq = 0.0_rt;
   if (phaseNum == 1){
      csq = EMM::parm->eos_gamma1*std::pow(1.0_rt/rho, 2.0)*(p+EMM::parm->eos_pinf1)/(1.0_rt/rho - EMM::parm->eos_b1);
   } else if (phaseNum == 2) {
      csq = EMM::parm->eos_gamma2*std::pow(1.0_rt/rho, 2.0)*(p+EMM::parm->eos_pinf2)/(1.0_rt/rho - EMM::parm->eos_b2);
   } else if (phaseNum == 3) {
      csq = EMM::parm->eos_gamma3*std::pow(1.0_rt/rho, 2.0)*(p+EMM::parm->eos_pinf3)/(1.0_rt/rho - EMM::parm->eos_b3);
   }
   return csq;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
TPF(
   int tableToInterpolate,
   amrex::Real T,
   amrex::Real p,
   int phaseNum
) noexcept
{
   using namespace amrex::literals;

   amrex::Real T1 = 0.0_rt;
   amrex::Real P1 = 0.0_rt;
   T1 = std::min(std::max(T, EMM::parm->TMINT1), EMM::parm->TMAXT1 - EMM::parm->DT1 / 2.0_rt);
   P1 = std::min(std::max(p, EMM::parm->PMINT1), EMM::parm->PMAXT1 - EMM::parm->DP1 / 2.0_rt);

   int II = 0;
   II = int((T1 - EMM::parm->TMINT1) / EMM::parm->DT1);
   if (II > EMM::parm->nnTP1 - 2)
   {
      II = II - 1;
   }

   int JJ = 0;
   JJ = int((P1 - EMM::parm->PMINT1) / EMM::parm->DP1);
   if (JJ > EMM::parm->mmTP1 - 2)
   {
      JJ = JJ - 1;
   }

   amrex::Real TMIN = 0.0_rt;
   amrex::Real TMAX = 0.0_rt;
   amrex::Real PMIN = 0.0_rt;
   amrex::Real PMAX = 0.0_rt;
   TMIN = EMM::TVEC1T[II][JJ];
   TMAX = EMM::TVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ + 1];
   PMIN = EMM::PVEC1T[II][JJ];
   PMAX = EMM::PVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ + 1];

   amrex::Real CT11 = 0.0_rt;
   amrex::Real CT12 = 0.0_rt;
   amrex::Real CT21 = 0.0_rt;
   amrex::Real CT22 = 0.0_rt;
   if(tableToInterpolate == ORHO){
      CT11 = EMM::RHOVEC1T[II][JJ];
      CT12 = EMM::RHOVEC1T[II][JJ + 1];
      CT21 = EMM::RHOVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ];
      CT22 = EMM::RHOVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ + 1];
   }else if (tableToInterpolate == OSOS){
      CT11 = EMM::SOSVEC1T[II][JJ];
      CT12 = EMM::SOSVEC1T[II][JJ + 1];
      CT21 = EMM::SOSVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ];
      CT22 = EMM::SOSVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ + 1];
   }else if (tableToInterpolate == OE){
      CT11 = EMM::EVEC1T[II][JJ];
      CT12 = EMM::EVEC1T[II][JJ + 1];
      CT21 = EMM::EVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ];
      CT22 = EMM::EVEC1T[std::min(II + 1, EMM::parm->nnTP1)][JJ + 1];
   }

   amrex::Real f1 = (TMAX - T1) / (TMAX - TMIN) * CT11 + (T1 - TMIN) / (TMAX - TMIN) * CT21;
   amrex::Real f2 = (TMAX - T1) / (TMAX - TMIN) * CT12 + (T1 - TMIN) / (TMAX - TMIN) * CT22;
   amrex::Real output = (PMAX - P1) / (PMAX - PMIN) * f1 + (P1 - PMIN) / (PMAX - PMIN) * f2;

   return output;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
   T_finder(amrex::Real alpha,
            amrex::Real rho,
            amrex::Real PG,
            amrex::Real TG,
            int phaseNum) noexcept
{
   amrex::Real TT = TG; // q(i, j, k, QT1);
   if (TT < 1.0_rt) TT = 300.0;
   TT = std::min(std::max(TT, EMM::parm->TMINT1 * 1.001), EMM::parm->TMAXT1 * 0.99);
   int zmax = 1000;

   amrex::Real epsT = 0.0_rt;
   amrex::Real rhoTTp = 0.0_rt;
   amrex::Real rhoT = 0.0_rt;
   amrex::Real drhodT = 0.0_rt;
   amrex::Real TTo = 0.0_rt;
   // amrex::Real alphaptf = 0.25;
   for (double alphaptf = 1.0_rt; alphaptf >= 0.1; alphaptf -= 0.1)
   // for (double alphaptf = 1.0_rt; alphaptf >= 0.1; alphaptf -= 0.05)
   {
      // TT = TG;
      for (int z = 1; z <= zmax; z++)
      {
            epsT = TT * 1E-3;
            rhoTTp = TPF(ORHO, TT + epsT, std::log10(PG), phaseNum);
            rhoT = TPF(ORHO, TT, std::log10(PG), phaseNum);
            drhodT = ((rhoTTp - rho) - (rhoT - rho)) / epsT + 1E-16;
            TTo = TT;
            TT = TTo - (rhoT - rho) / drhodT;
            TT = TTo * (1.0_rt - alphaptf) + TT * alphaptf;

            TT = std::min(std::max(TT, EMM::parm->TMINT1 * 1.01), EMM::parm->TMAXT1 * 0.99);
               // amrex::Print() << "alphaptf:   " << alphaptf << "\n";
               // amrex::Print() << "z:          " << z << "\n";
               // amrex::Print() << "PG:         " << PG << "\n";
               // amrex::Print() << "rho:        " << rho << "\n";
               // amrex::Print() << "rhoTTp:     " << rhoTTp << "\n";
               // amrex::Print() << "rhoT:       " << rhoT << "\n";
               // amrex::Print() << "epsT:       " << epsT << "\n";
               // amrex::Print() << "TT:         " << TT << "\n";
               // amrex::Print() << "TTo:        " << TTo << "\n";

            if (std::abs(TT - TTo)/std::abs(TTo) < 1E-10)
            // if (std::abs(rhoT-rho) < 0.001*rho)
            {
               return TT;
            }

      }
   }
   // amrex::Print() << "rhoTTp:     " << rhoTTp << "\n";
   // amrex::Print() << "rhoT:       " << rhoT << "\n";
   // amrex::Print() << "epsT:       " << epsT << "\n";
   // amrex::Print() << "TT:         " << TT << "\n";
   // amrex::Print() << "TTo:        " << TTo << "\n";
   // amrex::Print() << "PG:         " << PG << "\n";
   // amrex::Print() << "rho:        " << rho << "\n";
   // amrex::Print() << "TEMPERATURE DID NOT CONVERGE" << "\n";
   return TT;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
   p_finder(amrex::Real alpha1,
            amrex::Real alpha2,
            amrex::Real alpha3,
            amrex::Real rho1,
            amrex::Real rho2,
            amrex::Real rho3,
            amrex::Real p,
            amrex::Real T1G,
            amrex::Real T2G,
            amrex::Real T3G,
            amrex::Real eint
) noexcept
{
   BL_PROFILE("p_finder()");
   amrex::Real pg = p;
   amrex::Real p_init = p;

   // if(alpha1 > 1.0_rt-EMM::parm->alpha_min/100.0){alpha1 = 1.0_rt-EMM::parm->alpha_min/100.0;}
   // if(alpha1 < EMM::parm->alpha_min/100.0){alpha1 = EMM::parm->alpha_min/100.0;}
   // alpha2 = 1.0_rt - alpha1;
   if (alpha1 < 0.0_rt) {Print() << "PF negative VOF: " << alpha1 << "\n";};
   if (alpha2 < 0.0_rt) {Print() << "PF negative VOF: " << alpha2 << "\n";};
   if (alpha3 < 0.0_rt) {Print() << "PF negative VOF: " << alpha3 << "\n";};

   // for (double alpharint = 1.0_rt; alpharint >= 0.1; alpharint -= 0.05)
   for (double alpharint = 1.0_rt; alpharint >= 0.1; alpharint -= 0.1)
   {
      // pg = p_init;
      for (int Z = 0; Z < 2000 ; Z += 1)
      {
            amrex::Real eG3 = 0.0_rt;
            amrex::Real T3 = 0.0_rt;
            if(EMM::parm->tabulated3 == 0){
               eG3 = compute_NASG_ie(pg, rho3, 3);
            }else{
               T3 = T_finder(alpha3, rho3, pg, T3G, 3);
               T3 = std::min(std::max(T3, EMM::parm->TMINT1 * 1.02), EMM::parm->TMAXT1 * 0.98);
               eG3 = TPF(OE, T3, std::log10(pg), 3);
            }

            amrex::Real eG2 = 0.0_rt;
            amrex::Real T2 = 0.0_rt;
            if(EMM::parm->tabulated2 == 0){
               eG2 = compute_NASG_ie(pg, rho2, 2);
            }else{
               T2 = T_finder(alpha2, rho2, pg, T2G, 2);
               T2 = std::min(std::max(T2, EMM::parm->TMINT1 * 1.02), EMM::parm->TMAXT1 * 0.98);
               eG2 = TPF(OE, T2, std::log10(pg), 2);
            }

            amrex::Real eG1 = 0.0_rt;
            amrex::Real T1 = 0.0_rt;
            if(EMM::parm->tabulated1 == 0){
               eG1 = compute_NASG_ie(pg, rho1, 1);
            }else{
               T1 = T_finder(alpha1, rho1, pg, T1G, 1);
               T1 = std::min(std::max(T1, EMM::parm->TMINT1 * 1.02), EMM::parm->TMAXT1 * 0.98);
               eG1 = TPF(OE, T1, std::log10(pg), 1);
            }
            amrex::Real eps=1E-3*pg;

            amrex::Real ERR1 = alpha1*rho1*eG1 + alpha2*rho2*eG2 + alpha3*rho3*eG3 - eint;
            amrex::Real ptemp = pg + eps;

            if(EMM::parm->tabulated3 == 0){
               eG3 = compute_NASG_ie(ptemp, rho3, 3);
            }else{
               T3 = T_finder(alpha3, rho3, ptemp, T3G, 3);
               T3 = std::min(std::max(T3, EMM::parm->TMINT1 * 1.02), EMM::parm->TMAXT1 * 0.98);
               eG3 = TPF(OE, T3, std::log10(ptemp), 3);
            }

            if(EMM::parm->tabulated2 == 0){
               eG2 = compute_NASG_ie(ptemp, rho2, 2);
            }else{
               T2 = T_finder(alpha2, rho2, ptemp, T2G, 2);
               T2 = std::min(std::max(T2, EMM::parm->TMINT1 * 1.02), EMM::parm->TMAXT1 * 0.98);
               eG2 = TPF(OE, T2, std::log10(ptemp), 2);
            }

            eG1 = 0.0_rt;
            T1 = 0.0_rt;
            if(EMM::parm->tabulated1 == 0){
               eG1 = compute_NASG_ie(ptemp, rho1, 1);
            }else{
               T1 = T_finder(alpha1, rho1, ptemp, T1G, 1);
               T1 = std::min(std::max(T1, EMM::parm->TMINT1 * 1.02), EMM::parm->TMAXT1 * 0.98);
               eG1 = TPF(OE, T1, std::log10(ptemp), 1);
            }

            amrex::Real ERR2 = alpha1*rho1*eG1 + alpha2*rho2*eG2 + alpha3*rho3*eG3 - eint;

            amrex::Real derr = (ERR2-ERR1)/eps;
            amrex::Real pg0 = pg;
            pg = alpharint*(pg - ERR1/derr)+(1.0_rt-alpharint)*pg0;
            pg = std::min(std::max(pg, std::pow(10.0_rt,EMM::parm->PMINT1) * 1.01), std::pow(10.0_rt,EMM::parm->PMAXT1) * 0.99);
            // pg = std::min(std::max(pg, std::pow(10.0_rt,2300.0_rt) * 1.01), std::pow(10.0_rt,EMM::parm->PMAXT1) * 0.99);
            // pg = std::min(std::max(pg, std::pow(10.0_rt,EMM::parm->PMINT1) * 1.01), std::pow(10.0_rt,EMM::parm->PMAXT1) * 0.99);

            // amrex::Print() << "**************" << "\n";
            // amrex::Print() << "alpha1:       " << alpha1 << "\n";
            // amrex::Print() << "alpha2:       " << alpha2 << "\n";
            // amrex::Print() << "alpha3:       " << alpha3 << "\n";
            // amrex::Print() << "rho1:         " << rho1 << "\n";
            // amrex::Print() << "rho2:         " << rho2 << "\n";
            // amrex::Print() << "rho3:         " << rho3 << "\n";

            // amrex::Print() << "pg:           " << pg << "\n";
            if(std::abs(pg-pg0)/std::abs(pg0) < 1E-5){
            // if( ERR2 < 1E-3 && ERR1 < 1E-3 ){
               return pg;
               // break;
            }
      }
   }
   // amrex::Print() << "PRESSURE DID NOT CONVERGE" << "\n";
   return pg;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_pressure(
   amrex::Real mrho,
   amrex::Real e,
   amrex::Real es1,
   amrex::Real es2,
   amrex::Real es3,
   amrex::Real alpha1,
   amrex::Real alpha2,
   amrex::Real alpha3,
   amrex::Real rho1,
   amrex::Real rho2,
   amrex::Real rho3,
   amrex::Real Pprev,
   amrex::Real T1prev,
   amrex::Real T2prev,
   amrex::Real T3prev
) noexcept
{
   // int ll = 0;
   // amrex::Real d = 1E3;
   // if(alpha1 > 1.0_rt-2.0*EMM::parm->alpha_min/d){alpha1 = 1.0_rt-2.0*EMM::parm->alpha_min/d; ll = 1;}
   // if(alpha2 > 1.0_rt-2.0*EMM::parm->alpha_min/d){alpha2 = 1.0_rt-2.0*EMM::parm->alpha_min/d; ll = 1;}
   // if(alpha1 < EMM::parm->alpha_min/d){alpha1 = EMM::parm->alpha_min/d; ll = 1;}
   // if(alpha2 < EMM::parm->alpha_min/d){alpha2 = EMM::parm->alpha_min/d; ll = 1;}
   // if(ll == 1){
   //    alpha3 = 1.0_rt - alpha1 - alpha2;
   //    alpha1 = alpha1/(alpha1+alpha2+alpha3);
   //    alpha2 = alpha2/(alpha1+alpha2+alpha3);
   // }
   if (alpha1 < 0.0_rt) {Print() << "CP negative VOF: " << alpha1 << "\n";};
   if (alpha2 < 0.0_rt) {Print() << "CP negative VOF: " << alpha2 << "\n";};
   if (alpha3 < 0.0_rt) {Print() << "CP negative VOF: " << alpha3 << "\n";};

   amrex::Real p = 0.0_rt;
   // - alpha1*rho1*es1
   if (EMM::parm->tabulated == 0) {
      p = ( mrho*e - alpha1*rho1*es1 - alpha2*rho2*es2 - alpha3*rho3*es3 - ( 
         (alpha1*(1.0_rt-rho1*EMM::parm->eos_b1)*EMM::parm->eos_gamma1*EMM::parm->eos_pinf1)/(EMM::parm->eos_gamma1-1.0_rt)
         + (alpha2*(1.0_rt-rho2*EMM::parm->eos_b2)*EMM::parm->eos_gamma2*EMM::parm->eos_pinf2)/(EMM::parm->eos_gamma2-1.0_rt)
         + (alpha3*(1.0_rt-rho3*EMM::parm->eos_b3)*EMM::parm->eos_gamma3*EMM::parm->eos_pinf3)/(EMM::parm->eos_gamma3-1.0_rt))
         - alpha1*rho1*EMM::parm->eos_q1 - alpha2*rho2*EMM::parm->eos_q2 - alpha3*rho3*EMM::parm->eos_q3)
         / (alpha1*(1.0_rt-rho1*EMM::parm->eos_b1)/(EMM::parm->eos_gamma1-1.0_rt)
         +  alpha2*(1.0_rt-rho2*EMM::parm->eos_b2)/(EMM::parm->eos_gamma2-1.0_rt)
         +  alpha3*(1.0_rt-rho3*EMM::parm->eos_b3)/(EMM::parm->eos_gamma3-1.0_rt));
   }else{
      p = p_finder(alpha1, alpha2, alpha3, rho1, rho2, rho3, Pprev, T1prev, T2prev, T3prev, mrho*e - alpha1*rho1*es1 - alpha2*rho2*es2 - alpha3*rho3*es3);
   }

   return p;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_temperature(
   amrex::Real alpha,
   amrex::Real rho,
   amrex::Real p,
   amrex::Real Tprev,
   int phaseNum
) noexcept
{
   // amrex::Real d = 1E3;
   // if(alpha > 1.0_rt-2.0*EMM::parm->alpha_min/d){alpha = 1.0_rt-2.0*EMM::parm->alpha_min/d;}
   // if(alpha < EMM::parm->alpha_min/d){alpha = EMM::parm->alpha_min/d;}
   // if (alpha < 0.0_rt) {Print() << "CT negative VOF: " << alpha << "\n";}
   if (rho < 0.0_rt) {Print() << "CT negative density: " << rho << "\n";}

   amrex::Real T = 0.0_rt;
   if(phaseNum == 1 && EMM::parm->tabulated1 == 1){
      T = T_finder(alpha, rho, p, Tprev, 1);
   }else if (phaseNum == 1){
      T = (1.0_rt/rho-EMM::parm->eos_b1)*(p+EMM::parm->eos_pinf1)/EMM::parm->eos_cv1/(EMM::parm->eos_gamma1 - 1.0_rt);
   }
   if(phaseNum == 2 && EMM::parm->tabulated2 == 1){
      T = T_finder(alpha, rho, p, Tprev, 2);
   }else if (phaseNum == 2){
      T = (1.0_rt/rho-EMM::parm->eos_b2)*(p+EMM::parm->eos_pinf2)/EMM::parm->eos_cv2/(EMM::parm->eos_gamma2 - 1.0_rt);
   }
   if(phaseNum == 3 && EMM::parm->tabulated3 == 1){
      T = T_finder(alpha, rho, p, Tprev, 3);
   }else if (phaseNum == 3){
      T = (1.0_rt/rho-EMM::parm->eos_b3)*(p+EMM::parm->eos_pinf3)/EMM::parm->eos_cv3/(EMM::parm->eos_gamma3 - 1.0_rt);
   }
   return T;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_internal_energy(
   amrex::Real p,
   amrex::Real T,
   amrex::Real rho,
   int phaseNum
) noexcept
{
   amrex::Real eint = 0.0_rt;
   if(phaseNum == 1 && EMM::parm->tabulated1 == 1){
      eint = TPF(OE, T, std::log10(p), 1);
   }else if (phaseNum == 1){
      eint = (p+EMM::parm->eos_gamma1*EMM::parm->eos_pinf1)/(EMM::parm->eos_gamma1-1.0_rt)*(1.0_rt/rho - EMM::parm->eos_b1) + EMM::parm->eos_q1;
   }
   if(phaseNum == 2 && EMM::parm->tabulated2 == 1){
      eint = TPF(OE, T, std::log10(p), 2);
   }else if (phaseNum == 2){
      eint = (p+EMM::parm->eos_gamma2*EMM::parm->eos_pinf2)/(EMM::parm->eos_gamma2-1.0_rt)*(1.0_rt/rho - EMM::parm->eos_b2) + EMM::parm->eos_q2;
   }
   if(phaseNum == 3 && EMM::parm->tabulated3 == 1){
      eint = TPF(OE, T, std::log10(p), 3);
   }else if (phaseNum == 3){
      eint = (p+EMM::parm->eos_gamma3*EMM::parm->eos_pinf3)/(EMM::parm->eos_gamma3-1.0_rt)*(1.0_rt/rho - EMM::parm->eos_b3) + EMM::parm->eos_q3;
   }

   return eint;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
   amrex::Real
compute_SoS(
   amrex::Real p,
   amrex::Real T,
   amrex::Real rho,
   int phaseNum
) noexcept
{
   amrex::Real csq = 0.0_rt;
   if (phaseNum == 1 && EMM::parm->tabulated1 == 0){
      csq = EMM::parm->eos_gamma1*std::pow(1.0_rt/rho, 2.0)*(p+EMM::parm->eos_pinf1)/(1.0_rt/rho - EMM::parm->eos_b1);
   } else if(phaseNum == 1) {
      csq = std::pow(TPF(OSOS, T, std::log10(p), 1),2.0);
   }
   if (phaseNum == 2 && EMM::parm->tabulated2 == 0) {
      csq = EMM::parm->eos_gamma2*std::pow(1.0_rt/rho, 2.0)*(p+EMM::parm->eos_pinf2)/(1.0_rt/rho - EMM::parm->eos_b2);
   } else if (phaseNum == 2) {
      csq = std::pow(TPF(OSOS, T, std::log10(p), 2),2.0);
   }
   if (phaseNum == 3 && EMM::parm->tabulated3 == 0) {
      csq = EMM::parm->eos_gamma3*std::pow(1.0_rt/rho, 2.0)*(p+EMM::parm->eos_pinf3)/(1.0_rt/rho - EMM::parm->eos_b3);
   } else if (phaseNum == 3) {
      csq = std::pow(TPF(OSOS, T, std::log10(p), 3),2.0);
   }
   return csq;
}

// -------------------------------------------------------------------------------------------------
// Neo-Hookean Model:
// -------------------------------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_i1(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                       amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                       amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real i1 = 0.0_rt;

    i1 = std::pow(V11, 2.0) + std::pow(V12, 2.0) + std::pow(V13, 2.0) + std::pow(V21, 2.0) 
       + std::pow(V22, 2.0) + std::pow(V23, 2.0) + std::pow(V31, 2.0) + std::pow(V32, 2.0) + std::pow(V33, 2.0);

    return i1;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB11(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB11 = 0.0_rt;

    devB11 = std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0) + 
   (-std::pow(V11,2.0) - std::pow(V12,2.0) - std::pow(V13,2.0) - std::pow(V21,2.0) - 
      std::pow(V22,2.0) - std::pow(V23,2.0) - std::pow(V31,2.0) - std::pow(V32,2.0) - 
      std::pow(V33,2.0))/3.;

    return devB11;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB21(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB21 = 0.0_rt;

    devB21 = V11*V21 + V12*V22 + V13*V23;

    return devB21;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB31(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB31 = 0.0_rt;

    devB31 = V11*V31 + V12*V32 + V13*V33;

    return devB31;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB12(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB12 = 0.0_rt;

    devB12 = V11*V21 + V12*V22 + V13*V23;

    return devB12;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB22(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB22 = 0.0_rt;

   devB22 = std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0) + 
   (-std::pow(V11,2.0) - std::pow(V12,2.0) - std::pow(V13,2.0) - std::pow(V21,2.0) - 
     std::pow(V22,2.0) - std::pow(V23,2.0) - std::pow(V31,2.0) - std::pow(V32,2.0) - 
     std::pow(V33,2.0))/3.;

    return devB22;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB32(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB32 = 0.0_rt;

    devB32 = V21*V31 + V22*V32 + V23*V33;

    return devB32;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB13(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB13 = 0.0_rt;

    devB13 = V11*V31 + V12*V32 + V13*V33;

    return devB13;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB23(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB23 = 0.0_rt;

    devB23 = V21*V31 + V22*V32 + V23*V33;

    return devB23;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devB33(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devB33 = 0.0_rt;

    devB33 = std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0) + 
   (-std::pow(V11,2.0) - std::pow(V12,2.0) - std::pow(V13,2.0) - std::pow(V21,2.0) - 
      std::pow(V22,2.0) - std::pow(V23,2.0) - std::pow(V31,2.0) - std::pow(V32,2.0) - 
      std::pow(V33,2.0))/3.;

    return devB33;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_I2(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                       amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                       amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real I2 = 0.0_rt;

    I2 = std::pow(std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0) - 
      (-std::pow(V11*V21 + V12*V22 + V13*V23,2.0) + 
         (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0))*
          (std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0)))/
       (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
         2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
         std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
         2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
         2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
         std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
         2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
         std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
         2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
         2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
         2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
         2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
         std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
         2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
         std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)),2.0)/16.0 + 
   std::pow(V21*V31 + V22*V32 + V23*V33 - 
      ((V11*V21 + V12*V22 + V13*V23)*(V11*V31 + V12*V32 + V13*V33) - 
         (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0))*
          (V21*V31 + V22*V32 + V23*V33))/
       (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
         2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
         std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
         2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
         2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
         std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
         2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
         std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
         2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
         2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
         2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
         2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
         std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
         2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
         std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)),2.0)/8.0 + 
   std::pow(V11*V31 + V12*V32 + V13*V33 - 
      (-((std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0))*
            (V11*V31 + V12*V32 + V13*V33)) + 
         (V11*V21 + V12*V22 + V13*V23)*(V21*V31 + V22*V32 + V23*V33))/
       (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
         2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
         std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
         2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
         2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
         std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
         2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
         std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
         2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
         2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
         2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
         2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
         std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
         2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
         std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)),2.0)/8.0 + 
   std::pow(std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0) - 
      (-std::pow(V11*V31 + V12*V32 + V13*V33,2.0) + 
         (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0))*
          (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0)))/
       (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
         2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
         std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
         2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
         2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
         std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
         2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
         std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
         2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
         2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
         2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
         2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
         std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
         2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
         std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)),2.0)/16.0 + 
   std::pow(V11*V21 + V12*V22 + V13*V23 - 
      ((V11*V31 + V12*V32 + V13*V33)*(V21*V31 + V22*V32 + V23*V33) - 
         (V11*V21 + V12*V22 + V13*V23)*
          (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0)))/
       (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
         2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
         std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
         2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
         2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
         std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
         2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
         std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
         2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
         2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
         2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
         2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
         std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
         2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
         std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)),2.0)/8.0 + 
   std::pow(std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0) - 
      (-std::pow(V21*V31 + V22*V32 + V23*V33,2.0) + 
         (std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0))*
          (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0)))/
       (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
         2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
         std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
         2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
         2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
         std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
         2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
         std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
         2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
         2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
         2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
         2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
         std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
         2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
         std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)),2.0)/16.0;

    return I2;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe11(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe11 = 0.0_rt;

    devHe11 = (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0) - 
     (-std::pow(V21*V31 + V22*V32 + V23*V33,2.0) + 
        (std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0))*
         (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0)))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe11;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe21(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe21 = 0.0_rt;

    devHe21 = (V11*V21 + V12*V22 + V13*V23 - ((V11*V31 + V12*V32 + V13*V33)*
         (V21*V31 + V22*V32 + V23*V33) - 
        (V11*V21 + V12*V22 + V13*V23)*
         (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0)))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe21;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe31(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe31 = 0.0_rt;

    devHe31 = (V11*V31 + V12*V32 + V13*V33 - (-((std::pow(V21,2.0) + std::pow(V22,2.0) + 
             std::pow(V23,2.0))*(V11*V31 + V12*V32 + V13*V33)) + 
        (V11*V21 + V12*V22 + V13*V23)*(V21*V31 + V22*V32 + V23*V33))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe31;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe12(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe12 = 0.0_rt;

    devHe12 = (V11*V21 + V12*V22 + V13*V23 - ((V11*V31 + V12*V32 + V13*V33)*
         (V21*V31 + V22*V32 + V23*V33) - 
        (V11*V21 + V12*V22 + V13*V23)*
         (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0)))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe12;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe22(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe22 = 0.0_rt;

    devHe22 = (std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0) - 
     (-std::pow(V11*V31 + V12*V32 + V13*V33,2.0) + 
        (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0))*
         (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0)))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe22;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe32(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe32 = 0.0_rt;

    devHe32 = (V21*V31 + V22*V32 + V23*V33 - ((V11*V21 + V12*V22 + V13*V23)*
         (V11*V31 + V12*V32 + V13*V33) - 
        (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0))*
         (V21*V31 + V22*V32 + V23*V33))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe32;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe13(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe13 = 0.0_rt;

    devHe13 = (V11*V31 + V12*V32 + V13*V33 - (-((std::pow(V21,2.0) + std::pow(V22,2.0) + 
             std::pow(V23,2.0))*(V11*V31 + V12*V32 + V13*V33)) + 
        (V11*V21 + V12*V22 + V13*V23)*(V21*V31 + V22*V32 + V23*V33))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe13;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe23(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe23 = 0.0_rt;

    devHe23 = (V21*V31 + V22*V32 + V23*V33 - ((V11*V21 + V12*V22 + V13*V23)*
         (V11*V31 + V12*V32 + V13*V33) - 
        (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0))*
         (V21*V31 + V22*V32 + V23*V33))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe23;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_devHe33(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real devHe33 = 0.0_rt;

    devHe33 = (std::pow(V31,2.0) + std::pow(V32,2.0) + std::pow(V33,2.0) - 
     (-std::pow(V11*V21 + V12*V22 + V13*V23,2.0) + 
        (std::pow(V11,2.0) + std::pow(V12,2.0) + std::pow(V13,2.0))*
         (std::pow(V21,2.0) + std::pow(V22,2.0) + std::pow(V23,2.0)))/
      (std::pow(V13,2.0)*std::pow(V22,2.0)*std::pow(V31,2.0) - 
        2.0*V12*V13*V22*V23*std::pow(V31,2.0) + 
        std::pow(V12,2.0)*std::pow(V23,2.0)*std::pow(V31,2.0) - 
        2.0*std::pow(V13,2.0)*V21*V22*V31*V32 + 2.0*V12*V13*V21*V23*V31*V32 + 
        2.0*V11*V13*V22*V23*V31*V32 - 2.0*V11*V12*std::pow(V23,2.0)*V31*V32 + 
        std::pow(V13,2.0)*std::pow(V21,2.0)*std::pow(V32,2.0) - 
        2.0*V11*V13*V21*V23*std::pow(V32,2.0) + 
        std::pow(V11,2.0)*std::pow(V23,2.0)*std::pow(V32,2.0) + 2.0*V12*V13*V21*V22*V31*V33 - 
        2.0*V11*V13*std::pow(V22,2.0)*V31*V33 - 2.0*std::pow(V12,2.0)*V21*V23*V31*V33 + 
        2.0*V11*V12*V22*V23*V31*V33 - 2.0*V12*V13*std::pow(V21,2.0)*V32*V33 + 
        2.0*V11*V13*V21*V22*V32*V33 + 2.0*V11*V12*V21*V23*V32*V33 - 
        2.0*std::pow(V11,2.0)*V22*V23*V32*V33 + 
        std::pow(V12,2.0)*std::pow(V21,2.0)*std::pow(V33,2.0) - 
        2.0*V11*V12*V21*V22*std::pow(V33,2.0) + 
        std::pow(V11,2.0)*std::pow(V22,2.0)*std::pow(V33,2.0)))/4.0;

    return devHe33;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe11(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe11 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe11 = (-8. + 4.*devHe22 - 2.*devHe13*devHe31 + 1.*devHe13*devHe22*devHe31 - 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe11*
      (-4. + 1.*devHe23*devHe32 + devHe22*(2. - 1.*devHe33) + 2.*devHe33) + 
     devHe12*(-2.*devHe21 - 1.*devHe23*devHe31 + 1.*devHe21*devHe33))/
   (-8. + 4.*devHe22 + 2.*devHe13*devHe31 - 1.*devHe13*devHe22*devHe31 + 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe12*
      (2.*devHe21 + 1.*devHe23*devHe31 - 1.*devHe21*devHe33) + 
     devHe11*(4. - 1.*devHe23*devHe32 - 2.*devHe33 + 
        devHe22*(-2. + 1.*devHe33)));

    return exponentialdevHe11;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe21(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe21 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe21 = (0. - 4.*devHe23*devHe31 + devHe21*(-8. + 4.*devHe33))/
   (-8. + 4.*devHe22 + 2.*devHe13*devHe31 - 1.*devHe13*devHe22*devHe31 + 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe12*
      (2.*devHe21 + 1.*devHe23*devHe31 - 1.*devHe21*devHe33) + 
     devHe11*(4. - 1.*devHe23*devHe32 - 2.*devHe33 + 
        devHe22*(-2. + 1.*devHe33)));

   return exponentialdevHe21;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe31(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe31 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe31 = (0. + (-8. + 4.*devHe22)*devHe31 - 4.*devHe21*devHe32)/
   (-8. + 4.*devHe22 + 2.*devHe13*devHe31 - 1.*devHe13*devHe22*devHe31 + 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe12*
      (2.*devHe21 + 1.*devHe23*devHe31 - 1.*devHe21*devHe33) + 
     devHe11*(4. - 1.*devHe23*devHe32 - 2.*devHe33 + 
        devHe22*(-2. + 1.*devHe33)));

   return exponentialdevHe31;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe12(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe12 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe12 = (0. + 4.*devHe13*devHe32 + devHe12*(8. - 4.*devHe33))/
   (8. - 4.*devHe22 - 2.*devHe13*devHe31 + 1.*devHe13*devHe22*devHe31 - 
     1.*devHe13*devHe21*devHe32 - 2.*devHe23*devHe32 - 4.*devHe33 + 
     2.*devHe22*devHe33 + devHe11*
      (-4. + 1.*devHe23*devHe32 + devHe22*(2. - 1.*devHe33) + 2.*devHe33) + 
     devHe12*(-2.*devHe21 - 1.*devHe23*devHe31 + 1.*devHe21*devHe33));

   return exponentialdevHe12;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe22(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
    using namespace amrex::literals;
    amrex::Real exponentialdevHe22 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe22 = (-8. - 4.*devHe22 + 2.*devHe13*devHe31 + 1.*devHe13*devHe22*devHe31 - 
     1.*devHe13*devHe21*devHe32 - 2.*devHe23*devHe32 + 
     devHe11*(4. + 1.*devHe23*devHe32 + devHe22*(2. - 1.*devHe33) - 
        2.*devHe33) + 4.*devHe33 + 2.*devHe22*devHe33 + 
     devHe12*(-2.*devHe21 - 1.*devHe23*devHe31 + 1.*devHe21*devHe33))/
   (-8. + 4.*devHe22 + 2.*devHe13*devHe31 - 1.*devHe13*devHe22*devHe31 + 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe12*
      (2.*devHe21 + 1.*devHe23*devHe31 - 1.*devHe21*devHe33) + 
     devHe11*(4. - 1.*devHe23*devHe32 - 2.*devHe33 + 
        devHe22*(-2. + 1.*devHe33)));

   return exponentialdevHe22;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe32(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe32 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe32 = (0. - 4.*devHe12*devHe31 + (-8. + 4.*devHe11)*devHe32)/
   (-8. + 4.*devHe22 + 2.*devHe13*devHe31 - 1.*devHe13*devHe22*devHe31 + 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe12*
      (2.*devHe21 + 1.*devHe23*devHe31 - 1.*devHe21*devHe33) + 
     devHe11*(4. - 1.*devHe23*devHe32 - 2.*devHe33 + 
        devHe22*(-2. + 1.*devHe33)));

   return exponentialdevHe32;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe13(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe13 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe13 = (0. + devHe13*(8. - 4.*devHe22) + 4.*devHe12*devHe23)/
   (8. - 4.*devHe22 - 2.*devHe13*devHe31 + 1.*devHe13*devHe22*devHe31 - 
     1.*devHe13*devHe21*devHe32 - 2.*devHe23*devHe32 - 4.*devHe33 + 
     2.*devHe22*devHe33 + devHe11*
      (-4. + 1.*devHe23*devHe32 + devHe22*(2. - 1.*devHe33) + 2.*devHe33) + 
     devHe12*(-2.*devHe21 - 1.*devHe23*devHe31 + 1.*devHe21*devHe33));

   return exponentialdevHe13;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe23(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe23 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe23 = (0. - 4.*devHe13*devHe21 + (-8. + 4.*devHe11)*devHe23)/
   (-8. + 4.*devHe22 + 2.*devHe13*devHe31 - 1.*devHe13*devHe22*devHe31 + 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe12*
      (2.*devHe21 + 1.*devHe23*devHe31 - 1.*devHe21*devHe33) + 
     devHe11*(4. - 1.*devHe23*devHe32 - 2.*devHe33 + 
        devHe22*(-2. + 1.*devHe33)));

   return exponentialdevHe23;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_exponentialdevHe33(amrex::Real const V11, amrex::Real const V12, amrex::Real const V13,
                          amrex::Real const V21, amrex::Real const V22, amrex::Real const V23,
                          amrex::Real const V31, amrex::Real const V32, amrex::Real const V33) noexcept
{
   using namespace amrex::literals;
   amrex::Real exponentialdevHe33 = 0.0_rt;

   amrex::Real devHe11 = compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe21 = compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe31 = compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe12 = compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe22 = compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe32 = compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe13 = compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe23 = compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
   amrex::Real devHe33 = compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);

   exponentialdevHe33 = (-8. + 4.*devHe22 - 2.*devHe13*devHe31 + 1.*devHe13*devHe22*devHe31 - 
     1.*devHe13*devHe21*devHe32 - 2.*devHe23*devHe32 - 4.*devHe33 + 
     2.*devHe22*devHe33 + devHe11*
      (4. + 1.*devHe23*devHe32 + devHe22*(-2. - 1.*devHe33) + 2.*devHe33) + 
     devHe12*(2.*devHe21 - 1.*devHe23*devHe31 + 1.*devHe21*devHe33))/
   (-8. + 4.*devHe22 + 2.*devHe13*devHe31 - 1.*devHe13*devHe22*devHe31 + 
     1.*devHe13*devHe21*devHe32 + 2.*devHe23*devHe32 + 4.*devHe33 - 
     2.*devHe22*devHe33 + devHe12*
      (2.*devHe21 + 1.*devHe23*devHe31 - 1.*devHe21*devHe33) + 
     devHe11*(4. - 1.*devHe23*devHe32 - 2.*devHe33 + 
        devHe22*(-2. + 1.*devHe33)));

   return exponentialdevHe33;
}

}

#endif