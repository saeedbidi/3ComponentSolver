
#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "emm_prob_parm.H"
#include "EMM.H"

extern "C" {
    void amrex_probinit (const int* init,
                         const int* name,
                         const int* namelen,
                         const amrex_real* problo,
                         const amrex_real* probhi)
    {
        //amrex::ParmParse pp("prob");

        // pp.query("p_l", EMM::prob_parm->p_l);
        // pp.query("p_r", EMM::prob_parm->p_r);
        // pp.query("rho_l", EMM::prob_parm->rho_l);
        // pp.query("rho_r", EMM::prob_parm->rho_r);
        // pp.query("u_l", EMM::prob_parm->u_l);
        // pp.query("u_r", EMM::prob_parm->u_r);
    }
}
