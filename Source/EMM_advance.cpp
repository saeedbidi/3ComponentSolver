
#include "EMM.H"
#include "EMM_hydro_K.H"
#include "EMM_K.H"
#include <armadillo>

using namespace amrex;

Real
EMM::advance (Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EMM::advance()");

    for (int i = 0; i < num_state_data_types; ++i) {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);
    MultiFab dSdt(grids,dmap,NUM_STATE,0,MFInfo(),Factory());
    MultiFab Sborder(grids,dmap,NUM_STATE,NUM_GROW,MFInfo(),Factory());

    FluxRegister* fr_as_crse = nullptr;
    if (do_reflux && level < parent->finestLevel()) {
        EMM& fine_level = getLevel(level+1);
        fr_as_crse = fine_level.flux_reg.get();
    }

    FluxRegister* fr_as_fine = nullptr;
    if (do_reflux && level > 0) {
        fr_as_fine = flux_reg.get();
    }

    if (fr_as_crse) {
        fr_as_crse->setVal(0.0_rt);
    }

    // 1st order Time Integration
    FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    compute_dSdt(Sborder, dSdt, dt, iteration, fr_as_crse, fr_as_fine, time);
    MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    // compute_thermodynamics(S_new, Sborder);
    PlasticityUpdate(S_new, dt);

    // Time Integration with Runge-Kutta or Euler Half Time-Stepping
    // Half Time Stepping:
    // FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, 0.5_rt*dt, iteration, fr_as_crse, fr_as_fine, time);
    // MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, 0.5_rt*dt, dSdt, 0, 0, NUM_STATE, 0);
    // compute_thermodynamics(S_new, Sborder);
    // PlasticityUpdate(S_new, dt);

    // FillPatch(*this, Sborder, NUM_GROW, time+0.5_rt*dt, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, 0.5_rt*dt, iteration, fr_as_crse, fr_as_fine, time);
    // MultiFab::LinComb(S_new, 1.0_rt, S_old, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    // compute_thermodynamics(S_new, S_old);
    // PlasticityUpdate(S_new, dt);

    // Runge Kutta 2:
    // RK2 stage 1
    // FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, iteration, fr_as_crse, fr_as_fine, time);
    // // U^* = U^n + dt*dUdt^n
    // MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    // PlasticityUpdate(S_new, dt);

    // RK2 stage 2
    // After fillpatch Sborder = U^n+dt*dUdt^n
    // FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, 0.5_rt*dt, iteration, fr_as_crse, fr_as_fine, time);
    // // S_new = 0.5*(Sborder+S_old) = U^n + 0.5*dt*dUdt^n
    // MultiFab::LinComb(S_new, 0.5_rt, Sborder, 0, 0.5_rt, S_old, 0, 0, NUM_STATE, 0);
    // // S_new += 0.5*dt*dSdt
    // MultiFab::Saxpy(S_new, 0.5_rt*dt, dSdt, 0, 0, NUM_STATE, 0);
    // PlasticityUpdate(S_new, dt);
    // We now have S_new = U^{n+1} = (U^n+0.5*dt*dUdt^n) + 0.5*dt*dUdt^*
    // compute_thermodynamics(S_new, S_old);

    // //SSP RK3 stage 1
    // FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
    // MultiFab::LinComb(S_new, 1.0_rt, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);

    // //SSP RK3 stage 2
    // // After fillpatch Sborder = U^n+dt*dUdt^n
    // FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
    // MultiFab::LinComb(S_new, 0.25_rt, Sborder, 0, 0.75_rt, S_old, 0, 0, NUM_STATE, 0);
    // MultiFab::Saxpy(S_new, 0.25_rt*dt, dSdt, 0, 0, NUM_STATE, 0);

    // //SSP RK3 stage 3
    // FillPatch(*this, Sborder, NUM_GROW, time+0.5_rt*dt, State_Type, 0, NUM_STATE);
    // compute_dSdt(Sborder, dSdt, dt, fr_as_crse, fr_as_fine);
    // MultiFab::LinComb(S_new, 2.0_rt/3.0_rt, Sborder, 0, 1.0_rt/3.0_rt, S_old, 0, 0, NUM_STATE, 0);
    // MultiFab::Saxpy(S_new, (2.0_rt/3.0_rt)*dt, dSdt, 0, 0, NUM_STATE, 0);

    return dt;
}

void
EMM::compute_dSdt (MultiFab& S, MultiFab& dSdt, Real dt, int iteration,
                   FluxRegister* fr_as_crse, FluxRegister* fr_as_fine, Real time)
{
    BL_PROFILE("EMM::compute_dSdt()");

    const auto dx = geom.CellSizeArray();
    const auto dxinv = geom.InvCellSizeArray();
    const auto geomdata = geom.data();
    //auto dtdxinv = dt*dxinv;
    const int ncomp = NUM_STATE-4;
    const int nvar = NV;
    const int nprim = NPRIM;

    //Defined on the faces thanks to amrex::convert and IntVect::TheDimensionVector(idim)
    Array<MultiFab,AMREX_SPACEDIM> fluxes;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        fluxes[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), ncomp, 0);
    }

    Array<MultiFab,AMREX_SPACEDIM> qL; // May need to expand to include ghost cells
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        qL[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), ncomp, 1);
    }

    Array<MultiFab,AMREX_SPACEDIM> qR; // May need to expand to include ghost cells
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        qR[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), ncomp, 1);
    }

    // Array<MultiFab,AMREX_SPACEDIM> qL_THINC; // May need to expand to include ghost cells
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     qL_THINC[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
    //                         S.DistributionMap(), ncomp, 1);
    // }

    // Array<MultiFab,AMREX_SPACEDIM> qR_THINC; // May need to expand to include ghost cells
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     qR_THINC[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
    //                         S.DistributionMap(), ncomp, 1);
    // }

    Array<MultiFab,AMREX_SPACEDIM> US;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        US[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), 1, 0);
    }

    Array<MultiFab,AMREX_SPACEDIM> VS;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        VS[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), 1, 0);
    }

    Array<MultiFab,AMREX_SPACEDIM> WS;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        WS[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), 1, 0);
    }

    Array<MultiFab,AMREX_SPACEDIM> VF;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        VF[idim].define(amrex::convert(S.boxArray(),IntVect::TheDimensionVector(idim)),
                            S.DistributionMap(), 11, 0);
    }

    // Cell-centered for Velocity Divergence Nabla - U
    // Array<MultiFab,AMREX_SPACEDIM> H;
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     H[idim].define(S.boxArray(), S.DistributionMap(), ncomp, 0);
    // }
    // Array<MultiFab,AMREX_SPACEDIM> K;
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     K[idim].define(S.boxArray(), S.DistributionMap(), ncomp, 0);
    // }
    // Array<MultiFab,AMREX_SPACEDIM> M;
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     M[idim].define(S.boxArray(), S.DistributionMap(), ncomp, 0);
    // }

    // Cell-centered Velocities for elastic stretch tensor Divergence Nabla - V
    // Array<MultiFab,AMREX_SPACEDIM> UCC;
    // for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    //     UCC[idim].define(S.boxArray(), S.DistributionMap(), 9, 0);
    // }

    Parm const* lparm = parm.get();

    FArrayBox qtmp, UCCtmp, HCCtmp;
    for (MFIter mfi(S); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        auto sfab = S.array(mfi);
        auto const& dsdtfab = dSdt.array(mfi);

        AMREX_D_TERM(auto const& fxfab = fluxes[0].array(mfi);,
                     auto const& fyfab = fluxes[1].array(mfi);,
                     auto const& fzfab = fluxes[2].array(mfi););

        // Reconstructed States
        AMREX_D_TERM(auto const& qLxfab = qL[0].array(mfi);, // May need to expand to include ghost cells
                     auto const& qLyfab = qL[1].array(mfi);,
                     auto const& qLzfab = qL[2].array(mfi););

        AMREX_D_TERM(auto const& qRxfab = qR[0].array(mfi);, // May need to expand to include ghost cells
                     auto const& qRyfab = qR[1].array(mfi);,
                     auto const& qRzfab = qR[2].array(mfi););

        // Reconstructed States
        // AMREX_D_TERM(auto const& qLxfab_THINC = qL_THINC[0].array(mfi);, // May need to expand to include ghost cells
        //              auto const& qLyfab_THINC = qL_THINC[1].array(mfi);,
        //              auto const& qLzfab_THINC = qL_THINC[2].array(mfi););

        // AMREX_D_TERM(auto const& qRxfab_THINC = qR_THINC[0].array(mfi);, // May need to expand to include ghost cells
        //              auto const& qRyfab_THINC = qR_THINC[1].array(mfi);,
        //              auto const& qRzfab_THINC = qR_THINC[2].array(mfi););

        // Face velocities provied by the Riemann Solver
        AMREX_D_TERM(auto const& USxfab = US[0].array(mfi);,
                     auto const& USyfab = US[1].array(mfi);,
                     auto const& USzfab = US[2].array(mfi););

        AMREX_D_TERM(auto const& VSxfab = VS[0].array(mfi);,
                     auto const& VSyfab = VS[1].array(mfi);,
                     auto const& VSzfab = VS[2].array(mfi););

        AMREX_D_TERM(auto const& WSxfab = WS[0].array(mfi);,
                     auto const& WSyfab = WS[1].array(mfi);,
                     auto const& WSzfab = WS[2].array(mfi););

        AMREX_D_TERM(auto const& VFxfab = VF[0].array(mfi);,
                     auto const& VFyfab = VF[1].array(mfi);,
                     auto const& VFzfab = VF[2].array(mfi););

        //
        // AMREX_D_TERM(auto const& Hxfab = H[0].array(mfi);,
        //              auto const& Hyfab = H[1].array(mfi);,
        //              auto const& Hzfab = H[2].array(mfi););

        // AMREX_D_TERM(auto const& Kxfab = K[0].array(mfi);,
        //              auto const& Kyfab = K[1].array(mfi);,
        //              auto const& Kzfab = K[2].array(mfi););

        // AMREX_D_TERM(auto const& Mxfab = M[0].array(mfi);,
        //              auto const& Myfab = M[1].array(mfi);,
        //              auto const& Mzfab = M[2].array(mfi););

        // AMREX_D_TERM(auto const& UCCxfab = UCC[0].array(mfi);,
        //              auto const& UCCyfab = UCC[1].array(mfi);,
        //              auto const& UCCzfab = UCC[2].array(mfi););

        const Box& bxg2 = amrex::grow(bx,2);
        qtmp.resize(bxg2, nprim);
        
        Elixir qeli = qtmp.elixir();
        auto const& q = qtmp.array();
        
        UCCtmp.resize(bx, 11);
        Elixir UCCeli = UCCtmp.elixir();
        auto const& UCC = UCCtmp.array();

        HCCtmp.resize(bx, 11);
        Elixir HCCeli = HCCtmp.elixir();
        auto const& HCC = HCCtmp.array();

        // Print() << "Computing Primitive Variables" << "\n";
        amrex::ParallelFor(bxg2, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_ctoprim(i, j, k, sfab, q, geomdata, *lparm);
        });

        // Print() << "Computing non-conservative terms" << "\n";
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real c1sq = amrex::max(compute_SoS(q(i,j,k,QPRES), sfab(i,j,k,GT1), q(i,j,k,QRHO1), 1) + (4.0_rt/3.0_rt)*parm->eos_G01/parm->eos_rho01, parm->smallr);
            amrex::Real c2sq = amrex::max(compute_SoS(q(i,j,k,QPRES), sfab(i,j,k,GT2), q(i,j,k,QRHO2), 2) + (4.0_rt/3.0_rt)*parm->eos_G02/parm->eos_rho02, parm->smallr);
            amrex::Real c3sq = amrex::max(compute_SoS(q(i,j,k,QPRES), sfab(i,j,k,GT3), q(i,j,k,QRHO3), 3) + (4.0_rt/3.0_rt)*parm->eos_G03/parm->eos_rho03, parm->smallr);
            // amrex::Real c1sq = amrex::max(compute_SoS(q(i,j,k,QPRES), 300.0, q(i,j,k,QRHO1), 1), parm->smallr);
            // amrex::Real c2sq = amrex::max(compute_SoS(q(i,j,k,QPRES), 300.0, q(i,j,k,QRHO2), 2), parm->smallr);
            // amrex::Real c3sq = amrex::max(compute_SoS(q(i,j,k,QPRES), 300.0, q(i,j,k,QRHO3), 3), parm->smallr);

            amrex::Real alpha1 = q(i,j,k,QALPHA1);
            amrex::Real alpha2 = q(i,j,k,QALPHA2);
            amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
            amrex::Real rho1 = q(i,j,k,QRHO1); amrex::Real rho2 = q(i,j,k,QRHO2); amrex::Real rho3 = q(i,j,k,QRHO3);
            amrex::Real mrho = alpha1*rho1+alpha2*rho2+alpha3*rho3;
            amrex::Real cpsq = (1.0_rt)/(mrho*( alpha1/(rho1*c1sq) + alpha2/(rho2*c2sq) + alpha3/(rho3*c3sq) ));
            amrex::Real K1 = 0.0;
            amrex::Real K2 = 0.0;
            // if(iteration >= parm->K_activation){
            if(time >= parm->K_time){
                K1 = alpha1*( (mrho*cpsq)/(rho1*c1sq) - 1.0_rt);
                K2 = alpha2*( (mrho*cpsq)/(rho2*c2sq) - 1.0_rt);
            }

            HCC(i, j, k,0) = -(2.0/3.0)*q(i, j, k,QV11);
            HCC(i, j, k,1) = -(2.0/3.0)*q(i, j, k,QV21);
            HCC(i, j, k,2) = -(2.0/3.0)*q(i, j, k,QV31);
            HCC(i, j, k,3) = -(2.0/3.0)*q(i, j, k,QV12);
            HCC(i, j, k,4) = -(2.0/3.0)*q(i, j, k,QV22);
            HCC(i, j, k,5) = -(2.0/3.0)*q(i, j, k,QV32);
            HCC(i, j, k,6) = -(2.0/3.0)*q(i, j, k,QV13);
            HCC(i, j, k,7) = -(2.0/3.0)*q(i, j, k,QV23);
            HCC(i, j, k,8) = -(2.0/3.0)*q(i, j, k,QV33);
            HCC(i, j, k,9)  = - q(i, j, k,QALPHA1) - K1;
            HCC(i, j, k,10) = - q(i, j, k,QALPHA2) - K2;

            UCC(i, j, k,0) = q(i, j, k,QU);
            UCC(i, j, k,1) = q(i, j, k,QV);
            UCC(i, j, k,2) = q(i, j, k,QW);
            UCC(i, j, k,3) = q(i, j, k,QU);
            UCC(i, j, k,4) = q(i, j, k,QV);
            UCC(i, j, k,5) = q(i, j, k,QW);
            UCC(i, j, k,6) = q(i, j, k,QU);
            UCC(i, j, k,7) = q(i, j, k,QV);
            UCC(i, j, k,8) = q(i, j, k,QW);
            UCC(i, j, k,9) = 0.0;
            UCC(i, j, k,10) = 0.0;

            // Kyfab(i, j, k,UARHO1) = 0.0_rt;
            // Kyfab(i, j, k,UARHO2) = 0.0_rt;
            // Kyfab(i, j, k,UMX) = 0.0_rt;
            // Kyfab(i, j, k,UMY) = 0.0_rt;
            // Kyfab(i, j, k,UMZ) = 0.0_rt;
            // Kyfab(i, j, k,URHOE) = 0.0_rt;
            // Kyfab(i, j, k,GALPHA) = - q(i, j, k,QALPHA) - K;
            // Kyfab(i, j, k,GV11) = 0.0_rt;
            // Kyfab(i, j, k,GV21) = 0.0_rt;
            // Kyfab(i, j, k,GV31) = 0.0_rt;
            // Kyfab(i, j, k,GV12) = 0.0_rt;
            // Kyfab(i, j, k,GV22) = 0.0_rt;
            // Kyfab(i, j, k,GV32) = 0.0_rt;
            // Kyfab(i, j, k,GV13) = 0.0_rt;
            // Kyfab(i, j, k,GV23) = 0.0_rt;
            // Kyfab(i, j, k,GV33) = 0.0_rt;

#if (AMREX_SPACEDIM == 3)
            // Mzfab(i, j, k,UARHO1) = 0.0_rt;
            // Mzfab(i, j, k,UARHO2) = 0.0_rt;
            // Mzfab(i, j, k,UMX) = 0.0_rt;
            // Mzfab(i, j, k,UMY) = 0.0_rt;
            // Mzfab(i, j, k,UMZ) = 0.0_rt;
            // Mzfab(i, j, k,URHOE) = 0.0_rt;
            // Mzfab(i, j, k,GALPHA) = - q(i, j, k,QALPHA) - K ;
            // Mzfab(i, j, k,GV11) = 0.0_rt;
            // Mzfab(i, j, k,GV21) = 0.0_rt;
            // Mzfab(i, j, k,GV31) = 0.0_rt;
            // Mzfab(i, j, k,GV12) = 0.0_rt;
            // Mzfab(i, j, k,GV22) = 0.0_rt;
            // Mzfab(i, j, k,GV32) = 0.0_rt;
            // Mzfab(i, j, k,GV13) = 0.0_rt;
            // Mzfab(i, j, k,GV23) = 0.0_rt;
            // Mzfab(i, j, k,GV33) = 0.0_rt;
#endif
        });
        // const Box& bxg1 = amrex::grow(bx,1);
        
        // qLtmp.resize(bxg1,nprim);
        // Elixir qLeli = qLtmp.elixir();
        // auto const& qL = qLtmp.array();
        
        // qRtmp.resize(bxg1,nprim);
        // Elixir qReli = qRtmp.elixir();
        // auto const& qR = qRtmp.array();

        // x-direction
        int cdir = 0;
        const Box& xslpbx = amrex::grow(bx, cdir, 1); // i j k running on the cell centers here
        // Print() << "Computing PMUSCL and THINC X" << "\n";
        amrex::ParallelFor(xslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_PMUSCL_reconstruct_x(i, j, k, qLxfab, qRxfab, q, dt, geomdata, dxinv, *lparm);
            //EMM_THINC_reconstruct_x(i, j, k, qLxfab_THINC, qRxfab_THINC, q, nprim);
        });
        // amrex::ParallelFor(xslpbx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     //EMM_PMUSCL_reconstruct_x(i, j, k, qLxfab, qRxfab, q, dt, dxinv, *lparm);
        //     EMM_THINC_reconstruct_x(i, j, k, qLxfab_THINC, qRxfab_THINC, q, nprim);
        // });
        // Print() << "Computing TBV X" << "\n";
        // amrex::ParallelFor(bx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     EMM_TBV_x(i, j, k, qLxfab, qRxfab, qLxfab_THINC, qRxfab_THINC, q, nprim);
        // });
        // Print() << "Computing Riemann X" << "\n";
        const Box& xflxbx = amrex::surroundingNodes(bx,cdir); // i j k running on the face centers here
        amrex::ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_riemann_x(i, j, k, fxfab, USxfab, VSxfab, WSxfab, VFxfab, qLxfab, qRxfab, q, sfab, *lparm);
        });

        // y-direction
        cdir = 1;
        const Box& yslpbx = amrex::grow(bx, cdir, 1);
        // Print() << "Computing PMUSCL and THINC Y" << "\n";
        amrex::ParallelFor(yslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_PMUSCL_reconstruct_y(i, j, k, qLyfab, qRyfab, q, dt, geomdata, dxinv, *lparm);
            //EMM_THINC_reconstruct_y(i, j, k, qLyfab_THINC, qRyfab_THINC, q, nprim);
        });
        // Print() << "Computing TBV Y" << "\n";
        // amrex::ParallelFor(bx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     EMM_TBV_y(i, j, k, qLyfab, qRyfab, qLyfab_THINC, qRyfab_THINC, q, nprim);
        // });
        // Print() << "Computing Riemann Y" << "\n";
        const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
        amrex::ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_riemann_y(i, j, k, fyfab, USyfab, VSyfab, WSyfab, VFyfab, qLyfab, qRyfab, q, sfab, *lparm);
        });

        // z-direction
#if (AMREX_SPACEDIM == 3)
        cdir = 2;
        const Box& zslpbx = amrex::grow(bx, cdir, 1);
        //Print() << "Computing PMUSCL and THINC Z" << "\n";
        amrex::ParallelFor(zslpbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_PMUSCL_reconstruct_z(i, j, k, qLzfab, qRzfab, q, dt, geomdata, dxinv, *lparm);
            //EMM_THINC_reconstruct_z(i, j, k, qLzfab_THINC, qRzfab_THINC, q, nprim);
        });
        // Print() << "Computing TBV Z" << "\n";
        // amrex::ParallelFor(bx,
        // [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        // {
        //     EMM_TBV_z(i, j, k, qLzfab, qRzfab, qLzfab_THINC, qRzfab_THINC, q, nprim);
        // });
        const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
        //Print() << "Computing Riemann Z" << "\n";
        amrex::ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            EMM_riemann_z(i, j, k, fzfab, USzfab, VSzfab, WSzfab, VFzfab, qLzfab, qRzfab, q, sfab, *lparm);
        });
#endif

        amrex::ParallelFor(bx, NV,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            EMM_flux_to_dudt(i, j, k, n, dsdtfab, HCC, UCC,
                            AMREX_D_DECL(fxfab,fyfab,fzfab),
                            AMREX_D_DECL(USxfab,USyfab,USzfab),
                            AMREX_D_DECL(VSxfab,VSyfab,VSzfab),
                            AMREX_D_DECL(WSxfab,WSyfab,WSzfab), 
                            AMREX_D_DECL(VFxfab,VFyfab,VFzfab), dxinv);
            dsdtfab(i,j,k,GPRESS) = 0.0_rt;
            dsdtfab(i,j,k,GT1) = 0.0_rt;
            dsdtfab(i,j,k,GT2) = 0.0_rt;
            dsdtfab(i,j,k,GT3) = 0.0_rt;
        });

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
             EMM_axisymmetricAdd(i, j, k, dt, iteration, dsdtfab, q, sfab, geomdata, *lparm, time);
        });

        // don't have to do this, but we could
        qeli.clear();
        UCCeli.clear();
        HCCeli.clear(); // don't need them anymore
        //qLeli.clear();
        //qReli.clear();
    }

    if (fr_as_crse) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
            const Real scale = -dt*dA;
            fr_as_crse->CrseInit(fluxes[idim], idim, 0, 0, NCONS, scale, FluxRegister::ADD);
        }
    }

    if (fr_as_fine) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real dA = (idim == 0) ? dx[1]*dx[2] : ((idim == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
            const Real scale = dt*dA;
            fr_as_fine->FineAdd(fluxes[idim], idim, 0, 0, NCONS, scale);
        }
    }
}

void
EMM::compute_thermodynamics (MultiFab& Snew, MultiFab& Sborder)
{
    BL_PROFILE("EMM::thermodynamics()");
    //using namespace amrex::literals;

    Parm const* lparm = parm.get();
    const auto geomdata = geom.data();
    // MultiFab& cost = get_new_data(Cost_Type);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Snew,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // amrex::Real wt = amrex::second();
        //const Box& bx = mfi.growntilebox(ng);
        const Box& bx = mfi.tilebox();
        auto snewfab = Snew.array(mfi);
        auto soldfab = Sborder.array(mfi);

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
                EMM_thermo(i, j, k, snewfab, soldfab, *lparm);
        });
        // wt = (amrex::second() - wt) / bx.d_numPts();
        // cost[mfi].plus<RunOn::Host>(wt, bx);
    }
}

void
EMM::PlasticityUpdate (MultiFab& S, amrex::Real dt)
{
    BL_PROFILE("EMM::PlasticityUpdate()");
    //using namespace amrex::literals;

    Parm const* lparm = parm.get();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto sfab = S.array(mfi);
        
        int plasticity_on = 0;
         
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real alpha1 = sfab(i,j,k,GALPHA1); amrex::Real alpha2 = sfab(i,j,k,GALPHA2); amrex::Real alpha3 = 1.0_rt - alpha1 - alpha2;
            amrex::Real mrho = sfab(i,j,k,UARHO1)+sfab(i,j,k,UARHO2)+sfab(i,j,k,UARHO3);
            amrex::Real rho1 = sfab(i,j,k,UARHO1)/alpha1;
            amrex::Real rho2 = sfab(i,j,k,UARHO2)/alpha2;
            amrex::Real rho3 = sfab(i,j,k,UARHO3)/alpha3;
            amrex::Real ux = sfab(i,j,k,UMX)/mrho;
            amrex::Real uy = sfab(i,j,k,UMY)/mrho;
            amrex::Real uz = sfab(i,j,k,UMZ)/mrho;
            amrex::Real e = sfab(i,j,k,URHOE)/mrho - 0.5_rt*(ux*ux + uy*uy + uz*uz);
            amrex::Real PStrainP = sfab(i,j,k,UARP)/sfab(i,j,k,UARHO1);
            amrex::Real V11 = sfab(i,j,k,GV11); amrex::Real V12 = sfab(i,j,k,GV12); amrex::Real V13 = sfab(i,j,k,GV13);
            amrex::Real V21 = sfab(i,j,k,GV21); amrex::Real V22 = sfab(i,j,k,GV22); amrex::Real V23 = sfab(i,j,k,GV23);
            amrex::Real V31 = sfab(i,j,k,GV31); amrex::Real V32 = sfab(i,j,k,GV32); amrex::Real V33 = sfab(i,j,k,GV33);

            // Reinstates symmetry for flows without plasticity:
            arma::mat Ve = {{V11, V12, V13}, {V21, V22, V23},  {V31, V32, V33}};

            arma::mat VeVet = Ve*Ve.t();
            arma::mat U;
            arma::vec s;
            arma::mat V;

            arma::svd(U,s,V,VeVet);
            s(0) = std::sqrt(s(0));
            s(1) = std::sqrt(s(1));
            s(2) = std::sqrt(s(2));

            // arma::mat Vef = U*diagmat(s)*V.t();
            // sfab(i,j,k,GV11) = Vef(0,0);
            // sfab(i,j,k,GV21) = Vef(1,0);
            // sfab(i,j,k,GV31) = Vef(2,0);
            // sfab(i,j,k,GV12) = Vef(0,1);
            // sfab(i,j,k,GV22) = Vef(1,1);
            // sfab(i,j,k,GV32) = Vef(2,1);
            // sfab(i,j,k,GV13) = Vef(0,2);
            // sfab(i,j,k,GV23) = Vef(1,2);
            // sfab(i,j,k,GV33) = Vef(2,2);
            // amrex::Real I2    = compute_I2(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // amrex::Real I2np1 = I2;

            // if(I2 != 0.0 && plasticity_on == 1){
            //     // Idealised Plasticity Update:
            //     amrex::Real h1 = std::log(s(0));
            //     amrex::Real h2 = std::log(s(1));
            //     amrex::Real h3 = std::log(s(2));

            //     // Compute the mixture Shear:
            //     amrex::Real G1 = parm->eos_G01*std::pow(rho1/parm->eos_rho01,parm->eos_Beta1+1.0_rt);
            //     amrex::Real G2 = parm->eos_G02*std::pow(rho2/parm->eos_rho02,parm->eos_Beta2+1.0_rt);
            //     amrex::Real G = (alpha1*G1/parm->eos_Gama1 + alpha2*G2/parm->eos_Gama2)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2);
            //     // Compute the deviator of the stress tensor:
            //     amrex::Real Sigma11 = 2.0*G*compute_devHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma21 = 2.0*G*compute_devHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma31 = 2.0*G*compute_devHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma12 = 2.0*G*compute_devHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma22 = 2.0*G*compute_devHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma32 = 2.0*G*compute_devHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma13 = 2.0*G*compute_devHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma23 = 2.0*G*compute_devHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     amrex::Real Sigma33 = 2.0*G*compute_devHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            //     // Compute the Frobenius norm:
            //     amrex::Real FrobeniusNormStress = std::sqrt(Sigma11*Sigma11 + Sigma21*Sigma21 + Sigma31*Sigma31
            //                                               + Sigma12*Sigma12 + Sigma22*Sigma22 + Sigma32*Sigma32
            //                                               + Sigma13*Sigma13 + Sigma23*Sigma23 + Sigma33*Sigma33);
            //     Print() << "FrobeniusNormStress: " << FrobeniusNormStress << "\n";
            //     Print() << "PStrainP: " << PStrainP << "\n";
            //     Print() << "parm->eos_c1: " << parm->eos_c1 << "\n";
            //     Print() << "parm->eos_c2: " << parm->eos_c2 << "\n";
            //     Print() << "parm->eos_c3: " << parm->eos_c3 << "\n";
            //     Print() << "parm->eos_X0: " << parm->eos_X0 << "\n";
            //     Print() << "parm->eos_n: " << parm->eos_n << "\n";
            //     // Yield Stress
            //     // amrex::Real SigmaY = 0.2976E9;
            //     // Compute the new second invariant
                
            //     // if(std::sqrt(1.0_rt/3.0)*FrobeniusNormStress - SigmaY > 0.0){
            //     // if(std::sqrt(3.0)*G*I2 - SigmaY > 0.0){
            //     //     Print() << "PLASTICITY ACTIVATED" << "\n";
            //     //     I2np1 = SigmaY/(G*std::sqrt(3.0));
            //     // }
            //     // I2np1 = (alpha1*I2np1/parm->eos_Gama1)/(alpha1/parm->eos_Gama1 + alpha2/parm->eos_Gama2);
            //     // Work-hardening materials:
            //     amrex::Real f = 1E-9_rt;
            //     amrex::Real df = 1E9_rt;
            //     amrex::Real I2f = I2;
            //     amrex::Real I2fp = I2;
            //     Print() << "exponential: " << std::exp((-1.0_rt + (std::sqrt(3.0/2.0)*FrobeniusNormStress)/(parm->eos_c1 + parm->eos_c2*std::pow(PStrainP + std::sqrt(2.0/3.0)*(-I2f + I2),parm->eos_n)))/parm->eos_c3) << "\n";
            //     amrex::Real omega = 0.1_rt;
            //     int iteration = 0;
            //     // Print() << "PLASTICITY ACTIVATED 1" << "\n";
            //     do{
            //         f =  I2f - I2 + dt*parm->eos_X0*std::exp((1.0_rt/parm->eos_c3)*( (std::sqrt(3.0/2.0)*FrobeniusNormStress)/(parm->eos_c1+parm->eos_c2*std::pow(PStrainP-std::sqrt(2.0/3.0)*(I2f-I2),parm->eos_n)) - 1.0_rt));
                    
            //         df = 1.0_rt + (parm->eos_c2*dt*std::exp((-1.0_rt + (std::sqrt(3.0/2.0)*FrobeniusNormStress)/(parm->eos_c1 + parm->eos_c2*std::pow(PStrainP + std::sqrt(2.0/3.0)*(-I2f + I2),parm->eos_n)))/parm->eos_c3)*FrobeniusNormStress*std::pow(PStrainP + std::sqrt(2.0/3.0)*(-I2f + I2),-1.0_rt + parm->eos_n)*parm->eos_n*parm->eos_X0)/
            //         (parm->eos_c3*std::pow(parm->eos_c1 + parm->eos_c2*std::pow(PStrainP + std::sqrt(2.0/3.0)*(-I2f + I2),parm->eos_n),2.0));
                    
            //         Print() << "I2f: " << I2f << "\n";
            //         Print() << "I2:  " << I2 << "\n";
            //         Print() << "df:  " << df << "\n";
                    
            //         if(df == 0.0){f = 1E-40_rt; df = 1.0_rt;}
            //         I2f = I2fp - f/df;
            //         I2fp = I2f;

            //         iteration++;
            //         if(iteration==500){
            //             Print() << "NR FAILED" << "\n";
            //             break;
            //         }
            //     }while(std::fabs(f)>1e-10_rt);
            //     I2np1 = I2f;
            //     // Print() << "PLASTICITY ACTIVATED 2" << "\n";
            //     // Update the diag(h) with the values of the second invariant
            //     amrex::Real h1np1 = h1*I2np1/I2;
            //     amrex::Real h2np1 = h2*I2np1/I2;
            //     amrex::Real h3np1 = h3*I2np1/I2;
            //     // Print() << "PLASTICITY ACTIVATED 3" << "\n";
            //     // Update the diag(k) with the values of the hnp1
            //     s(0) = std::exp(h1np1);
            //     s(1) = std::exp(h2np1);
            //     s(2) = std::exp(h3np1);
            // }
            // Compute the new strech tensor
            arma::mat Venp1 = U*diagmat(s)*V.t();

            sfab(i,j,k,UARP) -= 0.0;//std::sqrt(2.0/3.0)*(I2np1-I2);
            sfab(i,j,k,GV11) = Venp1(0,0);
            sfab(i,j,k,GV21) = Venp1(1,0);
            sfab(i,j,k,GV31) = Venp1(2,0);

            sfab(i,j,k,GV12) = Venp1(0,1);
            sfab(i,j,k,GV22) = Venp1(1,1);
            sfab(i,j,k,GV32) = Venp1(2,1);

            sfab(i,j,k,GV13) = Venp1(0,2);
            sfab(i,j,k,GV23) = Venp1(1,2);
            sfab(i,j,k,GV33) = Venp1(2,2);

            if(sfab(i,j,k,GALPHA2) <= 0.1){
            // if(sfab(i,j,k,GALPHA1) < 1.0_rt - 0.95){
            // if(1.0_rt - sfab(i,j,k,GALPHA1) > 0.95){
                sfab(i,j,k,GV11) = 1.0_rt;
                sfab(i,j,k,GV21) = 0.0;
                sfab(i,j,k,GV31) = 0.0;
                sfab(i,j,k,GV12) = 0.0;
                sfab(i,j,k,GV22) = 1.0_rt;
                sfab(i,j,k,GV32) = 0.0;
                sfab(i,j,k,GV13) = 0.0;
                sfab(i,j,k,GV23) = 0.0;
                sfab(i,j,k,GV33) = 1.0_rt;
            }
        });
    }
}
            // sfab(i,j,k,GV11) = compute_exponentialdevHe11(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV21) = compute_exponentialdevHe21(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV31) = compute_exponentialdevHe31(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV12) = compute_exponentialdevHe12(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV22) = compute_exponentialdevHe22(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV32) = compute_exponentialdevHe32(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV13) = compute_exponentialdevHe13(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV23) = compute_exponentialdevHe23(V11,V12,V13,V21,V22,V23,V31,V32,V33);
            // sfab(i,j,k,GV33) = compute_exponentialdevHe33(V11,V12,V13,V21,V22,V23,V31,V32,V33);
