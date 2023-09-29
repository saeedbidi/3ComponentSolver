
#include <EMM.H>
#include <EMM_K.H>
#include <EMM_tagging.H>
#include <EMM_parm.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

#include <climits>

using namespace amrex;

constexpr int EMM::NUM_GROW;

BCRec     EMM::phys_bc;

int       EMM::verbose = 0;
IntVect   EMM::hydro_tile_size {AMREX_D_DECL(1024,16,16)};
Real      EMM::cfl       = 0.3_rt;
int       EMM::do_reflux = 1;
int       EMM::refine_max_dengrad_lev   = -1;
Real      EMM::refine_dengrad           = 1.0e10_rt;
Real      EMM::refine_vofgrad           = 1.0e10_rt;

Real      EMM::gravity = 0.0_rt;

// Gas
std::vector<std::vector<double>> EMM::TVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::PVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::RHOVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::EVEC1T(376, std::vector<double>(121));
std::vector<std::vector<double>> EMM::SOSVEC1T(376, std::vector<double>(121));

// std::vector<std::vector<double>> EMM::TVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::PVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::RHOVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::EVEC1T(374, std::vector<double>(100));
// std::vector<std::vector<double>> EMM::SOSVEC1T(374, std::vector<double>(100));

EMM::EMM ()
{}

EMM::EMM (Amr&            papa,
          int             lev,
          const Geometry& level_geom,
          const BoxArray& bl,
          const DistributionMapping& dm,
          Real            time)
    : AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    if (do_reflux && level > 0) {
        flux_reg.reset(new FluxRegister(grids,dmap,crse_ratio,level,NCONS));
    }

    buildMetrics();
}

EMM::~EMM ()
{}

void
EMM::init (AmrLevel& old)
{
    auto& oldlev = dynamic_cast<EMM&>(old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev.state[State_Type].curTime();
    Real prev_time = oldlev.state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    MultiFab& S_new = get_new_data(State_Type);
    FillPatch(old,S_new,0,cur_time,State_Type,0,NUM_STATE);
}

void
EMM::init ()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/static_cast<Real>(parent->MaxRefRatio(level-1));
    setTimeLevel(cur_time,dt_old,dt);

    MultiFab& S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NUM_STATE);
}

void
EMM::initData ()
{
    BL_PROFILE("EMM::initData()");

    const auto geomdata = geom.data();
    MultiFab& S_new = get_new_data(State_Type);

    Parm const* lparm = parm.get();
    //ProbParm const* lprobparm = prob_parm.get();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.validbox();
        auto sfab = S_new.array(mfi);

        amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            emm_initdata(i, j, k, sfab, geomdata, *lparm);//, *lprobparm);
        });
    }
}

void
EMM::computeInitialDt (int                    finest_level,
                       int                    sub_cycle,
                       Vector<int>&           n_cycle,
                       const Vector<IntVect>& ref_ratio,
                       Vector<Real>&          dt_level,
                       Real                   stop_time)
{
    //
    // Grids have been constructed, compute dt for all levels.
    //
    if (level > 0) {
        return;
    }
    
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        dt_level[i] = getLevel(i).initialTimeStep();
        n_factor   *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_level[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001_rt*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0_rt) {
        if ((cur_time + dt_0) > (stop_time - eps))
            dt_0 = stop_time - cur_time;
    }
    
    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
EMM::computeNewDt (int                    finest_level,
                   int                    sub_cycle,
                   Vector<int>&           n_cycle,
                   const Vector<IntVect>& ref_ratio,
                   Vector<Real>&          dt_min,
                   Vector<Real>&          dt_level,
                   Real                   stop_time,
                   int                    post_regrid_flag)
{
    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0) {
        return;
    }

    for (int i = 0; i <= finest_level; i++)
    {
        dt_min[i] = getLevel(i).estTimeStep();
    }

    if (post_regrid_flag == 1) 
    {
	//
	// Limit dt's by pre-regrid dt
	//
	for (int i = 0; i <= finest_level; i++)
	{
	    dt_min[i] = std::min(dt_min[i],dt_level[i]);
	}
    }
    else 
    {
	//
	// Limit dt's by change_max * old dt
	//
	static Real change_max = 1.1;
	for (int i = 0; i <= finest_level; i++)
	{
	    dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
	}
    }
    
    //
    // Find the minimum over all levels
    //
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001_rt*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0_rt) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
EMM::post_regrid (int lbase, int new_finest)
{
}

void
EMM::post_timestep (int iteration)
{
    BL_PROFILE("post_timestep");

    if (do_reflux && level < parent->finestLevel()) {
        MultiFab& S = get_new_data(State_Type);
        EMM& fine_level = getLevel(level+1);
        fine_level.flux_reg->Reflux(S, 1.0_rt, 0, 0, NCONS, geom);
    }

    if (level < parent->finestLevel()) {
        avgDown();
    }
}

void
EMM::postCoarseTimeStep (Real time)
{
    BL_PROFILE("postCoarseTimeStep()");

    // This only computes sum on level 0
    if (verbose >= 2) {
        //printTotal();

        const MultiFab& S_new = get_new_data(State_Type);
        const Real cur_time = state[State_Type].curTime();

        // MultiFab alpha(S_new.boxArray(), S_new.DistributionMap(), 1, 1);
        // FillPatch(*this, alpha, alpha.nGrow(), cur_time, State_Type, Alpha1, 1, 0);

        Real bubble_radius = 0.0;
        Real Vb = S_new.sum(GALPHA1,false);

        // Vb *= 8.0_rt*geom.CellSize()[0]*geom.CellSize()[1]*geom.CellSize()[2];
        Real bubble_vol = Vb*geom.CellSize()[0]*geom.CellSize()[1];
        // bubble_radius = 2.0_rt*std::pow(bubble_vol/(M_PI),0.5_rt);
        bubble_radius = 2.0_rt*std::pow(bubble_vol/(2.0_rt*M_PI),0.5_rt);
        // bubble_radius = std::pow(3.0_rt*Vb/(4.0_rt*M_PI),1.0/3.0);
        // 1D computations:
        // bubble_radius = 0.5_rt*Vb*geom.CellSize()[0];
        // bubble_radius = 0.5_rt*Vb*geom.CellSize()[1];

        // amrex::Print().SetPrecision(18) << "\n[EMM] " << time/1E-6 << " " << bubble_radius/1e-3 << "\n";
        amrex::Print().SetPrecision(18) << "\n[EMM] " << time/1E-6 << " " << bubble_radius << " " << bubble_vol << "\n";
    }
}

void
EMM::printTotal () const
{
    const MultiFab& S_new = get_new_data(State_Type);
    std::array<Real,5> tot;
    for (int comp = 0; comp < 5; ++comp) {
        tot[comp] = S_new.sum(comp,true) * geom.ProbSize();
    }
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
            // ParallelDescriptor::ReduceRealSum(tot.data(), 5, ParallelDescriptor::IOProcessorNumber());
            // amrex::Print().SetPrecision(17) << "\n[EMM] Total mass       is " << tot[0] << "\n"
            //                                 <<   "      Total x-momentum is " << tot[1] << "\n"
            //                                 <<   "      Total y-momentum is " << tot[2] << "\n"
            //                                 <<   "      Total z-momentum is " << tot[3] << "\n"
            //                                 <<   "      Total energy     is " << tot[4] << "\n";
#ifdef BL_LAZY
        });
#endif
}

void
EMM::post_init (Real)
{
    if (level > 0) return;
    for (int k = parent->finestLevel()-1; k >= 0; --k) {
        getLevel(k).avgDown();
    }

    if (verbose >= 2) {
        printTotal();
    }
}

void
EMM::post_restart ()
{
}

void
EMM::errorEst (TagBoxArray& tags, int, int, Real time, int, int)
{
    BL_PROFILE("EMM::errorEst()");

    if (level < refine_max_dengrad_lev)
    {
        MultiFab S_new(get_new_data(State_Type).boxArray(),get_new_data(State_Type).DistributionMap(), NUM_STATE, 1);
        const Real cur_time = state[State_Type].curTime();
        FillPatch(*this, S_new, S_new.nGrow(), cur_time, State_Type, Arho1, NUM_STATE, 0);

        const char   tagval = TagBox::SET;
        const Real dengrad_threshold = refine_dengrad;
        const Real vofgrad_threshold = refine_vofgrad;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto Sfab = S_new.array(mfi);
            auto tag = tags.array(mfi);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                EMM_tag_denerror(i, j, k, tag, Sfab, dengrad_threshold, vofgrad_threshold, tagval);
            });
        }
    }
}

void
EMM::read_params ()
{
    ParmParse pp("emm");

    pp.query("v", verbose);
 
    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
	for (int i=0; i<AMREX_SPACEDIM; i++) hydro_tile_size[i] = tilesize[i];
    }
   
    pp.query("cfl", cfl);

    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    pp.query("do_reflux", do_reflux);

    pp.query("refine_max_dengrad_lev", refine_max_dengrad_lev);
    pp.query("refine_dengrad", refine_dengrad);
    pp.query("refine_vofgrad", refine_vofgrad);

    // pp.query("eos_Gama1", parm->eos_Gama1);
    // pp.query("eos_Gama2", parm->eos_Gama2);
    // pp.query("eos_Gama3", parm->eos_Gama3);
    // pp.query("eos_K01", parm->eos_K01);
    // pp.query("eos_K02", parm->eos_K02);
    pp.query("eos_rho01", parm->eos_rho01);
    pp.query("eos_rho02", parm->eos_rho02);
    pp.query("eos_rho03", parm->eos_rho03);
    pp.query("eos_G01", parm->eos_G01);
    pp.query("eos_G02", parm->eos_G02);
    pp.query("eos_G03", parm->eos_G03);
    // pp.query("eos_alpha1", parm->eos_alpha1);
    // pp.query("eos_alpha2", parm->eos_alpha2);
    // pp.query("eos_Beta1", parm->eos_Beta1);
    // pp.query("eos_Beta2", parm->eos_Beta2);
    // pp.query("eos_T01", parm->eos_T01);
    // pp.query("eos_T02", parm->eos_T02);
    //Plasticity parameters
    // pp.query("eos_c1", parm->eos_c1);
    // pp.query("eos_c2", parm->eos_c2);
    // pp.query("eos_c3", parm->eos_c3);
    // pp.query("eos_n", parm->eos_n);
    // pp.query("eos_X0", parm->eos_X0);

    pp.query("eos_gamma1", parm->eos_gamma1);
    pp.query("eos_gamma2", parm->eos_gamma2);
    pp.query("eos_gamma3", parm->eos_gamma3);
    pp.query("eos_pinf1", parm->eos_pinf1);
    pp.query("eos_pinf2", parm->eos_pinf2);
    pp.query("eos_pinf3", parm->eos_pinf3);
    pp.query("eos_b1", parm->eos_b1);
    pp.query("eos_b2", parm->eos_b2);
    pp.query("eos_b3", parm->eos_b3);
    pp.query("eos_q1", parm->eos_q1);
    pp.query("eos_q2", parm->eos_q2);
    pp.query("eos_q3", parm->eos_q3);
    pp.query("eos_cv1", parm->eos_cv1);
    pp.query("eos_cv2", parm->eos_cv2);
    pp.query("eos_cv3", parm->eos_cv3);

    pp.query("alpha_min", parm->alpha_min);
    pp.query("K_activation", parm->K_activation);
    pp.query("eos_gamma1", parm->eos_gamma1);
    pp.query("eos_gamma2", parm->eos_gamma2);
    pp.query("eos_pinf1", parm->eos_pinf1);
    pp.query("eos_pinf2", parm->eos_pinf2);
    pp.query("alpha_min", parm->alpha_min);

    pp.query("tabulated", parm->tabulated);
    pp.query("tabulated1", parm->tabulated1);
    pp.query("tabulated2", parm->tabulated2);
    pp.query("tabulated3", parm->tabulated3);
    pp.query("tableRows1", parm->tableRows1);
    pp.query("tableColumns1", parm->tableColumns1);
    pp.query("nnTP1", parm->nnTP1);
    pp.query("mmTP1", parm->mmTP1);

    pp.query("K_time", parm->K_time);

    // ------------------------------------------------------------------
    // TABLES FOR THE 1st PHASE
    if(parm->tabulated == 1){
        std::ifstream file1("NEWAir_EoS_RKPRe.dat");
        // std::ifstream file1("IdealGas.dat");
        // std::ifstream file1("NEWAir_EoS_Helmholtze.dat");
        
        int rc;
        rc = parm->tableRows1 * parm->tableColumns1;
        // double A1[parm->tableRows1][parm->tableColumns1];
        std::vector<std::vector<double>> A1(parm->tableRows1, std::vector<double>(parm->tableColumns1));
        // double B1[rc];
        std::vector<double> B1(rc);
        int k = 0;
        for (int i = 0; i < parm->tableRows1; i++)
        {
            for (int j = 0; j < parm->tableColumns1; j++)
            {
                file1 >> A1[i][j];
                B1[k] = A1[i][j];
                k += 1;
            }
        }
        file1.close();

        const int m1 = 45496, n1 = 10;
        // const int m1 = 37400, n1 = 10;
        // auto Vee1 = new double[m1][n1];
        std::vector<std::vector<double>> Vee1(m1, std::vector<double>(n1));
        int i = 0;
        for (int j = 0; j < parm->tableColumns1; j++)
        {
            i = 0;
            for (int k = j; k < rc; k += 10)
            {
                Vee1[i][j] = B1[k];
                i += 1;
            }
        }

        k = 0;
        for (int i = 0; i < parm->nnTP1; i++)
        {
            for (int j = 0; j < parm->mmTP1; j++)
            {
                EMM::PVEC1T[i][j] = Vee1[k][0];
                EMM::TVEC1T[i][j] = Vee1[k][1];
                EMM::RHOVEC1T[i][j] = Vee1[k][2];
                EMM::EVEC1T[i][j] = Vee1[k][3];
                EMM::SOSVEC1T[i][j] = Vee1[k][7];
                k += 1;
            }
        }

        parm->TMINT1 = EMM::TVEC1T[0][0];
        parm->PMINT1 = EMM::PVEC1T[0][0];
        parm->TMAXT1 = EMM::TVEC1T[parm->nnTP1-1][parm->mmTP1-1];
        parm->PMAXT1 = EMM::PVEC1T[parm->nnTP1-1][parm->mmTP1-1];
        parm->DT1 = std::abs(EMM::TVEC1T[0][0] - EMM::TVEC1T[1][0]);
        parm->DP1 = std::abs(EMM::PVEC1T[0][0] - EMM::PVEC1T[0][1]);

        amrex::Print().SetPrecision(18) << "TMINT1:            " << parm->TMINT1 << "\n";
        amrex::Print().SetPrecision(18) << "PMINT1:            " << parm->PMINT1 << "\n";
        amrex::Print().SetPrecision(18) << "TMAXT1:            " << parm->TMAXT1 << "\n";
        amrex::Print().SetPrecision(18) << "PMAXT1:            " << parm->PMAXT1 << "\n";
        amrex::Print().SetPrecision(18) << "DT1:               " << parm->DT1 << "\n";
        amrex::Print().SetPrecision(18) << "DP1:               " << parm->DP1 << "\n";
    }
    parm->coord_type = amrex::DefaultGeometry().Coord();
    parm->Initialize();
}

void
EMM::avgDown ()
{
    BL_PROFILE("EMM::avgDown()");

    if (level == parent->finestLevel()) return;

    auto& fine_lev = getLevel(level+1);

    MultiFab& S_crse =          get_new_data(State_Type);
    MultiFab& S_fine = fine_lev.get_new_data(State_Type);

    amrex::average_down(S_fine, S_crse, fine_lev.geom, geom,
                        0, S_fine.nComp(), parent->refRatio(level));

    const int nghost = 0;
}

void
EMM::buildMetrics ()
{
    // make sure dx == dy == dz
    const Real* dx = geom.CellSize();
    if (std::abs(dx[0]-dx[1]) > 1.e-12_rt*dx[0] 
#if (AMREX_SPACEDIM == 3)
    || std::abs(dx[0]-dx[2]) > 1.e-12_rt*dx[0]
#endif
    ) {
        amrex::Abort("EMM: must have dx == dy == dz\n");
    }
}

Real
EMM::estTimeStep ()
{
    BL_PROFILE("EMM::estTimeStep()");

    const auto dx = geom.CellSizeArray();
    const MultiFab& S = get_new_data(State_Type);
    Parm const* lparm = parm.get();

    Real estdt = amrex::ReduceMin(S, 0,
    [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) noexcept -> Real
    {
        return EMM_estdt(bx, fab, dx, *lparm);
    });

    estdt *= cfl;
    ParallelDescriptor::ReduceRealMin(estdt);
    // estdt = 1E-11_rt;
    return estdt;
}

Real
EMM::initialTimeStep ()
{
    return estTimeStep();
}