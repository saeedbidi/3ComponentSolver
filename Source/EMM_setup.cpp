
#include <EMM.H>
#include <EMM_derive.H>

using namespace amrex;

int EMM::num_state_data_types = NUM_STATE_DATA_TYPE;
std::unique_ptr<Parm> EMM::parm;
//std::unique_ptr<ProbParm> EMM::prob_parm;

static Box the_same_box (const Box& b) { return b; }
//static Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }

//
// Components are:
//  Interior, Inflow, Outflow,  Symmetry,     SlipWall,     NoSlipWall
//
static int scalar_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_even
};

static int norm_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_odd,  BCType::reflect_odd,  BCType::reflect_odd
};

static int tang_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_odd
};

static
void
set_scalar_bc (BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
	bc.setLo(i,scalar_bc[lo_bc[i]]);
	bc.setHi(i,scalar_bc[hi_bc[i]]);
    }
}

static
void
set_x_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,norm_vel_bc[lo_bc[0]]);
    bc.setHi(0,norm_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

static
void
set_y_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)    
    bc.setLo(1,norm_vel_bc[lo_bc[1]]);
    bc.setHi(1,norm_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

static
void
set_z_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,norm_vel_bc[lo_bc[2]]);
    bc.setHi(2,norm_vel_bc[hi_bc[2]]);
#endif
}

void
EMM::variableSetUp ()
{
    parm.reset(new Parm{}); // This is deleted in EMM::variableCleanUp().
    //prob_parm.reset(new ProbParm{});

    read_params();

    bool state_data_extrap = false;
    bool store_in_checkpoint = true;
    desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
			   StateDescriptor::Point,NUM_GROW,NUM_STATE,
			   &cell_cons_interp,state_data_extrap,store_in_checkpoint);

    Vector<BCRec>       bcs(NUM_STATE);
    Vector<std::string> name(NUM_STATE);
    BCRec bc;
    int cnt = 0;
           set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "arho1";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "arho2";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "arho3";
    cnt++; set_x_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "xmom";
    cnt++; set_y_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "ymom";
    cnt++; set_z_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "zmom";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "rho_E";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "arp";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V11";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V21";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V31";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V12";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V22";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V32";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V13";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V23";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "V33";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "alpha1";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "alpha2";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "press";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "T1";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "T2";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "T3";

    StateDescriptor::BndryFunc bndryfunc(EMM_bcfill);
    bndryfunc.setRunOnGPU(true);  // I promise the bc function will launch gpu kernels.

    desc_lst.setComponent(State_Type,
			  Arho1,
			  name,
			  bcs,
              bndryfunc);

    num_state_data_types = desc_lst.size();

    // DEFINE DERIVED QUANTITIES
    // Density
    derive_lst.add("mdensity",IndexType::TheCellType(),1,
                   EMM_derden,the_same_box);
    derive_lst.addComponent("mdensity",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("mdensity",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("mdensity",desc_lst,State_Type,Arho3,1);

    // Pressure
    derive_lst.add("pressure",IndexType::TheCellType(),1,
                   EMM_derpres,the_same_box);
    derive_lst.addComponent("pressure",desc_lst,State_Type,RhoE,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,XMOM,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,YMOM,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,ZMOM,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,V33,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Press,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,T1,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,T2,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,T3,1);

    // derive_lst.add("pc",IndexType::TheCellType(),1,
    //                EMM_derpc,the_same_box);
    // derive_lst.addComponent("pc",desc_lst,State_Type,Alpha1,1);
    // derive_lst.addComponent("pc",desc_lst,State_Type,Arho1,1);
    // derive_lst.addComponent("pc",desc_lst,State_Type,Arho2,1);

    // derive_lst.add("ps",IndexType::TheCellType(),1,
    //                EMM_derps,the_same_box);
    // derive_lst.addComponent("ps",desc_lst,State_Type,Alpha1,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,Arho1,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,Arho2,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V11,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V21,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V31,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V12,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V22,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V32,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V13,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V23,1);
    // derive_lst.addComponent("ps",desc_lst,State_Type,V33,1);

    // derive_lst.add("pt",IndexType::TheCellType(),1,
    //                EMM_derpt,the_same_box);
    // derive_lst.addComponent("pt",desc_lst,State_Type,RhoE,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,Arho1,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,Arho2,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,XMOM,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,YMOM,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,ZMOM,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,Alpha1,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V11,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V21,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V31,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V12,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V22,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V32,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V13,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V23,1);
    // derive_lst.addComponent("pt",desc_lst,State_Type,V33,1);

    derive_lst.add("elS11",IndexType::TheCellType(),1,
                   EMM_derelS11,the_same_box);
    derive_lst.addComponent("elS11",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("elS11",desc_lst,State_Type,V33,1);

    derive_lst.add("elS22",IndexType::TheCellType(),1,
                   EMM_derelS22,the_same_box);
    derive_lst.addComponent("elS22",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("elS22",desc_lst,State_Type,V33,1);

    derive_lst.add("elS33",IndexType::TheCellType(),1,
                   EMM_derelS33,the_same_box);
    derive_lst.addComponent("elS33",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("elS33",desc_lst,State_Type,V33,1);

    derive_lst.add("elS21",IndexType::TheCellType(),1,
                   EMM_derelS21,the_same_box);
    derive_lst.addComponent("elS21",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("elS21",desc_lst,State_Type,V33,1);

    derive_lst.add("elS23",IndexType::TheCellType(),1,
                   EMM_derelS23,the_same_box);
    derive_lst.addComponent("elS23",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("elS23",desc_lst,State_Type,V33,1);

    derive_lst.add("elS31",IndexType::TheCellType(),1,
                   EMM_derelS31,the_same_box);
    derive_lst.addComponent("elS31",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("elS31",desc_lst,State_Type,V33,1);

    derive_lst.add("maxPrincipalStress",IndexType::TheCellType(),1,
                   EMM_derMaxPrincipalStress,the_same_box);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("maxPrincipalStress",desc_lst,State_Type,V33,1);

    derive_lst.add("minPrincipalStress",IndexType::TheCellType(),1,
                   EMM_derMinPrincipalStress,the_same_box);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("minPrincipalStress",desc_lst,State_Type,V33,1);

    derive_lst.add("maxShearStress",IndexType::TheCellType(),1,
                   EMM_derMaxShearStress,the_same_box);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("maxShearStress",desc_lst,State_Type,V33,1);

    derive_lst.add("strainEnergy",IndexType::TheCellType(),1,
                   EMM_derStrainEnergy,the_same_box);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V11,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V21,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V31,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V12,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V22,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V32,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V13,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V23,1);
    derive_lst.addComponent("strainEnergy",desc_lst,State_Type,V33,1);

    // Velocities
    derive_lst.add("x_velocity",IndexType::TheCellType(),1,
                   EMM_dervel,the_same_box);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,XMOM,1);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("x_velocity",desc_lst,State_Type,Arho3,1);

    derive_lst.add("y_velocity",IndexType::TheCellType(),1,
                   EMM_dervel,the_same_box);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,YMOM,1);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("y_velocity",desc_lst,State_Type,Arho3,1);

    derive_lst.add("z_velocity",IndexType::TheCellType(),1,
                   EMM_dervel,the_same_box);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,ZMOM,1);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("z_velocity",desc_lst,State_Type,Arho3,1);
    
    //Volume Fraction 3
    derive_lst.add("alpha3",IndexType::TheCellType(),1,
                   EMM_deralpha3,the_same_box);
    derive_lst.addComponent("alpha3",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("alpha3",desc_lst,State_Type,Alpha2,1);

    //Temprature 1
    derive_lst.add("T1o",IndexType::TheCellType(),1,
                   EMM_derT1,the_same_box);
    derive_lst.addComponent("T1o",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("T1o",desc_lst,State_Type,Arho1,1);
    derive_lst.addComponent("T1o",desc_lst,State_Type,Press,1);
    derive_lst.addComponent("T1o",desc_lst,State_Type,T1,1);

    //Temprature 2
    derive_lst.add("T2o",IndexType::TheCellType(),1,
                   EMM_derT2,the_same_box);
    derive_lst.addComponent("T2o",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("T2o",desc_lst,State_Type,Arho2,1);
    derive_lst.addComponent("T2o",desc_lst,State_Type,Press,1);
    derive_lst.addComponent("T2o",desc_lst,State_Type,T2,1);

    //Temprature 3
    derive_lst.add("T3o",IndexType::TheCellType(),1,
                   EMM_derT3,the_same_box);
    derive_lst.addComponent("T3o",desc_lst,State_Type,Alpha1,1);
    derive_lst.addComponent("T3o",desc_lst,State_Type,Alpha2,1);
    derive_lst.addComponent("T3o",desc_lst,State_Type,Arho3,1);
    derive_lst.addComponent("T3o",desc_lst,State_Type,Press,1);
    derive_lst.addComponent("T3o",desc_lst,State_Type,T3,1);
}

void
EMM::variableCleanUp ()
{
    parm.reset();
    //prob_parm.reset();
    desc_lst.clear();
    derive_lst.clear();
}
