!_______________________________________________________________________
!_______________________________________________________________________
! ============FTT:Steel model ========== Created by JF Mercure, F Knobloch, L van Duuren, and P Vercoulen
! Follows general equations of the FTT model (e.g. Mercure Energy Policy 2012)
! Adapted to the iron and steel industry case
! 
! FTT determines the technology mix
!________________________________________jm801@cam.ac.uk________________
    
! Headers
      SUBROUTINE IDFTTS(UPDATING, &

                    !FTT-S variables start by S
                    & BSTC,SEWA,SEWB,SEWS,SWSL,SEWG, &
                    & SEWK,SWKL,SEWI,SEWW,SWWL,SEWT, &
                    & SEWE,SWIY,STRT,SEWR,SICA,SICL, &
                    & SWSA,SWKA,SXSC,SXSF,SXSR,SPSP, &
                    & SEWC,SETC,SGC1,SGC2,STIM,SCIN, &
                    & SGC3,SG1L,SGD1,SGD2,SGD3,SCMM, &
                    & SLCI,SPMA,SWII,SWIG,SDWC,SEDW, &
                    & SEEM,SRDI,SEEI,SDIS,SMED,SHS1, &
                    & SMEF,SWIC,SWFC,SOMC,SCOC,SPRI, &
                    & SJEF,SWYL,SWIL,SWGL,SCOT,SPSA, &
                    & SG2L,SG3L,SD1L,SD2L,SD3L,SCML, &
                    & SJCO,SPSL,SPRC,SPCL,RSIY,SEMR, &
                    & SEMS,SXSS,SXLT,SXLR,SXRR,SKST, &
                    & STEF,SLT2,SMPT,SHS2,SEPF,SCFA, &
                    & SWIB,SEOL,SBEL,STEI,STSC,SMPL, &
                    & SXS1,SXS2,SXS3,SXS4,BQRY,SSSR, &
                    & SPRL,SWAP,SEIA,SXIM,SXEX,SITC, &
                    & SIPR,SWGI,STGI,SCOI,SIEF,SIEI, &
                    & SJFR,SPMT, &
                    ! FTT-P input to FTT-S (energy prices in 2013$)
                    & MEWP, &
                    !E3ME inputs to FTT-S
                    & FRCT,FR02,FR03,FROT,FR05,FR06, &
                    & FRGT,FRET,FR08,FR09,FR10,FRBT,FR12, &
                    & FR01,FR04,FR07,FR11,&
                    & EX,REX,PRSC,PFR0,PFRB,PFRC,PFRE,PFRG,PFRM, &
                    & FEDS,FETS,RTCA,REPP,FRY,FRY1,KRX,PYH,PYHX, &
					& FRY2, &
                    !E3ME output from FTT-S
                    !FTT region switch
                    & JFTS,                         &
                    !classifications
                    & NST,NC5,NJ,NR,NFU,NC,NSS,NSM,NXP,NY1,&
                    & NER,NY,NQ,NBQRY,NK, &
                    !Other E3ME variables
                    & IRUN,START,ITER,DATE,     &
                    & CHECK,CALIB,CALIBF,LASTIT,NOIT)
      
                    
      IMPLICIT NONE

! Dummy arguments 
      LOGICAL,INTENT(IN) :: CALIB,CALIBF,CHECK,LASTIT,UPDATING
      INTEGER,INTENT(IN) :: DATE,ITER,START,NST,NC5,NJ,NR,NFU,NC,NSS,NSM,NXP,NY1,NY,NQ,NBQRY,NK,NER
      INTEGER,INTENT(IN) :: IRUN,NOIT                                                               
     
      !-----------FTT variables--------------
      !---Exogenous variables
      !FTT
      REAL , INTENT(IN), DIMENSION(NST,NST) :: SEWA, SEWB,SWAP                       !Exchange matrix Aij, Spillover learning matrix Bij
      REAL , INTENT(INOUT), DIMENSION(NSM,NSM) :: SLCI                             !Final, intermediate and raw material consumption matrix (In the first column the data of the correct steelmaking technology gets selected (from SLSM) and in the second column the correct ironmaking technology gets selected (from SLIM).
      REAL , INTENT(IN), DIMENSION(NST,NR) :: SEWT,SKST!,SWIB                              !Subsidies/taxes on capital investment, Regulations, investment demand for baseline scenario.
      REAL , INTENT(INOUT), DIMENSION(NST,NR) :: SWIB
      REAL , INTENT(INOUT), DIMENSION(NR) :: SPSP, SDIS                                 !Exogenous projected steel production (kton)
      REAL , INTENT(IN), DIMENSION(NST,NSS) :: STIM                            !Interaction matrix between integrated steelproduction pathways and steelmaking technologies
      REAL , INTENT(INOUT), DIMENSION(NST,NR) :: SEWR, SWSA, SRDI, SEEI                     !Regulations, absolute exog production additions, R&D investments, EE investments.
      REAL , INTENT(IN), DIMENSION(NSM,NR) :: SPMA                               !Exogenous material prices for final and intermediate products, and raw materials -> Correct coupling to E3ME materials has to be implemented.
      REAL , INTENT(INOUT), DIMENSION(NSM,NR) :: SPMT                               !Growth rates derived from E3ME variables applied to SPMA 
      REAL , INTENT(IN), DIMENSION(NSM,NSS) :: SEEM                 !Material consumption (i.e. energy efficiency) constraints
      REAL , INTENT(INOUT), DIMENSION(NSM,NSS) :: SCMM                 !Costs and material parameters for each plant type. This includes precursor plants, ironmaking plants and steelmaking plants
      REAL , INTENT(IN), DIMENSION(NSM) :: SMEF, SMED,SMPT               !Material specific emission factors and energy densities resp.
      REAL , INTENT(INOUT), DIMENSION(NR) :: SPSA, RSIY   
      REAL , INTENT(INOUT), DIMENSION(NR, NY1) :: SHS1, SHS2   !Historical steel production for the period of 1918-1967, and 1968-2019, and 2001-2050 (for future calc). Used to calculated scrap availability
      REAL , INTENT(INOUT), DIMENSION(NR,NXP) :: SSSR, SXSS, SXLT, SXLR, SXRR, SLT2 !Parameters for scrap calculation: Sectoral split, lifetimes, loss rates, and recycling rates per product group (transport, machinary, construction, and products)
      !FTT-Power (energy prices)
      REAL , INTENT(IN), DIMENSION(NJ,NR) :: MEWP  !FTT-P endogenous marginal fuel prices (2013$/GJ) (for 1-hard coal, 3-crude oil, 7-gas, 8-electricity, 11-biofuels)
      !E3ME
      REAL , INTENT(IN), DIMENSION(NR,NY1) :: RTCA
      REAL , INTENT(IN), DIMENSION(NR) :: JFTS, PRSC, EX,REX,REPP
      REAL , INTENT(IN), DIMENSION(NFU,NR) :: FETS, FEDS, FRCT,FR01,FR02,FR03,FR04,FROT,FR05,FR06,FRGT,FR07, &    !All Fuel Use variables
                                               & FRET,FR08,FR09,FR10,FR11,FRBT,FR12, &
                                               & PFR0,PFRB,PFRC,PFRE,PFRG,PFRM,FRY   !Future fuel prices (by user) (0:average, B:biofuel(empty!), C:coal, E:electricity, G:gas, M:middle distillates)
      REAL , INTENT(INOUT), DIMENSION(NFU,NR) :: FRY1, FRY2 !FRY2 is a two-year FRY lag
      REAL , INTENT(INOUT), DIMENSION(NK,NR) :: KRX
      REAL , DIMENSION(NQ,NR,NBQRY) :: BQRY
      REAL , INTENT(IN), DIMENSION(NY,NR) :: PYH
      REAL , INTENT(INOUT), DIMENSION(NY, NR) :: PYHX
      !E3ME local variable to save on first iteration
      REAL, DIMENSION(25,71), SAVE :: fr1a,fr2a,fr3a,fr4a,fr5a,fr6a,fr7a,fr8a,fr9a,f10a,f11a,f12a
    
      !---Endogenous variables
      !FTT
      REAL , INTENT(INOUT), DIMENSION(NST,NR) :: SEWS, SEWG, SEWK, &            !Capacity shares, Production, Production capacities, 
                                        & SEWE, SEWI, SWIY, SWII,  &            !Emissions, (positive) capacity additions (kton/y), cumulative investment in total and by steel industry resp.in Mln$(2008) 
                                        & SWIG, SEWC, SETC, SGC1,  &            !cumulative investment by government in Mln$(2008) ,LCOS (base) ($(2008)/tcs), LCOS incl. policies ($(2008)/tcs), LCOS, incl. policies and gamma values ($(2008)/tcs).
                                        & SGC2, SGC3, SDWC, SGD1,  &
                                        & SGD2, SGD3, SEDW,  &  
                                        & SWIC, SWFC, SOMC, SCOC,  &                !Cost component of LTLCOS of Investment costs, O&M costs, Material consumption costs, CO2 tax costs resp. 
                                        & SWKA, SCOT,SCFA, &                       !Regulation on technologies, Exog capacity share additions (-), exog capacity additions (kton/y), Scrap consumption (t/tcs)
                                        & SITC, SWGI,SCOI,SIEF,SIEI
      REAL , INTENT(INOUT), DIMENSION(NST,NR,NC5) :: BSTC                       !Technology costs matrix
      REAL , INTENT(OUT), DIMENSION(NJ,NR) :: SJEF,SJCO,SJFR                             !total raw material consumption
      REAL , INTENT(INOUT), DIMENSION(NR, NY1) :: SXS1,SXS2,SXS3,SXS4 !Endogenous calculation of sector splits.
      REAL , INTENT(IN), DIMENSION(NSM,NR) :: STRT
      REAL , INTENT(OUT), DIMENSION(NR) :: SPRL              !Average steelprice LAG
      REAL , INTENT(OUT), DIMENSION(NR) :: SPRI,SPRC,SEMR,SEIA                     !Average steelprice (based on LTLCOS),Average steelprice (based on TLCOS),Y-o-y employment rate
      REAL , INTENT(OUT) , DIMENSION(NR) :: SXSC, SXSF, SXSR,SXIM,SXEX            !Endogenous domestic scrap availability, Future scrap availability, Scrap supply, scrap imports, scrap exports
      REAL , INTENT(OUT), DIMENSION(NST) :: SEWW                                !Cumulative global steelmaking capacity for learning
      REAL , INTENT(OUT), DIMENSION(NSS) :: SICA 
      REAL , INTENT(OUT), DIMENSION(NST,NR) :: SEMS,STEF,SCIN,SEPF,SEOL,SBEL, &                    !Absolute employment in FTE by technology, tech-specific EF (tCO2/tcs), capital investment costs ($(2008)/tcs), tech-specific employemtn factors (FTE/tcs).
                                        & STEI, STSC                                        !Tech specific EI (GJ/tcs), tech specific Scrap intensity (t/tcs)
      REAL , INTENT(OUT), DIMENSION(NR) :: SIPR, STGI
      !---Real FTT Lags (Go outside of FTT)
      !FTT
      REAL , INTENT(INOUT), DIMENSION(NST) :: SWWL
      REAL , INTENT(INOUT), DIMENSION(NST,NR) :: SWSL, SWKL, SG1L,   &          !S(t-1),CAP(t-1),LTLCOS(t-1)
                                        &  SG2L, SG3L, SD1L, SD2L,   &          !TMC(t-1),TPB(t-1),dLTLCOS(t-1),dTMC(t-1)
                                        &  SD3L, SWYL, SWIL, SWGL,   &         !dTPB(t-1), Lagged investments: total, industrial, governmental
                                        &  SMPL
      REAL , INTENT(INOUT), DIMENSION(NSM,NSS) :: SCML
      REAL , INTENT(OUT), DIMENSION(NR) :: SPSL, SPCL                               !Lagged total steel production (ktcs/y)
      REAL , INTENT(OUT), DIMENSION(NSS) :: SICL
      ! FTT local variables
      !Internal lags (for the time loop)
      REAL , ALLOCATABLE, SAVE :: BSTL(:,:,:)                                   !Costs(t-1) Persisting lag NOTE: DIMENSION(NST,NR,NC5)
      REAL , ALLOCATABLE, SAVE :: FP14(:,:), isFC(:,:), &             !2008 deflator values, 2014 energy prices, fuel cost trend switch (1=exogenous,2=E3ME prices,3=FTT-P prices, other: constant)
                                & FU14A(:,:), FU14B(:,:)                        !2014 values of FTT-H and E3ME residential fuel use 
      REAL , DIMENSION(NST,NR,NC5) :: BSTLt                                     !Technology costs matrix LAG
      REAL , DIMENSION(NR) :: RHUDt, SXSFt,SPSLt ,SPCLt, SPSAt, SPSAtl                                           !UD total interpolated between FTT time steps
      REAL , DIMENSION(NST,NR) :: SWSLt, SWKLt, SG1Lt, SG2Lt, &
                                & SG3Lt, SD1Lt, SD2Lt, SD3Lt, &
                                & SWYLt, SWILt, SWGLt, SMPLt
      REAL , DIMENSION(NSM,NSS) :: SCMLt
      REAL , DIMENSION(NFU,NR) :: FRY1t
      REAL , DIMENSION(NSS) :: SICLt   
      REAL , DIMENSION(NST) :: SWWLt                                            !SEWW(t-1)
      REAL , DIMENSION(NK,NR) :: KRXt
      !Other
      REAL , ALLOCATABLE, SAVE :: PRSC13(:),FRY19(:), SPRI19(:),PYH17(:,:)                                     !2008 and 2013 deflator values
      REAL , ALLOCATABLE, SAVE :: EX13(:),REX13(:),JSCAL(:,:),og_base(:),og_ratio19(:)                                      !
      REAL , DIMENSION(NST,NJ) :: JCONV
      REAL , DIMENSION(NJ,NR) :: FCE3, FCFT, SJEA                                     !fuel prices relative to 2014 (trend defined by switch isFC), E3ME and FTT fuel prices (matched to correct fuel type)
      REAL , DIMENSION(NST,NR) :: BI, SEWP                               ! SEWP is the relative fuel price change (€ in t/€ in 2014) 
      REAL , DIMENSION(NST,NR) :: dUk, dUkSk, dUkREG, dUkKST, endo_shares, endo_capacity, endo_gen, endo_eol !corrections total, from exog changes, from regs, from mandates, endogenous shares, endog capacity, endo gen
      REAL , DIMENSION(NST,NST) :: dSij, dSEij, F, FE, K1,K2,K3,K4, KE1, KE2, KE3, KE4                                  ! Share changes
      REAL , DIMENSION(NST,NR) :: isReg, dCap, SR, PIC_RD, PIC_EE, eol_replacements_t, eol_replacements, SEWIt
      REAL , DIMENSION(NST) :: dW, dSk, dGCAP, dGCAP2,CAP_Dif, C_INV,C_SUB,C_CT,C_MT,C_DS,mp_new,mp_old, SEWI0,dW_temp
      REAL , DIMENSION(NSS) :: SICA_LR
      REAL , DIMENSION(NR) :: GCO2, FBEE, GrowthRate1, GrowthRate2,Old_Employment,P_charcoal,P_biogas,P_hydrogen
      REAL , DIMENSION(NR) :: ScrapShortage, MinScrapDemand, MaxScrapDemand,MaxScrapSupply
      REAL , DIMENSION(NR) :: og_sim, ccs_share, og_fac, demand_weight
      INTEGER :: t, I, I2, J, K, DOFB, T1,T2, MAT, PriceSwitch, ProductionSwitch 
      REAL :: dt, invdt, Fij, dFij, dFEij, FEij, dFEji, FEji, Bidon, Basegrowth,SR_C,primary_iron_demand,primary_iron_supply
      REAL :: Utot, dUtot, gr_rate_corr, tScaling
!---------------------------------------------------------------------------
!---------------Setting Variables valid all years---------------------------
!---------------------------------------------------------------------------
      
DOFB = 0 !Switch for feedback of carbon tax revenue as energy efficiency investment
ProductionSwitch = 1 !Switch between exogenous production (in kton) and endogenous production (based on y-o-y growth of steel demand in monetary funds).

!Allocate the persisting LAGS and variables
!Persisting lags and variables are not passed outside of this function but persist between calls
!It avoids us creating additional 2 and 3D E3ME variables that are not needed outside
IF (.NOT. ALLOCATED(PRSC13)) THEN
    ALLOCATE(PRSC13(NR))
    PRSC13 = 0.0
ENDIF
IF (.NOT. ALLOCATED(EX13)) THEN
    ALLOCATE(EX13(NR))
    EX13 = 0.0
ENDIF
IF (.NOT. ALLOCATED(REX13)) THEN
    ALLOCATE(REX13(NR))
    REX13 = 0.0
ENDIF
IF (.NOT. ALLOCATED(BSTL)) THEN
    ALLOCATE(BSTL(NST,NR,NC5))
    BSTL = 0.0
ENDIF
IF (.NOT. ALLOCATED(isFC)) THEN
    ALLOCATE(isFC(NJ,NR))
    isFC = 0.0
ENDIF
IF (.NOT. ALLOCATED(FRY19)) THEN
    ALLOCATE(FRY19(NR))
    FRY19 = 0.0
ENDIF
IF (.NOT. ALLOCATED(SPRI19)) THEN
    ALLOCATE(SPRI19(NR))
    SPRI19 = 0.0
ENDIF
IF (.NOT. ALLOCATED(JSCAL)) THEN
    ALLOCATE(JSCAL(NJ,NR))
    JSCAL = 0.0
ENDIF
IF (.NOT. ALLOCATED(og_base)) THEN
    ALLOCATE(og_base(NR))
    og_base = 0.0
ENDIF
IF (.NOT. ALLOCATED(og_ratio19)) THEN
    ALLOCATE(og_ratio19(NR))
    og_ratio19 = 0.0
ENDIF

!-----General variables
!Time interval (quarterly)
dt = 1.0/NOIT
invdt = NOIT
gr_rate_corr = 3.0
tScaling = 10.0
!Calculate scrap availability for each year, and the endogenous product group split (SXS1-4)
IF (ITER==1) CALL IDFTTSScrap(NST,NXP,NR,NC5,NY1,NSM,NY,NQ,NBQRY,SHS1,SHS2,SXSC,BSTC,SSSR, SXSS,SXLT,SXLR,SXRR,SLT2,SPSA,SXS1,SXS2,SXS3,SXS4,BQRY,IRUN,DATE)

!save actuals fuel used in the first iteration to scale fuel-use
IF(ITER==1) THEN
  fr1a = FR01
  fr2a = FR02
  fr3a = FR03
  fr4a = FR04
  fr5a = FR05
  fr6a = FR06  
  fr7a = FR07
  fr8a = FR08
  fr9a = FR09
  f10a = FR10
  f11a = FR11
  f12a = FR12
ENDIF
!---------------------------------------------------------------------------
!---------------------------UPDATE------------------------------------------
!---------------------------------------------------------------------------

!-----For any year, IF in UPDATE mode we update all FTT lagged variables
IF ( UPDATING ) THEN
  
    SWSL = SEWS       !Shares LAG
    SWKL = SEWK       !Capacity LAG
    SG1L = SGC1       !TLCOH with Gamma LAG
    SD1L = SGD1       !SD of TLCOH LAG
    SG2L = SGC2       !TMC with Gamma LAG
    SD2L = SGD2       !SD of TMC LAG
    SG3L = SGC3       !TPB with Gamma LAG
    SD3L = SGD3       !SD of TPB LAG
    BSTL = BSTC       !Costs matrix LAG
    SWWL = SEWW       !Cumulative global steelmaking capacity LAG
    SWYL = SWIY       !Cumulative global investments LAG
    SWIL = SWII       !Cumulative global investments by the steel industry LAG
    SWGL = SWIG       !Cumulative global investments by government LAG
    SCML = SCMM       !Cost/material matrix all plants, used for reduction in EE LAG
    SPCL = SPRC       !Steel produce without carbon costs LAG
    SICL = SICA       !Global installed capacity of intermediate plants -lag
    FRY2 = FRY1       !Two-year FRY lag
    SMPL = SEMS       !Total employment by technology in FTE
    IF (DATE > 2019) SPRL = SPRC
    
    RETURN

ENDIF

!---------------------------------------------------------------------------
!-----------------------First year 2019   ----------------------------------
!---------------------------------------------------------------------------

!Before 2018, all variable are derived from the last historical data points 
!This is done in order to set the lags in 2019, the starting point of the FTT calculation
!NOTE: If we're updating we don't do the FTT calculation
IF (DATE < 2020) THEN
    DO J=1, NR
        !SEWG is in historic sheets, so no need to calculate that. 
        
        !Capacity (kton) (11th are capacity factors) 
        SEWK(:,J) = 0.0
        SEWK(:,J) = SEWG(:,J)/BSTC(:,J,12)
        SPSA(J) = SUM(SEWG(:,J))
        
        !'Historical' employment in th FTE
        SEMS(:,J) = SEWK(:,J) * BSTC(:,J,5)*1.1
        
        !In this preliminary model SPSP is historical production while SPSA is exogenous future production (based on E3ME baseline)
        !Total steel production by region (kton/y) = demand
        !SPSP(J) = SUM(SEWG(:,J))
        
        !Market capacity shares of steelmaking technologies:
        SEWS(:,J) = 0
        WHERE (SPSP(J) > 0.0 .AND. SEWK(:,J) > 0.0) SEWS(:,J) = SEWK(:,J)/SUM(SEWK(:,J))
        
        !Emissions (MtCO2/y) (13th is emissions factors tCO2/tcs)
        !A crude backwards calculation of emissions using simple emission factors
        SEWE(:,J) = SEWG(:,J)*BSTC(:,J,14)/1000
        
        !Regional average energy intensity (GJ/tcs)
        STEI(:,J) = BSTC(:,J,16)
        SEIA(J) = SUM(STEI(:,J)*SEWS(:,J))
        
        !ADAPT THIS PIECE OF CODE - What fuel user designation does iron and steel have in E3ME?
        !Total fuel use by iron & steel (fuel user 4)
        SJEF(:,J) = 0
        SJEF(1,J) = fr1a(4,J)    !Coal Use
        SJEF(2,J) = fr2a(4,J)      !Other Coal Use
        SJEF(3,J) = fr3a(4,J)      !Crude Use
        SJEF(4,J) = fr4a(4,J)       !Heavy Oil Use
        SJEF(5,J) = fr5a(4,J)      !Middle Distillates Use
        SJEF(6,J) = fr6a(4,J)       !Other gas Use
        SJEF(7,J) = fr7a(4,J)        !Natural gas Use
        SJEF(8,J) = fr8a(4,J)       !Electricity Use
        SJEF(9,J) = fr9a(4,J)       !Heat Use
        SJEF(10,J) = f10a(4,J)       !Waste Use
        SJEF(11,J) = f11a(4,J)       !Biofuels Use
        SJEF(12,J) = f12a(4,J)        !Hydrogen Use   
        
        SJCO(:,J) = 0
        SJCO(1,J) = fr1a(4,J)    !Coal Use
        SJCO(2,J) = fr2a(4,J)      !Other Coal Use
        SJCO(3,J) = fr3a(4,J)      !Crude Use
        SJCO(4,J) = fr4a(4,J)       !Heavy Oil Use
        SJCO(5,J) = fr5a(4,J)      !Middle Distillates Use
        SJCO(6,J) = fr6a(4,J)       !Other gas Use
        SJCO(7,J) = fr7a(4,J)        !Natural gas Use
        SJCO(8,J) = fr8a(4,J)       !Electricity Use
        SJCO(9,J) = fr9a(4,J)       !Heat Use
        SJCO(10,J) = f10a(4,J)       !Waste Use
        SJCO(11,J) = f11a(4,J)       !Biofuels Use
        SJCO(12,J) = f12a(4,J)        !Hydrogen Use
    ENDDO
ENDIF

IF (DATE == 2013 .AND. UPDATING) RETURN
IF (DATE == 2013 .AND. ITER ==1) THEN

    !2008 deflator value to rescale deflator PRSC which is in 2000
    PRSC13 = PRSC
    !Exchange rate local currency <-> Euro
    EX13 = EX
    REX13 = REX
ENDIF

IF (DATE == 2019 .AND. ITER ==1) THEN
    !Demand for steel in monetary funds
    FRY19 = FRY(4,:)
    
ENDIF


IF (DATE < 2019) RETURN
IF (DATE == 2019) THEN
    !PYH17 = PYH
    GrowthRate1 = 0.0
    !Connect historical data to future projections (only for DATE == 2014)
    IF (ProductionSwitch ==0) THEN
        SPSA = SPSP
    ELSEIF (ProductionSwitch == 1) THEN
        WHERE (FRY19 > 0) GrowthRate1 = 1 + ((FRY(4,:) - FRY19) / FRY19)/gr_rate_corr
        SPSA = SPSP * GrowthRate1
    ENDIF
    
    !Re-distribute materials:
    CALL IDFTTSRawMatDistr(NST,NSS,NR,NC5,NSM,SEWG,SXSC,SXSF,SXSR,SPSP,SPSA,SLCI,BSTC,STIM,SMEF,SMED,SCMM,STEF,STEI,STSC,IRUN,DATE,SCIN, SEPF,SXIM,SXEX,t)
    
    !Set 
    og_base = 1.0
    WHERE (SUM(SEWG(:,:), dim=1) > 0.0) og_base = SUM(SEWG(1:7,:), dim=1)/SUM(SEWG(:,:), dim=1)
    og_sim = og_base
    ccs_share = 0.0
    SJEF = 0.0
    SJCO = 0.0
    !$OMP PARALLEL DO PRIVATE(I)
    DO J = 1,NR
        !Regional average energy intensity (GJ/tcs)
        SEIA(J) = SUM(STEI(:,J)*SEWS(:,J))
        IF (SPSA(J) > 0.0) THEN
            IF (SUM(SEWG(:,J)) > 0.0) THEN
                og_sim (J) = SUM(SEWG(1:19,J))/SUM(SEWG(:,J))
                ccs_share(J) = ( SUM(SEWG(4:7,J)) + SUM(SEWG(10:11,J)) + SUM(SEWG(14:15,J)) + SUM(SEWG(18:19,J))  + SUM(SEWG(22:23,J)) ) / SUM(SEWG(:,J))
            ENDIF
            DO I = 1, NST
                !Calculate fuel consumption
                 SJEF(1,J) = SJEF(1,J) + BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                 SJEF(2,J) = SJEF(2,J) + BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                 SJEF(7,J) = SJEF(7,J) + BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                 SJEF(8,J) = SJEF(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                 SJEF(11,J) = SJEF(11,J) + (BSTC(I,J,41) * SEWG(I,J) * 1000 * SMED(19) * 1/41868 + BSTC(I,J,42)*SEWG(I,J) * 1000 * SMED(20) * 1/41868)
                 SJEF(12,J) = SJEF(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                 !Calculate fuel consumption and correct it for CCS and biobased technologies for emission calculation.
                 IF (BSTC(I,J,22) == 1) THEN
                    SJCO(1,J) = SJCO(1,J) + 0.1 * BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                    SJCO(2,J) = SJCO(2,J) + 0.1 * BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                    SJCO(7,J) = SJCO(7,J) + 0.1 * BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                    SJCO(8,J) = SJCO(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                    SJCO(11,J) = SJCO(11,J) - 0.9 * (BSTC(I,J,41) * SEWG(I,J) * 1000 * SMED(19) * 1/41868 + BSTC(I,J,42)*SEWG(I,J) * 1000 * SMED(20) * 1/41868)
                    SJCO(12,J) = SJCO(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                 ELSE
                    SJCO(1,J) = SJCO(1,J) + BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                    SJCO(2,J) = SJCO(2,J) + BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                    SJCO(7,J) = SJCO(7,J) + BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                    SJCO(8,J) = SJCO(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                    SJCO(11,J) = 0.0
                    SJCO(12,J) = SJCO(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                 ENDIF
            ENDDO
            SJEF(3,J) = fr3a(4,J)
            SJCO(3,J) = fr3a(4,J) * (1.0-ccs_share(J))
            SJEF(4,J) = fr4a(4,J)
            SJCO(4,J) = fr4a(4,J) * (1.0-ccs_share(J))
            SJEF(5,J) = fr5a(4,J)
            SJCO(5,J) = fr5a(4,J) * (1.0-ccs_share(J))
            SJEF(6,J) = fr6a(4,J)
            SJCO(6,J) = fr6a(4,J) * (1.0-ccs_share(J))
            SJEF(9,J) = fr9a(4,J)
            SJCO(9,J) = fr9a(4,J) * (1.0-ccs_share(J))
            SJEF(10,J) = f10a(4,J)
            SJCO(10,J) = f10a(4,J) * (1.0-ccs_share(J))
        ENDIF
        og_ratio19(J) = 0.0
        IF (SUM(SJEF(1:2,J)) > 0.0) og_ratio19(J) = SJEF(6,J) / SUM(SJEF(1:2,J))
        
        !Connect the E3ME and FTT-Power prices of materials to FTT-Steel prices if switch == 1
        !Convert units to $(2008)/GJ or $(2008)/t
        SPMT(1:11,J) = SPMA(1:11,J)
        SPMT(12,J) = MEWP(1,J) * SMED(12) !Hard coal PFRC(4,J) * EX(34) / (PRSC(J)/PRSC13(J)) / 41.868 / SMED(12)
        SPMT(13,J) = MEWP(2,J) * SMED(13) !Other coal
        SPMT(14,J) = MEWP(7,J) !Natural gas
        SPMT(15,J) = MEWP(8,J) / 3.6 !Electricity
        SPMT(16,J) = SPMA(16,J) 
        SPMT(17,J) = SPMA(17,J) 
        SPMT(18,J) = SPMA(18,J) 
        SPMT(19,J) = SPMA(19,J) 
        SPMT(20,J) = SPMA(20,J) 
        
        SPMT(1:11,J) = SPMA(1:11,J)
        SPMT(12,J) = PFRC(4,J) * REX(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J))) / 41.868 * SMED(12)
        SPMT(13,J) = PFRC(4,J) * REX(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J))) / 41.868 * SMED(12)
        SPMT(14,J) = PFRG(4,J) * REX(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J))) / 41.868 * SMED(12)
        SPMT(15,J) = PFRE(4,J) * REX(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J))) / 41.868 * SMED(12)
        SPMT(16,J) = SPMA(16,J) 
        SPMT(17,J) = SPMA(17,J) 
        SPMT(18,J) = SPMA(18,J) 
        SPMT(19,J) = SPMA(19,J) 
        SPMT(20,J) = SPMA(20,J) 
        
    ENDDO
    !$OMP END PARALLEL DO
    SJFR = SJEF
    
    SMPLt = SMPL
    
    WHERE (SUM(SMPL,dim=1) > 0.0) SEMR = SUM(SEMS,dim=1)/SUM(SMPL,dim=1)
    
    !SCOT = 0.0
    !Calculate LCOS from cost matrix BSTC
    CALL IDFTTLCOS(NST,NC5,NJ,NR,NFU,NC,NSS,NSM,NXP,NY1, IRUN,DATE,BSTC, &
                        & SEWT,STRT,SEWC,SETC,SGC1,SDWC,SGD1,SWIC,SWFC,SOMC, &
                        & SCOC,SPSP,SPSA,SPMT,SCMM,SMEF,SMED,STIM,SPRI, &
                        & SGC2,SGC3,SGD2,SGD3,SEWG,SEWS,REPP,FEDS,RTCA,FETS, &
                        & SCOT,PRSC,SPRC,SITC,SIPR,SWGI,STGI,SCOI,SIEF,SIEI,STEF,PRSC13,EX,EX13,REX13)
    
    !Calculate first SEWW point
    BI = MATMUL(SEWB,SEWK) 
    SEWW = SUM(BI, dim=2)

    SICA = 0.0
    DO T1 = 1,NST
        DO T2 = 1,NSS
            IF (STIM(T1,T2) == 1) THEN
                !Estimate installed capacities of intermediate plants
                IF (T2 < 8) THEN
                    SICA(T2) = SICA(T2) + 1.1 *SEWW(T1) * SUM(BSTC(T1,:,26+T2))/COUNT(SPSA > 0.0)
                !Estimate installed capacities of ironmaking plants
                ELSEIF (T2 > 7 .AND. T2 < 21) THEN
                    SICA(T2) =  SICA(T2) + 1.1 * SEWW(T1)
                !Estimate installed capacities of steelmaking plants
                ELSEIF (T2 > 20 .AND. T2 < 27)  THEN
                    SICA(T2) = SICA(T2) + SEWW(T1)
                !Estimate installed capacities of finishing plants. 
                !NOTE: That after this step it's not crude steel anymore. Therefore it is divided by 1.14
                ELSEIF (T2 == 27) THEN
                    SICA(T2) = SICA(T2) + SEWW(T1) / 1.14
                ENDIF
            ENDIF
        ENDDO
    ENDDO

    
    SEWI = SEWK - SWKL
    WHERE (SEWI < 0.0) SEWI = 0.0
    WHERE (BSTC(:,:,6)> 0.0) SEWI = SEWI + SWKL/BSTC(:,:,6)

    !Set average steel price in 2019
    SPRI19 = SPRI(:)
    !Find scaling factors for fuel use
    SJEA(:,:) = 0
    SJEA(1,:) = fr1a(4,:)    !Coal Use
    SJEA(2,:) = fr2a(4,:)      !Other Coal Use
    SJEA(3,:) = fr3a(4,:)      !Crude Use
    SJEA(4,:) = fr4a(4,:)       !Heavy Oil Use
    SJEA(5,:) = fr5a(4,:)      !Middle Distillates Use
    SJEA(6,:) = fr6a(4,:)       !Other gas Use
    SJEA(7,:) = fr7a(4,:)        !Natural gas Use
    SJEA(8,:) = fr8a(4,:)       !Electricity Use
    SJEA(9,:) = fr9a(4,:)       !Heat Use
    SJEA(10,:) = f10a(4,:)       !Waste Use
    SJEA(11,:) = f11a(4,:)       !Biofuels Use
    SJEA(12,:) = f12a(4,:)        !Hydrogen Use
    
    !$OMP PARALLEL DO
    DO I=1, NJ
        JSCAL(I,:) = 1.0
        WHERE (SJEF(I,:) > 0.001 .AND. SJEA(I,:) > 0.001) JSCAL(I,:) = SJEA(I,:)/SJEF(I,:)
    ENDDO
    !$OMP END PARALLEL DO
    JSCAL(11,:) = (JSCAL(1,:) + JSCAL(2,:))/2
    JSCAL(12,:) = (JSCAL(1,:) + JSCAL(2,:))/2    
    SJEF = SJEA
    SJCO = SJEA
    RETURN
ENDIF	 

!------Set Regulations
!isReg is defined as a maximum market share (SEWS) of a technology 
!isReg is not integer: to avoid sharp investment spikes (as SEWS goes below - above - below... SEWR)
!                      we assume 10% deviation from the regulation possible
!You can think of isReg as the proportion of investors foreseeing the regulation before (or after) it is reached

isReg = 0
WHERE (SEWR > 0.0) isReg = 1.0+TANH(1.5+10*(SWKL - SEWR)/SEWR)
WHERE (SEWR == 0.0) isReg = 1.0


!---------------------------------------------------------------------------
!----------Time Loop--------------------------------------------------------
!---------------------------------------------------------------------------

!-----Internal time loop lags: at t=1 they are the same as the lags coming from outside
!-----at t>1 they pass information between the end and beginning of the time loop

IF (( .NOT. UPDATING) .AND. (DATE > 2019 .OR. ITER > 1)) THEN
    SWSLt = SWSL       !Shares LAG
    SWKLt = SWKL       !Capacity LAG
    SG1Lt = SG1L       !TLCOH with Gamma LAG
    SD1Lt = SD1L       !SD of TLCOH LAG
    SG2Lt = SG2L       !TMC with Gamma LAG
    SD2Lt = SD2L       !SD of TMC LAG
    SG3Lt = SG3L       !TPB with Gamma LAG
    SD3Lt = SD3L       !SD of TPB LAG
    BSTLt = BSTL       !Costs matrix LAG
    SWWLt = SWWL       !Cumulative global steelmaking capacity LAG
    SWYLt = SWYL       !Cumulative global investments LAG
    SWILt = SWIL       !Cumulative global investments by the steel industry LAG
    SWGLt = SWGL       !Cumulative global investments by government LAG
    SCMLt = SCML       !Cost/material matrix all plants, used for reduction in EE LAG
    SPCLt = SPCL
    SICLt = SICL       !Global installed capacity of intermediate plants -lag
    SMPLt = SMPL       !Employment growth factors
    SWIY = 0.0     !Total yearly investment in real Euro
    SWIG = 0.0     !Total investment by goverment (subsidies, etc.)
ENDIF

IF((.NOT. UPDATING) .AND. (DATE > 2019)) KRXt = KRX !Parse KRX data to a time loop variable.
    
!Set steel demand before time-loop:
SPSA = 0.0
GrowthRate1 = 1.0
GrowthRate2 = 1.0
!Here, we need to account for the fact we're in iter==1. FRY(4,:) needs to be forecasted forwad for the year ahead.
!WHERE (FRY19 > 0) GrowthRate1 = 1 + ((FRY(4,:) - FRY19) / FRY19)/gr_rate_corr
WHERE (FRY19 > 0 .AND. FRY2(4,:) > 0) GrowthRate1 = 1 + ((FRY1(4,:)*(1+(FRY1(4,:)-FRY2(4,:))/FRY2(4,:)) - FRY19) / FRY19)/gr_rate_corr
WHERE (FRY19 > 0) GrowthRate2 = 1 + ((FRY1(4,:) - FRY19) / FRY19)/gr_rate_corr
SPSA = SPSP * GrowthRate1 
IF (ITER == 1) SPSL = SPSP * GrowthRate2  !Lagged steel demand
!---------------------------------------------------------------------------
!--------------FTT SIMULATION-----------------------------------------------
!---------------------------------------------------------------------------
!------TIME LOOP!!: we calculate quarterly: t=1 means the end of the first quarter
IF (ITER==1) THEN
    DO t = 1, invdt
        SPSAt = 0.0
        !Time-step of steel demand
        SPSAt = SPSL + (SPSA-SPSL) * t/invdt
        SPSAtl = SPSL + (SPSA-SPSL) * (t-1)/invdt
    
        primary_iron_supply = 0.0
        primary_iron_demand = 0.0
        !primary_iron_supply = SUM( SWGI(:,J) * (0.9 - BSTC(:,J,12)) )
        !primary_iron_demand = (1-BSTC(26,J,25)/1.1) * SWKL(26,J) * BSTC(26,J,12)
    
        !If there's not enough scrap to supply the scrap route or there's not enough iron supply to meet the gap of scrap supply, then regulate scrap. 
        !This is more of a weak regulation. 
        WHERE (.NOT. (SEWR(26,:) > SWKLt(26,:) .AND. SEWR(26,:) > -1.0)) isReg(26,:) = 1.0- TANH(2*1.25*(BSTLt(26,:,26)-0.5)/0.5)
        WHERE (isReg(26,:) > 1.0) isReg(26,:) = 1.0
    
        DO J = 1, NR

            !Calculate share exchanges dSij
            dSij = 0.0
            dSEij = 0.0
            F = .5;
            FE = .5;
            IF (SPSAt(J) > 0.0) THEN
                ! i) Calculate end-of-lifetime share changes
                !$OMP PARALLEL DO PRIVATE(K, dFij, Fij)
                DO I = 1, NST     
                    !Only calculate for non-zero shares (SWSL>0) and no exogenous capacity
                    IF (SWSLt(I,J)>0.0) THEN  ! .AND. SG1Lt(I,J)/=0.0 .AND. SDTC(I,J)/=0.0
                        dSij(I,I) = 0   !Diagonal should be zero
                        DO K = 1, I-1
                            !Investor choices matrix Fij 
                            !Only calculate for non-zero shares (SWSL>0)
                            IF (SWSLt(K,J)>0.0) THEN   !.AND. SG1Lt(K,J)/=0.0 .AND. SDTC(K,J)/=0.0
                                !NOTE: TANH(1.25 X) is a cheap approx way to reproduce the normal CDF, i.e. ERF(X)), 1.414 = sqrt(2)
                                !Propagating width of variations in perceived costs
                                dFij = 1.414*SQRT(SD1Lt(I,J)*SD1Lt(I,J) + SD1Lt(K,J)*SD1Lt(K,J))
                                !Preferences based on cost differences by technology pairs (in log space)
                                Fij = 0.5*(1+TANH(1.25*(SG1Lt(K,J)-SG1Lt(I,J))/dFij))
                                !Preferences are either from investor choices (Fij) or enforced by regulations (SEWR)
                                F(I,K) = Fij*(1.0-isReg(I,J))*(1.0-isReg(K,J)) + isReg(K,J)*(1.0-isReg(I,J)) + .5*(isReg(I,J)*isReg(K,J))
                                F(K,I) = (1.0-Fij)*(1.0-isReg(K,J))*(1.0-isReg(I,J)) + isReg(I,J)*(1.0-isReg(K,J)) + .5*(isReg(K,J)*isReg(I,J))
                                !-------Shares equation!! Core of the model!!------------------ 
                                !(see eq 1 in Mercure EP 48 799-811 (2012) )
                                !dSij(I,K) = SWSLt(I,J)*SWSLt(K,J)*(SEWA(I,K)*F(I,K) - SEWA(K,I)*F(K,I))*dt/tScaling
                                !------Runge-Kutta Algorithm (RK4) implemented by RH 5/10/22, do not remove the divide-by-6!--------!
                                K1(I,K) = SWSLt(I,J)*SWSLt(K,J)*(SEWA(I,K)*F(I,K) - SEWA(K,I)*F(K,I))
                                K1(K,I) = - K1(I,K)
                                K2(I,K) = (SWSLt(I,J)+0.5*K1(I,K)*dt)*(SWSLt(K,J)+0.5*K1(K,I)*dt)*(SEWA(I,K)*F(I,K) - SEWA(K,I)*F(K,I))
                                K2(K,I) = - K2(I,K)
                                K3(I,K) = (SWSLt(I,J)+0.5*K2(I,K)*dt)*(SWSLt(K,J)+0.5*K2(K,I)*dt)*(SEWA(I,K)*F(I,K) - SEWA(K,I)*F(K,I))
                                K3(K,I) = - K3(I,K)
                                K4(I,K) = (SWSLt(I,J)+K3(I,K)*dt)*(SWSLt(K,J)+K3(K,I)*dt)*(SEWA(I,K)*F(I,K) - SEWA(K,I)*F(K,I))
                                K4(K,I) = - K4(I,K)
                        
                                dSij(I,K) = dt*(K1(I,K)+2*K2(I,K)+2*K3(I,K)+K4(I,K))/6/tScaling
                                !------End of RK4 Alogrithm!------------------------------------------------------------------------!                           
                                dSij(K,I) = -dSij(I,K)
                                !-------Shares equation!! Core of the model!!------------------ 
                            ENDIF
                        ENDDO
                    ENDIF
                ENDDO
                !$OMP END PARALLEL DO
                !The set payback period equals the scrapping rate. This number is usually higher than the substitution rate from above calculation.
                !This is due to more plants being eligible for scrapping. However, the investor preference should be much lower than the one from above.
                SR(:,J) = 1/BSTC(:,J,20) - 1/BSTC(:,J,6)
                !Constant used to reduce the magnitude of premature scrapping (there's evidence that is less likely to happen in iron and steel industry)
                SR_C = 2.5
                !$OMP PARALLEL DO PRIVATE(K, dFEij, dFEji, FEij, FEji)
                DO I = 1, NST   
                    !Only calculate for non-zero shares (SWSLt>0), only if scrapping decision rate > 0
                    IF (SWSLt(I,J)>0.0) THEN 
                        dSEij(I,I) = 0   !Diagonal should be zero
                        DO K = 1, I-1 
                            !Investor choices matrix FEij 
                            !Only calculate for non-zero shares (SWSLt>0), only if scrapping decision rate > 0 
                            IF (SWSLt(K,J)>0.0) THEN
                                    !NOTE: TANH(1.25 X) is a cheap approx way to reproduce the normal CDF, i.e. ERF(X)), 1.414 = sqrt(2)
                                   !Propagating width of variations in perceived costs
                                    dFEij = 1.414*SQRT(SD2Lt(K,J)*SD2Lt(K,J) + SD3Lt(I,J)*SD3Lt(I,J))
                                    dFEji = 1.414*SQRT(SD2Lt(I,J)*SD2Lt(I,J) + SD3Lt(K,J)*SD3Lt(K,J))
                                    !Preferences based on cost differences by technology pairs (asymmetric!)
                                    FEij = 0.5*(1+TANH(1.25*(SG2Lt(K,J)-SG3Lt(I,J))/dFEij))
                                    FEji = 0.5*(1+TANH(1.25*(SG2Lt(I,J)-SG3Lt(K,J))/dFEji))
                                    !Preferences are either from investor choices (FEij) or enforced by regulations (HREG)
                                    FE(I,K) = FEij*(1.0-isReg(I,J))
                                    FE(K,I) = FEji*(1.0-isReg(K,J))
                                    !-------Shares equation!! Core of the model!!------------------ 
                                    !dSEij(I,K) = SWSLt(I,J)*SWSLt(K,J)*(SWAP(I,K)*FE(I,K)*SR(K,J)/SR_C - SWAP(K,I)*FE(K,I)*SR(I,J)/SR_C)*dt
                                    !------Runge-Kutta Algorithm (RK4) implemented by RH 5/10/22, do not remove the divide-by-6!--------!
                                    KE1(I,K) = SWSLt(I,J)*SWSLt(K,J)*(SWAP(I,K)*FE(I,K)*SR(K,J)/SR_C - SWAP(K,I)*FE(K,I)*SR(I,J)/SR_C)
                                    KE1(K,I) = - KE1(I,K)
                                    KE2(I,K) = (SWSLt(I,J)+0.5*KE1(I,K)*dt)*(SWSLt(K,J)+0.5*KE1(K,I)*dt)*(SWAP(I,K)*FE(I,K)*SR(K,J)/SR_C - SWAP(K,I)*FE(K,I)*SR(I,J)/SR_C)
                                    KE2(K,I) = - KE2(I,K)
                                    KE3(I,K) = (SWSLt(I,J)+0.5*KE2(I,K)*dt)*(SWSLt(K,J)+0.5*KE2(K,I)*dt)*(SWAP(I,K)*FE(I,K)*SR(K,J)/SR_C - SWAP(K,I)*FE(K,I)*SR(I,J)/SR_C)
                                    KE3(K,I) = - KE3(I,K)
                                    KE4(I,K) = (SWSLt(I,J)+KE3(I,K)*dt)*(SWSLt(I,J)+KE3(I,K)*dt)*(SWAP(I,K)*FE(I,K)*SR(K,J)/SR_C - SWAP(K,I)*FE(K,I)*SR(I,J)/SR_C)
                                    KE4(K,I) = - KE4(I,K)
                        
                                    dSEij(I,K) = dt*(KE1(I,K)+2*KE2(I,K)+2*KE3(I,K)+KE4(I,K))/6/tScaling
                                    !------End of RK4 Alogrithm!------------------------------------------------------------------------! 
                                    !dSEij(K,I) = -dSEij(I,K)
                                    !-------Shares equation!! Core of the model!!------------------ 
                            ENDIF 
                        ENDDO
                    ENDIF
                ENDDO
                !$OMP END PARALLEL DO
                
                !Calulate endogenous shares!
                DO I = 1, NST
                    endo_shares(I,J) = 0.0
                    endo_capacity(I,J) = 0.0
                ENDDO
                endo_shares(:,J) =  SWSLt(:,J) + SUM(dSij,dim=2) + SUM(dSEij,dim=2)
                endo_eol(:,J) =  SUM(dSij,dim=2) 
                endo_capacity(:,J) = SPSAt(J)/SUM(endo_shares(:,J)*BSTC(:,J,12))*endo_shares(:,J)
                !Note: for steel, shares are shares of generation
                endo_gen(:,J) = endo_capacity(:,J) * BSTC(:,J,12)
                
                demand_weight(J) = SUM(endo_shares(:,J)*BSTC(:,J,12))
                
                Utot = SUM(endo_capacity(:,J))*demand_weight(J)
                dUk(:,J) = 0 
                dUkSK(:,J) = 0
                dUkREG(:,J) = 0
                dUkKST(:,J) = 0
                !Kickstart in terms of SPSAt
                WHERE (SWKA(:,J) < 0.0 ) dUkKST(:,J) = SKST(:,J) * Utot * dt
                
                !Regulations have priority over kickstarts
                WHERE (dUkKST(:,J)/BSTC(:,J,12) + endo_capacity(:,J) > SEWR(:,J) .AND. SEWR(:,J)>= 0.0) dUkKST(:,J) = 0.0
                
                !Shares are shares of demand, divided by average capacity factor 
                !Regulation is done in terms of shares of raw demand (no weighting)
                !Correct for regulations using difference between endogenous demand and demand from last time step with endo shares
                WHERE((endo_capacity(:,J)*demand_weight(J) - endo_shares(:,J)*SPSAtl(J)) > 0.0) dUkREG(:,J) = -(endo_capacity(:,J)*demand_weight(J) - endo_shares(:,J)*SPSAtl(J))*isReg(:,J)
                
                !Calculate demand subtractions based on exogenous capacity after regulations and kickstart, to prevent subtractions being too large and causing negatve shares.
                !Convert to direct shares of SPSAt - no weighting!
                WHERE (SWKA(:,J) < endo_capacity(:,J)) dUkSK(:,J) = ((SWKA(:,J) - endo_capacity(:,J))*demand_weight(J) - dUkREG(:,J) - dUkKST(:,J))*(t/invdt)
                !If SWKA is a target and is larger than the previous year's capacity, treat as a kick-start based on previous year's capacity. Small additions will help the target be met. 
                WHERE (SWKA(:,J) > endo_capacity(:,J) .AND. SWKA(:,J) > SWKL(:,J)) dUkSK(:,J) = (SWKA(:,J) - endo_capacity(:,J))*demand_weight(J)*(t/invdt)
                !Regulations have priority over exogenous capacity
                WHERE (SWKA(:,J) < 0 .OR. (SEWR(:,J) >= 0.0 .AND. SWKA(:,J) > SEWR(:,J)) ) dUkSK(:,J) = 0.0
       
                
                dUk(:,J) = dUkREG(:,J) + dUkSK(:,J) + dUkKST(:,J)
                dUtot  = SUM(dUk(:,J))
                
                !Use modified shares of demand and total modified demand to recalulate market shares
                !This method will mean any capacities set to zero will result in zero shares
                !It avoids negative shares
                !All other capacities will be stretched, depending on the magnitude of dUtot and how much of a change this makes to total capacity/demand
                !If dUtot is small and implemented in a way which will not under or over estimate capacity greatly, SWKA is fairly accurate

                !Market share changes due to exogenous settings and regulations
                IF (SUM(endo_capacity(:,J)*demand_weight(J) + dUk(:,J)) > 0) SWSA(:,J) = dUk(:,J)/SUM(endo_capacity(:,J)*demand_weight(J) + dUk(:,J))
                !New market shares
                IF (SUM(endo_capacity(:,J)*demand_weight(J) + dUk(:,J)) > 0) SEWS(:,J) = (endo_capacity(:,J)*demand_weight(J) + dUk(:,J))/SUM(endo_capacity(:,J)*demand_weight(J) + dUk(:,J))
                
                !Changes due to end-of-lifetime replacements
                SEOL(:,J) = SUM(dSij,dim=2)
                !Changes due to premature scrapping
                SBEL(:,J) = SUM(dSEij,dim=2)
            
                !--Main variables once we have new shares:--
                !Steel production capacity per technology (kton)
                SEWK(:,J) = SPSAt(J)/SUM(SEWS(:,J)*BSTC(:,J,12))*SEWS(:,J)
                !Actual steel production per technology (kton) (capacity factors column 12)
                SEWG(:,J) = SEWK(:,J)*BSTC(:,J,12)
                !Emissions (MtCO2/y) (14th is emissions factors tCO2/tcs)
                SEWE(:,J) = SEWG(:,J)*STEF(:,J)/1e3
                
            ENDIF
       
        
            !EOL replacements based on shares growth
            eol_replacements_t(:,J) = 0.0
            eol_replacements(:,J) = 0.0
            IF (t==1) SEWI(:,J) = 0.0
            SEWIt(:,J) = 0.0
                
            WHERE (endo_eol(:,J) >= 0 .AND. BSTC(:,J,6)>0) eol_replacements_t(:,J) = SWKL(:,J)*dt/BSTC(:,J,6)
            
                  
            WHERE ((-SWSLt(:,J)*dt/BSTC(:,J,6) < endo_eol(:,J) < 0) .AND. BSTC(:,J,6)>0) eol_replacements_t(:,J) = (SEWS(:,J)-SWSLt(:,J) + SWSLt(:,J)*dt/BSTC(:,J,6))*SWKL(:,J)
           
            
            !Capcity growth
            WHERE (SEWK(:,J)-SWKLt(:,J) > 0) 
                SEWIt(:,J) = SEWK(:,J)-SWKLt(:,J) + eol_replacements_t(:,J)  
            ELSEWHERE  
                SEWIt(:,J) = eol_replacements_t(:,J)  
            ENDWHERE
                
            !Capcity growth, add each time step to get total at end of loop
            SEWI(:,J) = SEWI(:,J) + SEWIt(:,J)
        
            !Connect the E3ME and FTT-Power prices of materials to FTT-Steel prices if switch == 1
            !Convert units to $(2008)/GJ or $(2008)/t
            SPMT(1:11,J) = SPMA(1:11,J)
            SPMT(12,J) = MEWP(1,J) * SMED(12) !Hard coal
            SPMT(13,J) = MEWP(2,J) * SMED(13) !Other coal
            SPMT(14,J) = MEWP(7,J) !Natural gas
            SPMT(15,J) = MEWP(8,J) / 3.6 !Electricity
            IF (J .LE. 33) THEN
                SPMT(16,J) = SPMA(16,J) !* PYH(6,J)/PYH17(6,J)
                SPMT(17,J) = SPMA(17,J) !* PYH(6,J)/PYH17(6,J)
            ELSE
                SPMT(16,J) = SPMA(16,J) !* PYH(4,J)/PYH17(4,J)
                SPMT(17,J) = SPMA(17,J) !* PYH(4,J)/PYH17(4,J)
            ENDIF
            SPMT(18,J) = SPMA(18,J)  !Hydrogen
            SPMT(19,J) = SPMA(19,J) !Charcoal
            SPMT(20,J) = SPMA(20,J) !Biogass
            !Calculate lagged employment
            !Old_Employment(J) = SUM(SWKLt(:,J) * BSTLt(:,J,5)) * 1000
        ENDDO
    
    
        !Cumulative investment for learning cost reductions
        !(Learning knowledge is global! Therefore we sum over regions)
        !BI = MATMUL(SEWB,SEWIt)          !Investment spillover: spillover matrix B
        !dW = SUM(BI,dim=2)         !Total new investment dW (see after eq 3 Mercure EP48)
        
        SEWI0 = SUM(SEWIt, dim=2)
        DO I=1, NST
            dW_temp = SEWI0
            dW(I) = DOT_PRODUCT(dW_temp, SEWB(I, :))
        ENDDO
        
	    !Cumulative capacity for learning
        SEWW = SWWLt + dW
      
        !Update technology costs for both the carbon price and for learning
        !Some Costs do not change
        BSTC(:,:,1:22) = BSTLt(:,:,1:22)
        !BSTLt(:,:,22:40) = 0.0
        SCMM = SCMLt
    
        !$OMP PARALLEL DO
        DO J = 1, NR
            !Switch: Do governments feedback xx% of their carbon tax revenue as energy efficiency investments?
            !Government income due to carbon tax in mln$(2008) 13/3/23 RSH: is this 2008 or 2013 as the conversion suggests?
            GCO2(J) = SUM(SEWG(:,J) * BSTLt(:,J,14)) *(REPP(J)*FETS(4,J) + RTCA(J,DATE-2000)*FEDS(4,J)) * REX(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J)))  *3.66/1000        
            IF (DOFB ==1) THEN
                IF (J == 36 .OR. J == 41 .OR. J == 48 .OR. J == 49) THEN
                    !10% feedback of revenue as energy efficiency investment
                    FBEE(J) = 0.1 * GCO2(J)
                    SEEI(:,J) = SEWS(:,J) * FBEE(J)
                    WHERE (SEEI(:,J) < 0.0) SEEI(:,J) = 0.0
                ENDIF
            ENDIF
                !Calculate perceived increase in capacity (PIC)
            PIC_RD(:,J) = SRDI(:,J)*1e6/BSTLt(:,J,1)*dt
            PIC_EE(:,J) = SEEI(:,J)*1e6/BSTLt(:,J,1)*dt
        ENDDO
        !$OMP END PARALLEL DO

        !Recalculate 'capacity' additions that include spillover, R&D, and EE investments.
        WHERE (SEWW > 0.0) dGCAP = (SEWW - SWWLt + SUM(PIC_RD,dim=2)/1000)/ SEWW 
        WHERE (SEWW > 0.0) dGCAP2 = (SEWW - SWWLt + SUM(PIC_RD/2 + PIC_EE, dim=2)/1000)/ SEWW 
        WHERE ( dGCAP < 0.0) dGCAP = 0.0
        WHERE ( dGCAP2 < 0.0) dGCAP2 = 0.0

        !$OMP PARALLEL DO
        DO I = 1, NST
            !New investment costs from past learning (column 8 are NEGATIVE learning exponents) (eq 2-3 EP48)
            IF (dGCAP(I) > 0.0001) THEN
                !BSTC(I,:,1) = BSTLt(I,:,1) + lOG(1-BSTC(I,:,7)) /lOG(2) *(SEWW-SWWLt)* BSTLt(I,:,1)
                BSTC(I,:,1) = BSTLt(I,:,1) * (1+BSTC(I,:,8) * dGCAP(I))
                BSTC(I,:,2) = BSTLt(I,:,1) * 0.3
            ENDIF
        ENDDO
        !$OMP END PARALLEL DO
    
        SCIN = BSTC(:,:,1)
        SICA = 0.0
    

        DO T1 = 1,NST
            DO T2 = 1,NSS
                IF (STIM(T1,T2) == 1) THEN
                    !Estimate installed capacities of intermediate plants (first 7 columns of SICA matrix)
                    IF (T2 < 8) THEN
                        SICA(T2) = SICA(T2) + 1.1 *SEWW(T1) * SUM(BSTLt(T1,:,26+T2))/COUNT(SPSAt>0.0)
                        !SICA_LR = -0.015
                    !Estimate installed capacities of ironmaking plants (column 8 to 20 of SICA matrix)
                    ELSEIF (T2 > 7 .AND. T2 < 21) THEN
                        SICA(T2) =  SICA(T2) + 1.1 * SEWW(T1)
                        !SICA_LR = BSTLt(T1,1,9)
                    !Estimate installed capacities of steelmaking plants (column 21 to 26 of SICA matrix)
                    ELSEIF (T2 > 20 .AND. T2 < 27)  THEN
                        SICA(T2) = SICA(T2) + SEWW(T1)
                        !SICA_LR = BSTLt(T1,1,9)
                    !Estimate installed capacities of finishing plants (column 27 of SICA matrix)
                    !NOTE: That after this step it's not crude steel anymore. Therefore it is divided by 1.14
                    ELSEIF (T2 == 27) THEN
                        SICA(T2) = SICA(T2) + SEWW(T1) / 1.14
                        !SICA_LR = -0.015
                    ENDIF
                ENDIF
            ENDDO
        ENDDO
       SICA_LR = -0.015
        !Calculate learning in terms of energy/material consumption 
        !$OMP PARALLEL DO PRIVATE(T2)
        DO MAT = 1, NSM
            DO T2 = 1,NSS
                IF (SICA(T2)-SICLt(T2) > 0.0) SCMM(MAT,T2) = (SCMLt(MAT,T2) - SEEM(MAT,T2)) * (1.0 + SICA_LR(T2) * (SICA(T2)-SICLt(T2))/SICLt(T2) ) + SEEM(MAT,T2)
            ENDDO
        ENDDO
       !$OMP END PARALLEL DO

        !Update material and cost input output matrix
        SLCI(:,5:11) = SCMM(:,1:7)
    
        !Re-distribute materials:
        IF (ITER < 10 .AND. t == invdt) CALL IDFTTSRawMatDistr(NST,NSS,NR,NC5,NSM,SEWG,SXSC,SXSF,SXSR,SPSP,SPSAt,SLCI,BSTC,STIM,SMEF,SMED,SCMM,STEF,STEI,STSC,IRUN,DATE,SCIN, SEPF,SXIM,SXEX,t)
    
        !Regional average energy intensity (GJ/tcs)
        SEIA = SUM(STEI*SEWS,dim=1)
    
        !Calculate bottom-up employment growth rates
        SEMS = SEWK * BSTC(:,:,5)*1.1
        WHERE (SUM(SMPLt,dim=1) > 0.0) SEMR = SUM(SEMS, dim=1) / SUM(SMPL,dim=1)
    
        !STEEL Calculate steelproduction costs from cost matrix BSTC
        CALL IDFTTLCOS(NST,NC5,NJ,NR,NFU,NC,NSS,NSM,NXP,NY1, IRUN,DATE,BSTC, &
                            & SEWT,STRT,SEWC,SETC,SGC1,SDWC,SGD1,SWIC,SWFC,SOMC, &
                            & SCOC,SPSP,SPSAt,SPMT,SCMM,SMEF,SMED,STIM,SPRI, &
                            & SGC2,SGC3,SGD2,SGD3,SEWG,SEWS,REPP,FEDS,RTCA,FETS, &
                            & SCOT,PRSC,SPRC,SITC,SIPR,SWGI,STGI,SCOI,SIEF,SIEI,STEF,PRSC13,EX,EX13,REX13)
    
        CAP_DIF = 0.0
        C_INV = 0.0
        C_SUB = 0.0
        C_CT = 0.0
        C_MT = 0.0
        C_DS = 0.0
    
        og_sim = 0.0
        SJEF = 0.0
        SJCO = 0.0
        og_sim = og_base
        og_fac = 1.0
        ccs_share = 0.0

        !$OMP PARALLEL DO PRIVATE(I)
        DO J = 1,NR
            IF (SPSAt(J) > 0.0) THEN
                ! Technologies 1 to 23 are most likely to use other gas.
                ! The share of production of these technologies decides the share of other gas actuals in the system.
                !IF (SUM(SEWG(:,J)) > 0.0) og_sim(J) = (SUM(SEWG(1:7,J)) + SUM(SEWG(16:23,J)))/ SUM(SEWG:,J)
                IF (SUM(SEWG(:,J)) > 0.0) THEN
                    og_sim(J) = SUM(SEWG(1:7,J))/SUM(SEWG(:,J))
                    ccs_share(J) = ( SUM(SEWG(4:7,J)) + SUM(SEWG(10:11,J)) + SUM(SEWG(14:15,J)) + SUM(SEWG(18:19,J))  + SUM(SEWG(22:23,J)) ) / SUM(SEWG(:,J))
                ENDIF
                DO I = 1, NST
                    !Calculate fuel consumption
                     SJEF(1,J) = SJEF(1,J) + BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                     SJEF(2,J) = SJEF(2,J) + BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                     SJEF(7,J) = SJEF(7,J) + BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                     SJEF(8,J) = SJEF(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                     SJEF(11,J) = SJEF(11,J) + (BSTC(I,J,41) * SEWG(I,J) * 1000 * SMED(19) * 1/41868 + BSTC(I,J,42)*SEWG(I,J) * 1000 * SMED(20) * 1/41868)
                     SJEF(12,J) = SJEF(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                     !Calculate fuel consumption and correct it for CCS and biobased technologies for emission calculation.
                     IF (BSTC(I,J,22) == 1) THEN
                        SJCO(1,J) = SJCO(1,J) + 0.1 * BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                        SJCO(2,J) = SJCO(2,J) + 0.1 * BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                        SJCO(7,J) = SJCO(7,J) + 0.1 * BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                        SJCO(8,J) = SJCO(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                        SJCO(11,J) = SJCO(11,J) - 0.9 * (BSTC(I,J,41) * SEWG(I,J) * 1000 * SMED(19) * 1/41868 + BSTC(I,J,42)*SEWG(I,J) * 1000 * SMED(20) * 1/41868)
                        SJCO(12,J) = SJCO(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                     ELSE
                        SJCO(1,J) = SJCO(1,J) + BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                        SJCO(2,J) = SJCO(2,J) + BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                        SJCO(7,J) = SJCO(7,J) + BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                        SJCO(8,J) = SJCO(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                        SJCO(11,J) = 0.0
                        SJCO(12,J) = SJCO(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                     ENDIF
                ENDDO
                SJFR(:,J) = SJEF(:,J)
                IF (og_base(J) > 0.0) og_fac(J) = og_sim(J) / og_base(J)
                SJEF(3,J) = og_fac(J) * fr3a(4,J)
                SJCO(3,J) = og_fac(J) * fr3a(4,J) * (1.0 - ccs_share(J))
                SJEF(4,J) = og_fac(J) * fr4a(4,J)
                SJCO(4,J) = og_fac(J) * fr4a(4,J) * (1.0 - ccs_share(J))
                SJEF(5,J) = og_fac(J) * fr5a(4,J)
                SJCO(5,J) = og_fac(J) * fr5a(4,J) * (1.0 - ccs_share(J))
                SJEF(6,J) = og_ratio19(J) * SUM(SJEF(1:2,J))
                SJCO(6,J) = og_ratio19(J) * SUM(SJEF(1:2,J)) * (1.0 - ccs_share(J))
                SJEF(9,J) = og_fac(J) * fr9a(4,J)
                SJCO(9,J) = og_fac(J) * fr9a(4,J) * (1.0 - ccs_share(J))
                SJEF(10,J) = og_fac(J) * f10a(4,J)
                SJCO(10,J) = og_fac(J) * f10a(4,J) * (1.0 - ccs_share(J))
                SJEF(:,J) = SJEF(:,J)*JSCAL(:,J)
                SJCO(:,J) = SJCO(:,J)*JSCAL(:,J)
            ENDIF
        
            !Calculate total investment demand
            SWIY(:,J) = SWIY(:,J) + (SEWI(:,J) * dt * BSTC(:,J,1)) / 1000 / REX13(34) * EX13(J) / PRSC13(J) !* PRSC(J))
            !Difference in investment costs between scenario and baseline. Connected to KRX.
            RSIY(J) = SUM(SWIY(:,J) - SWIB(:,J))
        
            !Government investment
            SWIG(:,J) = SWIG(:,J) + ((SEWT(:,J)*(SEWI(:,J) * dt * BSTC(:,J,1)/ 1000)) + SRDI(:,J) + SEEI(:,J)) / REX13(34) * EX13(J) / PRSC13(J)
            !Depending on the scenario, decide where SWIG goes to. Doesn't have to be KRX necessarily. 
            !IF (J<34) KRX(58,J) = KRXt(58,J) + SUM(SWIG(:,J))
            !IF (J>33) KRX(38,J) = KRXt(38,J) + SUM(SWIG(:,J))
        
        ENDDO
        !$OMP END PARALLEL DO
    
        !For time loop: update lags
        SWSLt = SEWS       !Shares LAG
        SWKLt = SEWK       !Capacity LAG
        SG1Lt = SGC1       !TLCOH with Gamma LAG
        SD1Lt = SGD1       !SD of TLCOH LAG
        SG2Lt = SGC2       !TMC with Gamma LAG
        SD2Lt = SGD2       !SD of TMC LAG
        SG3Lt = SGC3       !TPB with Gamma LAG
        SD3Lt = SGD3       !SD of TPB LAG
        BSTLt = BSTC       !Costs matrix LAG
        SWWLt = SEWW       !Cumulative global steelmaking capacity LAG
        !SWYLt = SWIY       !Cumulative global investments LAG
        SWILt = SWII       !Cumulative global investments by the steel industry LAG
        SWGLt = SWIG       !Cumulative global investments by government LAG
        SCMLt = SCMM       !Cost/material matrix all plants, used for reduction in EE LAG
        SPCLt = SPRC       !Steel produce without carbon costs LAG
        SICLt = SICA       !Global installed capacity of intermediate plants -lag
        SMPLt = SEMS
    
    ENDDO !-------------------------end of time loop----------------------------------------------------------
ENDIF
!---------------------------------------------------------------------------
!----------ITER > 1---------------------------------------------------------
!---------------------------------------------------------------------------
IF (ITER > 1) THEN
    !Here, we need to account for the fact we're in iter>1. GrowthRate1 is updated.
    WHERE (FRY19 > 0) GrowthRate1 = 1 + ((FRY(4,:) - FRY19) / FRY19)/gr_rate_corr

    SPSA = SPSP * GrowthRate1 
    DO J = 1,NR
        IF (SPSA(J) > 0.0) THEN
            !--Main variables once we have new shares:--
            !Steel production capacity per technology (kton)
            SEWK(:,J) = SPSA(J)/SUM(SEWS(:,J)*BSTC(:,J,12))*SEWS(:,J)
            !Actual steel production per technology (kton) (capacity factors column 12)
            SEWG(:,J) = SEWK(:,J)*BSTC(:,J,12)
            !Emissions (MtCO2/y) (14th is emissions factors tCO2/tcs)
            SEWE(:,J) = SEWG(:,J)*STEF(:,J)/1e3
       
            !EOL replacements based on shares growth
            eol_replacements_t(:,J) = 0.0
            eol_replacements(:,J) = 0.0
                
            WHERE ((SEWS(:,J)-SWSL(:,J) >= 0) .AND. BSTC(:,J,6) > 0.0) eol_replacements(:,J) = SWKL(:,J)/BSTC(:,J,6)
                  
            WHERE ((-SWSL(:,J)/BSTC(:,J,6) < SEWS(:,J)-SWSL(:,J) < 0) .AND. BSTC(:,J,6) > 0.0) eol_replacements(:,J) = (SEWS(:,J)-SWSL(:,J) + SWSL(:,J)/BSTC(:,J,6))*SWKL(:,J)
             
            !Capcity growth
            WHERE (SEWK(:,J)-SWKL(:,J) > 0) 
                SEWI(:,J) = SEWK(:,J)-SWKL(:,J) + eol_replacements(:,J)
            ELSEWHERE  
                SEWI(:,J) = eol_replacements(:,J)
            ENDWHERE
        ENDIF
        
        !Connect the E3ME and FTT-Power prices of materials to FTT-Steel prices if switch == 1
        !Convert units to $(2008)/GJ or $(2008)/t
        SPMT(1:11,J) = SPMA(1:11,J)
        SPMT(12,J) = MEWP(1,J) * SMED(12) !Hard coal
        SPMT(13,J) = MEWP(2,J) * SMED(13) !Other coal
        SPMT(14,J) = MEWP(7,J) !Natural gas
        SPMT(15,J) = MEWP(8,J) / 3.6 !Electricity
        IF (J .LE. 33) THEN
            SPMT(16,J) = SPMA(16,J) !* PYH(6,J)/PYH17(6,J)
            SPMT(17,J) = SPMA(17,J) !* PYH(6,J)/PYH17(6,J)
        ELSE
            SPMT(16,J) = SPMA(16,J) !* PYH(4,J)/PYH17(4,J)
            SPMT(17,J) = SPMA(17,J) !* PYH(4,J)/PYH17(4,J)
        ENDIF
        SPMT(18,J) = SPMA(18,J)  !Hydrogen
        SPMT(19,J) = SPMA(19,J) !Charcoal
        SPMT(20,J) = SPMA(20,J) !Biogass
    ENDDO

    !----------------------------------------------
    !No learning outside the time loop! Allow cumulative investment to be recalculated, but learning is time dependent
    !so must be based on projections
    !----------------------------------------------
    
    !Cumulative investment for learning cost reductions
    !(Learning knowledge is global! Therefore we sum over regions)
    !BI = MATMUL(SEWB,SEWI)          !Investment spillover: spillover matrix B
    !dW = SUM(BI,dim=2)          !Total new investment dW (see after eq 3 Mercure EP48)
    
    SEWI0 = SUM(SEWI, dim=2)
    DO I=1, NST
        dW_temp = SEWI0
        dW(I) = DOT_PRODUCT(dW_temp, SEWB(I, :))
    ENDDO
    
    !Cumulative capacity for learning
    SEWW = SWWL + dW
      

    
    !$OMP PARALLEL DO
    DO J = 1, NR
        !Switch: Do governments feedback xx% of their carbon tax revenue as energy efficiency investments?
        !Government income due to carbon tax in mln$(2008) 13/3/23 RSH: is this 2008 or 2013 as the conversion suggests?
        GCO2(J) = SUM(SEWG(:,J) * BSTL(:,J,14)) *(REPP(J)*FETS(4,J) + RTCA(J,DATE-2000)*FEDS(4,J)) * REX(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J)))  *3.66/1000        
        IF (DOFB ==1) THEN
            IF (J == 36 .OR. J == 41 .OR. J == 48 .OR. J == 49) THEN
                !10% feedback of revenue as energy efficiency investment
                FBEE(J) = 0.1 * GCO2(J)
                SEEI(:,J) = SEWS(:,J) * FBEE(J)
                WHERE (SEEI(:,J) < 0.0) SEEI(:,J) = 0.0
            ENDIF
        ENDIF
    ENDDO
    !$OMP END PARALLEL DO


    !Re-distribute materials:
    IF (ITER < 10) CALL IDFTTSRawMatDistr(NST,NSS,NR,NC5,NSM,SEWG,SXSC,SXSF,SXSR,SPSP,SPSA,SLCI,BSTC,STIM,SMEF,SMED,SCMM,STEF,STEI,STSC,IRUN,DATE,SCIN, SEPF,SXIM,SXEX,t)
    
    !Regional average energy intensity (GJ/tcs)
    SEIA = SUM(STEI*SEWS,dim=1)
    
    !Calculate bottom-up employment growth rates
    SEMS = SEWK * BSTC(:,:,5)*1.1
    WHERE (SUM(SMPL,dim=1) > 0.0) SEMR = SUM(SEMS, dim=1) / SUM(SMPL,dim=1)
    
    !STEEL Calculate steelproduction costs from cost matrix BSTC
    CALL IDFTTLCOS(NST,NC5,NJ,NR,NFU,NC,NSS,NSM,NXP,NY1, IRUN,DATE,BSTC, &
                    & SEWT,STRT,SEWC,SETC,SGC1,SDWC,SGD1,SWIC,SWFC,SOMC, &
                    & SCOC,SPSP,SPSAt,SPMT,SCMM,SMEF,SMED,STIM,SPRI, &
                    & SGC2,SGC3,SGD2,SGD3,SEWG,SEWS,REPP,FEDS,RTCA,FETS, &
                    & SCOT,PRSC,SPRC,SITC,SIPR,SWGI,STGI,SCOI,SIEF,SIEI,STEF,PRSC13,EX,EX13,REX13)
    
    CAP_DIF = 0.0
    C_INV = 0.0
    C_SUB = 0.0
    C_CT = 0.0
    C_MT = 0.0
    C_DS = 0.0
    
    og_sim = 0.0
    SJEF = 0.0
    SJCO = 0.0
    og_sim = og_base
    og_fac = 1.0
    ccs_share = 0.0

    !$OMP PARALLEL DO PRIVATE(I)
    DO J = 1,NR
    IF (SPSA(J) > 0.0) THEN
        ! Technologies 1 to 23 are most likely to use other gas.
        ! The share of production of these technologies decides the share of other gas actuals in the system.
        !IF (SUM(SEWG(:,J)) > 0.0) og_sim(J) = (SUM(SEWG(1:7,J)) + SUM(SEWG(16:23,J)))/ SUM(SEWG:,J)
        IF (SUM(SEWG(:,J)) > 0.0) THEN
            og_sim(J) = SUM(SEWG(1:7,J))/SUM(SEWG(:,J))
            ccs_share(J) = ( SUM(SEWG(4:7,J)) + SUM(SEWG(10:11,J)) + SUM(SEWG(14:15,J)) + SUM(SEWG(18:19,J))  + SUM(SEWG(22:23,J)) ) / SUM(SEWG(:,J))
        ENDIF
        DO I = 1, NST
            !Calculate fuel consumption
                SJEF(1,J) = SJEF(1,J) + BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                SJEF(2,J) = SJEF(2,J) + BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                SJEF(7,J) = SJEF(7,J) + BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                SJEF(8,J) = SJEF(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                SJEF(11,J) = SJEF(11,J) + (BSTC(I,J,41) * SEWG(I,J) * 1000 * SMED(19) * 1/41868 + BSTC(I,J,42)*SEWG(I,J) * 1000 * SMED(20) * 1/41868)
                SJEF(12,J) = SJEF(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                !Calculate fuel consumption and correct it for CCS and biobased technologies for emission calculation.
                IF (BSTC(I,J,22) == 1) THEN
                SJCO(1,J) = SJCO(1,J) + 0.1 * BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                SJCO(2,J) = SJCO(2,J) + 0.1 * BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                SJCO(7,J) = SJCO(7,J) + 0.1 * BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                SJCO(8,J) = SJCO(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                SJCO(11,J) = SJCO(11,J) - 0.9 * (BSTC(I,J,41) * SEWG(I,J) * 1000 * SMED(19) * 1/41868 + BSTC(I,J,42)*SEWG(I,J) * 1000 * SMED(20) * 1/41868)
                SJCO(12,J) = SJCO(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                ELSE
                SJCO(1,J) = SJCO(1,J) + BSTC(I,J,34) * SEWG(I,J) * 1000 * SMED(12) * 1/41868
                SJCO(2,J) = SJCO(2,J) + BSTC(I,J,35) * SEWG(I,J) * 1000 * SMED(13) * 1/41868
                SJCO(7,J) = SJCO(7,J) + BSTC(I,J,36) * SEWG(I,J) * 1000 * SMED(14) * 1/41868
                SJCO(8,J) = SJCO(8,J) + BSTC(I,J,37) * SEWG(I,J) * 1000 * SMED(15) * 1/41868
                SJCO(11,J) = 0.0
                SJCO(12,J) = SJCO(12,J) + BSTC(I,J,40) * SEWG(I,J) * 1000 * SMED(18) * 1/41868
                ENDIF
        ENDDO
        SJFR(:,J) = SJEF(:,J)
        IF (og_base(J) > 0.0) og_fac(J) = og_sim(J) / og_base(J)
        SJEF(3,J) = og_fac(J) * fr3a(4,J)
        SJCO(3,J) = og_fac(J) * fr3a(4,J) * (1.0 - ccs_share(J))
        SJEF(4,J) = og_fac(J) * fr4a(4,J)
        SJCO(4,J) = og_fac(J) * fr4a(4,J) * (1.0 - ccs_share(J))
        SJEF(5,J) = og_fac(J) * fr5a(4,J)
        SJCO(5,J) = og_fac(J) * fr5a(4,J) * (1.0 - ccs_share(J))
        SJEF(6,J) = og_ratio19(J) * SUM(SJEF(1:2,J))
        SJCO(6,J) = og_ratio19(J) * SUM(SJEF(1:2,J)) * (1.0 - ccs_share(J))
        SJEF(9,J) = og_fac(J) * fr9a(4,J)
        SJCO(9,J) = og_fac(J) * fr9a(4,J) * (1.0 - ccs_share(J))
        SJEF(10,J) = og_fac(J) * f10a(4,J)
        SJCO(10,J) = og_fac(J) * f10a(4,J) * (1.0 - ccs_share(J))
        SJEF(:,J) = SJEF(:,J)*JSCAL(:,J)
        SJCO(:,J) = SJCO(:,J)*JSCAL(:,J)
    ENDIF
        
    !Calculate total investment demand
    SWIY(:,J) = (SEWI(:,J) * BSTC(:,J,1)) / 1000 / REX13(34) * EX13(J) / PRSC13(J) !* PRSC(J))
    !Difference in investment costs between scenario and baseline. Connected to KRX.
    RSIY(J) = SUM(SWIY(:,J) - SWIB(:,J))
        
    !Government investment
    SWIG(:,J) = ((SEWT(:,J)*(SEWI(:,J) * BSTC(:,J,1)/ 1000)) + SRDI(:,J) + SEEI(:,J)) / REX13(34) * EX13(J) / PRSC13(J)
    !Depending on the scenario, decide where SWIG goes to. Doesn't have to be KRX necessarily. 
    !IF (J<34) KRX(58,J) = KRXt(58,J) + SUM(SWIG(:,J))
    !IF (J>33) KRX(38,J) = KRXt(38,J) + SUM(SWIG(:,J))
        
    ENDDO
    !$OMP END PARALLEL DO    
ENDIF

! Calculate the price feedback and add to PYHX (multiplicative exogenous
DO J=1,NR
    
    
    IF (J < 34) THEN
        IF (SPRL(J) > 0.0 .AND. SPRC(J) > 0.0) PYHX(17,J) = PYHX(17,J) * ( 0.333 * (SPRC(J)-SPRL(J))/SPRL(J) ) +1.0
    ELSE
        IF (SPRL(J) > 0.0 .AND. SPRC(J) > 0.0) PYHX(14,J) = PYHX(14,J) * ( 0.333 * (SPRC(J)-SPRL(J))/SPRL(J) ) +1.0
    ENDIF
    
    
ENDDO

END        
