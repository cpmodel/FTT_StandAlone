!_______________________________________________________________________
!_______________________________________________________________________
! ============FTT:Steel model ========== Created by JF Mercure, F Knobloch, L. van Duuren, and P. Vercoulen
! Follows general equations of the FTT model (e.g. Mercure Energy Policy 2012)
! Adapted to the iron and steel industry.
! 
! FTT determines the technology mix
!________________________________________jm801@cam.ac.uk________________

!---------------Levelised cost of steel production calculation function-----------------

! Headers
      SUBROUTINE IDFTTLCOS(NST,NC5,NJ,NR,NFU,NC,NSS,NSM,NXP,NY1, IRUN,DATE,BSTC, &
                        & SEWT,STRT,SEWC,SETC,SGC1,SDWC,SGD1,SWIC,SWFC,SOMC, &
                        & SCOC,SPSP,SPSA,SPMT,SCMM,SMEF,SMED,STIM,SPRI, &
                        & SGC2,SGC3,SGD2,SGD3,SEWG,SEWS,REPP,FEDS,RTCA,FETS, &
                        & SCOT,PRSC,SPRC,SITC,SIPR,SWGI,STGI,SCOI,SIEF,SIEI,STEF,PRSC13,EX,EX13,REX13)
      IMPLICIT NONE
        
      INTEGER,INTENT(IN) :: NST,NC5,NJ,NR,NFU,NC,NSS,NSM,NXP,NY1
      INTEGER,INTENT(IN) :: IRUN,DATE  
      
! Dummy arguments 
      !-----------FTT variables--------------
      REAL , INTENT(IN), DIMENSION(NST,NR,NC5) :: BSTC                  !Matrix of technology costs
      REAL , INTENT(IN), DIMENSION(NST,NR) :: SEWT, SEWG, SEWS             !subsidy, fuel tax
      REAL , INTENT(OUT), DIMENSION(NST,NR) :: SEWC, SETC, SGC1, SDWC, SGD1, &  !LCOS, without, with policies and gamma resp., dLCOS, dLCOS with policies
                                       & SWIC, SWFC, SOMC, SCOC, & !Investment, Material, O&M, and CO2 tax cost components of TLCOS resp. 
                                       & SGC2, SGC3, SGD2, STEF, & 
                                       & SGD3, SITC, SWGI,SCOI,SIEF,SIEI       ! ...,LCOI, Production of Iron,CO2 cost per ton iron, CO2 emissions per ton iron, Energy intensity  
      REAL , INTENT(INOUT), DIMENSION(NST,NR) :: SCOT
      REAL , INTENT(IN), DIMENSION(NR) :: SPSP,SPSA !Projected steel production (kton)   
      REAL , INTENT(OUT), DIMENSION(NR) :: SPRI,SPRC,SIPR,STGI
      REAL , INTENT(IN), DIMENSION(NSM,NR) :: STRT !Material subsidy or tax
      REAL , INTENT(INOUT), DIMENSION(NSM,NR) :: SPMT !Material prices, 
      REAL , INTENT(IN), DIMENSION(NSM,NSS) :: SCMM !Material subsidy or tax
      REAL , INTENT(IN), DIMENSION(NSM) :: SMEF, SMED
      REAL , INTENT(IN), DIMENSION(NST,NSS) :: STIM
      
      !-----------E3ME variables--------------
      REAL , INTENT(IN), DIMENSION(NFU,NR) :: FEDS, FETS
      REAL , INTENT(IN), DIMENSION(NR) :: REPP, PRSC,PRSC13,EX,EX13, REX13
      REAL , INTENT(IN), DIMENSION(NR,NY1) :: RTCA 
            
! Local variables
      ! FTT local variables
      INTEGER :: t, I, J, K, A
      !REAL , DIMENSION(NST) :: TF
      REAL :: IC, dIC, OM, dOM, L, B, r, Gam, CF, dCF, EF, dEF, TotCost, dTotCost, & 
                    & NPV1, NPV2, NPV1p, NPV1o, NPV2p, dNPV, dNPVp, &
                    & LCOS, LCOSprice, dLCOS, TLCOS, dTLCOS, LTLCOS, ICC, FCC, OMC, CO2C
      
      REAL , DIMENSION(NSM) :: MU, dMU, MP,SPMATax
      !REAL , DIMENSION(NST,NR) :: PS
      INTEGER :: P, M, Q, t_I
      REAL , DIMENSION(NST) ::  IC_I, OM_I, L_I, B_I, CF_I, EF_I, EF_I_Fossil, EF_I_Biobased, EI_I, r_I, SP_I_2, Sub_I, TotCost_I, TaxTotCost_I
      REAL :: It_I, St_I, Ft_I, FTt_I, OMt_I, CO2t_I, NPV1p_I, NPV2p_I, mean_price_I
      REAL , DIMENSION(NST) :: S, PS, PLCOS, PB, dPB, It, St, Pt, dIt, OMt, dOMt, Ft, dFt, FTt, CO2t, dCO2t, TaxTotCost !Values need to be stored for calculation of TMC, dTMC, TPB and dTPB
      REAL , DIMENSION(NST) :: TMC, dTMC, TPB, dTPB
      REAL , DIMENSION(NSM) :: Inventory_I
      REAL , DIMENSION(NR) :: iron_demand, iron_supply, local_average_price
      REAL :: EF_sec_route,mean_EF_I
    
DO J = 1, NR
    IF (SPSA(J) > 0.0) THEN
        !Calculate cost of ironmaking, price of iron production, inventory, emission factors    
        SPMATax = 0.0
        SPMATax = (1+STRT(:,J)) * SPMT(:,J)
        S = SEWS(:,J)               !Market shares
        !LCOI Calculation starts here
        TotCost_I = 0.0             !Total cost due to material/fuel consumption
        TaxTotCost_I = 0.0          !Total cost due to material/fuel consumption inc. additional tax/subsidies on materials
        IC_I = 0.0                  !Investment costs 
        OM_I = 0.0                  !O&M costs
        EF_I = 0.0                  !Emission factors - total
        EF_I_Fossil = 0.0           !Emission factors - fossils only
        EF_I_Biobased = 0.0         !Emission factors - biobased only
        EI_I =0.0                   !Energy intensity
        CF_I = 0.0                  !Capacity factors
        SITC(:,J) = 0.0                  !Levelised Cost of Iron
        r_I = 0.0                   !Discount rate
        L_I = 0.0                   !Lifetime
        B_I = 0.0                   !Leadtime
        Sub_I = 0.0                 !Subsidies/tax on investment costs
        SWGI(:,J) = 0.0                  !Iron production per ironmaking technology    
    
        DO Q = 1,NST
            DO P = 1,NSS
                !Not all integrated steelmaking routes have an ironmaking step and the condition below skips those routes.
                !Also, only a specific range of the NSS classification is required
                IF ( (STIM(Q,P) == 1) .AND. (P .GE. 8) .AND. (P .LE. 20) ) THEN
                    Inventory_I = 0.0           !Inventory for ironmaking 
                    !First, add ironmaking
                    Inventory_I(:) = SCMM(:,P) 
                    !Second, add sinter and pellet
                    Inventory_I(:) = Inventory_I(:) + Inventory_I(7) * SCMM(:,3) + Inventory_I(8) * SCMM(:,4) + Inventory_I(9) * SCMM(:,5) + Inventory_I(10) * SCMM(:,6)
                    !Third, add coke
                    Inventory_I(:) = Inventory_I(:) + Inventory_I(5) * SCMM(:,1) + Inventory_I(6) * SCMM(:,2)
                    !Fourth, add oxygen
                    Inventory_I(:) = Inventory_I(:) + Inventory_I(11) * SCMM(:,7)                
                
                    IF (BSTC(Q,J,22) == 1) THEN
                        !Increase fuel consumption if CCS
                        Inventory_I(12:15) = 1.1 * Inventory_I(12:15)
                        Inventory_I(18:20) = 1.1 * Inventory_I(18:20)
                    ENDIF
                
                    IC_I(Q) = SCMM(21,P)
                    OM_I(Q) = SCMM(22,P)
                    EF_I(Q) = SUM(Inventory_I*SMEF)
                    EF_I_Fossil(Q) = SUM(Inventory_I(1:15)*SMEF(1:15))
                    EF_I_Biobased(Q) = SUM(Inventory_I(16:24)*SMEF(16:24))
                    SIEI(Q,J) = SUM(Inventory_I*SMED)
                    CF_I(Q) = 0.9    
                    TotCost_I(Q) = SUM(Inventory_I * SPMT(:,J))
                    TaxTotCost_I(Q) = SUM(Inventory_I * SPMATax)    
                    r_I(Q) = BSTC(Q,J,10)
                    L_I(Q) = BSTC(Q,J,6)
                    B_I(Q) = BSTC(Q,J,7)
                    Sub_I(Q) = SEWT(Q,J)    
                    SWGI(Q,J) = SEWG(Q,J) * CF_I(Q)
                    SIEF(Q,J) = 0.0
                    IF (BSTC(Q,J,22) == 1) THEN
                        SIEF(Q,J) = 0.1 * EF_I(Q) - EF_I_Biobased(Q)
                    ELSE
                        SIEF(Q,J) = EF_I_Fossil(Q)            
                    ENDIF    
                
                    !Calculation of levelised cost starts here
                    IF (SPSA(J) > 0.0) THEN
                        It_I = 0.0 ! investment cost ($(2008)/tpi)), mean
                        St_I = 0.0 ! upfront subsidy/tax on investment cost ($(2008)/tpi))
                        Ft_I = 0.0 ! fuel cost ($(2008)/tpi)), mean
                        FTt_I = 0.0 ! fuel tax/subsidy (($(2008)/tpi)))
                        OMt_I = 0.0 ! O&M cost (($(2008)/tpi))), mean
                        NPV1p_I = 0.0 ! Discounted costs for the TLCOI calculation
                        NPV2p_I = 0.0 ! Denominator for the TLCOI calculation 
                    
                        DO t_I = 0, (B_I(Q) + L_I(Q)) -1
            
                            IF (t_I < B_I(Q)) THEN
                                !Investment costs are divided over each building year
                                It_I = IC_I(Q)/(CF_I(Q)*B_I(Q))
                                !Tech-specific subsidy or tax
                                St_I = Sub_I(Q) * It_I
                                !No material cost, tax/subsidy, O&M costs, CO2 tax during construction
                                Ft_I = 0.0
                                FTt_I = 0.0
                                OMt_I = 0.0
                                SCOI(Q,J) = 0.0
                            ELSE
                                !No Investment costs or subsidy or tax after construction
                                It_I = 0.0
                                St_I = 0.0
                                !Material costs
                                Ft_I = TotCost_I(Q)
                                !Subsidy or tax on materials
                                FTt_I = TaxTotCost_I(Q)
                                !Operation and maintenance costs
                                OMt_I = OM_I(Q)
                                !Costs due to CO2 tax 
                                !NEW CALCULATION OF SCOT FOR PI!!!!!
                                SCOI(Q,J) = EF_I(Q) *(REPP(J)*FETS(4,J) + RTCA(J,DATE-2000)*FEDS(4,J))  * REX13(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J))) /3.66
                            ENDIF
                            NPV1p_I = NPV1p_I + (It_I + St_I + FTt_I + OMt_I + SCOI(Q,J) )/(1+r_I(Q))**t_I
                            NPV2p_I = NPV2p_I+ 1/(1+r_I(Q))**t_I 
                
                        ENDDO
                        SITC(Q,J) = NPV1p_I/NPV2p_I !Levelised cost of ironmaking, this is used to determine the cost of pig iron which may be used in the Scrap-EAF route
                    ELSE
                        SITC(Q,J) = 0.0
                    ENDIF
                ENDIF   
            ENDDO
        ENDDO
        !The average price of pig iron is the levelised costs of ironmaking times the share of each ironmaking technology
        SIPR(J) = 0.0 
    
        !Total production of intermediate iron product
        STGI(J) = SUM(SWGI(:,J))
        IF (STGI(J) > 0.0) THEN
            SIPR(J) = SUM(SITC(:,J)*SWGI(:,J)/SUM(SWGI(:,J)))
        ENDIF
    ENDIF
ENDDO    


!Global average price:
mean_price_I = SUM(SIPR*STGI/SUM(STGI))
local_average_price = 0.0
iron_demand = 0.0
iron_supply = 0.0

!Global average EF for intermediate iron (to connect to imported iron by Scrap - EAF route)
mean_EF_I = SUM(SIEF*SWGI)/SUM(STGI)

DO J=1,NR
    IF (SPSA(J) > 0.0) THEN
        SPMATax = 0.0
        SPMATax = (1+STRT(:,J)) * SPMT(:,J)
        EF_sec_route = 0.0
        iron_demand(J) =  BSTC(26,J,25)*SEWG(26,J)
        IF (STGI(J) > SUM(BSTC(:,J,24)*SEWG(:,J))) iron_supply(J) =  STGI(J) - SUM(BSTC(:,J,24)*SEWG(:,J))
    
        IF (iron_demand(J) > 0.0) THEN
            !The price of intermediate iron can only be used directly if there's enough iron production 
            IF (SIPR(J) > 0.0 .AND. SIPR(J) < 3*SPMT(3,J) .AND. iron_supply(J) >= iron_demand(J) ) THEN
                SPMT(3,J) = SIPR(J)
            !If there isn't enough domestic supply of iron, then a part of the price has to be calculated from global price and global EF of iron production resp.
            ELSEIF (SIPR(J) > 0.0 .AND. SIPR(J) < 3*SPMT(3,J) .AND. iron_supply(J) > 0.0 .AND. iron_demand(J) > iron_supply(J)) THEN
                local_average_price(J) = ( (iron_demand(J) - iron_supply(J)) * mean_price_I + iron_supply(J) * SIPR(J) ) / iron_demand(J)
                SPMT(3,J) = local_average_price(J)
            !If there's no iron production (only MOE or Scrap-EAF present) or local price is ridiculously high, then take global average.
            ELSE
                SPMT(3,J) =  mean_price_I
            ENDIF

            !Emissions related to iron production that is used in the Scrap-EAF route need to be allocated to this route
            !There's enough domestic supply to cover the domestic demand
            IF (iron_supply(J) >= iron_demand(J)) THEN
                EF_sec_route = SUM(SIEF(:,J)*SWGI(:,J)/STGI(J))
            !If there's some iron supply, but not enough demand,partly use the global mean
            ELSEIF ( iron_supply(J) < iron_demand(J) .AND. iron_supply(J) > 0.0 ) THEN
                EF_sec_route = ( (iron_demand(J) - iron_supply(J)) * mean_EF_I + SUM(SIEF(:,J)*SWGI(:,J)/STGI(J)) * iron_supply(J) )  / iron_demand(J)
            !If there's no iron supply at all, use global mean completely.
            ELSE
                EF_sec_route = mean_EF_I 
            ENDIF
        ENDIF
        !(not turn on)OMP PARALLEL DO PRIVATE(IC, dIC, OM, dOM, L, B, r, Gam, CF, dCF, EF_sec_route, EF, dEF, TotCost, dTotCost, TaxTotCost, SPMATax, It, dIt, OMt, dOMt, St, Ft, dFt, FTt, CO2t, dCO2t, NPV1, NPV2, NPV1p, NPV2p, dNPV, dNPVp, ICC, OMC, FCC, CO2C, Pt,t)
        !LCOS Calculation starts here
        DO I = 1,NST
            IF (SPSA(J) > 0.0) THEN 
            
                !Initialize CostMatrix variables
                IC = BSTC(I,J,1)
                dIC = BSTC(I,J,2)
                OM = BSTC(I,J,3)
                dOM = BSTC(I,J,4)
                L = BSTC(I,J,6)
                B = BSTC(I,J,7)
                r = BSTC(I,J,10)
                Gam = BSTC(I,J,11)
                CF = BSTC(I,J,12)
                dCF = BSTC(I,J,13)
                !Adjust emission factor for Scrap-EAF due to emissions.
                IF (I.EQ.26) STEF(I,J) = BSTC(I,J,14) + BSTC(I,J,25) * EF_sec_route
                EF = STEF(I,J)
                dEF = BSTC(I,J,15)
            
                !Only select raw materials from inventory
                !MU(:) = BSTC(I,J,14:37)
                !dMU = 0.3*MU
                !MP = SPMA(J,:)
            
                TotCost = 0.0
                dTotCost = 0.0
                TaxTotCost = 0.0
                DO K = 1, NSM - 4
                    TotCost = TotCost + BSTC(I,J,22+K)*SPMT(K,J)
                    TaxTotCost(I) = TaxTotCost(I) + BSTC(I,J,22+K)*SPMATax(K)
                    dTotCost = dToTCost + 0.1*BSTC(I,J,22+K)*SPMT(K,J)
                ENDDO

                It = 0.0 ! investment cost ($(2008)/tcs)), mean
                dIt(I) = 0.0 ! investment cost ($(2008)/tcs)), SD
                St = 0.0 ! upfront subsidy/tax on investment cost ($(2008)/tcs))
                Ft = 0.0 ! fuel cost ($(2008)/tcs)), mean
                dFt(I) = 0.0 ! fuel cost ($(2008)/tcs)), SD
                FTt = 0.0 ! fuel tax/subsidy (($(2008)/tcs)))
                OMt = 0.0 ! O&M cost (($(2008)/tcs))), mean
                dOMt(I) = 0.0 ! O&M cost ($(2008)/tcs)), SD
                CO2t = 0.0 ! CO2 cost ($(2008)/tcs)), mean
                dCO2t(I) = 0.0 ! CO2 cost ($(2008)/tcs)), SD
                NPV1 = 0.0 ! Discounted costs for the LCOS calculation
                NPV2 = 0.0 ! Denominator for the LCOS calculation 
                NPV1o = 0.0
                NPV1p = 0.0 ! Discounted costs for the TLCOS/LTLCOS calculation
                NPV2p = 0.0 ! Denominator for the TLCOS/LTLCOS calculation
                dNPV = 0.0 ! SD for the TLCOS/LTLCOS calculation
                dNPVp = 0.0 ! SD for the TLCOS/LTLCOS calculation
                ICC = 0.0 ! Investment cost component of LTLCOS
                FCC = 0.0 ! Fuel/Material cost component of LTLCOS
                OMC = 0.0 ! O&M cost component of LTLCOS
                CO2C = 0.0 ! CO2 tax cost component of LTLCOS
                Pt = 0.0 !Production of steel (0 during leadtime, 1 during lifetime)
            
                DO t=0, (B+L)-1
                    IF (t < B) THEN
                        !Investment costs are divided over each building year
                        It(I) = IC/(CF*B) 
                        dIt(I) = dIC/(CF*B)
                        !Tech-specific subsidy or tax
                        St(I) = SEWT(I,J)*It(I)
                        !No material cost, tax/subsidy, O&M costs, CO2 tax during construction
                        Ft(I) =0.0
                        dFt(I) = 0.0
                        FTt(I) = 0.0
                        OMt(I) = 0.0
                        dOMt(I) = 0.0
                        CO2t(I) = 0.0
                        dCO2t(I) = 0.0
                        Pt(I) = 1.0 !No production
                    ELSE
                        !No Investment costs or subsidy or tax after construction
                        It(I) = 0
                        dIt(I) = 0
                        St(I) = 0
                        !Material costs
                        Ft(I) = TotCost
                        dFt(I) = dTotCost
                        !Material subsidy or tax
                        FTt(I) = TaxTotCost(I)
                        !Operation and maintenance
                        OMt(I) = OM
                        dOMt(I) = dOM
                        !CO2 tax
                        !NOTE: We use the emission factors stored in BSTC rather than the ones stored in STEF. 
                        !EFs in STEF are altered to take into account the emissions due to iron production that is used in the scrap-EAF route.
                        !These emissions do not take place directly in the Scrap-EAF route and therefore this technology does not bare the costs.
                        SCOT(I,J) = BSTC(I,J,14)*(REPP(J)*FETS(4,J) + RTCA(J,DATE-2000)*FEDS(4,J)) * REX13(34) / (PRSC(J)*EX13(J)/(PRSC13(J)*EX(J))) /3.66
                        dCO2t(I) = 0.1 * SCOT(I,J) 
                        Pt(I) = 1.0
                    ENDIF
                
                    NPV1 = NPV1 + (It(I) + Ft(I) + OMt(I))/(1+r)**t
                    NPV1p = NPV1p + (It(I) + St(I) + FTt(I) + OMt(I) + SCOT(I,J))/(1+r)**t
                    NPV1o = NPV1o + (It(I) + St(I) + FTt(I) + OMt(I))/(1+r)**t
               
                    dNPV = dNPV + SQRT(dIt(I)**2 + dFt(I)**2 + dOMt(I)**2)/(1+r)**t
                    dNPVp = dNPVp + SQRT(dIt(I)**2 + dFt(I)**2 + dOMt(I)**2 + dCO2t(I)**2)/(1+r)**t
                
                    ICC = ICC + (It(I) + St(I))/(1+r)**t
                    FCC = FCC + (FTt(I))/(1+r)**t
                    OMC = OMC + (OMt(I))/(1+r)**t
                    CO2C = CO2C + (SCOT(I,J))/(1+r)**t
                
                    NPV2 = NPV2 + Pt(I)/(1+r)**t
                    NPV2p = NPV2p + Pt(I)/(1+r)**t
                ENDDO
            
                LCOS = NPV1/NPV2
                dLCOS = dNPV/NPV2
                LCOSprice = NPV1o/NPV2
                TLCOS = NPV1p/NPV2p
                dTLCOS = dNPVp/NPV2p
                LTLCOS = TLCOS + Gam

                !Variables for endogenous scrapping
                PB(I) = BSTC(I,J,20) ! payback threshold
                dPB(I) = 0.3* PB(I)
                It(I) = BSTC(I,J,1)/(CF) 
                dIt(I) = BSTC(I,J,2)/(CF)
                St(I) = SEWT(I,J)*It(I)
                FTt(I) = TaxTotCost(I)
                dFt(I) = dTotCost
                OMt(I) = OM
                dCO2t(I) = 0.1 * SCOT(I,J) 
            
                !Marginal cost calculation (for endogenous scrapping) (fuel cost+OM cost+fuel and CO2 tax policies)   
                TMC(I) = Ft(I) + OMt(I) + FTt(I) + SCOT(I,J)
                dTMC(I) = SQRT(dFt(I)**2 + dOMt(I)**2)
            
                !Payback cost calculation (for endogenous scrapping) (TMC+(investment cost+investment subsidy)/payback threshold)   
                TPB(I) = TMC(I) + (It(I) + St(I))/PB(I)
                dTPB(I) = SQRT(dFt(I)**2 + dOMt(I)**2 + dIt(I)**2/PB(I)**2 + (It(I)**2/PB(I)**4)*dPB(I)**2)
            
                SGC2(I,J) = TMC(I) + Gam
                SGC3(I,J) = TPB(I) + Gam
                SGD2(I,J) = dTMC(I)
                SGD3(I,J) = dTPB(I)
            
                !For the cost components we exclude the gamma value as we assume the gamma value is accordingly distributed over the cost components
                SWIC(I,J) = (ICC/NPV2p)  ! Investment cost component of TLCOS
                SWFC(I,J) = (FCC/NPV2p)  ! Material cost component of TLCOS
                SOMC(I,J) = (OMC/NPV2p)  ! O&M cost component of TLCOS
                SCOC(I,J) = (CO2C/NPV2p) ! CO2 cost component of TLCOS
            
                SEWC(I,J) = LCOS
                SETC(I,J) = LCOSprice
                SGC1(I,J) = LTLCOS

            
                SDWC(I,J) = dLCOS
                SGD1(I,J) = dTLCOS
            
            ELSE
                LCOS = 0
                dLCOS = 0
                TLCOS = 0
                dTLCOS = 0
                LTLCOS = 0
            
                SWIC(I,J) = 0
                SWFC(I,J) = 0
                SOMC(I,J) = 0
                SCOC(I,J) = 0
            
                SEWC(I,J) = 0
                SETC(I,J) = 0
                SGC1(I,J) = 0
                SGC2(I,J) = 0
                SGC3(I,J) = 0
            
                SDWC(I,J) = 0
                SGD1(I,J) = 0
                SGD2(I,J) = 0
                SGD3(I,J) = 0
            ENDIF
        ENDDO !DO statement: end of NST loop
        !OMP END PARALLEL DO

        IF (SPSA(J) > 0.0) THEN
            WHERE (SGC1(:,J) .NE. 0.0)
                PLCOS = SGC1(:,J)
            ENDWHERE
            SPRI(J) =  SUM(PLCOS * S) / SUM(S)
    
            WHERE (SETC(:,J) .NE. 0.0)
                PLCOS = SETC(:,J)
            ENDWHERE
            SPRC(J) =  SUM(PLCOS * S) / SUM(S)
        ENDIF
    ENDIF
ENDDO !DO statement: end of NR loop



END