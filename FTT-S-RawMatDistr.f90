!_______________________________________________________________________
!_______________________________________________________________________

    ! Headers
      SUBROUTINE IDFTTSRawMatDistr(NST,NSS,NR,NC5,NSM,SEWG,SXSC,SXSF,SXSR,SPSP,SPSA,SLCI,BSTC,STIM,SMEF,SMED,SCMM,STEF,STEI,STSC,IRUN,DATE,SCIN, SEPF,SXIM,SXEX,t)
      IMPLICIT NONE
      
      INTEGER,INTENT(IN) :: NST, NSS, NR, NC5, NSM  
      INTEGER,INTENT(IN) :: IRUN,DATE,t  
      
      !Output variables: SSCR, SCOA, SNAG, SELC,SIRO, SLIS,SHYD,SBCH,SBGA, SXSR

      REAL , INTENT(IN), DIMENSION(NST, NR) :: SEWG !Regulations, Tech specific steel production
      REAL , INTENT(IN), DIMENSION(NR) :: SXSC, SPSP, SPSA !Scrap availability (kton), Remaining scrap due to a surplus (kton)
      REAL , INTENT(INOUT), DIMENSION(NR) :: SXSF, SXSR,SXIM,SXEX
      REAL , INTENT(INOUT), DIMENSION(NSM,NSM) :: SLCI                 !Final, intermediate and raw material consumption matrix (In the first column the data of the correct steelmaking technology gets selected (from SLSM) and in the second column the correct ironmaking technology gets selected (from SLIM).
      REAL , INTENT(INOUT), DIMENSION(NST,NR,NC5) :: BSTC           !CostMatrix
      REAL , INTENT(IN), DIMENSION(NST,NSS) :: STIM                 !Interaction matrix between individual plants and integrated steelmaking routes
      REAL , INTENT(IN), DIMENSION(NSM) :: SMEF, SMED               !Material specific emission factors and energy densities resp.
      REAL , INTENT(IN), DIMENSION(NSM,NSS) :: SCMM                 !Costs and material parameters for each plant type. This includes precursor plants, ironmaking plants and steelmaking plants
      REAL , INTENT(OUT), DIMENSION(NST,NR) :: STEF,SCIN,SEPF, STEI, STSC
      
      
      !Local variables
      REAL , DIMENSION(NST,3) :: MetalInput    
      REAL , DIMENSION(NSM, NST) :: Inventory
      REAL , DIMENSION(NSM,NSM) :: II, Subtracting, Inverse !Identity matrix
      REAL , DIMENSION(NSM) :: FD !Final Demand vector
      REAL , DIMENSION(NST) :: EF2, EMPL, INV, OM !Maximum Scrap input
      REAL , DIMENSION(NR) :: MaxScrapDemand,MaxScrapDemandP,MaxScrapDemandS,ScraplimitTrade,ScrapShortage,ScrapAbundance,ScrapSupply
      REAL , DIMENSION(NST,NR) :: MSSPP
      
      REAL ::  PITCSR, EF, EF_Fossil, EF_Biobased !Maximum share of scrap in primary production technologies (t/tcs), Pig iron to crude steel ratio, Maximum scrap demand in a region (t)
      INTEGER :: I, J, x, A, Tech, MAT, MAT1, MAT2, B

MetalInput = 0.0
MaxScrapDemand = 0.0
MaxScrapDemandP = 0.0
ScraplimitTrade = 0.0
MSSPP = 0.0
PITCSR = 1.1; !Pig iron (or DRI) to crude steel ratio. Rationale: pig iron/DRI 
!has a higher content of carbon, and this is removed in the steelmaking
!process (to about 0.0001 - 0.005%wt)
SXSF = 0.0


!!!SIMPLE TREATMENT OF SCRAP TRADE. FLOWS ARE ONLY A FUNCTION OF SCRAP SHORTAGES FOR THE SECONDARY STEELMAKING ROUTE.
MSSPP = BSTC(:,:,17)
MaxScrapDemand = SUM(MSSPP*SEWG, dim=1)
MaxScrapDemandP = SUM(MSSPP(1:24,:)*SEWG(1:24,:), dim=1)
ScraplimitTrade = MSSPP(26,:)*SEWG(26,:)
IF (t == 1) THEN
    ScrapShortage = 0.0
    ScrapAbundance = 0.0
    SXIM = 0.0
    SXEX = 0.0
    SXSR = 0.0

    WHERE (SXSC < ScraplimitTrade) ScrapShortage = ScraplimitTrade - SXSC
    WHERE (SXSC > MaxScrapDemand) ScrapAbundance = SXSC - MaxScrapDemand

    !If there's a global abundance of scrap then the shortages can simply be met.
    IF (SUM(ScrapShortage) .LE. SUM(ScrapAbundance) .AND. SUM(ScrapShortage) > 0.0) THEN
        WHERE (ScrapShortage > 0.0) SXIM = ScrapShortage
    ENDIF

    !If the supply of scrap is insufficient to meet global demand then weight import according the ratio of abundance and shortage.   
    IF (SUM(ScrapShortage) .GT. SUM(ScrapAbundance) .AND. SUM(ScrapShortage) > 0.0) THEN    
        WHERE (ScrapShortage > 0.0) SXIM = ScrapShortage * (SUM(ScrapAbundance) / SUM(ScrapShortage))
    ENDIF

    !Countries export scrap according to their weights to the global abundance.
    WHERE (SXSC > MaxScrapDemand) SXEX = SUM(SXIM) * (ScrapAbundance / SUM(ScrapAbundance))
ENDIF    

!Total scrap supply is home generated scrap (SXSC) plus imports minus exports    
SXSR = SXSC + SXIM - SXEX    

DO J = 1, NR
    IF (SPSA(J) > 0.0) THEN
        !$OMP PARALLEL DO
        DO I = 1, NST-2
            !There's enough scrap to meet the maximum scrap demand
            IF (SXSR(J) >= MaxScrapDemand(J)) THEN

                MetalInput(I,1) = (1.0 - 0.09 - MSSPP(I,J)) * PITCSR 
                MetalInput(I,2) = 0.0
                MetalInput(I,3) = MSSPP(I,J) +0.09
            
                MetalInput(26,1) = 0.0
                MetalInput(26,2) = 0.0
                MetalInput(26,3) = MSSPP(26,J) +0.09
            !There's not enough scrap to feed into all the technologies, but there's 
            !enough scrap to feed into the Scrap-EAF route.             
            ELSEIF  ( SXSR(J) < MaxScrapDemand(J) .AND. SXSR(J) >= ScraplimitTrade(J) ) THEN ! .AND. SUM(MSSPP(1:24,J)) > 0.0) THEN
                MetalInput(I,2) = 0.0
                !MetalInput(I,3) = 0.09 + (SXSR(J)/MaxScrapDemand) - SXSR(J)
                IF (SUM(SEWG(1:24,J) * MSSPP(1:24,J)) > 0.0) THEN
                    MetalInput(I,3) = 0.09 + (SXSR(J)-ScraplimitTrade(J))/MaxScrapDemandP(J) * MSSPP(I,J)
                ELSE
                    MetalInput(I,3) = MSSPP(I,J)/2 +0.09
                ENDIF
                MetalInput(I,1) = (1.0 - MetalInput(I,3)) * PITCSR
                
                MetalInput(26,1) = 0.0
                MetalInput(26,2) = 0.0
                MetalInput(26,3) = MSSPP(26,J) +0.09

            !There's not enough scrap available to meet the demand, so all available
            !scrap will be fed into the Scrap-EAF route.    
            ELSEIF (SXSR(J) < MaxScrapDemand(J) .AND. SXSR(J) < SEWG(26,J)*(1-0.09)) THEN
            
                MetalInput(I,1) = PITCSR * (1.0 - 0.09)
                MetalInput(I,2) = 0.0
                MetalInput(I,3) = 0.09

                MetalInput(26,1) = 0.0
                MetalInput(26,2) = (1-0.09-SXSR(J)/SEWG(26,J))*PITCSR
                MetalInput(26,3) = 0.09+SXSR(J)/SEWG(26,J)  
            ENDIF
        ENDDO
        !$OMP END PARALLEL DO
    ENDIF
    
    !Initialize identity matrix and final demand vector
    II = 0
    FORALL(x = 1:NSM) 
        II(x,x) = 1 
    ENDFORALL
    !Initialize Final Demand Vector
    FD = 0.0
    FD(1) = 1.0
    
    Inventory = 0.0
    !Select the correct ironmaking technology. Not applicable for techs 25 and
    !26.    
    DO A = 1, (NST)
        DO B = 1, (NSS)
            IF ((STIM(A,B) == 1) .AND. (B>7) .AND. (B<21)) THEN
                !Select the correct ironmaking technology. Not applicable for techs 25 and
                !26.
                IF (A<25) THEN
                    SLCI(:,2) = SCMM(:,B)
                ELSE
                    SLCI(:,2) = 0.0   
                ENDIF
            ENDIF
            IF ((STIM(A,B) == 1) .AND. (B>20) .AND. (B<27)) THEN
                !Now select the correct steelmaking technology.
                SLCI(:,1) = SCMM(:,B)
                SLCI(2,1) = MetalInput(A,1)
                SLCI(3,1) = MetalInput(A,2)
                SLCI(4,1) = MetalInput(A,3)
            ENDIF
        ENDDO
        
        
        !The inventory matrix (SLCI) is now completely filled. Now calculate the inventory vector per technology.
        !First, we add the steelmaking step to the inventory vector. 
        Inventory(:,A) = SLCI(:,1) 
        !Add ironmaking step scaled to the need of PIP
        Inventory(:,A) = Inventory(:,A) + SLCI(2,1) * SLCI(:,2)
        !Add Sinter and Pellet inventory      
        Inventory(:,A) = Inventory(:,A) + (Inventory(7,A) * SCMM(:,3) + Inventory(8,A) * SCMM(:,4) + Inventory(9,A) * SCMM(:,5) + Inventory(10,A) * SCMM(:,6))
        !Add Coke inventory
        Inventory(:,A) = Inventory(:,A) + Inventory(5,A) * SCMM(:,1) + Inventory(6,A) * SCMM(:,2)
        !Add Oxygen inventory
        Inventory(:,A) = Inventory(:,A) + Inventory(11,A) * SCMM(:,7)
        !Add finishing step
        Inventory(:,A) = Inventory(:,A) + SCMM(:,27)/1.14
        
        !Material based Emission Intensity        
        EF = SUM(Inventory(:,A) * SMEF)
        EF_Fossil = SUM(Inventory(1:15,A) * SMEF(1:15))
        EF_Biobased = SUM(Inventory(16:24,A) * SMEF(16:24))
        
        !From here on: Put the results of the inventory calculation into the CostMatrix
        !Material based Energy Intensity
        BSTC(A,J,16) = SUM(Inventory(:,A) * SMED)
        STEI(A,J) = BSTC(A,J,16)
        STSC(A,J) = Inventory(4,A)
        
        DO MAT = 1, NSM-4
            BSTC(A,J,22+MAT) = Inventory(MAT,A)
        ENDDO

        !Select CCS technologies
        IF (BSTC(A,J,22) == 1) THEN
            !Increase fuel consumption
            Inventory(12:15,A) = 1.1 * Inventory(12:15,A)
            Inventory(18:20,A) = 1.1 * Inventory(18:20,A)
            !Capital investment costs
            IF (DATE == 2017) SCIN(A,J) = Inventory(21,A) + 50 * 0.9 * EF
            
            !Adjust O&M Costs due to CO2 transport and storage
            BSTC(A,J,3) = Inventory(22,A) + 5.0 * 0.9 * EF
            BSTC(A,J,4) = BSTC(A,J,3) * 0.3
            
            !Adjust EF for CCS utilising technologies
            BSTC(A,J,14) = 0.1 * EF - EF_Biobased
            STEF(A,J) = 0.1 * EF - EF_Biobased
            BSTC(A,J,15) = BSTC(A,J,14) *0.1
            
            !Adjust total Employment for complete steelmaking route (increase by 5% due to CCS)
            BSTC(A,J,5) = 1.05 * Inventory(24,A)
            SEPF(A,J) = 1.05 * Inventory(24,A)
            
            !Adjust electricity consumption due to CCS use
            BSTC(A,J,37) = BSTC(A,J,37) * 1.1
            
        ELSE
            !Capital investment costs
            IF (DATE == 2017) SCIN(A,J) = Inventory(21,A)

            !OM and dOM
            BSTC(A,J,3) = Inventory(22,A)
            BSTC(A,J,4) = BSTC(A,J,3) *0.3
            
            !EF and dEF
            BSTC(A,J,14) = EF - EF_Biobased
            STEF(A,J) = EF - EF_Biobased
            BSTC(A,J,15) = BSTC(A,J,14)*0.1
            
            !Employment
            BSTC(A,J,5) = Inventory(24,A)
            SEPF(A,J) = Inventory(24,A)
        ENDIF   
    ENDDO
    EF2 = BSTC(:,J,14)
    OM = BSTC(:,J,3)
    EMPL = Inventory(24,:)
ENDDO
END
