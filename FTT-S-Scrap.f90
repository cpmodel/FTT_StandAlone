!_______________________________________________________________________
!_______________________________________________________________________
! ============FTT:Steel model ========== Created by JF Mercure, F Knobloch, L. van Duuren, and P. Vercoulen
! Follows general equations of the FTT model (e.g. Mercure Energy Policy 2012)
! Adapted to the iron and steel industry.
! 
! FTT determines the technology mix
!________________________________________jm801@cam.ac.uk________________

!---------------Scrap availability calculation function-----------------
    
    SUBROUTINE IDFTTSScrap(NST,NXP,NR,NC5,NY1,NSM,NY,NQ,NBQRY,SHS1,SHS2,SXSC,BSTC,SSSR, SXSS,SXLT,SXLR,SXRR,SLT2,SPSA,SXS1,SXS2,SXS3,SXS4,BQRY,IRUN,DATE)
    IMPLICIT NONE
    
    INTEGER,INTENT(IN) :: NST,NXP,NR,NC5,NY1,NSM,NY,NQ,NBQRY
    INTEGER,INTENT(IN) :: IRUN,DATE
    
    REAL , INTENT(INOUT), DIMENSION(NR, NY1) :: SHS1, SHS2   !Historical steel production for the period of 1918-1917, and future steel production (2000-2100). Used to calculated scrap availability
    REAL , INTENT(INOUT), DIMENSION(NR, NY1) :: SXS1,SXS2,SXS3,SXS4 !Endogenous calculation of sector splits.
    REAL , INTENT(INOUT), DIMENSION(NR) :: SPSA
    REAL , INTENT(OUT), DIMENSION(NR) :: SXSC
    REAL , INTENT(INOUT), DIMENSION(NST,NR,NC5) :: BSTC
    REAL , INTENT(INOUT), DIMENSION(NR,NXP) :: SSSR, SXSS, SXLT, SXLR, SXRR, SLT2
    REAL , DIMENSION(NQ,NR,NBQRY) :: BQRY
    INTEGER , DIMENSION(NY) :: non_EU_mask, EU_mask
    REAL , DIMENSION(NY,NR) :: SteelUse,SteelUse0,SteelUse1,SteelUse2,SteelUse3,SteelUse4
    REAL , DIMENSION(NR) :: TotalSteelUse
    REAL :: TestB1_1, TestB1_2, TestB2_1, TestB2_2, TEST_SPLIT, TEST_TOTAL
    INTEGER :: J, PG, RetrieveDate, I
    SSSR = SXSS
    IF (DATE > 2017) SHS2(:,DATE - 2000) = SPSA
    SXSC = 0.0
    non_EU_mask =   (/ 3,3,3,2,3,3,3,3,3,3,3,3,2,0,0,3,4,1,1,2,4,3,3,3,3,1,4,4,3,1,1,3,4,4,4,4,4,4,4,4,4,4,2,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 /)
    EU_mask =       (/ 3,3,1,3,3,3,3,3,3,3,3,3,3,3,3,3,0,0,4,4,2,1,1,4,3,3,3,3,3,3,3,3,3,3,1,1,3,1,3,4,3,3,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4 /)
    DO J = 1,NR
        DO PG = 1, NXP
            RetrieveDate = NINT(DATE - SXLT(J,PG)) 
            
            !Calculate scrap availability
            !SXSCtot = SXSCpg + HSP(T-LTpg) * RRpg * SSpg * (1-LRpg) for all pg
            IF (RetrieveDate < 1918) THEN
                !If data has to be retrieved from a date at which is before the period included on the databank, then make an assumption what the steelproduction would have been
                !Assumed growth rate of 3.5%
                SXSC(J) = SXSC(J) + SHS1(J,1)*(0.965)**(1918 - RetrieveDate) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG))
            ELSEIF (RetrieveDate < 2017) THEN
                SXSC(J) = SXSC(J) + SHS1(J,RetrieveDate - 1917) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                IF (RetrieveDate+1 < 2017) SXSC(J) = SXSC(J) + SHS1(J,RetrieveDate+1 - 1917) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                IF (RetrieveDate+2 < 2017) SXSC(J) = SXSC(J) + SHS1(J,RetrieveDate+2 - 1917) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                IF (RetrieveDate+3 < 2017) SXSC(J) = SXSC(J) + SHS1(J,RetrieveDate+3 - 1917) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.15
                IF (RetrieveDate+4 < 2017) SXSC(J) = SXSC(J) + SHS1(J,RetrieveDate+4 - 1917) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.1
                IF (RetrieveDate+5 < 2017) SXSC(J) = SXSC(J) + SHS1(J,RetrieveDate+5 - 1917) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.1
                IF (RetrieveDate+6 < 2017) SXSC(J) = SXSC(J) + SHS1(J,RetrieveDate+6 - 1917) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.05
                IF (RetrieveDate+1 .GE. 2017) SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+1 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                IF (RetrieveDate+2 .GE. 2017) SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+2 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                IF (RetrieveDate+3 .GE. 2017) SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+3 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.15
                IF (RetrieveDate+4 .GE. 2017) SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+4 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.1
                IF (RetrieveDate+5 .GE. 2017) SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+5 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.1
                IF (RetrieveDate+6 .GE. 2017) SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+6 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.05
            !If endogenous sector split calculation is going to be included then it should be fed in here to replace SXSS
            ELSEIF (RetrieveDate .GE. 2017) THEN
                SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+1 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+2 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.2
                SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+3 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.15
                SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+4 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.1
                SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+5 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.1
                SXSC(J) = SXSC(J) + SHS2(J,RetrieveDate+6 - 2000) * SXRR(J,PG) * SSSR(J,PG) * (1-SXLR(J,PG)) * 0.05
                
            ENDIF
        ENDDO
    ENDDO
END
    
            
            
        