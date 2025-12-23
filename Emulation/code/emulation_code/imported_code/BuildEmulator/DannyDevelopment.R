#A function to obtain the data field from ncdfs and to get the data in the right order, adjusting for missingness
#INPUT: DataDir a string with the directory that contains the ncdf data
#NOTE: It is assumed we are working in the directory that contains the parameter data
GetDataAndDesign <- function(DataDir="~/Documents/CanadaData/MSLP/", prefix="psl", variable="psl"){
  param.names <- c("FACAUT", "UICEFAC", "CT", "CBMF", "WEIGHT", "DRNGAT", "RKXMIN", "XINGLE", "ALMC", "ALF", "TAUS1", "CSIGMA", "CVSG")
  param.lows <- c(0.5, 600, 0.75e-03, 0.01, 0.1, 0, 0.01, 10, 10, 1e06, 1800, 0.05, 5e-04)
  param.highs <- c(5, 7500, 1.3e-03, 0.06, 1, 0.1, 0.5, 50, 1000, 1e09, 21600, 0.5, 1e-02)
  which.logs <- c(9, 10, 13)
  param.defaults <- c(2.5, 4800, 1e-03, 0.03, 0.75, 0.005, 0.1, 10, 600, 5e08, 6*3600, 0.2, 3e-03)
  plows <- param.lows
  plows[which.logs] <- log10(plows[which.logs])
  phighs <- param.highs
  phighs[which.logs] <- log10(phighs[which.logs])
  StandardParams_U <- param.defaults
  StandardParams_U[which.logs] <- log10(StandardParams_U[which.logs])
  StandardParams_U <- ((StandardParams_U-plows)/(phighs-plows))*2 - 1
  tid0 <- "dma_1-000"
  Stands <- data.frame(tid0,t(StandardParams_U))
  names(Stands) <- c("t_IDs",param.names)
  load("tMasterWave1_US.RData")
  tID1 <- "dma"
  labels <- c("0","1","2","3","4","5","6","7","8","9")
  counter <- 1
  first <- 2
  second <- 1
  third <- 1
  t_IDs <- rep(tID1,65)
  twave <- "1"
  while(counter<66){
    tnew <- paste(labels[third],labels[second],labels[first],sep="")
    t_IDs[counter] <- paste(t_IDs[counter],twave,sep="_")
    t_IDs[counter] <- paste(t_IDs[counter],tnew,sep="-")
    first <- first + 1
    if(first>10){
      second <- second+1
      first <- 1
      if(second > 10){
        third <- third+1
        second <- 1
      }
    }
    counter <- counter + 1
  }
  tMasterWave1_US$t_IDs <- t_IDs
  tMasterWave1_US <- rbind(tMasterWave1_US, Stands)
  nc000 <- nc_open(file=paste(DataDir, prefix, "_Amon_DevAM4-4_dma_1-000_r1i1p1_200301-200812.nc", sep=""))
  psl000 <- ncvar_get(nc000,variable)
  PSLwave1all <- array(NA, dim = c(dim(psl000),66)) 
  PSLwave1all[,,,1] <- psl000
  numbers <- 1:66
  numbers <- sapply(numbers, function(e) paste("0",numbers[e],sep=""))
  numbers[1:9] <- sapply(1:9, function(e) paste("0",numbers[e],sep=""))
  numbers
  Missing <- c()
  k <- 2
  while(k <= 66){
    filename <- paste(DataDir, prefix, "_Amon_DevAM4-4_dma_1-", numbers[k-1], "_r1i1p1_200301-200812.nc", sep="")
    newNC <- try(nc_open(file=filename),silent=TRUE)
    if(inherits(newNC,"try-error")){
      Missing <- c(Missing,numbers[k-1])
      k <- k+1
    }
    else{ 
      PSLwave1all[,,,k] <- ncvar_get(newNC,variable)
      k <- k+1
    }
  }
  MissingMembers <- which(numbers %in% Missing)
  wave1Design <- tMasterWave1_US[-MissingMembers,]
  N1 <- length(wave1Design[,1])
  wave1Design <- wave1Design[c(N1,1:(N1-1)),]
  PSLwave1all <- PSLwave1all[,,,-(MissingMembers+1)]
  dim(PSLwave1all) <- c(prod(dim(PSLwave1all)[1:2]),dim(PSLwave1all)[3:4])
  OriginalField <- aperm(PSLwave1all,c(1,3,2))
  return(list(Design=wave1Design[,-1], FieldData = OriginalField))
}

#EXAMPLE CALL - Tested 27/07/16
#ourData <- GetDataAndDesign()

#Get 3D data using pressure and lat (zonally integrating over longitude)
GetDataAndDesignZonalPressure <- function(DataDir="~/Documents/CanadaData/MSLP/", prefix="psl", variable="psl"){
  param.names <- c("FACAUT", "UICEFAC", "CT", "CBMF", "WEIGHT", "DRNGAT", "RKXMIN", "XINGLE", "ALMC", "ALF", "TAUS1", "CSIGMA", "CVSG")
  param.lows <- c(0.5, 600, 0.75e-03, 0.01, 0.1, 0, 0.01, 10, 10, 1e06, 1800, 0.05, 5e-04)
  param.highs <- c(5, 7500, 1.3e-03, 0.06, 1, 0.1, 0.5, 50, 1000, 1e09, 21600, 0.5, 1e-02)
  which.logs <- c(9, 10, 13)
  param.defaults <- c(2.5, 4800, 1e-03, 0.03, 0.75, 0.005, 0.1, 10, 600, 5e08, 6*3600, 0.2, 3e-03)
  plows <- param.lows
  plows[which.logs] <- log10(plows[which.logs])
  phighs <- param.highs
  phighs[which.logs] <- log10(phighs[which.logs])
  StandardParams_U <- param.defaults
  StandardParams_U[which.logs] <- log10(StandardParams_U[which.logs])
  StandardParams_U <- ((StandardParams_U-plows)/(phighs-plows))*2 - 1
  tid0 <- "dma_1-000"
  Stands <- data.frame(tid0,t(StandardParams_U))
  names(Stands) <- c("t_IDs",param.names)
  load("tMasterWave1_US.RData")
  tID1 <- "dma"
  labels <- c("0","1","2","3","4","5","6","7","8","9")
  counter <- 1
  first <- 2
  second <- 1
  third <- 1
  t_IDs <- rep(tID1,65)
  twave <- "1"
  while(counter<66){
    tnew <- paste(labels[third],labels[second],labels[first],sep="")
    t_IDs[counter] <- paste(t_IDs[counter],twave,sep="_")
    t_IDs[counter] <- paste(t_IDs[counter],tnew,sep="-")
    first <- first + 1
    if(first>10){
      second <- second+1
      first <- 1
      if(second > 10){
        third <- third+1
        second <- 1
      }
    }
    counter <- counter + 1
  }
  tMasterWave1_US$t_IDs <- t_IDs
  tMasterWave1_US <- rbind(tMasterWave1_US, Stands)
  nc000 <- nc_open(file=paste(DataDir, prefix, "_Amon_DevAM4-4_dma_1-000_r1i1p1_200301-200812.nc", sep=""))
  initArray <- ncvar_get(nc000,variable)
  Pressure <- ncvar_get(nc000, "plev")
  lat <- ncvar_get(nc000, "lat")
  zonalArray <- apply(initArray, MARGIN = 2:4, mean)
  Arraywave1all <- array(NA, dim = c(dim(zonalArray),66)) 
  Arraywave1all[,,,1] <- zonalArray
  numbers <- 1:66
  numbers <- sapply(numbers, function(e) paste("0",numbers[e],sep=""))
  numbers[1:9] <- sapply(1:9, function(e) paste("0",numbers[e],sep=""))
  numbers
  Missing <- c()
  k <- 2
  while(k <= 66){
    filename <- paste(DataDir, prefix, "_Amon_DevAM4-4_dma_1-", numbers[k-1], "_r1i1p1_200301-200812.nc", sep="")
    newNC <- try(nc_open(file=filename),silent=TRUE)
    if(inherits(newNC,"try-error")){
      Missing <- c(Missing,numbers[k-1])
      k <- k+1
    }
    else{ 
      Arraywave1all[,,,k] <- apply(ncvar_get(newNC,variable), MARGIN=2:4, mean)
      k <- k+1
    }
  }
  MissingMembers <- which(numbers %in% Missing)
  wave1Design <- tMasterWave1_US[-MissingMembers,]
  N1 <- length(wave1Design[,1])
  wave1Design <- wave1Design[c(N1,1:(N1-1)),]
  Arraywave1all <- Arraywave1all[,,,-(MissingMembers+1)]
  dim(Arraywave1all) <- c(prod(dim(Arraywave1all)[1:2]),dim(Arraywave1all)[3:4])
  OriginalField <- aperm(Arraywave1all,c(1,3,2))
  return(list(Design=wave1Design[,-1], FieldData = OriginalField))
}

#Example call: Tested 11/8/16
#ourDataTA <- GetDataAndDesignZonalPressure(DataDir="~/Documents/CanadaData/ta/", prefix="ta", variable="ta")

#Convert the 3D spatio-temporal data into spatial data, by extracting the required monthly average and get the svd basis
#INPUTS:
#OriginalField is a data array with the first dimension the length of the field (so a strung out map), the second the ensemble size and the third in months from the start of the simulation.
#which.months Jan = 1, Dec = 12. DJF is then c(12,1,2) but consecutive months will be taken (e.g. c(12, 13, 14)), 
#num.years: how many of the last n years of the simulation to keep
#scaling: A scaling to apply for numerical tractability
#OUTPUTS: List containing:
#tBasis: the required svd basis
#CentredField the correctly averaged field with the ensemble mean removed from each ensemble member and then scaled by the scaling
#EnsembleMean The mean field of the correctly averaged ensemble. (So e.g. the ensemble mean for DJF)
#scaling: The used scaling from the input to the function.
ExtractCentredScaledDataAndBasis <- function(OriginalField, which.months, num.years, scaling=1){
  tmonths <- seq(from=which.months[1],by=1,length.out = length(which.months))
  allYears <- dim(OriginalField)[3]
  while(tmonths[length(tmonths)] + 12 <= allYears){
    l1 <- length(tmonths)
    tmonths <- c(tmonths, c(tmonths[c(l1-2,l1-1,l1)])+12)
  }
  needed.months <- num.years*length(which.months)
  if(needed.months < length(tmonths))
    tmonths <- tmonths[-c(1:(length(tmonths) - needed.months))]
  NewArray <- OriginalField[,,tmonths]
  MeanField <- apply(NewArray, MARGIN=c(1,2), mean)
  #MeanField is now the uncentered, unscaled field we want to emulate. The last step is to centre, scale and extract the basis
  EnsMean <- apply(MeanField, MARGIN=1, mean)
  CentredField <- MeanField
  for(i in 1:dim(MeanField)[2]){
    CentredField[,i] <- CentredField[,i] - EnsMean
  }
  CentredField <- CentredField/scaling
  Basis <- svd(t(CentredField))$v
  return(list(tBasis=Basis, CentredField = CentredField, EnsembleMean = EnsMean, scaling=scaling, Months=tmonths))
}

#Example call: Tested 27/7/16
#DJFdata <- ExtractCentredScaledDataAndBasis(OriginalField = ourData$FieldData, which.months=c(12,1,2), num.years=5, scaling=10^4)
#MAMdata <- ExtractCentredScaledDataAndBasis(OriginalField = ourData$FieldData, which.months=c(3,4,5), num.years=5, scaling=10^4)
#JJAdata <- ExtractCentredScaledDataAndBasis(OriginalField = ourData$FieldData, which.months=c(6,7,8), num.years=5, scaling=10^4)
#SONdata <- ExtractCentredScaledDataAndBasis(OriginalField = ourData$FieldData, which.months=c(9,10,11), num.years=5, scaling=10^4)

#Change to separate temporal reduction and basis calculation (data cleaning vs every-application calculations.)
ExtractData <- function(OriginalField, which.months, num.years){
  tmonths <- seq(from=which.months[1],by=1,length.out = length(which.months))
  allYears <- dim(OriginalField)[3]
  while(tmonths[length(tmonths)] + 12 <= allYears){
    l1 <- length(tmonths)
    tmonths <- c(tmonths, c(tmonths[c(l1-2,l1-1,l1)])+12)
  }
  needed.months <- num.years*length(which.months)
  if(needed.months < length(tmonths))
    tmonths <- tmonths[-c(1:(length(tmonths) - needed.months))]
  NewArray <- OriginalField[,,tmonths]
  MeanField <- apply(NewArray, MARGIN=c(1,2), mean)
  #MeanField is now the uncentered, unscaled field we want to emulate. The last step is to centre, scale and extract the basis
  return(list(Data = MeanField))
}

CentreAndBasis <- function(MeanField, scaling=1, weightinv = NULL){
  EnsMean <- apply(MeanField, MARGIN=1, mean)
  CentredField <- MeanField
  for(i in 1:dim(MeanField)[2]){
    CentredField[,i] <- CentredField[,i] - EnsMean
  }
  CentredField <- CentredField/scaling
  Basis <- wsvd(t(CentredField), weightinv = weightinv)$v
  return(list(tBasis=Basis, CentredField = CentredField, EnsembleMean = EnsMean, scaling=scaling))
}



#Code for plotting a basis
#INPUTS: 
#Basis: The basis whose columns you want to plot
#num.vectors: The number of basis vectors you want to plot (currently an integer)
#breakVec: An optional vector containing the breaks for the plot image.
#col.fun: A colour function defaulting to tim.colors from fields.
#lat, lon ordered latitude and longitude as output from the relevant ncdf (see example).
plotBasis <- function(Basis, num.vectors, ylat, xlon, breakVec=NULL, col.fun=tim.colors, latlon=TRUE){
  tRange <- range(Basis)
  if(is.null(breakVec))
    breakVec <- pretty(tRange,65)
  cols <- col.fun(length(breakVec)-1)
  mar <- par("mar")
  tmar <- mar
  tmar[4] <- tmar[2]
  tmar[2] <- 1
  tmar <- tmar/4
  tmar[4] <- 3
  tmar[1] <- 0.3
  tmar[3] <- tmar[1]
  if(num.vectors<5){
    N <- 2
    M <- 2
  }
  else if(num.vectors < 10){
    N <- 3
    M <- 3
  }
  else if(num.vectors < 13){
    N <- 3
    M <- 4
  }
  else if(num.vectors < 17){
    N <- 4
    M <- 4
  }
  else if(num.vectors < 21){
    N <- 5
    M <- 4
  }
  else if(num.vectors < 26){
    N <- 5
    M <- 5
  }
  else{
    stop("Choose 25 basis vectors or less, or change code")
  }
  key <- rep(1, M)
  numMatrix <- t(matrix(2:(N*M+1),nrow=N,ncol=M))
  layout(cbind(numMatrix, key), widths=c(rep(0.3,N), lcm(1.6)))
  par(mar=tmar,las=1)
  plot.new()
  plot.window(c(0,1), range(breakVec), xaxs="i", yaxs="i")
  rect(0, breakVec[-length(breakVec)],1,breakVec[-1],col=cols)
  axis(4)
  par(mar=rep(0.1,4),usr=c(0,1,0,1))
  for(k in 2:(num.vectors+1)){
    basisVec <- Basis[,(k-1)]
    dim(basisVec) <- c(length(xlon),length(ylat))
    plot.new()
    if(latlon)
      plot.window(c(-180,180),c(-90,90))
    else
      plot.window(range(xlon)[c(2,1)], range(ylat)[c(2,1)])
    .filled.contour(xlon[order(xlon)],ylat[order(ylat)],basisVec[order(xlon),order(ylat)],breakVec,cols)
    rect(range(xlon)[1], range(ylat)[1],range(xlon)[2], range(ylat)[2])
    if(latlon)
      map("world",add=T,wrap=FALSE,interior=FALSE)
  }
}

#Example Call: Tested 27/7/16
#nc000 <- nc_open(file="~/Documents/CanadaData/MSLP/psl_Amon_DevAM4-4_dma_1-000_r1i1p1_200301-200812.nc")
#lat <- ncvar_get(nc000,"lat")
#lon <- ncvar_get(nc000,"lon")
#lon[lon>180] <- lon[lon>180] - 360
#plotBasis(Basis=DJFdata$tBasis, 9, lat, lon, pretty(range(DJFdata$tBasis[,1:9]),65))


#Function to compute the coefficients of an ensemble by projecting it onto a given basis.
#INPUTS:
#SpatialFieldsEnsemble: The ensemble consisting of an l x n matrix where l is the number of gridboxes in the field and n is the number of ensemble members.
#basis: The l x nprime matrix of basis vectors where nprime <= n
#orthogonal: Boolean to indicate whether the basis vectors are orthogonal.
#When time is not in the ensemble due to pre-averaging
#OUTPUT: nprime x n matrix of coefficients for the given ensemble
StandardCoefficients <- function(SpatialFieldsEnsemble, basis, orthogonal=TRUE){
  EnsDims <- dim(SpatialFieldsEnsemble)
  n <- EnsDims[2]
  l <- EnsDims[1]
  nprime <- dim(basis)[2]
  if(is.null(nprime))
    nprime <- 1
  OrthogCoeffs <- tensor(basis,SpatialFieldsEnsemble,1,1)
  if(orthogonal){
    return(OrthogCoeffs)
  }
  else{
    Qbasis <- chol(crossprod(basis))
    t1 <- backsolve(Qbasis, OrthogCoeffs, transpose =TRUE) #n' x n
    t2 <- backsolve(Qbasis, diag(nprime), transpose=TRUE) #n' x n'
    return(tensor(t2,t1,2,1))
  }
}

#Example call: (first 10 coefficients) Tested 29/07/16
#DJFcoefs <- StandardCoefficients(DJFdata$CentredField, DJFdata$tBasis[,1:10], orthogonal=FALSE)

#Reconstruct a spatial field from a set of basis vectors and relevant coefficients
#INPUTS:
#coefficients: nprime x n matrix, 
#basis: l x nprime matrix
Reconstruct <- function(coefficients, basis){
  tensor(basis, coefficients, 2, 1)
}

#Example call: Tested 29/07/16
#Reconstructed <- Reconstruct(DJFcoefs, DJFdata$tBasis[,1:10])

#INPUTS:
#Coefficients: Coefficients of basis computed using StandardCoefficients
#EnsembleData: List generated by ExtractCentredScaledDataAndBasis
#which.plot. Which ensemble member to plot
#OriginalScale: Boolean. Do you want to compute the basis back on the original scale including the ensemble mean?
plot.EnsembleMemberReconstruction <- function(coefficients, EnsembleData, which.plot=1, OriginalScale=TRUE){
  nprime <- dim(coefficients)[1]
  trecon <- Reconstruct(coefficients=coefficients, basis=EnsembleData$tBasis[,1:nprime])
  VarianceExplained <- crossprod(c(trecon))/crossprod(c(EnsembleData$CentredField))
  VarianceExplained.String <- paste("Ensemble variance explained = ", 100*VarianceExplained, "%", sep="" )
  print(VarianceExplained.String)
  EnsembleField <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconField <- trecon[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  par(mfrow=c(3,1))
  tdims <- c(length(lon), length(lat))
  tanom <- EnsembleField - ReconField
  image.plot(lon[order(lon)],lat, matrix(EnsembleField,nrow=tdims[1],ncol=tdims[2])[order(lon),],main=paste("Ensemble member",which.plot,sep=" "))
  map("world",add=T)
  image.plot(lon[order(lon)],lat, matrix(ReconField,nrow=tdims[1],ncol=tdims[2])[order(lon),], main="Reconstruction")
  map("world",add=T)
  image.plot(lon[order(lon)],lat, matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),], main = "Anomaly")
  map("world",add=T)
}

#Example call: Tested 29/07/16
#plot.EnsembleMemberReconstruction(DJFcoefs, DJFdata, 1) 

#Function to extract multiple observation data sets centred using the mean of an ensemble and scaled using the same scaling as our ensemble data
#INPUTS:
#DataFiles: A string vector of file locations for data fields
#which.obs: String declaring the field name in the ncdf
#EnsembleData: Post-processed ensemble data in list form as output by ExtractCentredScaledDataAndBasis
#orignalLat: The value of lat extracted for the ensemble (we need lat and lon to match)
#orignalLon: The value of lon extracted for the ensemble (we need lat and lon to match)
#OUTPUT:
#A list of observation vectors each centred with the mean of the ensemble as the EnsembleData was and then scaled in the same way.
GetObservations <- function(DataFiles, which.obs="psl", EnsembleData, originalLat=lat, originalLon=lon, permanent.scaling=1){
    numFiles <- length(DataFiles)
    ObsList <- list()
    for(i in 1:numFiles){
      ncdatatemp <- nc_open(file=DataFiles[i])
      psltemp <- ncvar_get(ncdatatemp, which.obs)/permanent.scaling
      datlon <- ncvar_get(ncdatatemp, "lon")
      datlon[datlon>180] <- datlon[datlon>180] - 360
      ldl <- length(datlon)
      if((ldl>length(originalLon))&(datlon[1]==datlon[ldl])){
        warning("Repeated Longitude, removing last row of data")
        psltemp <- psltemp[-ldl,,]
      }
      datlat <- ncvar_get(ncdatatemp, "lat")
      latlen <- length(datlat)
      if((latlen>length(originalLat))&(datlat[1]==datlat[latlen])){
        warning("Repeated Latitude, removing last column of data")
        psltemp <- psltemp[,-latlen,]
      }
      dim(psltemp) <- c(prod(dim(psltemp)[c(1,2)]),dim(psltemp)[3])
      ttimes <- ncvar_get(ncdatatemp, "time")
      mytimes <- (which(ttimes>=200309)[1]):(which(ttimes<=200808)[length(which(ttimes<=200808))])
      psltemp <- psltemp[,mytimes]
      psltemp <- psltemp[,EnsembleData$Months-8]
      psltempMean <- apply(psltemp,MARGIN = 1, mean)
      ObsList <- c(ObsList, list((psltempMean - EnsembleData$EnsembleMean)/EnsembleData$scaling))
    }
    return(ObsList)
}

#Example Call: Tested 29/07/16
#tDataFiles <- c("~/Documents/CanadaData/MSLP/psl_Amon_ERA-Interim_reanalysis_r1i1p1_197901-201510.nc", "~/Documents/CanadaData/MSLP/psl_Amon_MERRA_reanalysis_r1i1p1_197901-201511.nc", "~/Documents/CanadaData/MSLP/psl_Amon_NCEP2_reanalysis_r1i1p1_197901-201512.nc")
#tObs <- GetObservations(DataFiles=tDataFiles, which.obs="psl", EnsembleData = DJFdata)
#tObsJJA <- GetObservations(DataFiles=tDataFiles, which.obs="psl", EnsembleData = JJAdata)

GetObservationsZonalPressures <- function(DataFiles, which.obs="psl", EnsembleData, originalLat=lat, originalLon=lon, originalPressure=Pressure, permanent.scaling=1){
  numFiles <- length(DataFiles)
  ObsList <- list()
  for(i in 1:numFiles){
    ncdatatemp <- nc_open(file=DataFiles[i])
    psltemp <- ncvar_get(ncdatatemp, which.obs)/permanent.scaling
    datlon <- ncvar_get(ncdatatemp, "lon")
    datlon[datlon>180] <- datlon[datlon>180] - 360
    ldl <- length(datlon)
    if((ldl>length(originalLon))&(datlon[1]==datlon[ldl])){
      warning("Repeated Longitude, removing last row of data")
      psltemp <- psltemp[-ldl,,,]
    }
    datlat <- ncvar_get(ncdatatemp, "lat")
    latlen <- length(datlat)
    if((latlen>length(originalLat))&(datlat[1]==datlat[latlen])){
      warning("Repeated Latitude, removing last column of data")
      psltemp <- psltemp[,-latlen,,]
    }
    datpressure <- ncvar_get(ncdatatemp, "Z")
    prelen <- length(datpressure)
    if((prelen>length(originalPressure))&(datpressure[1]==datpressure[prelen])){
      warning("Repeated Pressure, removing last columnBar of data")
      psltemp <- psltemp[,,-prelen,]
    }
    tzonal <- apply(psltemp, MARGIN=2:4, mean)
    dim(tzonal) <- c(prod(dim(tzonal)[c(1,2)]),dim(tzonal)[3])
    ttimes <- ncvar_get(ncdatatemp, "time")
    mytimes <- (which(ttimes>=200309)[1]):(which(ttimes<=200808)[length(which(ttimes<=200808))])
    tzonal <- tzonal[,mytimes]
    tzonal <- tzonal[,EnsembleData$Months-8]
    psltempMean <- apply(tzonal,MARGIN = 1, mean)
    ObsList <- c(ObsList, list((psltempMean - EnsembleData$EnsembleMean)/EnsembleData$scaling))
  }
  return(ObsList)
}

#Plot an ensemble member and an observation field and their anomaly.
#INPUTS:
#ObsField: List of Observation fields as output by GetObservations
#EnsembleData: Post-processed ensemble data in list form as output by ExtractCentredScaledDataAndBasis
#which.obs: Integer: which member of the observations list
#which.plot: Integer: which ensemble member
#OriginalScale: Not yet coded
#AnomBreaks: Vector of breaks for plotting (see ?image)
#AnomCol: Vector of colours for plotting (1 lower than breaks, see ?image) 
plot.DataEnsembleMember <- function(ObsField, EnsembleData, which.obs = 1, which.plot=1, OriginalScale = TRUE, AnomCols=NULL, AnomBreaks=NULL,add.contours=FALSE,Anomaly.Only=FALSE,tcontours=seq(from=-40,by=2,to=40), contour.zero=FALSE, ...){
  EnsembleField <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- EnsembleField - ObsToScale
  print(range(tanom))
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  tdims <- c(length(lon), length(lat))
  if(!Anomaly.Only){
    par(mfrow=c(3,1))
    image.plot(lon[order(lon)],lat, matrix(EnsembleField,nrow=tdims[1],ncol=tdims[2])[order(lon),],main=paste("Ensemble member",which.plot,sep=" "))
    map("world",add=T)
    image.plot(lon[order(lon)],lat, matrix(ObsToScale,nrow=tdims[1],ncol=tdims[2])[order(lon),], main="Observations")
    map("world",add=T)
  }
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  image.plot(lon[order(lon)],lat, matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = TRUE, ...)
  map("world",add=T, wrap=TRUE,interior = FALSE)
  if(add.contours){
    if(!contour.zero)
      tcontours <- tcontours[-which(tcontours==0)]
    contour(x=lon[order(lon)],y=lat, z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE,levels=tcontours,lty=1+(tcontours<0),lwd=0.8)
  }
}

#Example Call: Tested 31/07/16
#plot.DataEnsembleMember(tObs,DJFdata, 1,1)
#plot.DataEnsembleMember(tObsJJA,JJAdata, 3,1, AnomBreaks = GenerateBreaks(WhiteRange = c(-300,300), ColRange = c(-1200,1200), Length.Colvec = 7,DataRange=c(-1800,1550)), AnomCols = AnomalyColours(7),add.contours = TRUE, Anomaly.Only = TRUE)

plot.DataEnsembleMemberPressure <- function(ObsField, EnsembleData, Pressure, which.obs = 1, which.plot=1, AnomCols=NULL, AnomBreaks=NULL,add.contours=FALSE,Anomaly.Only=FALSE, scale.log=TRUE, tmain="", tcontour=seq(from=-40,by=1,len=100)){
  EnsembleField <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- EnsembleField - ObsToScale
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  tdims <- c(length(lat), length(Pressure))
  if(!Anomaly.Only){
    par(mfrow=c(3,1))
    image.plot(lon[order(lon)],lat, matrix(EnsembleField,nrow=tdims[1],ncol=tdims[2])[order(lon),],main=paste("Ensemble member",which.plot,sep=" "))
    map("world",add=T)
    image.plot(lon[order(lon)],lat, matrix(ObsToScale,nrow=tdims[1],ncol=tdims[2])[order(lon),], main="Observations")
    map("world",add=T)
  }
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  plot.new()
  if(scale.log){
    plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  else{
    plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  title(main=tmain, xlab="Lattitude", ylab="Pressure")
  Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
  Pvec <- Pressure[order(Pressure)]/100
  Pvec <- as.character(Pvec)
  LabVec <- rep("",length(Pvec))
  if(scale.log){
    LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
    Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
  }
  else{
    LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
    Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
  }
  box()
  #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
  if(add.contours){
    if(scale.log){
      contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    }
    else{
      contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
    }
  }
}

#Example Call: See DeckTA.R Tested 11/8/16

#Function to create a data frame with the inputs and outputs together ready for emulation
#INPUTS: 
#Design. The Ensemble design in the order that fields appear in EnsembleData
#EnsembleData: #EnsembleData: Post-processed ensemble data in list form as output by ExtractCentredScaledDataAndBasis
#HowManyBasisVectors: Integer saying how many coefficients to put in the data frame.
#Noise: Boolean. Add a column of uniform random numbers called 'Noise' to the data?
#OUTPUTS: Data frame ready to be called 'tData' in Danny's emulator code.
GetEmulatableData <- function(Design, EnsembleData, HowManyBasisVectors, Noise=TRUE){
  if(Noise){
    Noise <- runif(length(Design[,1]),-1,1)
    Design <- cbind(Design, Noise)
  }
  tcoefs <- StandardCoefficients(EnsembleData$CentredField, EnsembleData$tBasis[,1:HowManyBasisVectors], orthogonal=FALSE)
  tData <- cbind(Design, t(tcoefs))
  ln <- length(names(tData))
  names(tData)[(ln-HowManyBasisVectors+1):ln] <- paste("C",1:HowManyBasisVectors,sep="")
  tData
}

#Example Call: Tested 1/8/16
#tData <- GetEmulatableData(Design = ourData$Design, EnsembleData = DJFdata, HowManyBasisVectors = 10)

#Function to generate initial emulators for each of the first few basis coefficients in the spatial field
#Note each emulator needs diagnosing, tuning and validating
#INPUTS:
#tData: The data frame to emulate, format as output by GetEmulatableData
#HowManyEmulators: NULL (build all emulators) or Integer n: build the first n.
#OUTPUT:
#A list each element contains the usual emulator list that can be tuned, diagnosed and run with EMULATOR.gp
InitialFieldEmulators <- function(tData, HowManyEmulators){
  twd <- getwd()
  setwd("~/Dropbox/ORCAcode")
  source("EmulateCode.R")
  setwd(twd)
  lastCand <- which(names(tData)=="Noise")
  tfirst <- lastCand + 1
  if(is.null(HowManyEmulators))
    HowManyEmulators <- length(names(tData)) - lastCand
  lapply(1:HowManyEmulators, function(k) try(BuildEmulator(Response=names(tData)[lastCand+k], tData=tData, CandidateInputs=names(tData)[1:lastCand], CandidateFactors=NULL, fit.gp=TRUE, fit.correlation.params=TRUE, dString="tData", cov.fun="powerexp", Half.Lengths=rep(1,lastCand-1), Half.Factors=NULL, Powers=1.9, Matern.Root=NULL, maxdf=ceiling(length(tData[,1])/10), Nugget=NULL, CLS=NULL, num.long=5000, MAP=TRUE, MAP.time=30),silent=TRUE))
}

NoiseTry <- function(tData, Response){
  N <- length(tData[,1])
  tData$Noise <- runif(N,-1,1)
  lastCand <- which(names(tData)=="Noise")
  BuildEmulator(Response=Response, tData=tData, CandidateInputs=names(tData)[1:lastCand], CandidateFactors=NULL, fit.gp=TRUE, fit.correlation.params=TRUE, dString="tData", cov.fun="powerexp", Half.Lengths=rep(1,lastCand-1), Half.Factors=NULL, Powers=1.9, Matern.Root=NULL, maxdf=ceiling(length(tData[,1])/10), Nugget=NULL, CLS=NULL, num.long=5000, MAP=TRUE, MAP.time=30)
}

#Example call: Tested 1/8/16
#EmulatorsList <- InitialFieldEmulators(tData, NULL)

#Extract the standard deviations for the random coefficients of the residual basis vectors
#INPUTS:
#EnsembleData: List from ExtractCentredScaledDataAndBasis
#num.basisVecs: How many basis vectors are we fitting on?
Truncation.SD <- function(EnsembleData, num.basisVecs){
  ALLcoefs <- StandardCoefficients(EnsembleData$CentredField, EnsembleData$tBasis, orthogonal=FALSE)
  Truncated.Coefs <- ALLcoefs[-c(1:num.basisVecs),]
  nT <- length(Truncated.Coefs[,1])
  sds <- sapply(1:nT, function(k) sd(Truncated.Coefs[k,]))
  sds
}

#Example call: Tested 5/8/16
#trunc.sds <- Truncation.SD(JJAdata, num.basisVecs=10)

#Function to generate LOO coefficients for emulators and to plot them
#INPUTS:
#EmulatorList: A list of Emulators
#tData: Emulatable Data (see GetEmulatableData)
#Which.Em: Which member of EmulatorList are we assessing?
#sds: How many standard deviations big are the error bars?
#NamesToPlot: The last 3 plots can be of the output vs a named input. Integers or names can be specified as well as NULL (To give the first 3 active variables)
LOOcoefficients <- function(EmulatorList, tData, Which.Em = 1, sds=2, NamesToPlot=NULL){
  par(mfrow=c(2,2),mar=c(4,4,1,1))
  if(is.null(NamesToPlot)){
    tplot1 <- EmulatorList[[Which.Em]]$Names[1]
    tplot2 <- EmulatorList[[Which.Em]]$Names[2]
    tplot3 <- EmulatorList[[Which.Em]]$Names[3]
  }
  else if(is.integer(NamesToPlot)){
    tplot1 <- EmulatorList$Names[NamesToPlot[1]]
    tplot2 <- EmulatorList$Names[NamesToPlot[2]]
    tplot3 <- EmulatorList$Names[NamesToPlot[3]]
  }
  else{
    tplot1 <- NamesToPlot[1]
    tplot2 <- NamesToPlot[2]
    tplot3 <- NamesToPlot[3]
  }
  Noise.Index <- which(names(tData)=="Noise")
  tresponse <- Noise.Index + Which.Em
  tLOOS0 <- LOOplotMaker(tData=tData, tEmulator = EmulatorList[[Which.Em]], sds=sds, num.inputs=Noise.Index-1, which.response=tresponse, which.plot=NULL, LOOs=NULL)
  tLOOS <- LOOplotMaker(tData=tData, tEmulator = EmulatorList[[Which.Em]], sds=sds, num.inputs=Noise.Index-1, which.response=tresponse, which.plot=tplot1, LOOs=tLOOS0)
  tLOOS <- LOOplotMaker(tData=tData, tEmulator = EmulatorList[[Which.Em]], sds=sds, num.inputs=Noise.Index-1, which.response=tresponse, which.plot=tplot2, LOOs=tLOOS0)
  tLOOS <- LOOplotMaker(tData=tData, tEmulator = EmulatorList[[Which.Em]], sds=sds, num.inputs=Noise.Index-1, which.response=tresponse, which.plot=tplot3, LOOs=tLOOS0)
}

#Example call: Tested 01/08/16
#LOOcoefficients(EmulatorList = EmulatorsList, tData = tData, Which.Em = 10, sds=2, NamesToPlot = NULL)

#Function to plot a LOO emulator of the projection onto the field and compare with the ensemble projection.
#INPUTS: 
#Emulator: the usual form of emulator with all of it elements. Note this could be changed to an expectation and variance at one point.
#EnsembleData: Post-processed ensemble data in list form as output by ExtractCentredScaledDataAndBasis
#TruncatedSDs: Called from Truncation.SD
#which.plot: Integer to say which ensemble member to emulate and plot
LOO.field <- function(Emulators, tData, EnsembleData, TruncatedSDs, which.plot=1, AnomCols=NULL, AnomBreaks=NULL, add.contours=FALSE, Anomaly.Only=FALSE, Uncertainty=TRUE, tcontours=seq(from=-40,by=1,len=100), contour.zero=TRUE,...){
  nEms <- length(Emulators)
  Noise.Index <- which(names(tData)=="Noise")
  LOOcoefsDat <- sapply(1:length(Emulators), function(e) LOO(loo.index = which.plot, Emulator = Emulators[[e]], tData=tData, num.inputs = Noise.Index-1, is.GP = TRUE))
  LOOcoefs <- unlist(LOOcoefsDat[1,])
  LOOcoefsSd <- sqrt(unlist(LOOcoefsDat[2,]))
  trecon <- Reconstruct(coefficients=LOOcoefs, basis=EnsembleData$tBasis[,1:nEms])
  EnsembleField <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconField <- trecon*EnsembleData$scaling + EnsembleData$EnsembleMean
  tdims <- c(length(lon), length(lat))
  tanom <- EnsembleField - ReconField
  par(mar=c(5,4,2,1))
  if(!Anomaly.Only){
    par(mfrow=c(3,1))
    image.plot(lon[order(lon)],lat, matrix(EnsembleField,nrow=tdims[1],ncol=tdims[2])[order(lon),],main=paste("Ensemble member",which.plot,sep=" ",xlab=""))
    map("world",add=T,wrap=TRUE, interior = FALSE)
    image.plot(lon[order(lon)],lat, matrix(ReconField,nrow=tdims[1],ncol=tdims[2])[order(lon),], main="Reconstruction", xlab="")
    map("world",add=T,wrap=TRUE,interior = FALSE)
  }
  if(Uncertainty){
    par(mfrow=c(3,1))
  }
  image.plot(lon[order(lon)],lat, matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols, lab.breaks=c("",AnomBreaks[-c(1,length(AnomBreaks))],""), horizontal = 0+Anomaly.Only, xlab="", ...)
  map("world",add=T,wrap=TRUE, interior = FALSE)
  if(add.contours){
    if(!contour.zero)
      tcontours <- tcontours[-which(tcontours==0)]
    contour(x=lon[order(lon)],y=lat, z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE,levels=tcontours,lty= 1 +(tcontours<0) )
  }
  if(Uncertainty){
    trecon.upper <- Reconstruct(coefficients = c(LOOcoefs + 2*LOOcoefsSd, 2*TruncatedSDs), basis=EnsembleData$tBasis)
    trecon.lower <- Reconstruct(coefficients = c(LOOcoefs - 2*LOOcoefsSd, (-2)*TruncatedSDs), basis=EnsembleData$tBasis)
    ReconUpper <- trecon.upper*EnsembleData$scaling + EnsembleData$EnsembleMean
    ReconLower <- trecon.lower*EnsembleData$scaling + EnsembleData$EnsembleMean
    tanom.upper <- EnsembleField - ReconUpper
    tanom.lower <- EnsembleField - ReconLower
    image.plot(lon[order(lon)],lat, matrix(tanom.upper,nrow=tdims[1],ncol=tdims[2])[order(lon),],  breaks=AnomBreaks, col=AnomCols, lab.breaks=c("",AnomBreaks[-c(1,length(AnomBreaks))],""), horizontal = 0+Anomaly.Only,xlab="")
    map("world",add=T,wrap=TRUE, interior = FALSE)
    if(add.contours)
      contour(x=lon[order(lon)],y=lat, z=matrix(tanom.upper,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE, levels=tcontours,lty= 1 +(tcontours<0))
    image.plot(lon[order(lon)],lat, matrix(tanom.lower,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols, lab.breaks=c("",AnomBreaks[-c(1,length(AnomBreaks))],""), horizontal = 0+Anomaly.Only,xlab="")
    map("world",add=T,wrap=TRUE, interior = FALSE)
    if(add.contours)
      contour(x=lon[order(lon)],y=lat, z=matrix(tanom.lower,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE, levels=tcontours,lty= 1 +(tcontours<0))
  }
}

LOO.field.zonal <- function(Emulators, tData, EnsembleData, ObsField, which.obs=1, TruncatedSDs, which.plot=1, which.zone="longitude", lat=lat, lon=lon, ...){
  nEms <- length(Emulators)
  Noise.Index <- which(names(tData)=="Noise")
  LOOcoefsDat <- sapply(1:length(Emulators), function(e) LOO(loo.index = which.plot, Emulator = Emulators[[e]], tData=tData, num.inputs = Noise.Index-1, is.GP = TRUE))
  LOOcoefs <- unlist(LOOcoefsDat[1,])
  LOOcoefsSd <- sqrt(unlist(LOOcoefsDat[2,]))
  trecon <- Reconstruct(coefficients=LOOcoefs, basis=EnsembleData$tBasis[,1:nEms])
  trecon.upper <- Reconstruct(coefficients = c(LOOcoefs + 2*LOOcoefsSd, 2*TruncatedSDs), basis=EnsembleData$tBasis)
  trecon.lower <- Reconstruct(coefficients = c(LOOcoefs - 2*LOOcoefsSd, (-2)*TruncatedSDs), basis=EnsembleData$tBasis)
  EnsembleField <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconField <- trecon*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconUpper <- trecon.upper*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconLower <- trecon.lower*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tdims <- c(length(lon), length(lat))
  trueanom <- EnsembleField - ObsToScale
  tanom <- ReconField - ObsToScale
  tanom.upper <- ReconUpper - ObsToScale
  tanom.lower <- ReconLower - ObsToScale
  par(mar=c(5,4,2,1), mfrow=c(1,1))
  dim(tanom) <- tdims
  dim(trueanom) <- tdims
  dim(tanom.lower) <- tdims
  dim(tanom.upper) <- tdims
  otherZoneMargin <- ifelse(which.zone=="longitude",2,1) #Counter intuitive but used in apply
  ZonalAnomTrue <- apply(trueanom, MARGIN=c(otherZoneMargin), mean)
  ZonalAnom <- apply(tanom, MARGIN=c(otherZoneMargin), mean)
  ZonalAnomLow <- apply(tanom.lower, MARGIN=c(otherZoneMargin), mean)
  ZonalAnomUpper <- apply(tanom.upper, MARGIN=c(otherZoneMargin), mean)
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  latloni <- ifelse(which.zone=="longitude",1,2)
  dirmat <- list(lat,lon)
  latlon <- dirmat[[latloni]]
  plot(latlon, ZonalAnom, type='l', xlim=c(latlon[length(latlon)],latlon[1]), ...)
  abline(h=0,lty=1)
  points(latlon, ZonalAnomLow, type='l', lty=2)
  points(latlon, ZonalAnomUpper, type='l', lty=2)
  points(latlon, ZonalAnomTrue, type="l",col=2)
}

#Example call: Tested 01/08/16
#tBreaks <- GenerateBreaks(WhiteRange = c(-2,2), ColRange = c(-10,10), Length.Colvec = 7, DataRange=c(-14,14))
#trunc.sds <- Truncation.SD(JJAdata, num.basisVecs=10)
#trunc.sds[] <- 0
#LOO.field(Emulators=EmulatorsList, tData=tData, EnsembleData = JJAdata, TruncatedSDs = trunc.sds, which.plot=10, AnomBreaks = tBreaks, AnomCols = AnomalyColours(7),add.contours = TRUE, Anomaly.Only = TRUE)

LOO.fieldPressure <- function(Emulators, tData, EnsembleData, Pressure, TruncatedSDs, which.plot=1, AnomCols=NULL, AnomBreaks=NULL,add.contours=FALSE,Anomaly.Only=FALSE, scale.log=TRUE, tmain="", tcontours=seq(from=-40,by=1,len=100), Uncertainty=TRUE, contour.zero=TRUE){
  nEms <- length(Emulators)
  Noise.Index <- which(names(tData)=="Noise")
  LOOcoefsDat <- sapply(1:length(Emulators), function(e) LOO(loo.index = which.plot, Emulator = Emulators[[e]], tData=tData, num.inputs = Noise.Index-1, is.GP = TRUE))
  LOOcoefs <- unlist(LOOcoefsDat[1,])
  LOOcoefsSd <- sqrt(unlist(LOOcoefsDat[2,]))
  trecon <- Reconstruct(coefficients=LOOcoefs, basis=EnsembleData$tBasis[,1:nEms])
  trecon.upper <- Reconstruct(coefficients = c(LOOcoefs + 2*LOOcoefsSd, 2*TruncatedSDs), basis=EnsembleData$tBasis)
  trecon.lower <- Reconstruct(coefficients = c(LOOcoefs - 2*LOOcoefsSd, (-2)*TruncatedSDs), basis=EnsembleData$tBasis)
  EnsembleField <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconField <- trecon*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconUpper <- trecon.upper*EnsembleData$scaling + EnsembleData$EnsembleMean
  ReconLower <- trecon.lower*EnsembleData$scaling + EnsembleData$EnsembleMean
  tdims <- c(length(lon), length(lat))
  tanom <- EnsembleField - ReconField
  tanom.upper <- EnsembleField - ReconUpper
  tanom.lower <- EnsembleField - ReconLower
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  tdims <- c(length(lat), length(Pressure))
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  if(Uncertainty){
    par(mar=c(4,4,2,1),mfrow=c(3,1))
  }
  plot.new()
  if(scale.log){
    plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  else{
    plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  title(main=paste("LOO", tmain, "Ensemble member", which.plot, sep=" "), xlab="Lattitude", ylab="Pressure")
  Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
  Pvec <- Pressure[order(Pressure)]/100
  Pvec <- as.character(Pvec)
  LabVec <- rep("",length(Pvec))
  if(scale.log){
    LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
    Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
  }
  else{
    LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
    Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
  }
  box()
  #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
  if(add.contours){
    if(!contour.zero)
      tcontours <- tcontours[-which(tcontours==0)]
    if(scale.log){
      contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontours, lty=1+(tcontours<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    }
    else{
      contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontours, lty=1+(tcontours<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
    }
  }
  if(Uncertainty){
    tanom1 <- tanom.upper
    tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
    tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
    plot.new()
    if(scale.log){
      plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    else{
      plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    title(main=paste(tmain,"Upper",sep=" "), xlab="Lattitude", ylab="Pressure")
    Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
    Pvec <- Pressure[order(Pressure)]/100
    Pvec <- as.character(Pvec)
    LabVec <- rep("",length(Pvec))
    if(scale.log){
      LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
      Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
    }
    else{
      LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
      Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
    }
    box()
    #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    if(add.contours){
      if(scale.log){
        contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom.upper,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontours, lty=1+(tcontours<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
      }
      else{
        contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom.upper,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontours, lty=1+(tcontours<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
      }
    }
    tanom1 <- tanom.lower
    tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
    tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
    plot.new()
    if(scale.log){
      plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    else{
      plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    title(main=paste(tmain,"Lower",sep=" "), xlab="Lattitude", ylab="Pressure")
    Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
    Pvec <- Pressure[order(Pressure)]/100
    Pvec <- as.character(Pvec)
    LabVec <- rep("",length(Pvec))
    if(scale.log){
      LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
      Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
    }
    else{
      LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
      Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
    }
    box()
    #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    if(add.contours){
      if(scale.log){
        contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom.lower,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontours, lty=1+(tcontours<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
      }
      else{
        contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom.lower,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontours, lty=1+(tcontours<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
      }
    }
  }
}
#Function to generate anomaly colours for the CCCMA plot deck
#INPUT: n, odd integer, number of colours (minimum is 7)
AnomalyColours <- function(n=7){
  tn <- min(c(n,7))
  intpalette(c("cyan4", "cyan1", "paleturquoise1", "white", "rosybrown2", "red2", "orangered4"), tn)
}

#Example call: Tested 02/08/16
#anomCols <- AnomalyColours()

LWPcolours <- function(n=13){
  tn <- min(c(n,13))
  intpalette(c("blue", "deepskyblue3", "cyan1", "green1", "lawngreen", "olivedrab1", "snow2", "yellow","gold", "orange2", "chocolate1", "orangered", "red"),tn)
}


#Function to generate the right breaks for a Canada style plot
#INPUTS: 
#WhiteRange: A vector giving the lower and upper data values for white
#ColRange: A Vector giving the lower and upper data values for the bluest and reddest colours to start
#Length.Colvec: How many colours are required? Breaks will be 1 longer.
#DataRange: Everything above and below the limits of ColRange will get the darkest blue/red. If the Data are 
#particularly large or psoitively/negatively skewed, this may need altering from the default.
#Another reason to alter may be to identify exceedingly anomalous values.
#OUTPUT: A vector of breaks for colour plotting.
GenerateBreaks <- function(WhiteRange, ColRange, Length.Colvec, DataRange =c(-99999999,99999999)){
  lbs <- Length.Colvec+1
  lhs <- c(DataRange[1], ColRange[1])
  rhs <- c(ColRange[2], DataRange[2])
  restLHS <- seq(from=ColRange[1],to=WhiteRange[1], length.out = (Length.Colvec-1)/2 )[-c(1,(Length.Colvec-1)/2)]
  restRHS <- seq(from=WhiteRange[2],to=ColRange[2], length.out = (Length.Colvec-1)/2 )[-c(1,(Length.Colvec-1)/2)]
  c(lhs,restLHS,WhiteRange,restRHS,rhs)
}

#Example call: Tested 02/8/16
#CustomBreaks <- GenerateBreaks(WhiteRange = c(-2,2), ColRange = c(-10,10), Length.Colvec = 7)



#Maybe it's time to look at Zonal plots too.....

#INPUTS:
#EnsembleData: Scaled and centred ensemble data from the right months and averages (see ExtractCentredScaledDataAndBasis)
#which.zone: Either "lattitude" or "longitude" to specify which cooridnate direction we are averaging in.
#lat: Vector of latitudes in the ensemble
#lon: Vector of longitudes in the ensemble
#scaling: An optional scaling for computational robustness
GetZonalEnsembleData <- function(EnsembleData, which.zone="longitude", lat, lon, scaling=1){
  stopifnot(which.zone=="lattitude" | which.zone=="longitude")
  tdims <- c(length(lon), length(lat))
  OriginalData <- EnsembleData$CentredField * EnsembleData$scaling
  N <- dim(EnsembleData$CentredField)
  for(k in 1:N[2]){
    OriginalData[,k] <- OriginalData[,k] + EnsembleData$EnsembleMean
  }
  dim(OriginalData) <- c(tdims, N[2])
  otherZoneMargin <- ifelse(which.zone=="longitude",2,1) #Counter intuitive but used in apply
  ZonalData <- apply(OriginalData, MARGIN=c(otherZoneMargin,3), mean)
  #Zonal Data should now be a length.zone x n matrix
  EnsMean <- apply(ZonalData, MARGIN=1, mean)
  CentredData <- ZonalData
  for(i in 1:dim(ZonalData)[2]){
    CentredData[,i] <- CentredData[,i] - EnsMean
  }
  CentredData <- CentredData/scaling
  Basis <- svd(t(CentredData))$v
  return(list(tBasis=Basis, CentredField = CentredData, EnsembleMean = EnsMean, scaling=scaling))
}

#Example call: Tested 03/8/16
#ZonalJJA <- GetZonalEnsembleData(JJAdata, "longitude",lat, lon, 1)

#Get the zonal averages of the obs
#INPUTS: 
#ObservationData: Obs data output from GetObservations
#FieldEnsemble: Scaled and centred ensemble data from the right months and averages (see ExtractCentredScaledDataAndBasis)
#ZonalEnsemble: Output from GetZonalEnsembleData
#which.zone: (see GetZonalEnsembleData)
#lat: Vector of latitudes in the ensemble
#lon: Vector of longitudes in the ensemble
GetZonalObservations <- function(ObservationData, FieldEnsemble, ZonalEnsemble, which.zone, lat, lon){
  stopifnot(which.zone=="lattitude" | which.zone=="longitude")
  OriginalFields <- lapply(ObservationData, function(e) (e*FieldEnsemble$scaling)+FieldEnsemble$EnsembleMean)
  UnpackField <- function(field){
    dim(field) <- c(length(lon), length(lat))
    field
  }
  OriginalFields <- lapply(OriginalFields, function(e) UnpackField(e))
  otherZoneMargin <- ifelse(which.zone=="longitude",2,1)
  ZonalData <- lapply(OriginalFields, function(e) apply(e, MARGIN=otherZoneMargin, mean))
  lapply(ZonalData, function(e) (e-ZonalEnsemble$EnsembleMean)/ZonalEnsemble$scaling)
}

#Example call:
#zonalJJAobs <- GetZonalObservations(tObsJJA, JJAdata, ZonalJJA, which.zone="longitude", lat, lon)

#Plot an ensemble member's zonal anomaly from all 3 data sets (could add legend in future)
#INPUTS:
#EnsembleData: Output from GetZonalEnsembleData
#ObsData: Output from GetZonalObservations
#which.plot: Which ensemble member to plot
#latlon: Vector of lattitude or longitude values to plot against
#... Arguments to plot
plot.EnsembleMemberZonal <- function(EnsembleData, ObsData, which.plot=1, latlon, ...){
  EnsembleMember <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling
  ZonalAnomaly <- EnsembleMember - ObsData[[1]]*EnsembleData$scaling
  plot(latlon, ZonalAnomaly, type='l', xlim=c(latlon[length(latlon)],latlon[1]), ...)
  abline(h=0,lty=2)
  ZonalAnomaly2 <- EnsembleMember - ObsData[[2]]*EnsembleData$scaling
  points(latlon, ZonalAnomaly2, type='l',col=2)
  ZonalAnomaly3 <- EnsembleMember - ObsData[[3]]*EnsembleData$scaling
  points(latlon, ZonalAnomaly3, type='l',col=4)
}

#Example call: Tested 03/8/16
#plot.EnsembleMemberZonal(ZonalJJA, zonalJJAobs, 1, latlon=lat, xlab="Lattitude")

#Function to plot a zonal mean by averaging the 2D field directly
plot.DataEnsembleMemberZonal <- function(ObsField, EnsembleData, which.obs = 1, which.plot=1, which.zone="longitude", lat=lat, lon=lon, ...){
  EnsembleField <- EnsembleData$CentredField[,which.plot]*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- EnsembleField - ObsToScale
  stopifnot(which.zone=="lattitude" | which.zone=="longitude")
  tdims <- c(length(lon), length(lat))
  dim(tanom) <- tdims
  otherZoneMargin <- ifelse(which.zone=="longitude",2,1) #Counter intuitive but used in apply
  ZonalAnom <- apply(tanom, MARGIN=c(otherZoneMargin), mean)
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  latloni <- ifelse(which.zone=="longitude",1,2)
  dirmat <- list(lat,lon)
  latlon <- dirmat[[latloni]]
  plot(latlon, ZonalAnom, type='l', xlim=c(latlon[length(latlon)],latlon[1]), ...)
  abline(h=0,lty=1)
}


#Plot an ensemble member's zonal anomaly from all 3 data sets (could add legend in future)
#INPUTS:
#Coefficients: Coefficients of an ensemble member projected onto a basis (note these could be emulator output)
#EnsembleData: Output from GetZonalEnsembleData
#ObsData: Output from GetZonalObservations
#which.plot: Which ensemble member to plot
#latlon: Vector of lattitude or longitude values to plot against
#... Arguments to plot
plot.EnsembleMemberReconstructionZonal <- function(coefficients, EnsembleData, ObsData, latlon, which.plot=1, ...){
  nprime <- dim(coefficients)[1]
  Reconstruction <- Reconstruct(coefficients=coefficients, basis=EnsembleData$tBasis[,1:nprime])[,which.plot]*EnsembleData$scaling
  ReconstructionAnomaly <- Reconstruction - ObsData[[1]]*EnsembleData$scaling
  plot(latlon, ReconstructionAnomaly, type='l', xlim=c(latlon[length(latlon)],latlon[1]), ...)
  abline(h=0,lty=2)
  ReconstructionAnomaly2 <- Reconstruction - ObsData[[2]]*EnsembleData$scaling
  points(latlon, ReconstructionAnomaly2, type='l',col=2)
  ReconstructionAnomaly3 <- Reconstruction - ObsData[[3]]*EnsembleData$scaling
  points(latlon, ReconstructionAnomaly3, type='l',col=4)
}

#Example call: Tested 03/8/16
#plot.EnsembleMemberReconstructionZonal(StandardCoefficients(ZonalJJA$CentredField, ZonalJJA$tBasis[,1:5], orthogonal=FALSE), ZonalJJA, zonalJJAobs, 1, latlon=lat, xlab="Lattitude")

#Need new EMULATOR.gp.spatial function (maybe only for HM or some sort of uncertainty plot)

#Plot an ensemble member and an observation field and their anomaly.
#INPUTS:
#x: Vector of inputs to be evaluated by the Emulator
#ObsField: List of Observation fields as output by GetObservations
#EnsembleData: Post-processed ensemble data in list form as output by ExtractCentredScaledDataAndBasis
#names.data: The names of the ensemble parameters.
#which.obs: Integer: which member of the observations list
#OriginalScale: Not yet coded
#AnomBreaks: Vector of breaks for plotting (see ?image)
#AnomCol: Vector of colours for plotting (1 lower than breaks, see ?image) 
plot.DataEmulator <- function(x, Emulator, ObsField, EnsembleData, names.data, which.obs = 1, AnomCols=NULL, AnomBreaks=NULL, add.contours=FALSE, Anomaly.Only=FALSE, tcontours=seq(from=-40,by=2,to=40), contour.zero=TRUE, ...){
  colnames(x) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  #Coef.lower <- Coef.exp - 2*sqrt(Coef.var)
  #Coef.upper <- Coef.exp + 2*sqrt(Coef.var)
  #pred.lower <- Reconstruct(coefficients=Coef.lower, basis=EnsembleData$tBasis[,1:length(Emulator)])
  #pred.lower <- pred.lower*EnsembleData$scaling + EnsembleData$EnsembleMean
  #tanom.lower <- pred.lower - ObsToScale
  #pred.upper <- Reconstruct(coefficients=Coef.upper, basis=EnsembleData$tBasis[,1:length(Emulator)])
  #pred.upper <- pred.upper*EnsembleData$scaling + EnsembleData$EnsembleMean
  #tanom.upper <- pred.upper - ObsToScale
  #AnomBreaks = GenerateBreaks(WhiteRange = c(-2,2), ColRange = c(-10,10), Length.Colvec = 7, DataRange=round(range(range(tanom),range(tanom.lower),range(tanom.upper))))
  par(mar=c(3,3,4,2))
  tdims <- c(length(lon), length(lat))
  if(!Anomaly.Only){
    par(mfrow=c(2,1), mar=c(3,3,2,2))
    image.plot(lon[order(lon)],lat, matrix(pred,nrow=tdims[1],ncol=tdims[2])[order(lon),], ...)
    map("world",add=T,wrap=TRUE,interior=FALSE)
  }
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  image.plot(lon[order(lon)],lat, matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = Anomaly.Only, ...)
  map("world",add=T, wrap=TRUE, interior = FALSE)
  if(add.contours){
    if(!contour.zero)
      tcontours <- tcontours[-which(tcontours==0)]
    contour(x=lon[order(lon)],y=lat, z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE,levels=tcontours, lty=1+(tcontours<0),lwd=0.8)
  }
  #Want uncertainty plots. Ideally upper and lower 95% ranges...
  #For now. +/- 2sd (can add the basis uncertainty here too, but must add to emulators first)
  
  #tanom1 <- tanom.lower
#  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
#  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
#    image.plot(lon[order(lon)],lat, matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = TRUE, ...)
#  map("world",add=T, wrap=TRUE, interior = FALSE)
#  if(add.contours){
#    if(!contour.zero)
#      tcontours <- tcontours[-which(tcontours==0)]
#    contour(x=lon[order(lon)],y=lat, z=matrix(tanom.lower,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE,levels=tcontours, lty=1+(tcontours<0),lwd=0.8)
#  }
#  tanom1 <- tanom.upper
#  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
#  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
#  image.plot(lon[order(lon)],lat, matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = TRUE, ...)
#  map("world",add=T, wrap=TRUE)
#  if(add.contours){
#    if(!contour.zero)
#      tcontours <- tcontours[-which(tcontours==0)]
#    contour(x=lon[order(lon)],y=lat, z=matrix(tanom.upper,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE,levels=tcontours, lty=1+(tcontours<0),lwd=0.8)
#  }
}


#Example call: Tested 5/8/16.
#newDesign <- 2*as.data.frame(randomLHS(100, 13)) - 1
#colnames(newDesign) <- names(ourData$Design)
#tBreaks <- GenerateBreaks(WhiteRange = c(-2,2), ColRange = c(-10,10), Length.Colvec = 7, DataRange=c(-14,14))
#plot.DataEmulator(x=newDesign[15,], Emulator=EmulatorsList, ObsField=tObsJJA, EnsembleData=JJAdata, names.data=names(ourData$Design), which.obs=1, AnomBreaks = tBreaks, AnomCols = AnomalyColours(7), add.contours = TRUE, Anomaly.Only = FALSE)
#debug(plot.DataEmulator)

#Plot an ensemble member and an observation field and their anomaly.
#INPUTS:
#x: Vector of inputs to be evaluated by the Emulator
#ObsField: List of Observation fields as output by GetObservations
#EnsembleData: Post-processed ensemble data in list form as output by ExtractCentredScaledDataAndBasis
#names.data: The names of the ensemble parameters.
#which.obs: Integer: which member of the observations list
#OriginalScale: Not yet coded
#AnomBreaks: Vector of breaks for plotting (see ?image)
#AnomCol: Vector of colours for plotting (1 lower than breaks, see ?image) 
plot.DataEmulatorZonal <- function(x, Emulator, ObsField, EnsembleData, names.data, which.obs = 1, which.zone="longitude", lat=lat, lon=lon, Uncertainty=FALSE, ...){
  colnames(x) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  Coef.lower <- Coef.exp - 2*sqrt(Coef.var)
  Coef.upper <- Coef.exp + 2*sqrt(Coef.var)
  pred.lower <- Reconstruct(coefficients=Coef.lower, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred.lower <- pred.lower*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom.lower <- pred.lower - ObsToScale
  pred.upper <- Reconstruct(coefficients=Coef.upper, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred.upper <- pred.upper*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom.upper <- pred.upper - ObsToScale
  tdims <- c(length(lon), length(lat))
  dim(tanom) <- tdims
  otherZoneMargin <- ifelse(which.zone=="longitude",2,1) #Counter intuitive but used in apply
  ZonalAnom <- apply(tanom, MARGIN=c(otherZoneMargin), mean)
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  latloni <- ifelse(which.zone=="longitude",1,2)
  dirmat <- list(lat,lon)
  latlon <- dirmat[[latloni]]
  plot(latlon, ZonalAnom, type='l', xlim=c(latlon[length(latlon)],latlon[1]), ...)
  abline(h=0,lty=1)
  #Want uncertainty plots. Ideally upper and lower 95% ranges...
  #For now. +/- 2sd (can add the basis uncertainty here too, but must add to emulators first)
  if(Uncertainty){
    dim(tanom.upper) <- tdims
    dim(tanom.lower) <- tdims
    ZonalLow <- apply(tanom.lower, MARGIN = c(otherZoneMargin), mean)
    ZonalHigh <- apply(tanom.upper, MARGIN = c(otherZoneMargin), mean)
    points(latlon, ZonalLow, type='l', lty=2)
    points(latlon, ZonalHigh, type='l', lty=2)
  }
}

#Example call: Tested 5/8/16.
#newDesign <- 2*as.data.frame(randomLHS(100, 13)) - 1
#colnames(newDesign) <- names(ourData$Design)
#tBreaks <- GenerateBreaks(WhiteRange = c(-2,2), ColRange = c(-10,10), Length.Colvec = 7, DataRange=c(-14,14))
#plot.DataEmulator(x=newDesign[15,], Emulator=EmulatorsList, ObsField=tObsJJA, EnsembleData=JJAdata, names.data=names(ourData$Design), which.obs=1, AnomBreaks = tBreaks, AnomCols = AnomalyColours(7), add.contours = TRUE, Anomaly.Only = FALSE)
#debug(plot.DataEmulator)

plot.DataEmulatorPressure <- function(x, Emulator, ObsField, EnsembleData, Pressure, names.data, which.obs = 1, which.plot=1, AnomCols=NULL, AnomBreaks=NULL,add.contours=FALSE,Anomaly.Only=FALSE, scale.log=TRUE, tmain="", tcontour=seq(from=-40,by=1,len=100), Uncertainty=TRUE){
  colnames(x) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  Coef.lower <- Coef.exp - 2*sqrt(Coef.var)
  Coef.upper <- Coef.exp + 2*sqrt(Coef.var)
  pred.lower <- Reconstruct(coefficients=Coef.lower, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred.lower <- pred.lower*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom.lower <- pred.lower - ObsToScale
  pred.upper <- Reconstruct(coefficients=Coef.upper, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred.upper <- pred.upper*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom.upper <- pred.upper - ObsToScale
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  tdims <- c(length(lat), length(Pressure))
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  if(Uncertainty){
    par(mar=c(4,4,2,1),mfrow=c(3,1))
  }
  plot.new()
  if(scale.log){
    plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  else{
    plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  title(main=tmain, xlab="Lattitude", ylab="Pressure",cex.main=0.9)
  Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
  Pvec <- Pressure[order(Pressure)]/100
  Pvec <- as.character(Pvec)
  LabVec <- rep("",length(Pvec))
  if(scale.log){
    LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
    Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
  }
  else{
    LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
    Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
  }
  box()
  #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
  if(add.contours){
    if(scale.log){
      contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    }
    else{
      contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
    }
  }
  if(Uncertainty){
   tanom1 <- tanom.upper
    tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
    tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
    plot.new()
    if(scale.log){
      plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    else{
      plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    title(main=paste(tmain,"Upper",sep=" "), xlab="Lattitude", ylab="Pressure")
    Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
    Pvec <- Pressure[order(Pressure)]/100
    Pvec <- as.character(Pvec)
    LabVec <- rep("",length(Pvec))
    if(scale.log){
      LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
      Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
    }
    else{
      LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
      Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
    }
    box()
    #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    if(add.contours){
      if(scale.log){
        contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
      }
      else{
        contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
      }
    }
    tanom1 <- tanom.lower
    tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
    tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
    plot.new()
    if(scale.log){
      plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    else{
      plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
      .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
    }
    title(main=paste(tmain,"Lower",sep=" "), xlab="Lattitude", ylab="Pressure")
    Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
    Pvec <- Pressure[order(Pressure)]/100
    Pvec <- as.character(Pvec)
    LabVec <- rep("",length(Pvec))
    if(scale.log){
      LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
      Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
    }
    else{
      LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
      Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
    }
    box()
    #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    if(add.contours){
      if(scale.log){
        contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
      }
      else{
        contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
      }
    }
  }
}
#Function to compare anomaly plots for two proposed ensemble members via their emulators
#INPUTS:
#x1: 1st vector of inputs to be evaluated by the Emulator
#x2: 2nd vector of inputs to be evaluated by the Emulator
#ObsField: List of Observation fields as output by GetObservations
#EnsembleData: Post-processed ensemble data in list form as output by ExtractCentredScaledDataAndBasis
#names.data: The names of the ensemble parameters.
#which.obs: Integer: which member of the observations list
#OriginalScale: Not yet coded
#AnomBreaks: Vector of breaks for plotting (see ?image)
#AnomCol: Vector of colours for plotting (1 lower than breaks, see ?image) 
CompareAnomalies <- function(x1, x2, Emulator, ObsField, EnsembleData, names.data, which.obs = 1, AnomCols=NULL, AnomBreaks=NULL, add.contours=FALSE, tcontours=seq(from=-40,by=2,to=40), contour.zero=TRUE, tmain="", ...){
  colnames(x1) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x1[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  tdims <- c(length(lon), length(lat))
  par(mfrow=c(2,1),mar=c(4,3,1,1))
  image.plot(lon[order(lon)],lat, matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = FALSE, main=paste(tmain, "Parameters X1", sep=" "), ...)
  map("world",add=T, wrap=TRUE, interior = FALSE)
  if(add.contours){
    if(!contour.zero)
      tcontours <- tcontours[-which(tcontours==0)]
    contour(x=lon[order(lon)],y=lat, z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE, levels=tcontours, lty=1+(tcontours<0),lwd=0.8)
  }
  colnames(x2) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x2[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  image.plot(lon[order(lon)],lat, matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = FALSE, main=paste(tmain, "Parameters X1", sep=" "), ...)
  map("world",add=T, wrap=TRUE, interior = FALSE)
  if(add.contours){
    contour(x=lon[order(lon)],y=lat, z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE, levels=tcontours, lty=1+(tcontours<0),lwd=0.8)
  }
}

#Example call: Data required generated in SAT.R. Tested 11/08/16
#CompareAnomalies(x1=ourDataSAT$Design[1,], x2=newDesign[which.min(JJAscoresMVsat$impl),], Emulator=EmulatorsListSAT, ObsField=tObsJJASAT, EnsembleData=JJArotSAT, names.data=names(ourDataSAT$Design), which.obs=3, AnomBreaks = tBreaksSAT, AnomCols = AnomalyColours(7), add.contours = TRUE)

CompareAnomaliesZonal <- function(x1, x2, Emulator, ObsField, EnsembleData, names.data, which.obs = 1, which.zone="longitude", lat=lat, lon=lon, ...){
  colnames(x1) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x1[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  tdims <- c(length(lon), length(lat))
  dim(tanom) <- tdims
  otherZoneMargin <- ifelse(which.zone=="longitude",2,1) #Counter intuitive but used in apply
  ZonalAnom <- apply(tanom, MARGIN=c(otherZoneMargin), mean)
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  latloni <- ifelse(which.zone=="longitude",1,2)
  dirmat <- list(lat,lon)
  latlon <- dirmat[[latloni]]
  plot(latlon, ZonalAnom, type='l', xlim=c(latlon[length(latlon)],latlon[1]), ...)
  abline(h=0,lty=1)
  colnames(x2) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x2[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  dim(tanom) <- tdims
  ZonalAnom <- apply(tanom, MARGIN=c(otherZoneMargin), mean)
  points(latlon, ZonalAnom, type='l', col=2)
}

CompareAnomaliesPressure <- function(x1, x2, Emulator, ObsField, EnsembleData, Pressure, names.data, which.obs = 1, which.plot=1, AnomCols=NULL, AnomBreaks=NULL,add.contours=FALSE,Anomaly.Only=FALSE, scale.log=TRUE, tmain="", tcontour=seq(from=-40,by=1,len=100), Uncertainty=TRUE){
  colnames(x1) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x1[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  par(mar=c(4,4,2,1),mfrow=c(2,1))
  tdims <- c(length(lat), length(Pressure))
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  plot.new()
  if(scale.log){
    plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  else{
    plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  title(main=paste(tmain, "new x1", sep=" "), xlab="Lattitude", ylab="Pressure")
  Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
  Pvec <- Pressure[order(Pressure)]/100
  Pvec <- as.character(Pvec)
  LabVec <- rep("",length(Pvec))
  if(scale.log){
    LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
    Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
  }
  else{
    LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
    Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
  }
  box()
  #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
  if(add.contours){
    if(scale.log){
      contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    }
    else{
      contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
    }
  }
  colnames(x2) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x2[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  tdims <- c(length(lat), length(Pressure))
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  plot.new()
  if(scale.log){
    plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  else{
    plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  title(main=paste(tmain, "new x2", sep=" "), xlab="Lattitude", ylab="Pressure")
  Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
  Pvec <- Pressure[order(Pressure)]/100
  Pvec <- as.character(Pvec)
  LabVec <- rep("",length(Pvec))
  if(scale.log){
    LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
    Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
  }
  else{
    LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
    Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
  }
  box()
  #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
  if(add.contours){
    if(scale.log){
      contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    }
    else{
      contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
    }
  }
}


#Function that compares an actual ensemble member with an emulator
CompareAnomaliesActual <- function(x1, which.member=1, Emulator, ObsField, EnsembleData, names.data, which.obs = 1, AnomCols=NULL, AnomBreaks=NULL, add.contours=FALSE, tcontours=seq(from=-40,by=2,to=40), contour.zero=TRUE, tmain="", ...){
  EnsembleField <- EnsembleData$CentredField[,which.member]*EnsembleData$scaling + EnsembleData$EnsembleMean
  colnames(x1) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x1[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  EnsAnom <- EnsembleField - ObsToScale
  tanom <- pred - ObsToScale
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  tanom0 <- EnsAnom
  tanom0[tanom0>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom0[tanom0<AnomBreaks[1]] <- AnomBreaks[1]
  tdims <- c(length(lon), length(lat))
  par(mfrow=c(2,1),mar=c(3,3,1,1))
  image.plot(lon[order(lon)],lat, matrix(tanom0,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = FALSE, main=paste(tmain, "Ensemble member", which.member , sep=" "), ...)
  map("world",add=T, wrap=TRUE, interior = FALSE)
  if(add.contours){
    if(!contour.zero)
      tcontours <- tcontours[-which(tcontours==0)]
    contour(x=lon[order(lon)],y=lat, z=matrix(EnsAnom,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE,levels=tcontours, lty=1+(tcontours<0),lwd=0.8)
  }
  image.plot(lon[order(lon)],lat, matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lon),], breaks=AnomBreaks, col=AnomCols,lab.breaks=AnomBreaks,horizontal = FALSE, main=paste(tmain, "Parameters X1", sep=" "), ...)
  map("world",add=T, wrap=TRUE, interior = FALSE)
  if(add.contours){
    contour(x=lon[order(lon)],y=lat, z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lon),],add=TRUE, levels=tcontours, lty=1+(tcontours<0),lwd=0.8)
  }
}

CompareAnomaliesZonalActual <- function(x1, which.member, Emulator, ObsField, EnsembleData, names.data, which.obs = 1, which.zone="longitude", lat=lat, lon=lon, ...){
  EnsembleField <- EnsembleData$CentredField[,which.member]*EnsembleData$scaling + EnsembleData$EnsembleMean
  colnames(x1) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x1[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  EnsAnom <- EnsembleField - ObsToScale
  tanom <- pred - ObsToScale
  tdims <- c(length(lon), length(lat))
  dim(tanom) <- tdims
  otherZoneMargin <- ifelse(which.zone=="longitude",2,1) #Counter intuitive but used in apply
  ZonalAnom <- apply(tanom, MARGIN=c(otherZoneMargin), mean)
  dim(EnsAnom) <- tdims
  ZonalEnsAnom <- apply(EnsAnom, MARGIN=c(otherZoneMargin), mean)
  par(mar=c(4,4,2,1),mfrow=c(1,1))
  latloni <- ifelse(which.zone=="longitude",1,2)
  dirmat <- list(lat,lon)
  latlon <- dirmat[[latloni]]
  plot(latlon, ZonalAnom, type='l', xlim=c(latlon[length(latlon)],latlon[1]), col=2, ...)
  abline(h=0,lty=1)
  points(latlon, ZonalEnsAnom, type='l', col=1)
}

CompareAnomaliesPressureActual <- function(x1, which.member=1, Emulator, ObsField, EnsembleData, Pressure, names.data, which.obs = 1, which.plot=1, AnomCols=NULL, AnomBreaks=NULL,add.contours=FALSE,Anomaly.Only=FALSE, scale.log=TRUE, tmain="", tcontour=seq(from=-40,by=1,len=100), Uncertainty=TRUE){
  EnsembleField <- EnsembleData$CentredField[,which.member]*EnsembleData$scaling + EnsembleData$EnsembleMean
  colnames(x1) <- names.data
  EmOutput <- lapply(1:length(Emulator), function(e) unlist(EMULATOR.gp(x1[1,], Emulator[[e]])))
  Coef.exp <- unlist(lapply(EmOutput, function(e) e[1]))
  Coef.var <- unlist(lapply(EmOutput, function(e) e[2]))
  pred <- Reconstruct(coefficients=Coef.exp, basis=EnsembleData$tBasis[,1:length(Emulator)])
  pred <- pred*EnsembleData$scaling + EnsembleData$EnsembleMean
  ObsToScale <- ObsField[[which.obs]]*EnsembleData$scaling + EnsembleData$EnsembleMean
  tanom <- pred - ObsToScale
  EnsAnom <- EnsembleField - ObsToScale
  par(mar=c(4,4,2,1),mfrow=c(2,1))
  tdims <- c(length(lat), length(Pressure))
  tanom1 <- tanom
  tanom1[tanom1>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom1[tanom1<AnomBreaks[1]] <- AnomBreaks[1]
  tanom0 <- EnsAnom
  tanom0[tanom0>AnomBreaks[length(AnomBreaks)]] <- AnomBreaks[length(AnomBreaks)]
  tanom0[tanom0<AnomBreaks[1]] <- AnomBreaks[1]
  plot.new()
  if(scale.log){
    plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom0,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  else{
    plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom0,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  title(main=paste(tmain, "Ensemble member", which.member, sep=" "), xlab="Lattitude", ylab="Pressure")
  Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
  Pvec <- Pressure[order(Pressure)]/100
  Pvec <- as.character(Pvec)
  LabVec <- rep("",length(Pvec))
  if(scale.log){
    LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
    Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
  }
  else{
    LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
    Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
  }
  box()
  #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
  if(add.contours){
    if(scale.log){
      contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(EnsAnom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    }
    else{
      contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(EnsAnom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
    }
  }
  plot.new()
  if(scale.log){
    plot.window(range(lat)[c(2,1)], range(log(Pressure))[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],log(Pressure[order(Pressure)]), matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  else{
    plot.window(range(lat)[c(2,1)], range(Pressure)[c(2,1)], "",xaxs="i",yaxs="i")
    .filled.contour(lat[order(lat)],Pressure[order(Pressure)], matrix(tanom1,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)], levels=AnomBreaks, col=AnomCols)
  }
  title(main=paste(tmain, "new x1", sep=" "), xlab="Lattitude", ylab="Pressure")
  Axis(x=lat[order(lat)], at=c(-90,-60,-30,0,30,60,90), side=1)
  Pvec <- Pressure[order(Pressure)]/100
  Pvec <- as.character(Pvec)
  LabVec <- rep("",length(Pvec))
  if(scale.log){
    LabVec[c(1:12,14,17,19,23,26,37)] <- Pvec[c(1:12,14,17,19,23,26,37)]
    Axis(log(Pressure[order(Pressure)]),at=log(Pressure[order(Pressure)]),labels=LabVec,side=2,las=1)
  }
  else{
    LabVec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)] <- Pvec[c(1,9,11,13,15,17,19:27,29,31,33,35,37)]
    Axis(Pressure[order(Pressure)],at=Pressure[order(Pressure)],labels=LabVec,side=2,las=1)
  }
  box()
  #contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE,levels=AnomBreaks+ceiling(AnomBreaks/2),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
  if(add.contours){
    if(scale.log){
      contour(x=lat[order(lat)],y=log(Pressure[order(Pressure)]), z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(log(Pressure))[c(2,1)], ylog=TRUE)
    }
    else{
      contour(x=lat[order(lat)],y=Pressure[order(Pressure)], z=matrix(tanom,nrow=tdims[1],ncol=tdims[2])[order(lat),order(Pressure)],add=TRUE, levels=tcontour, lty=1+(tcontour<0),xlim=range(lat)[c(2,1)],ylim=range(Pressure)[c(2,1)])
    }
  }
}

ParametersPlot <- function(NewParameters, StandardModel, ParameterNames, param.lows = c(0.5, 600, 0.75e-03, 0.01, 0.1, 0, 0.01, 10, 10, 1e06, 1800, 0.05, 5e-04), param.highs = c(5, 7500, 1.3e-03, 0.06, 1, 0.1, 0.5, 50, 1000, 1e09, 21600, 0.5, 1e-02), which.logs = c(9, 10, 13), param.defaults = c(2.5, 4800, 1e-03, 0.03, 0.75, 0.005, 0.1, 10, 600, 5e08, 6*3600, 0.2, 3e-03)){
  par(mfrow=c(1,1),mar=c(4,1,1,1))
  eps <- 1/(length(ParameterNames)*10)
  conversion <- function(anX,lows,highs){
    ((anX+1)/2)*(highs-lows) +lows
  }
  tlows <- param.lows
  thighs <- param.highs
  tlows[which.logs] <- log10(param.lows[which.logs])
  thighs[which.logs] <- log10(param.highs[which.logs])
  Scaled <- unlist(sapply(1:length(param.lows), function(i) conversion(NewParameters[i],tlows[i], thighs[i])))
  Scaled[which.logs] <- 10^Scaled[which.logs]
  plot(0,0,type = 'l', xlim=c(-1.25,1.81), ylim=c(0,1.05), yaxt="n", ylab=ParameterNames[i],bty="n",xaxt="n",xlab="")
  for(i in 1:length(ParameterNames)){
    if(NewParameters[i] < StandardModel[i])
      rect(NewParameters[i],0+(i-1)/length(ParameterNames)+eps,StandardModel[i],0+i/length(ParameterNames),col="cyan4")
    else
      rect(StandardModel[i],0+(i-1)/length(ParameterNames)+eps,NewParameters[i],0+i/length(ParameterNames),col="orangered4")
    rect(StandardModel[i],0+(i-1)/length(ParameterNames)+eps,StandardModel[i],0+i/length(ParameterNames),col=2,lwd=3)
    text(-1.2, 0+(i)/(length(ParameterNames))- 1/(2*length(ParameterNames)) + eps,labels=ParameterNames[i])
    text(1.47, 0+(i)/(length(ParameterNames))- 1/(2*length(ParameterNames)) + eps,labels= paste(param.lows[i]),pos=4,cex=0.65)
    text(1.71, 0+(i)/(length(ParameterNames))- 1/(2*length(ParameterNames)) + eps,labels= paste(param.highs[i]),pos=4,cex=0.65)
    text(1.3, 0+(i)/(length(ParameterNames))- 1/(2*length(ParameterNames)) + eps,labels= paste(param.defaults[i]),pos=4,cex=0.65)
    text(1.05, 0+(i)/(length(ParameterNames))- 1/(2*length(ParameterNames)) + eps,labels= paste(signif(Scaled[i],2)),pos=4,cex=0.65)
  }
  Axis(x=c(-1,1), at=seq(from=-1,to=1,by=0.5), side=1)
  text(c(1.05,1.3,1.47,1.71), 1.05, labels=c("New", "Stan", "Min", "Max"), cex=0.75,pos=4,)
}
#ParametersPlot(NewParameters = ourDataRLUTCS$Design[3,], StandardModel=ourDataRLUTCS$Design[1,], ParameterNames = names(ourDataRLUTCS$Design))
