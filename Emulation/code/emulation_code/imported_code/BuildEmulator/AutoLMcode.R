ProduceMainEffects <- function(name,order){
	if(order<1)
	  return(NA)
	else if(order<2)
	  return(name)
	else{
		terms <- c(name, sapply(2:order, function(i) paste("I(",name,"^",i,")",sep=""),simplify=TRUE))
		return(paste(terms,collapse="+"))
		}
	}
pme <- Vectorize(ProduceMainEffects)

FitLM <- function(DataString, ResponseString, Names, mainEffects, Interactions, Factors=NULL, FactorInteractions=NULL, ThreeWayInters=NULL, tData, Fouriers=NULL){
	stopifnot(is.list(ThreeWayInters)||is.null(ThreeWayInters))
	#stopifnot(length(Names)==length(mainEffects))
	if(!is.null(Interactions)){
		if(!is.matrix(Interactions))
	   		stop("Interactions should be matrix")
		#stopifnot(length(Interactions[,1])==length(Names))
		#stopifnot(length(Interactions[1,])==length(Names))
	}
  if(sum(mainEffects)<1)
    mainEffectPoly <- "1"
  else
	  mainEffectPoly <- paste(na.omit(pme(Names,mainEffects)),collapse="+")
	if(sum(Interactions)<1)
		Interactions <- NULL
	if(!is.null(Interactions)){
		stopifnot(all(diag(Interactions)<1))
		#Create Interaction terms
		indexMarkers <- which(Interactions>0)
		produceInteraction <- function(indexMarker){
			i <- (indexMarker - 1)%%length(Names) + 1
			j <- 1 + (indexMarker - i)/length(Names)
			paste("I(",Names[i],"*",Names[j],")",sep="")
			}
		pI <- Vectorize(produceInteraction)
		interactionPoly <- paste(pI(indexMarkers),collapse="+")
		tPoly <- paste(mainEffectPoly,interactionPoly,sep="+")
	}
	else{
		tPoly <- mainEffectPoly
	}
	if(!is.null(Fouriers)){
	  #Fouriers needs to be a list of 2*k matrices (below), one for each variable in Names/tData
	  #Fouriers will be a 2*k matrix containing 1's or 0's indicating 
	  #which Fourier Frequencies are to be fitted, with the following conventions:
	  #A 1 at Fouriers[1,j] indicates I(sin(j*pi*x)) is in the model (0 means it is not)
	  #A 1 at Fouriers[2,j] indicates I(cos(j*pi*x)) is in the model (0 means it is not)
	  #k is the largest frequency we try to fit (the covariance function should handle the rest)
	  #WARNING: Assume variable is on [-1,1]
	  warning("Fitting fourier basis assumes variables are on [-1,1]. Rescale if not")
	  for(i in 1:length(Fouriers)){ #Go through each matrix in the list of matrices
	   if(sum(Fouriers[[i]])<1)
	     Fouriers[[i]] <- NULL
	    else{
	     tmpsines <- sapply(1:dim(Fouriers[[i]])[2], function(j) paste("I(sin(pi*",names(Fouriers)[i],"*",j,"))",sep=""))
	     tsins <- tmpsines[which(Fouriers[[i]][1,]>0)]
	     tmpcosines <- sapply(1:dim(Fouriers[[i]])[2], function(j) paste("I(cos(pi*",names(Fouriers)[i],"*",j,"))",sep=""))
	     tcos <- tmpcosines[which(Fouriers[[i]][2,]>0)]
	     tsinePoly <- paste(tsins,collapse="+")
	     if(tsinePoly != "")
	       tPoly <- paste(tPoly,tsinePoly,sep="+")
	     tcosinePoly <- paste(tcos,collapse="+")
	     if(tcosinePoly != "")
	       tPoly <- paste(tPoly,tcosinePoly,sep="+")
	    }
	  }
	}
	 if(!is.null(Factors)){
  	    facts <- sapply(1:length(Factors), function(i) paste("as.factor(",Factors[i],")",sep=""),simplify=TRUE)
  	    FactorPoly <- paste(facts,collapse="+")
  	    tPoly <- paste(tPoly,FactorPoly,sep="+")
  	    if(!is.null(FactorInteractions)){
  	    	if(!all(FactorInteractions==0)){#BUG IF MORE THAN ONE FACTOR
	  	    	#FactorInteractions is a matrix that is nfactors x length(names)
  		    	#To get the factor interactions we apply pme to each row of the matrix and then wrap it in "as.factor(correctFactor):(pmeBit)
  	    		FactInteracts <- sapply(1:length(FactorInteractions[,1]), function(i) if(mean(FactorInteractions[i,])!=0){paste("as.factor(",attr(FactorInteractions,"dimnames")[[1]][i],"):(",paste(na.omit(pme(Names,FactorInteractions[i,])),collapse="+"),")",sep="")},simplify=TRUE)
  	    		FInteractBit <- paste(FactInteracts,collapse="+")
  	    		tPoly <- paste(tPoly,FInteractBit,sep="+")
  	    		}
  	    }
  	}
  	if(!is.null(ThreeWayInters)){
  		if(any(lapply(ThreeWayInters,function(e) sum(e))<1)){
  			ThreeWayInters[which(lapply(ThreeWayInters,function(e) sum(e))<1)] <- NULL
  		}
  		if(length(ThreeWayInters)<1){
  			ThreeWayInters <- NULL
  		}
  	}
  	if(!is.null(ThreeWayInters)){
  		#ThreeWayInters is a list of 2-way interaction matrices, each having name of the 3rd interaction
  		produceInteraction3 <- function(indexMarker,name){
		i <- (indexMarker - 1)%%length(Names) + 1
		j <- 1 + (indexMarker - i)/length(Names)
		paste("I(",name,"*",Names[i],"*",Names[j],")",sep="")
		}
	   pI3 <- Vectorize(produceInteraction3)
	   threeWayVec <- c()
  		for(k in 1:length(ThreeWayInters)){
  			tempName <- names(ThreeWayInters)[k]
  			tmpIndexMarkers <- which(ThreeWayInters[[k]]>0)
  			threeWayVec <- c(threeWayVec,pI3(tmpIndexMarkers,tempName))
  			}
  		threeWays <- paste(threeWayVec,collapse="+")
  		tPoly <- paste(tPoly,threeWays,sep="+")
  		}
  	eval(parse(text=paste("lm(",ResponseString,"~",tPoly,",data=",DataString,")",sep="")))
	}

#FitLM allows us to fit large LMs by specifying matrices that indicate what terms should be present.

#Now we want to be able to delete terms by looking at the fits and removing terms (one at a time) that do not contribute, subject to certain rules on what can and cannot be removed. 
#So, if the highest order of a main effect is inactive the order can be reduced by 1 in mainEffects, but ONLY if that term is not present in any interactions.
#An interaction may be removed if there are no higher order interactions containing it. (So if for every matrix in the list ThreeWayInters, an entry is zero on both the upper and lower triangle, the corresponding term in Interactions is allowed to be set to zero.)
#Three way interactions can always be set to zero.
#Factor interactions can always be set to zero.
#A Factor name can be removed if all elements of its row in Factor Interactions is zero.

#Next we require a method for finding candidate terms to delete.
#Idea is to use the sum of squares explained from the anova. Take anova(lm)$"Sum Sq" and then order it so that the original indices are retained. (so maybe a matrix with index and sum_Sq is created and then sorted by sum sq.) With a tolerance of (say) 1e-3, we look at the term associated with the min(sum_sq). If we are allowed to delete it we do and refit. If not, we go to the next smallest and so on until the sum sq is > than tol and we reconsider the new model.

RemoveTerm <- function(linModel, Tolerance, Names, mainEffects, Interactions, Factors=NULL, FactorInteractions=NULL, ThreeWayInters=NULL, Fouriers=NULL){
	sumsqs <- data.frame(sumsq=anova(linModel)$"Sum Sq",index=1:length(anova(linModel)$"Sum Sq"))
	sumsqs <- sumsqs[do.call(order,sumsqs),]
	Irremovable <- TRUE
	k <- 1
	while(Irremovable){
		if(sumsqs[k,1]>=Tolerance){
			Irremovable <- FALSE
		return(list(removed=FALSE,mainEffects=mainEffects,Interactions=Interactions,Factors=Factors,FactorInteractions=FactorInteractions,ThreeWayInters=ThreeWayInters))
		}
		else{
		term <- attr(anova(linModel),"row.names")[sumsqs[k,2]]
		if(term=="Residuals"){
			Irremovable <- TRUE
		}
		else{
		on_sin <- strsplit(term, "sin")[[1]]
		on_cos <- strsplit(term, "cos")[[1]]
		if(length(on_sin)>1){
		  #We have a Fourier term sin(kx)
		  tbigsplit <- strsplit(strsplit(on_sin,")")[[2]][1] , " \\* ")[[1]]
		  Fvar <- tbigsplit[2]
		  tk <- as.numeric(tbigsplit[3])
		  Fouriers[[Fvar]][1,tk] <- 0
		  Irremovable <- FALSE
		}
		else if(length(on_cos)>1){
		  #We have a Fourier term cos(kx)
		  tbigsplit <- strsplit(strsplit(on_cos,")")[[2]][1] , " \\* ")[[1]]
		  Fvar <- tbigsplit[2]
		  tk <- as.numeric(tbigsplit[3])
		  Fouriers[[Fvar]][2,tk] <- 0
		  Irremovable <- FALSE
		}
		else{  
		onstar <- strsplit(term," \\* ")[[1]]
		if(length(onstar)>2){
			#Term is a 3 way interaction
			firstWay <- strsplit(onstar[1],"\\(")[[1]][2]
			thirdWay <- strsplit(onstar[3],"\\)")[[1]][1]
			ThreeWayInters[[firstWay]][which(Names==onstar[2]),which(Names==thirdWay)] <- 0
			Irremovable <- FALSE
			}
		else if(length(onstar)>1){
			#Term is a 2 way interaction
			first <- strsplit(onstar[1],"\\(")[[1]][2]
			second <- strsplit(onstar[2],"\\)")[[1]][1]
			#Need to check there are no 3 way interactions containing it.
			var1 <- which(Names==first)
			var2 <- which(Names==second)
			if(!is.null(ThreeWayInters)){
			    if(all(sapply(1:length(ThreeWayInters),function(j) ThreeWayInters[[j]][var1,var2] < 1,simplify=TRUE))){
			       Interactions[var1,var2] <- 0
			       Irremovable <- FALSE
			    }
			}
			else{
				Interactions[var1,var2] <- 0
			    Irremovable <- FALSE
			}
		}
		else{
			#Term could be a factor/factor interaction/power or individual.
			onInt <- strsplit(onstar,":")[[1]]
			if(length(onInt)>1){
				#Term is a factor interaction. Want to make sure that the second term is the factor which it almost always is by default.
				if(length(strsplit(onInt[2],"\\.")[[1]])<2){
					onInt <- c(onInt[2],onInt[1])
				}
				tFactor <- strsplit(strsplit(onInt[2],"\\(")[[1]][2],"\\)")[[1]]
				#We need to find out if it is an interaction with a linear or power variable.
				onIntP <- strsplit(onInt[1],"\\^")[[1]]
				if(length(onIntP)<2){
					#We have a linear term
					if(FactorInteractions[which(attr(FactorInteractions,"dimnames")[[1]]==tFactor),which(Names==onInt[1])]<2){
						#There are no higher powers of this factor so safe to delete.
				       FactorInteractions[which(attr(FactorInteractions,"dimnames")[[1]]==tFactor),which(Names==onInt[1])] <- 0
				       Irremovable <- FALSE
				    }
				}
				else{
					#It has a power. Ideally we would remove just this power. However, under the current (very very slick) matrix system this is not allowed. We therefore only remove the term if it is the highest power. We do now need to remove the brackets from the variable.
					tVariable <- strsplit(onIntP[1],"\\(")[[1]][2]
					tPower <- as.numeric(strsplit(onIntP[2],"\\)")[[1]][1])
					if(!FactorInteractions[which(attr(FactorInteractions,"dimnames")[[1]]==tFactor),which(Names==tVariable)] > tPower){
#						We can remove this factor interaction as it is with the highest power. 
					    FactorInteractions[which(attr(FactorInteractions,"dimnames")[[1]]==tFactor),which(Names==tVariable)] <- FactorInteractions[which(attr(FactorInteractions,"dimnames")[[1]]==tFactor),which(Names==tVariable)] - 1
					    Irremovable <- FALSE
					    }
					}
			}
			else{
				#Term is a factor, individual or power
				onOpenBracket <- strsplit(onInt,"\\(")[[1]]
				if(length(onOpenBracket)<2){
					#Term is an individual. We must check it doesn't appear in any type of interaction term or power term before removing it.
					variableNum <- which(Names==onOpenBracket)
					if((mainEffects[variableNum]<2)){
						#There are no higher power terms
						if(all(Interactions[variableNum,]<1)&all(Interactions[,variableNum]<1)){
							#There are no interactions
							if(!any(names(ThreeWayInters)==onOpenBracket)){
								#No three way matrices named after onOpenBracket
								if(all(lapply(ThreeWayInters, function(e) (all(e[variableNum,]<1))&(all(e[,variableNum]<1))))){
								#No three way interactions
								    if(all(FactorInteractions[,variableNum]<1)){
								    	#No factor interactions and hence OK to delete
								    	mainEffects[variableNum] <- 0
								    	Irremovable <- FALSE
								    	}
								     }
								}
							}
						}
					}
				else{
					#Term is a factor or power
					onPower <- strsplit(onOpenBracket[2],"\\^")[[1]]
					if(length(onPower)>1){
						#Term is a power
						tPower <- as.numeric(strsplit(onPower[2], "\\)")[[1]][1])
						tVar <- onPower[1]
						tNum <- which(Names==tVar)
						#Need to check there are no higher powers
						if(!mainEffects[tNum]>tPower){
							#No higher powers
							#Need to check no higher or equivalent powers in the factor interactions.
							if(!any(FactorInteractions[,tNum]>=tPower)){
							   mainEffects[tNum] <- tPower-1
							   Irremovable <- FALSE
							   }
							}
						}
					else{
						#Term is a factor. Need to check there are no factor interactions. Even though FitLM deletes rows in the FactorInteractions matrix that are all 0, we keep them (as they are automatically handled and are useful for keeping the code simple)
						tFactor <- strsplit(onPower,"\\)")[[1]][1]
						if(all(FactorInteractions[which(attr(FactorInteractions,"dimnames")[[1]]==tFactor),]<1)){
							#No factor interactions
							Factors <- Factors[-which(Factors==tFactor)]
							if(length(Factors)<1)
								Factors <- NULL
							Irremovable <- TRUE
							}
						}
					}
				}
			} 
			}}}
			k <- k+1
		}
			return(list(removed=TRUE,what=term,mainEffects=mainEffects,Interactions=Interactions,Factors=Factors,FactorInteractions=FactorInteractions,ThreeWayInters=ThreeWayInters,Fouriers=Fouriers))

	}

#This function removes up to N terms (within the specified tolerance), and returns the final linear model as well as the different matrices in a list.

removeNterms <- function(N, linModel, dataString, responseString, Tolerance, Names, mainEffects, Interactions, Factors=NULL, FactorInteractions=NULL, ThreeWayInters=NULL,tData, Fouriers=NULL){
	dontStop <- TRUE
	j <- 1
	while(dontStop&(j<=N)){
		tmpRm <- RemoveTerm(linModel=linModel, Tolerance=Tolerance, Names=Names, mainEffects=mainEffects, Interactions=Interactions, Factors=Factors, FactorInteractions=FactorInteractions, ThreeWayInters=ThreeWayInters, Fouriers = Fouriers)
		if(!tmpRm$removed){
			dontStop <- FALSE
		}
		else{
			print(paste(j,tmpRm$what,sep=" "))
			mainEffects <- tmpRm$mainEffects
			Interactions <- tmpRm$Interactions
			Factors <- tmpRm$Factors
			FactorInteractions <- tmpRm$FactorInteractions
			ThreeWayInters <- tmpRm$ThreeWayInters
			if(!is.null(ThreeWayInters)){
				tsums <- unlist(lapply(ThreeWayInters, function(e) sum(e)))
				threeoldnames <- names(ThreeWayInters)
				for(l in which(tsums<1)){
					ThreeWayInters[[threeoldnames[l]]] <- NULL
				}
				if(length(ThreeWayInters) < 1)
					ThreeWayInters <- NULL
			}
			Fouriers <- tmpRm$Fouriers
			if(!is.null(Fouriers)){
			  tsums <- unlist(lapply(Fouriers, function(e) sum(e)))
			  oldFnames <- names(Fouriers)
			  for(l in which(tsums<1)){
			    Fouriers[[oldFnames[l]]] <- NULL
			  }
			  if(length(Fouriers)<1)
			    Fouriers <- NULL
			}
			linModel <- FitLM(dataString, responseString, Names, mainEffects, Interactions, Factors, FactorInteractions, ThreeWayInters,tData,Fouriers=Fouriers)
			j <- j+1
		}	
	}
	return(list(Names=Names,linModel=linModel,mainEffects=mainEffects,Interactions=Interactions,Factors=Factors,FactorInteractions=FactorInteractions,ThreeWayInters=ThreeWayInters, Fouriers=Fouriers))
}

#The above is the stepwise delete stuff. For it to work well, want to try a sort of stepwise add that introduces new variables to see if they affect the RSS enough to justify their inclusion.

#The way this will work will be to allow either a new term to be added  with all valid interactions and factor interactions included, or to allow the order of a term to be increased with the order of factor interactions increased. May also allow a variable to have three way interactions later.

addNewTerms <- function(Name, oldNames, mainEffects, Interactions, Factors, FactorInteractions, ThreeWayInters, Fouriers, add.order=FALSE,is.NewFactor=FALSE,try.NewThreeWay=FALSE, FourierList = NULL){
  if(!is.null(FourierList)){
    #FourierList contains the required terms "order" (of the term to be added) and
    #sincos which is either "None" (add both fourier terms of the given order for Name)
    #"sin" or "cos" (add the sin or cos fourier term of the given order for Name)
    Fouriers <- AddFourier(Name=Name, order=FourierList$order, sincos=FourierList$sincos, Fouriers=Fouriers)
  }
	else if(try.NewThreeWay){
		ThreeWayInters[[Name]] <- Interactions
		if(length(ThreeWayInters)>1){
			listNames <- names(ThreeWayInters)
			for(k in 1:length(listNames)){
				for(j in 1:length(listNames)){
					if(j != k){
					   tRow <- which(oldNames == listNames[j])
					   tmpMat <- ThreeWayInters[[k]] + t(ThreeWayInters[[k]])
	   				   tVars <- which(tmpMat[tRow,]>0)
					   totherRow <- which(oldNames == listNames[k])
					   ThreeWayInters[[j]][totherRow,tVars] <- 0
					   ThreeWayInters[[j]][tVars,totherRow] <- 0
					}
				}
			}
		   }
		}
	else if((Name %in% oldNames)){
	   mainEffects[which(oldNames==Name)] <- mainEffects[which(oldNames==Name)] + 1
	   if(!is.null(Factors)){
	       FactorInteractions[,which(oldNames==Name)] <- FactorInteractions[,which(oldNames==Name)] + 1
	       }
	   }
	else if(is.NewFactor){
		if(!is.null(mainEffects)){
			if(is.null(Factors)){
				FactorInteractions <- t(mainEffects)
			}
			else{
				FactorInteractions <- rbind(FactorInteractions,mainEffects)
			}
			Factors <- c(Factors,Name)
			rownames(FactorInteractions) <- Factors
		}
		else
			Factors <- c(Factors,Name)
		}	
	else{
		oldNames <- c(oldNames,Name)
		mainEffects <- c(mainEffects,1)
		if(length(mainEffects)<2){
			Interactions <- NULL
		}
		else if(length(mainEffects)<3){
			Interactions <- rbind(c(0,0),c(1,0))
		}
		else{
			m <- length(Interactions[,1])
			Interactions <- rbind(Interactions,rep(1,m))
			Interactions <- cbind(Interactions,rep(0,m+1))
		}
		if(!is.null(ThreeWayInters)){
		    ThreeWayInters <- lapply(ThreeWayInters, function(e) {tmp <- rbind(e,rep(1,m)); e <- cbind(tmp,rep(0,m+1))})
		    if(length(ThreeWayInters)>1){
		    	for(k in 2:length(ThreeWayInters)){
		    		j <- sapply(1:(k-1), function(m) which(oldNames==names(ThreeWayInters)[m]))
		    		thisName <- which(oldNames==names(ThreeWayInters)[k])
		    		for(l in 1:length(j)){
		    			if(j[l] < thisName){
				    		ThreeWayInters[[k]][j[l],] <- 0
				    		ThreeWayInters[[k]][-thisName,j[l]] <- 0
				    	}
				    	else{
				    		ThreeWayInters[[k]][,j[l]] <- 0
				    		ThreeWayInters[[k]][j[l],-thisName] <- 0
				    	}
			    	}
		    	}
		    }
		}
		if(!is.null(Factors))
		    FactorInteractions <- cbind(FactorInteractions,rep(1,length(Factors)))
	}
	return(list(Names=oldNames,mainEffects=mainEffects,Interactions=Interactions,Factors=Factors,FactorInteractions=FactorInteractions,ThreeWayInters=ThreeWayInters, Fouriers=Fouriers))
}

AddFourier <- function(Name, order, sincos = "None", Fouriers){
  if(sincos == "None")
    t1s <- c(1,2)
  else if(sincos == "sin")
    t1s <- 1
  else if(sincos == "cos")
    t1s <- 2
  else
    stop("sincos should be either `None', `sin' or `cos'")
  tname <- which(names(Fouriers)==Name)
  if(length(tname)<1){
    Fouriers[[Name]] <- matrix(0,nrow=2,ncol=order)
    Fouriers[[Name]][t1s,order] <- 1
  }
  else if(dim(Fouriers[[Name]])[2] < order){
    k <- order - dim(Fouriers[[Name]])[2]
    Fouriers[[Name]] <- cbind(Fouriers[[Name]],matrix(0,nrow=2,ncol=k))
    Fouriers[[Name]][t1s, order] <- 1
  }
  else{
    Fouriers[[Name]][t1s, order] <- 1
  }
  Fouriers
}

FitAddedModel <- function(newTermsList,DataString,ResponseString,tData){
	tLM <- FitLM(DataString=DataString, ResponseString=ResponseString, Names=newTermsList$Names, mainEffects=newTermsList$mainEffects, Interactions=newTermsList$Interactions, Factors=newTermsList$Factors, FactorInteractions=newTermsList$FactorInteractions, ThreeWayInters=newTermsList$ThreeWayInters, Fouriers=newTermsList$Fouriers, tData)
	#print(summary(tLM)$sigma)
	tLM
}

AddToModel <- function(Name,is.NewFactor=FALSE,try.NewThreeWay=FALSE,ModelSpecificationList){
	msl <- ModelSpecificationList
	newTerms <- addNewTerms(Name, msl$Names, msl$mainEffects, msl$Interactions, msl$Factors, msl$FactorInteractions, msl$ThreeWayInters, msl$Fouriers, add.order,is.NewFactor,try.NewThreeWay,FourierList = msl$FourierList)
	linModel <- FitAddedModel(newTerms,msl$DataString,msl$ResponseString, msl$tData)
	return(list(linModel=linModel,Names=newTerms$Names,mainEffects=newTerms$mainEffects,Interactions=newTerms$Interactions,Factors=newTerms$Factors,FactorInteractions=newTerms$FactorInteractions,ThreeWayInters=newTerms$ThreeWayInters,DataString=msl$DataString,ResponseString=msl$ResponseString,tData=msl$tData, Fouriers=newTerms$Fouriers, BestFourier=msl$BestFourier, maxOrder=msl$maxOrder))
}

AddBest <- function(candidateNames,candidateFactors,ModelSpecificationList){
	msl <- ModelSpecificationList
	tdfs <- summary(msl$linModel)$df
	if(tdfs[2]<= (tdfs[1]+tdfs[2])/2){
		print("No further terms permitted with the given degrees of freedom")
	  msl$Break <- TRUE
		return(msl)
	}
	else{
		currentRSS <- summary(msl$linModel)$sigma
		NewRSS <- sapply(1:length(candidateNames), function(i){tmp <- AddToModel(candidateNames[i],FALSE,FALSE,msl); summary(tmp$linModel)$sigma},simplify=TRUE)
		ThreeWayRSS <- unlist(sapply(1:length(candidateNames), function(k){if(length(msl$Names)>1){if(candidateNames[k]%in%msl$Names){if(msl$mainEffects[which(msl$Names==candidateNames[k])]>1){if(is.null(msl$ThreeWayInters[[candidateNames[k]]])){tmp <- AddToModel(candidateNames[k],FALSE,TRUE,msl); summary(tmp$linModel)$sigma}}}}},simplify=TRUE))
		if(!is.null(candidateFactors)){
		NewRSSFact <- sapply(1:length(candidateFactors), function(i){tmp <- AddToModel(candidateFactors[i],TRUE,FALSE,msl); summary(tmp$linModel)$sigma},simplify=TRUE)
		AllNewRSS <- c(NewRSS,NewRSSFact)
		}
		else{
			AllNewRSS <- NewRSS
		}
		if(!is.null(ThreeWayRSS)){
			AllNewRSS <- c(AllNewRSS, ThreeWayRSS)
		}
		if(!is.null(msl$BestFourier) & msl$BestFourier){
		  BestFourierModel <- TryFouriers(candidateNames, msl, msl$maxOrder)
		  BestFRSS <- summary(BestFourierModel$linModel)$sigma
		  AllNewRSS <- c(AllNewRSS,BestFRSS)
		}
		Reductions <- currentRSS - AllNewRSS
		if(!any(Reductions > 0)){
			print("No permitted terms improve the fit")
			msl$Break <- TRUE
			return(msl)
		}
		tMax <- which.max(Reductions)
		if(tMax <= length(candidateNames)){
			if(candidateNames[tMax]=="Noise"){
				print("Noise fitted, stopping algorithm")
				msl$Break <- TRUE
				return(msl)
			}
			else{
				print(paste("Max reduction is ",Reductions[tMax]," using ", candidateNames[tMax],sep=""))
				return(AddToModel(candidateNames[tMax],FALSE,FALSE,msl))
			}
		}
		else if(tMax == length(Reductions) & msl$BestFourier){
		  if(!is.null(BestFourierModel$Fouriers$Noise)){
		    print("Fourier Noise fitted, stopping algorithm")
		    msl$Break <- TRUE
		    return(msl)
		  }
		  else{
		    print("Max reduction is with the Fourier term")
		    return(BestFourierModel)
		  }
		}
		else if(is.null(ThreeWayRSS)||(tMax <= length(candidateNames)+length(candidateFactors))){
			print(paste("Max reduction is ", Reductions[tMax]," using factor ", candidateFactors[tMax-length(candidateNames)],sep=""))
			return(AddToModel(candidateFactors[tMax-length(candidateNames)],TRUE,FALSE,msl))
		}
		else{
			possNames <- msl$Names[which(msl$mainEffects>1)]
			notPossNames <- names(msl$ThreeWayInters)
			if(!is.null(notPossNames))
			    possNames <- possNames[-which(possNames%in%notPossNames)]
			tAddition <- possNames[tMax-length(candidateNames)-length(candidateFactors)]
			print(paste("Max reduction is ", Reductions[tMax], " using Three Way Interactions with ", tAddition, sep=""))
			return(AddToModel(tAddition,FALSE,TRUE,msl))
		}
	}	
}

TryFouriers <- function(candidateNames, ModelSpecificationList, maxOrder=NULL){
  msl <- ModelSpecificationList
  tdfs <- summary(msl$linModel)$df
  if(tdfs[2]<= (tdfs[1]+tdfs[2])/2){
    print("No further Fourier terms permitted with the given degrees of freedom")
    return(msl)
  }
  if(is.null(maxOrder))
    maxOrder <- min((tdfs[1]+tdfs[2])/4,10)
  currentRSS <- summary(msl$linModel)$sigma
  NewRSS <- sapply(1:maxOrder, function(k) sapply(1:length(candidateNames), function(i){msl$FourierList <- list(order=k, sincos="None"); tmp <- AddToModel(candidateNames[i],FALSE,FALSE,msl); summary(tmp$linModel)$sigma},simplify=TRUE))
  ReductionMatrix <- currentRSS - NewRSS
  tMax <- arrayInd(which.max(ReductionMatrix), .dim = dim(ReductionMatrix))
  if(!any(ReductionMatrix > 0)){
    print("No permitted Fourier terms improve the fit")
    msl$Break <- TRUE
    msl$FourierList <- NULL
    return(msl)
  }
  torder <- tMax[1,2]
  tName <- candidateNames[tMax[1,1]]
  print(paste("Max Fourier reduction is ",ReductionMatrix[tMax]," using sin(pi*",torder,"*", tName, ") and cos(pi*",torder,"*", tName, ")",sep=""))
  msl$FourierList <- list(order=torder,sincos="None")
  AddToModel(tName,FALSE,FALSE,msl)
}
