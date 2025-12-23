packages <- c('GenSA', 'parallel', 'cvTools', 'fields', 'mco', 'far', 'lhs', 'tensor', 'matrixcalc', 'rstan')
sapply(packages, require, character.only = TRUE)


#### Emulation ####
# The function that fits the linear model and Gaussian process components of a scalar emulator, 
# where `EMULATE' is a function that fits a regression model.
# Generally used inside other functions.
#### Problem - don't want it to rename as y, instead select an output name ####
FitEmulatorPars <- function(response, training, validation, cands, canfacs = NULL, ...){
  train <- training[, c(cands, canfacs, response)]
  val <- validation[,c(cands, canfacs, response)]
  regmodel <- EMULATE(response, train, tcands = cands, tcanfacs = canfacs, ...)
  whichnoise <- which(cands=="Noise")
  #names(d.initial) <- cands[-whichnoise]
  whichnoise <- which(names(train)=="Noise")
  train <- train[,-whichnoise]
  val <- val[,-whichnoise]
  active <- regmodel$Names
  if (is.null(regmodel$Factors) == FALSE){active <- c(active, regmodel$Factors)}
  t <- dim(train)[2]
  colnames(train)[t] <- colnames(val)[t] <- "y"
  #startpar <- c(rep(1, length(active)))
  #names(startpar) <- active
  #opt <- GenSA(NULL, MaxLikelihood, lower = c(rep(0.1,length(active)),0.001), upper = c(rep(10,length(active)),0.999),
  #      data = train[,c(active, "y")], regmodel = regmodel$linModel, control=list(max.call=100))
  #print(opt)
  check <- 0
  count <- 0
  while (check == 0 & count < 2){
    bestpars <- matrix(numeric(10*(length(active)+1)), nrow = 10)
    maxl <- c(rep(0,10))
    for (i in 1:10){
      opt <- GenSA(NULL, MaxLikelihood, lower = c(rep(0.1,length(active)),0.001), upper = c(rep(10,length(active)),0.999),
                   data = train[,c(active, "y")],  regmodel = regmodel$linModel, control=list(max.call=100))
      bestpars[i,] <- opt$par
      maxl[i] <- opt$value
    }
    bestl <- which.min(maxl)
    #print(maxl[bestl])
    #print(bestpars[bestl,])
    opt2 <- GenSA(bestpars[bestl,], CrossValPars, lower = c(rep(0.1,length(active)),0.001), upper = c(rep(10,length(active)),0.999),
                  data = train[,c(active, "y")], val = val[,c(active, "y")],  regmodel = regmodel$linModel, control=list(max.call=10))
    if (opt2$value > 99998){
      check <- 0
      count <- count + 1
    }
    else {check <- 1}
  }
  if (count == 2){
    count2 <- 0
    while (opt2$value > 99998 & count2 < 20){
      opt2 <- GenSA(NULL, CrossValPars, lower = c(rep(0.1,length(active)),0.001), upper = c(rep(10,length(active)),0.999),
                    data = train[,c(active, "y")], val = val[,c(active, "y")],  regmodel = regmodel$linModel, control=list(max.call=10))
      #print(c(opt2$par, opt2$value))
      if (opt2$value > 99998){
        count2 <- count2 + 1
      }
    }
  }
  if (opt2$value > 99998){print("Warning: emulator failed cross-validation or prediction test")}
  p <- t - 1
  d.op <- opt2$par[1:length(active)]
  nu.op <- opt2$par[length(opt2$par)]
  temp.em <- em(train[,active],train$y,d.op,nu.op,regmodel$linModel,active=NULL)
  cvem <- cv.em(temp.em,dim(train)[1])
  par(mfrow=c(1,2), mar=c(4,2,2,2))
  cv.plot(cvem)
  val.pred(temp.em, val[,active], val$y)
  fulldata <- rbind(train, val)
  regmodel.full <- lm(update(as.formula(regmodel$linModel), y~.),data=fulldata)
  emulator <- em(fulldata[,active],fulldata[,p+1],d.op,nu.op,regmodel.full,active=NULL)
  emulator$x <- fulldata[,1:p]
  emulator$y <- fulldata[,p+1]
  emulator$valid <- ifelse(opt2$value > 99998, 0, 1)
  return(emulator)
}

# Fitting emulator given data, correlation lengths etc.
# x - matrix of inputs
# y - vector of outputs
# d - correlation lengths (vector)
# nu - nugget (scalar)
# regmodel - a linear model object. Can be constant.
# type = "BR" or "LS" - fit Bayesian regression or least squares model
# active - can specify which of the columns of x are active
#### Fix inputs so consistent ####
em <- function(x, y, d, nu, regmodel, type = "BR", active = NULL, ...){
  require(MASS)
  if (length(d) == 1){
    x <- as.data.frame(x)
    colnames(x)[1] <- names(d)
  }
  names(d) <- names(x)
  if (is.null(active) == TRUE){ ### active = NULL means that all variables in x are active
    x.ac <- x
    active <- names(x)
  }
  else {
    x.ac <- x[,active]
    d <- d[active]
  }
  if (length(active) == 1){
    x.ac <- as.data.frame(x.ac)
    colnames(x.ac) <- active
  }
  if (type == "LS") {
    if (nu == 1){
      pred = predict(regmodel, x.ac)
      error = y - pred
      H = model.matrix(regmodel, x.ac)
      n = dim(x.ac)[1]
      q = dim(H)[2]
      sigma2 = (summary(regmodel)$sigma)^2
      return(list(x=x,y=y,regmodel=regmodel,nu=nu,error=error,H=H,n=n,q=q,sigma2=sigma2,type=type,active=active))
    }
    else {
      A = calcA(x.ac,d,nu,...)
      pred = predict(regmodel, x)
      error = y - pred
      Q = chol(A)
      H = model.matrix(regmodel, x)
      n = dim(x)[1]
      q = dim(H)[2]
      x.c = backsolve(Q, error, transpose = TRUE)
      sigma2 = (summary(regmodel)$sigma)^2
      return(list(x=x,y=y,regmodel=regmodel,d=d,nu=nu,Q=Q,error=error,H=H,n=n,q=q,x.c=x.c,sigma2=sigma2,type=type,
                  active=active,...))
    }
  }
  else if (type == "BR"){
    if (nu == 1){
      pred = predict(regmodel, x.ac)
      error = y - pred
      H = model.matrix(regmodel, x.ac)
      n = dim(x.ac)[1]
      q = dim(H)[2]
      sigma2 = as.numeric(((n-q-2)^(-1))*(t(error)%*%error))
      return(list(x=x,y=y,regmodel=regmodel,nu=nu,error=error,H=H,n=n,q=q,sigma2=sigma2,type=type,active=active))
    }
    else {
      A = calcA(x.ac,d,nu,...)
      data <- as.data.frame(cbind(x.ac,y))
      if (is.vector(x) == TRUE){
        tempName <- names(summary(regmodel)$aliased)[2]
        names(data) <- c(tempName, "y")
      }
      glsmodel = lm.gls(update.formula(formula(regmodel),y~.),data,W=A,inverse=TRUE)
      pred = pred.gls(glsmodel, x.ac)
      error = y - pred
      Q = chol(A)
      H = model.matrix(regmodel, x.ac)
      n = dim(x.ac)[1]
      q = dim(H)[2]
      x.c = backsolve(Q, error, transpose = TRUE)
      h.c = backsolve(Q, H, transpose = TRUE)
      comp2 = chol2inv(chol(crossprod(h.c, h.c)))
      sigma2 = as.numeric(((n-q-2)^(-1))*crossprod(x.c,x.c))
      return(list(x=x,y=y,regmodel=glsmodel,d=d,nu=nu,Q=Q,error=error,H=H,n=n,q=q,x.c=x.c,h.c=h.c,comp2=comp2,sigma2=sigma2,
                  type=type,active=active,...))
    }
  }
  else {
    stop("Type must be 'LS' (Least Squares) or 'BR' (Bayes Regression)")
  }                     
}

# GLS model predictor (required for em)
pred.gls <- function(glsmodel,data){
  tt <- terms(glsmodel)
  Terms <- delete.response(tt)
  mm <- model.frame(Terms, data, xlev = glsmodel$xlevels)
  x0 <- model.matrix(Terms, mm, contrasts.arg = glsmodel$contrasts)
  prediction <- x0 %*% glsmodel$coefficients
  return(prediction)
}

# Fit multiple emulators, generally to a set of basis coefficients
# tData - standard data object, including `Noise' column
# NumberOfEms - how many emulators fitting (e.g. first 5 basis coefficients)
# TrainingProportion - proportion of the data that should be used to train the emulator, sampled randomly from tData. Remainder of data used as validation set
#### Sample multiple noise vectors - this would be within tData ####
BasisEmulators <- function(tData, NumberOfEms, TrainingProportion = 0.75, ...){
  lastCand <- which(names(tData)=="Noise")
  n <- dim(tData)[1]
  n1 <- ceiling(n*TrainingProportion)
  samp <- sample(1:n, n)
  lapply(1:NumberOfEms, function(k) FitEmulatorPars(response=names(tData)[lastCand+k], training=tData[samp[1:n1],], validation=tData[samp[-(1:n1)],], cands=names(tData)[1:lastCand], canfacs=NULL, ...))
}

# Evaluating em object at new data, returning expectation and variance
# em - an emulator object
# new - a new design
#### ... - options passed to calctx/calcA - surely this should be within em object? ####
pred.em <- function(em, newDesign, ...){
  x <- em$x
  n <- em$n
  regmodel <- em$regmodel
  nu <- em$nu
  error <- em$error
  H <- em$H
  q <- em$q
  sigma2 <- em$sigma2
  type <- em$type
  active <- em$active
  if (length(active) == 1){
    x.ac <- as.data.frame(x[,active])
    #if (dim(x.ac)[2] == 1){
    #  x.ac <- x.ac
    #}
    #else {
    #  x.ac <- x.ac[,active]
    #}
    names(x.ac) <- active
  }
  else {
    x.ac <- x[,active]
  }
  if (is.null(dim(newDesign))){
    newDesign <- newDesign
    new.ac <- as.data.frame(newDesign)
    colnames(new.ac) <- active
  }
  else if ((dim(newDesign)[1]*dim(newDesign)[2]) == 1){
    newDesign <- newDesign
    new.ac <- as.data.frame(newDesign)
    colnames(new.ac) <- active
  }
  else if ((dim(newDesign)[1]*dim(newDesign)[2]) == dim(newDesign)[2] & length(active) == 1){
    #new <- new[,names(x)]
    new.ac <- as.data.frame(newDesign[,active])
    names(new.ac) <- active
  }
  else {
    newDesign <- newDesign[,names(x)]
    new.ac <- as.data.frame(newDesign[,active])
    if (length(active) == 1){
      names(new.ac) <- active
    }
  }
  if (type == "BR"){
    if (nu == 1){
      postm <- predict(regmodel,new.ac)
      tt <- terms(regmodel)
      Terms <- delete.response(tt)
      mm <- model.frame(Terms, new.ac, xlev = regmodel$xlevels)
      hx <- model.matrix(Terms, mm, contrasts.arg = regmodel$contrasts)
      postcov <- sigma2 * (diag(dim(new.ac)[1]) + (hx)%*%chol2inv(chol(t(H)%*%H))%*%t(hx))
    }
    else {
      d <- em$d
      tx <- calctx(x.ac,new.ac,d,nu,...)
      Q <- em$Q
      x.c <- em$x.c
      h.c <- em$h.c
      comp2 <- em$comp2
      y.c <- backsolve(Q, tx, transpose = TRUE)
      mean.adj <- crossprod(y.c,x.c)
      cov.adj <- crossprod(y.c,y.c)
      priormean <- pred.gls(regmodel, new.ac)
      priorcov <- calcA(new.ac, d, nu,...)
      postm <- priormean + mean.adj
      postcov <- priorcov - cov.adj                                   
      tt <- terms(regmodel)
      Terms <- delete.response(tt)
      mm <- model.frame(Terms, new.ac, xlev = regmodel$xlevels)
      hx <- model.matrix(Terms, mm, contrasts.arg = regmodel$contrasts)
      comp1 <- hx - crossprod(y.c, h.c)
      postcov <- sigma2 * (postcov + comp1 %*% comp2 %*% t(comp1))
    }
  }
  else if (type == "LS"){
    if (nu == 1){
      preds <- predict(regmodel,new.ac,interval="prediction",level=0.95)
      postm <- preds[,1]
      postcov <- ((preds[,3] - preds[,1])/qt(0.975,n-q))^2
    }
    else {
      d <- em$d
      tx <- calctx(x.ac,new.ac,d,nu,...)
      Q <- em$Q
      x.c <- em$x.c
      y.c <- backsolve(Q, tx, transpose = TRUE)
      mean.adj <- crossprod(y.c,x.c)
      cov.adj <- crossprod(y.c,y.c)
      priormean <- pred.gls(regmodel, new.ac)
      priorcov <- calcA(new.ac, d, nu,...)
      postm <- priormean + mean.adj
      postcov <- sigma2 * (priorcov - cov.adj)                                   
    }
  }
  return(list(post.m = postm, post.cov = diag(postcov)))
}

# Predicting the output for a list of em objects (e.g. the output given by BasisEmulators)
# ems - list of emulator objects
# design - matrix of inputs to make predictions at
#### Is 1000 the optimal split? ####
MultiPred <- function(ems, design){
  Expectation <- matrix(0, nrow = dim(design)[1], ncol = length(ems))
  Variance <- matrix(0, nrow = dim(design)[1], ncol = length(ems))
  s <- floor(dim(design)[1]/1000)
  if (s > 0){
    for (i in 1:s){
      EmOutput <- mclapply(1:length(ems), function(e) pred.em(ems[[e]], design[(1000*(i-1) + 1):(1000*i),]))
      for (j in 1:length(ems)){
        Expectation[(1000*(i-1) + 1):(1000*i),j] <- EmOutput[[j]]$post.m
        Variance[(1000*(i-1) + 1):(1000*i),j] <- EmOutput[[j]]$post.cov
      }
    }
  }
  r <- s*1000 - dim(design)[1]
  if (r > 0){
    EmOutput <- mclapply(1:length(ems), function(e) pred.em(ems[[e]], design[(1000*s + 1):(1000*s + r),]))
    for (j in 1:length(ems)){
      Expectation[(1000*s + 1):(1000*s + r),j] <- EmOutput[[j]]$post.m
      Variance[(1000*s + 1):(1000*s + r),j] <- EmOutput[[j]]$post.cov
    }
  }
  return(list(Expectation=Expectation, Variance=Variance))
}

# Cross-validation used in the fitting of the Gaussian process correlation lengths
CrossValPars <- function(x, data, val, regmodel, type = "BR"){
  d <- x[-length(x)]
  nu <- x[length(x)]
  p <- length(d)
  n <- dim(data)[1]
  colnames(data)[p+1] <- colnames(val)[p+1] <- "y"
  new.em <- em(data[,1:p],data[,p+1],d,nu,regmodel,type)
  new.cv <- cv.em(new.em, K=n)
  new.outside95 <- sum(abs(data[,p+1]-new.cv$cvmean) >= qt(0.975,new.cv$df)*new.cv$cvse)
  check1 <- is.valid(new.outside95, n, 0.05)
  pred <- pred.em(new.em, val[,1:p])
  new.outside95v <- sum((abs(val[,(p+1)]-pred$post.m)>=qt(0.975,new.em$n - new.em$q)*sqrt(pred$post.cov)))
  check2 <- is.valid(new.outside95v, dim(val)[1], 0.05)
  output <- mean(new.cv$cvse) + ifelse(check1 == FALSE, 99999, 0) + ifelse(check2 == FALSE, 99999, 0)
  return(c(output, mean(new.cv$cvse), check1, check2))
}

# Cross-validate emulator, splitting data into K groups
# em - emulator object
# K - number of groups to split the data into (defaults to K = n, leave-one-out cross-validation)
cv.em <- function(em, K = NULL){
  require(cvTools)
  data <- data.frame(em$x[,em$active],em$y) 
  if (length(em$active) == 1){
    colnames(data)[1] <- em$active
  }
  n <- em$n
  d <- em$d
  nu <- em$nu
  type <- em$type
  if (is.null(K) == TRUE){
    K <- n
  }
  if (n == K){
    basis <- attr(terms(em$regmodel),"term.labels")
    if (length(basis)==0) basis = "1"
    cv <- mclapply(1:n, function(i) leave.one.out(i,data,d,nu,basis,type))
    cv <- mclapply(cv,function(x) do.call(cbind,x))
    cv <- do.call(rbind,cv)
    cvmean <- cv[,1]
    cvse <- cv[,2]
    df <- n - length(basis) - 1
    return(list(x=em$x,y=em$y,cvmean=cvmean,cvse=cvse,df=df))
  }
  else {
    folds <- cvFolds(n, K, R = 1, type = "random")
    basis <- attr(terms(em$regmodel),"term.labels")
    if (length(basis)==0) basis = "1"
    cvmean <- c(rep(0,n))
    cvse <- c(rep(0,n))
    for(i in 1:K){
      train <- data[folds$subsets[folds$which != i], ]
      trainx <- train[,-length(train)]
      trainy <- train[,length(train)]
      val <- data[folds$subsets[folds$which == i], ]
      valx <- val[,-length(val)]
      valy <- val[,length(val)]
      mod <- paste("trainy~", paste(basis,collapse="+"))
      newlm <- lm(as.formula(mod), data = trainx)
      newem <- em(trainx,trainy,d,nu,newlm,type)
      pred <- pred.em(newem,valx,valy)
      cvmean[folds$subsets[folds$which == i]] <- pred$post.m
      cvse[folds$subsets[folds$which == i]] <- sqrt((pred$post.cov))
    }
    df <- summary(newlm)$df[2]
    return(list(x=em$x,y=em$y,cvmean=cvmean,cvse=cvse,folds=folds,df=df))
  }
}

# Refitting emulator leaving out ensemble member i
leave.one.out <- function(i,data,d,nu,basis,type){
  p <- dim(data)[2]
  train <- data[-i, ]
  trainx <- train[,1:(p-1)]
  trainy <- train[,p]
  val <- data[i, ]
  valx <- val[,1:(p-1)]
  valy <- val[,p]
  if (p == 2){
    trainx <- as.data.frame(trainx)
    valx <- as.data.frame(valx)
    names(trainx) <- names(valx) <- names(data)[1]
  }
  mod <- paste("trainy~", paste(basis,collapse="+"))
  newlm <- lm(as.formula(mod), data = trainx)
  newem <- em(trainx,trainy,d,nu,newlm,type)
  pred <- pred.em(newem,valx,valy)
  cvmean <- pred$post.m
  cvse <- sqrt((pred$post.cov))
  return(list(cvmean=cvmean,cvse=cvse))
}

# Gives the negative logLikelihood
MaxLikelihood <- function(x, data, regmodel, type = "BR"){
  d <- x[-length(x)]
  nu <- x[length(x)]
  p <- length(d)
  n <- dim(data)[1]
  colnames(data)[p+1] <- "y"
  new.em <- em(data[,1:p],data[,p+1],d,nu,regmodel,type)
  A <- t(new.em$Q) %*% new.em$Q
  logL <- -0.5*determinant(A, logarithm = TRUE)$modulus -0.5*determinant(crossprod(new.em$h.c, new.em$h.c), logarithm=TRUE)$modulus - ((new.em$n - new.em$q)/2)*log(new.em$sigma2)
  output <- -logL
  return(output)
}

# Does the emulator pass validation, using a binomial argument
is.valid <- function(outside, n, tolerance = 0.05, level = 0.05){
  ind <- TRUE
  check1 <- 1 - pbinom(outside-1, n, level)
  if (check1 < tolerance){ind <- FALSE}
  check2 <- pbinom(outside, n, level)
  if (check2 < tolerance){ind <- FALSE}
  return(ind)
}

# Cross-validation plot - takes output of cv.em
cv.plot <- function(cvem,level=0.95){
  require(sfsmisc)
  mean <- cvem$cvmean
  se <- cvem$cvse
  y <- cvem$y
  df <- cvem$df
  z <- level + (1-level)/2
  upp <- max(c(mean+qt(z,df)*se,y))
  low <- min(c(mean-qt(z,df)*se,y))
  errbar(mean,mean,mean + qt(z,df)*se, mean - qt(z,df)*se,pch=18,
         main = "Cross-validation", xlab = "Prediction", ylab="Observation", ylim = c(low,upp))
  points(mean,y,pch=19,col = ifelse(abs(y-mean)<=qt(z,df)*se,"green","red"))
}

# Plots against each input parameter instead, press enter to cycle through
cv.par <- function(cvem,level=0.95){
  mean <- cvem$cvmean
  se <- cvem$cvse
  x <- cvem$x
  y <- cvem$y
  df <- cvem$df
  z <- level + (1-level)/2
  p <- dim(x)[2]
  upp <- max(c(mean+qt(z,df)*se,y))
  low <- min(c(mean-qt(z,df)*se,y))
  for (i in 1:p){cat ("Press [enter] to continue")
    line <- readline()
    errbar(x[,i],mean,mean + qt(z,df)*se, mean - qt(z,df)*se,pch=18, xlab=names(x)[i],ylab="Prediction", ylim = c(low,upp)) 
    points(x[,i],y,pch=19,col = ifelse(abs(y-mean)<=qt(z,df)*se,"green","red"))}
}


# Plots predictions for validation data set. 'model' can be lm or em object
val.pred <- function(model, valdata, valobs, level=0.95){
  if (is.null(model$coefficients) == F){
    pred <- predict(model,valdata,interval="prediction",level=level)
    upp <- max(c(pred[,3],valobs))
    low <- min(c(pred[,2],valobs))
    errbar(pred[,1],pred[,1],pred[,3],pred[,2],cap = 0.015,pch=20,ylim=c(low,upp),
           main="Validation for regression",xlab = "Prediction",ylab="Observation")
    points(pred[,1],valobs,pch=19,col = ifelse(valobs>pred[,3] | valobs<pred[,2],"red","green"))}
  else {
    pred <- pred.em(model,valdata)
    mean <- pred$post.m
    se <- sqrt(pred$post.cov)
    z <- level + (1-level)/2
    df <- model$n - model$q
    upp <- max(c(mean+qt(z,df)*se,valobs))
    low <- min(c(mean-qt(z,df)*se,valobs))
    errbar(mean,mean,mean + qt(z,df)*se, mean - qt(z,df)*se,pch=18, xlab="Prediction",ylab="Observation", ylim = c(low,upp)) 
    points(mean,valobs,pch=19,col = ifelse(abs(valobs-mean)<=qt(z,df)*se,"green","red"))
  }
}

# Calculating the correlation matrices required in em
# t(x)
calctx <- function(x, new, d, nu = 0, cov = "gauss", d2 = c(rep(2,length(d)))){
  n <- dim(x)[1] # number of training points
  m <- dim(new)[1]
  if (is.null(m) == T){
    m <- 1
  }
  p <- dim(x)[2] # number of parameters
  if (is.null(p) == T){
    p <- 1
  }
  stopifnot(p == length(d))
  if (p == 1 & m == 1){
    new <- new
  }
  else {
    new <- new[,names(x)]
  } 
  if (p == 1){
    x <- x
    new <- new
  }
  else {
    for (i in 1:p){
      x[,i] <- as.numeric(as.character(x[,i]))
    }
    for (i in 1:p){
      new[,i] <- as.numeric(as.character(new[,i]))
    }
  }  
  if (cov == "gauss") {
    leng <- as.matrix(rdist(scale(x,center=FALSE,scale=d),scale(new,center=FALSE,scale=d))) 
    tx <- (1 - nu)*exp(-(leng^2))
  }
  else if (cov == "Matern3_2") {
    tx <- matrix(c(rep(1,n*m)),nrow=n)
    for (i in 1:p){
      dis <- as.matrix(rdist(scale(x[,i],center=FALSE,scale=d[i]/sqrt(3)),scale(new[,i],center=FALSE,scale=d[i]/sqrt(3))))
      dis <- (matrix(c(rep(1,n*m)),nrow=n) + dis) * exp(-dis)
      tx <- tx * dis
    }
    tx <- (1 - nu)*tx
  }
  else if (cov == "Matern5_2") {
    tx <- matrix(c(rep(1,n*m)),nrow=n)
    for (i in 1:p){
      dis <- as.matrix(rdist(scale(x[,i],center=FALSE,scale=d[i]/sqrt(5)),scale(new[,i],center=FALSE,scale=d[i]/sqrt(5))))
      dis <- (matrix(c(rep(1,n^2)),nrow=n) + dis + (dis^2)/3) * exp(-dis)
      tx <- tx * dis
    }
    tx <- (1 - nu)*tx
  }                                                                                 
  else if (cov == "powerexp") {
    tx <- matrix(c(rep(0,n^2)),nrow=n)
    for (i in 1:p){
      dis <- as.matrix((rdist(scale(x[,i],center=FALSE,scale=d[i]),scale(new[,i],center=FALSE,scale=d[i])))^d2[i])                                              
      tx <- tx + dis
    }
    tx <- (1 - nu)*exp(-tx)
  }
  else {
    stop("Covariance function must be 'gauss' or 'exp' or 'Matern3_2' or 'Matern5_2' or 'powerexp'")
  }                     
  return(tx)
}

# A
calcA <- function(x, d, nu, cov = "gauss", d2 = c(rep(2,length(d)))){
  n <- dim(x)[1] # number of training points
  p <- dim(x)[2] # number of parameters
  if (is.null(n) == TRUE){
    n <- length(x)
  }
  if (is.null(p) == TRUE){
    p <- 1
  }
  stopifnot(p == length(d))
  if (p == 1){
    x <- x
  }
  else {
    for (i in 1:p){
      x[,i] <- as.numeric(as.character(x[,i]))
    }
  }
  if (cov == "gauss") {
    leng <- as.matrix(dist(scale(x,center=FALSE,scale=d),method="euclidean",diag=TRUE,upper=TRUE))
    A <- nu*diag(n) + (1 - nu)*exp(-(leng^2))
  }
  else if (cov == "exp") {
    leng <- matrix(c(rep(0,n^2)),nrow=n)
    leng <- as.matrix(dist(scale(x,center=FALSE,scale=d),method="euclidean",diag=TRUE,upper=TRUE))
    A <- nu*diag(n) + (1 - nu)*exp(-leng)
  }
  else if (cov == "Matern3_2") {
    A <- matrix(c(rep(1,n^2)),nrow=n)
    for (i in 1:p){
      dis <- as.matrix(dist(scale(x[,i],center=FALSE,scale=d[i]/sqrt(3)),method="euclidean",diag=TRUE,upper=TRUE))
      dis <- (matrix(c(rep(1,n^2)),nrow=n) + dis) * exp(-dis)
      A <- A * dis 
    }
    A <- nu*diag(n) + (1 - nu)*A
  }
  else if (cov == "Matern5_2") {
    A <- matrix(c(rep(1,n^2)),nrow=n)
    for (i in 1:p){
      dis <- as.matrix(dist(scale(x[,i],center=FALSE,scale=d[i]/sqrt(5)),method="euclidean",diag=TRUE,upper=TRUE))
      dis <- (matrix(c(rep(1,n^2)),nrow=n) + dis + (dis^2)/3) * exp(-dis)
      A <- A * dis
    }
    A <- nu*diag(n) + (1 - nu)*A
  }
  else if (cov == "powerexp") {
    A <- matrix(c(rep(0,n^2)),nrow=n)
    for (i in 1:p){
      dis <- as.matrix((dist(scale(x[,i],center=FALSE,scale=d[i]),method="euclidean",diag=TRUE,upper=TRUE))^d2[i])                                              
      A <- A + dis
    }
    A <- nu*diag(n) + (1 - nu)*exp(-A)
  }
  else {
    stop("Covariance function must be 'gauss' or 'exp' or 'Matern3_2' or 'Matern5_2' or 'powerexp'")
  }                     
  return(A)
}





GetEmulatableDataWeighted <- function(Design, EnsembleData, HowManyBasisVectors, Noise=TRUE, weightinv = NULL){
  if(Noise){
    Noise <- runif(length(Design[,1]),-1,1)
    Design <- cbind(Design, Noise)
  }
  #tcoefs <- StandardCoefficients(EnsembleData$CentredField, EnsembleData$tBasis[,1:HowManyBasisVectors], orthogonal=FALSE)
  coeffs <- CalcScores(data = EnsembleData$CentredField, basis = EnsembleData$tBasis[,1:HowManyBasisVectors], weightinv = weightinv)
  tData <- cbind(Design, coeffs)
  ln <- length(names(tData))
  names(tData)[(ln-HowManyBasisVectors+1):ln] <- paste("C",1:HowManyBasisVectors,sep="")
  tData
}







# Function that gives variance given by deleted basis vectors (Wilkinson 2010)
# DataBasis - object containing basis, centred field etc.
# q - where the basis is truncated
DiscardedBasisVariance <- function(DataBasis, q, weightinv = NULL){
  BasMinusQ <- DataBasis$tBasis[,-(1:q)]
  DeletedCoeffs <- CalcScores(DataBasis$CentredField, BasMinusQ, weightinv)
  EstVar <- apply(DeletedCoeffs, 2, var)
  DeletedBasisVar <- BasMinusQ %*% diag(EstVar) %*% t(BasMinusQ)
  return(DeletedBasisVar)
}


# Given a matrix (or vector) of implausibilities 'impl', this function counts how many are in NROY space for each basis size, and each measure max2, max3 etc.
UV.HM <- function(impl, I = 3){
  nroys <- impl < I
  d <- dim(impl)[2]
  NROYsize <- matrix(c(rep(0,d*(d+1))), nrow = d)
  for (i in 1:d){
    NROYsize[i,1] <- sum(nroys[,i])
  }
  for (i in 2:d){
    nroy <- apply(nroys[,1:i], 1, sum)
    for (j in 1:i){
      NROYsize[i,j+1] <- sum(nroy > (i - j))
    }
  }
  NROYsize <- NROYsize/dim(impl)[1]*100
  colnames(NROYsize) <- c("UV", paste("max", 1:d))
  rownames(NROYsize) <- paste("basis", 1:d)
  output <- NULL
  output$UV <- NROYsize[,1]
  output$max <- NROYsize[d,-1]
  if (d > 5){
    output$max <- output$max[1:5]
  }
  return(output)
}


#### Calibration ####

Calibrate <- function(Ems, DataBasis, Obs, Error, Disc, Prior.x=logPriorDist, ...){
  q <- length(Ems)
  V2 <- DataBasis$tBasis[,-(1:q)]
  DeletedScores <- CalcScores(DataBasis$CentredField, V2)
  EstVar <- apply(DeletedScores, 2, var)
  BasisVar <- V2 %*% diag(EstVar) %*% t(V2)
  Error <- Error + BasisVar
  Posterior <- metrop(CalLikelihood, Ems=ems, Prior.x = Prior.x, DataBasis = DataBasis, Obs = obs, Error = Error, Disc = Disc, ...)
  return(Posterior)
}

CalLikelihood <- function(x, Ems, DataBasis, Obs, Error, Disc, VarNames, Prior.x=logPriorDist){
  x <- as.data.frame(x)
  if (dim(x)[1] > dim(x)[2]) x <- as.data.frame(t(x))
  colnames(x) <- VarNames
  q <- length(Ems)
  Basis <- DataBasis$tBasis[,1:q]
  logPrior <- Prior.x(x)
  EmOutput <- lapply(1:q, function(e) pred.em(Ems[[e]], x))
  Expectation <- Variance <- numeric(q)
  for (i in 1:q){
    Expectation[i] <- EmOutput[[i]]$post.m
    Variance[i] <- EmOutput[[i]]$post.cov
  }
  Expectation <- Recon(Expectation, Basis)
  Variance <- Basis %*% diag(Variance) %*% t(Basis)
  V <- Error + Disc + Variance
  Q <- chol(V)
  y <- backsolve(Q, as.vector(Obs - Expectation)) # tranpose = TRUE causes break
  y <- crossprod(y,y)
  logL <- as.numeric(-0.5*determinant(V, logarithm=TRUE)$modulus - 0.5*y + logPrior)
  return(logL)
}

# Uniform prior on inputs
logUnifDist <- function(x){
  if (any(x < -1) || any(x > 1))
    return (-Inf)
  else {
    return (1)
  }
}


CalImplModel <- function(x, ImplData, prior.x = logBoundPrior){
  n <- dim(ImplData)[1]
  Ic <- ImplData$Ic
  If <- ImplData$If
  x <- as.data.frame(t(x))
  logPrior <- as.numeric(logBoundPrior(x))
  if (logPrior == -Inf){
    logL <- -Inf
    return(logL)
  }
  else {
    w <- numeric(n)
    for (i in 1:n){w[i] <- as.numeric(exp(x[1] + x[2]*log(If[i])))}
    logC <- 0
    for (i in 1:n){
      sigmai <- x[5]/w[i]
      newlogC <- as.numeric(-0.5*log(2*sigmai*pi) - (1/(2*sigmai))*(Ic[i] - (x[3] + x[4]*If[i]))^2)
      logC <- logC + newlogC
    }
    logL <- logC + logPrior
    if (is.na(logL) == TRUE){logL <- -Inf}
    if (logL == +Inf){logL <- -Inf}
    return(logL)
  }
}

logBoundPrior <- function(x){
  if (x[4] <= 0 | x[2] >= 0 | x[5] <= 0 | x[1] < -10 | x[1] > 10 | x[3] > 0 | x[3] < -50){return(-Inf)}
  else {
    logb1 <- 9*log(x[4]) - x[4]
    loga1 <- 9*log(-x[2]) + x[2]
    logb0 <- -(x[3]^2)
    loga0 <- -(x[1]^2)
    sigmas <- log(1/x[5])
    total <- logb1 + loga1 + logb0 + loga0 + sigmas
    return (total) 
  }
}

#### Sampling based on implausibility ####
# Essentially, a Latin hypercube in impl space
# Also want to ensure that have good spread in parameter space
# Hence maximise minimum distance for samples
# DesignNROY - design in the current NROY space for a large sample
# Impl - vector of implausibilities for the design
# SampleSize - size of sample within NROY space
# Intervals - number of segments to divide implausibility space into. Defaults to SampleSize
# RepeatSamples - how many times to sample before returning maximin
ImplSample <- function(DesignNROY, Impl, SampleSize, Intervals = NULL, RepeatSamples = 100){
  n <- SampleSize
  if (is.null(Intervals)){
    Intervals <- n
  }
  maxImpl <- max(Impl)
  minImpl <- min(Impl) - 0.000001 # so that it is possible to select the run with the minimum implausibility
  rangeImpl <- maxImpl - minImpl
  RunsPerInterval <- n / Intervals
  if (! RunsPerInterval == as.integer(RunsPerInterval)){
    stop("Sample size not divisible by number of intervals")
  }
  R <- RepeatSamples
  NewDesign <- array(0, dim = c(R, n, dim(DesignNROY)[2]))
  ChosenPoints <- matrix(0, nrow = R, ncol = n)
  MinDist <- numeric(R)
  for (j in 1:R){
    for (i in 1:Intervals){
      IntervalIndex <- which(Impl > minImpl + (rangeImpl/Intervals)*(i-1) & Impl <= minImpl + (rangeImpl/Intervals)*i)
      if (length(IntervalIndex) < RunsPerInterval){
        stop(paste("Not enough runs in interval ", i, sep = ""))
      }
      SampleIndex <- sample(IntervalIndex, RunsPerInterval)
      NewDesign[j, (RunsPerInterval*(i-1) + 1):(RunsPerInterval*i), ] <- as.matrix(DesignNROY[SampleIndex,])
      ChosenPoints[j, (RunsPerInterval*(i-1) + 1):(RunsPerInterval*i)] <- SampleIndex
    }
    MinDist[j] <- min(dist(NewDesign[j,,])) # minimum distance between 2 points in design j
  }
  WhichMax <- which.max(MinDist)
  return(list(NewDesign = NewDesign[WhichMax,,], ChosenPoints = ChosenPoints[WhichMax,]))
}

