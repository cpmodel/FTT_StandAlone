get.predict.mats <- function(object){
  #tt <- terms(object)
  if(!inherits(object, "lm"))
    warning("Need lm object")
  #Terms <- delete.response(tt)
  n <- length(object$residuals)
  p <- object$rank
  p1 <- seq_len(p)
  piv <- qr(object)$pivot[p1]
  res.var <- {
    r <- object$residuals
    w <- object$weights
    rss <- sum(if(is.null(w)) r^2 else r^2 * w)
    df <- object$df.residual
    rss/df
  }
  XRinvRHS <- qr.solve(qr.R(qr(object))[p1, p1])
  return(list(piv=piv,res.var=res.var,p=p,XRinvRHS=XRinvRHS,n=n))
}

custom.predict <- function(object, newdata, required.list){
  tt <- terms(object)
  Terms <- delete.response(tt)
  m <- model.frame(Terms, newdata, na.action=na.pass, xlev=object$xlevels)
  X <- model.matrix(Terms, m, contrasts.arg=object$contrasts)
  mmDone <- FALSE
  beta <- object$coefficients
  predictor <- drop(X[, required.list$piv, drop=FALSE] %*% beta[required.list$piv])
  XRinv <- X[, required.list$piv]%*%required.list$XRinvRHS
  ip <- drop(XRinv^2 %*% rep(required.list$res.var, required.list$p))
  predictor <- cbind(predictor, sqrt(ip+required.list$res.var))
  colnames(predictor) <- c("fit", "sd")
  predictor
}