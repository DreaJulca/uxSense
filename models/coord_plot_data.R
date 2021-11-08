libchk <- try(library(data.table))
if(class(libchk) == 'try-error'){
  install.packages('data.table', repos = 'https://cloud.r-project.org/')
  library(data.table)
}

libchk <- try(library(jsonlite))
if(class(libchk) == 'try-error'){
  install.packages('jsonlite', repos = 'https://cloud.r-project.org/')
  library(jsonlite)
}



#!/usr/bin/env Rscript
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  process(args[1])
}


process <- function(vidpath) {
  options("scipen"=-100, digits=15)
  shortvidpath <- gsub(tolower(getwd()), '', tolower(vidpath), fixed = T) 
  posefile <- gsub('//', '/', paste0(vidpath, '/all_pose_estimates.csv'), fixed = T)
  pose <- try(fread(posefile))
  if(class(pose)[1] == 'try-error'){
    print(pose)
    print('retrying...')
    process(vidpath)
  } else {
    xrngs <- lapply(0:16, function(i){
      rng <- eval(parse(text=paste0('pose[,range(x_', i, ')]')))
      breaks <- seq(rng[1], rng[2], by=abs(rng[2]-rng[1])/20)
      thisCut <- as.data.table(eval(parse(text=paste0('table(cut(pose[, x_', i, '] , breaks, right=FALSE))'))))
      setnames(thisCut, old = c('V1', 'N'), new = c('Range', 'Freq'))
      
      #Get mins and maxes
      spltrngs <- strsplit(thisCut[,Range], split = ',')
      rmins <- unlist(lapply(spltrngs, function(strng){return(as.numeric(gsub(')', '', gsub('[', '', strng[1], fixed = T), fixed = T)))}))
      rmaxs <- unlist(lapply(spltrngs, function(strng){return(as.numeric(gsub(')', '', gsub('[', '', strng[2], fixed = T), fixed = T)))}))
      thisCut[,Min := rmins]
      thisCut[,Max := rmaxs]
      
      return(thisCut)
    })
    
    names(xrngs) <- paste0('x_', 0:16)
    
    yrngs <- lapply(0:16, function(i){
      rng <- eval(parse(text=paste0('pose[,range(y_', i, ')]')))
      breaks <- seq(rng[1], rng[2], by=abs(rng[2]-rng[1])/20)
      thisCut <- as.data.table(eval(parse(text=paste0('table(cut(pose[, y_', i, '] , breaks, right=FALSE))'))))
      setnames(thisCut, old = c('V1', 'N'), new = c('Range', 'Freq'))

      #Get mins and maxes
      spltrngs <- strsplit(thisCut[,Range], split = ',')
      rmins <- unlist(lapply(spltrngs, function(strng){return(as.numeric(gsub(')', '', gsub('[', '', strng[1], fixed = T), fixed = T)))}))
      rmaxs <- unlist(lapply(spltrngs, function(strng){return(as.numeric(gsub(')', '', gsub('[', '', strng[2], fixed = T), fixed = T)))}))
      thisCut[,Min := rmins]
      thisCut[,Max := rmaxs]
      
      return(thisCut)
    })
    
    names(yrngs) <- paste0('y_', 0:16)
    
    zrngs <- lapply(0:16, function(i){
      rng <- eval(parse(text=paste0('pose[,range(z_', i, ')]')))
      breaks <- seq(rng[1], rng[2], by=abs(rng[2]-rng[1])/20)
      thisCut <- as.data.table(eval(parse(text=paste0('table(cut(pose[, z_', i, '] , breaks, right=FALSE))'))))
      setnames(thisCut, old = c('V1', 'N'), new = c('Range', 'Freq'))

      #Get mins and maxes
      spltrngs <- strsplit(thisCut[,Range], split = ',')
      rmins <- unlist(lapply(spltrngs, function(strng){return(as.numeric(gsub(')', '', gsub('[', '', strng[1], fixed = T), fixed = T)))}))
      rmaxs <- unlist(lapply(spltrngs, function(strng){return(as.numeric(gsub(')', '', gsub('[', '', strng[2], fixed = T), fixed = T)))}))
      thisCut[,Min := rmins]
      thisCut[,Max := rmaxs]
      
      return(thisCut)
    })
    
    names(zrngs) <- paste0('z_', 0:16)
    
    pose[,frame:=.I]
    
    newposefile <- gsub('assets/', 'public/skelframes/', posefile, fixed = T)
    options("scipen"=-100, digits=15)
    write.table(pose, newposefile, sep = ',', row.names = F)
    options("scipen"=-100, digits=15)
    writeLines(toJSON(c(xrngs, yrngs, zrngs), digits=15), gsub('all_pose_estimates.csv', 'pose_histogram_data.json', newposefile, fixed = T))

  }
  
}

main()