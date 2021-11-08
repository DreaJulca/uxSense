libchk <- try(library(data.table))
if(class(libchk) == 'try-error'){
  install.packages('data.table', repos = 'https://cloud.r-project.org/')
  library(data.table)
}


#!/usr/bin/env Rscript
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  #however... we may want to use stdin at some point.
    infile <- args[1]
    #infile should be a folder with the text files
    process(infile)
}


process <- function(infile) {
  pubdir <- gsub('assets/', 'public/skelframes/', infile, fixed=T)
  actfiles <- paste0(infile, '/actions/')
  actfiles <- gsub('//', '/', actfiles, fixed = T)
  allfiles <- paste0(actfiles, '/', list.files(actfiles))
  allfiles <- gsub('//', '/', allfiles, fixed = T)

  bestactions <- rbindlist(lapply(allfiles, function(file){
	#print(paste0("Reading ", file))
    x <- fread(file, skip = 2, col.names = c('prob', 'logit', 'action'))
    spltfile <- strsplit(gsub(infile, '', file, fixed = T), split = '_')
    startframe <- as.numeric(spltfile[[1]][2])
    endframe <- as.numeric(gsub('.txt', '', spltfile[[1]][4], fixed = T))
    
    x[,start := startframe]
    x[,end := endframe]

    return(x[,.(maxprob=max(prob), prob, logit, action, start, end),by=list(start,end)][prob==maxprob])
  }), use.names = T)[order(start)][,.(prob, logit, action, start, end)]
  #bestactions[,maxprob:=NULL]
  write.table(bestactions, file = paste0(pubdir, '/actions_best.csv'), sep = ',', row.names = F)
   
  allactions <- rbindlist(lapply(allfiles, function(file){
    x <- fread(file, skip = 2, col.names = c('prob', 'logit', 'action'))
    spltfile <- strsplit(gsub(infile, '', file, fixed = T), split = '_')
    startframe <- as.numeric(spltfile[[1]][2])
    endframe <- as.numeric(gsub('.txt', '', spltfile[[1]][4], fixed = T))
    
    x[,start := startframe]
    x[,end := endframe]
    
    return(x)
  }), use.names = T)[order(start)]
 
  write.table(allactions, file = paste0(pubdir, '/actions_all.csv'), sep = ',', row.names = F)
  
}

main()