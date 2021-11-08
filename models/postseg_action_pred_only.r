libchk <- try(library(jsonlite))
if(class(libchk) == 'try-error'){
  install.packages('jsonlite', repos = 'https://cloud.r-project.org/')
  library(jsonlite)
}


#!/usr/bin/env Rscript
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
	#however... we may want to use stdin at some point.
  if (length(args) == 1) {
    process(file("stdin"), args[1])
  } else {
	  infile <- args[1]
	  outfile <- args[2]
	  labelfile <- args[3]
    process(infile , outfile, labelfile)
  }
}


process <- function(infile, outfile, labelfile) {
	segfnl <- fromJSON(readLines(outfile))

	vidpath <- gsub('/all_pose_estimates.csv', '.mp4', infile, fixed = T) 
	shortvidpath <- gsub(
	  gsub(
	    '/', 
	    '\\', 
	    getwd(), 
	    fixed = T
	   ), 
	  '',
	  gsub(
	    tolower(getwd()), 
	    '', 
	    tolower(vidpath), 
	    fixed = T
	   ), fixed = T
	 )
	
	#actionBatchFile <- paste0(getwd(), "/assets/action_ienn_batch.sh")
	#file.create(actionBatchFile)

  #For now, we always want to have variable window size; can use fixed size if suspect an error.
	fixedFrameWidth <- FALSE
	lagSize <- 2
	leadSize <- 2
	for(i in (1+lagSize):(length(segfnl)-leadSize)){
	  doRun <- TRUE;
	  if(fixedFrameWidth){
	    if(segfnl[i] > 125){
	      startSeg <- segfnl[i] - 125
	      endSeg <- segfnl[i] + 124
	    } else {
	      startSeg <- 1
	      endSeg <- 250
	      if(i > 1){
	        doRun <- FALSE;
	      }
	    }
	    
	    if(segfnl[i] < segfnl[length(segfnl)] - 125){
	      startSeg <- segfnl[i] - 125
	      endSeg <- segfnl[i] + 124
	    } else {
	      startSeg <- segfnl[length(segfnl)] - 249
	      endSeg <- segfnl[length(segfnl)]
	      if(i < length(segfnl)){
	        doRun <- FALSE;
	      }
	    }
	  } else {
	    startSeg <- segfnl[(i-lagSize)]
	    endSeg <- segfnl[(i+leadSize)]
	  }
	  
		cmdstr <- paste0("python models/kinetics-i3d/evaluate.py --vid ", vidpath, " --startframe ", startSeg, " --endframe ", endSeg)
		#write(cmdstr, file=actionBatchFile, append=TRUE)
        print('****************************************************************')
        print(cmdstr)
    if(doRun){
      system(cmdstr)
    }
		#file.rename(gsub('.mp4', paste0('/actions/frames_', (i-2))))
	}

	cmdstrfnl <- paste0('Rscript models/coord_plot_data.R ', gsub('.mp4', '', vidpath, fixed=T))
	system(cmdstrfnl)
	cmdstrfnl <- paste0('Rscript models/tabulate_actions.R ', gsub('.mp4', '', vidpath, fixed=T))
	system(cmdstrfnl)
}

main()