library(parallel)
libchk <- try(library(jsonlite))
if(class(libchk) == 'try-error'){
	install.packages('jsonlite', repos = 'https://cloud.r-project.org/')
	library(jsonlite)
}

libchk <- try(library(data.table))
if(class(libchk) == 'try-error'){
	install.packages('data.table', repos = 'https://cloud.r-project.org/')
	library(data.table)
}

libchk <- try(library(bit64))
if(class(libchk) == 'try-error'){
	install.packages('bit64', repos = 'https://cloud.r-project.org/')
	library(bit64)
}

libchk <- try(library(ecp))
if(class(libchk) == 'try-error'){
	install.packages('ecp', repos = 'https://cloud.r-project.org/')
	library(ecp)
}


#!/usr/bin/env Rscript
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
	# test if there is at least one argument: if not, return an error
	if (length(args)!=3) {
	  stop("Exactly three arguments must be supplied. Syntax: (> Rscript models/segment.R input_file.csv output_file.json skeleton_joint_labels.json).n", call.=FALSE)
	}
	
	#however... we may want to use stdin at some point.
  if (length(args) == 1) {
    process(file("stdin"), args[1])
  } else {
	  infilerel <- args[1]
	  outfile <- args[2]
	  labelfile <- args[3]
    process(infilerel , outfile, labelfile)
  }
}


process <- function(infilerel, outfile, labelfile) {
  infile <- paste0(getwd(), '/', infilerel)
	#chunkSize is the number of frames to consider at any given time. 
	# We will go up to about 30 seconds worth of frames, based on avg vid fps (29.97 fps)
	chunkSize <- 500
	x <- fread(infile)
	x[,frame:=.I]
	
	#subset for model; suppose we just want hands, normalized to head, and the trajectory of the head
	#WARNING: I call it head lag, but it is actually neck (closest to core position)
	headLag <- data.table(
		frame = x[2:(dim(x)[1]), frame], 
		head_lag_x = x[1:(dim(x)[1] - 1), x_1], 
		head_lag_y = x[1:(dim(x)[1] - 1), y_1], 
		head_lag_z = x[1:(dim(x)[1] - 1), z_1]
	)
	
	setkey(headLag, key = frame)
	setkey(x, key = frame)
	
	x_traject <- headLag[x]
	
	getMinVals <- function(varnames){
		minVal <- min(
			as.numeric(
				as.matrix(
					x_traject[,lapply(.SD, 'min'), .SDcols = varnames]
				)
			), 
			na.rm = T
		)
	
		return(minVal)
	}
	
	
	alln <- names(x_traject)
	xn <- alln[grepl('x_', alln, fixed = T) ]
	yn <- alln[grepl('y_', alln, fixed = T) ]
	zn <- alln[grepl('z_', alln, fixed = T) ]
	
	minXVal <- getMinVals(xn)
	minYVal <- getMinVals(yn)
	minZVal <- getMinVals(zn)
	
	# Normalize 3D axes to start at 0
	x_traject[, (xn) := lapply(.SD, '-', minXVal), .SDcols = xn]
	x_traject[, (yn) := lapply(.SD, '-', minYVal), .SDcols = yn]
	x_traject[, (zn) := lapply(.SD, '-', minZVal), .SDcols = zn]
	
	
	getAngleABC <- function(A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z){
		#In pseudo-code, the vector BA (call it v1) is:
		#v1 = {A.x - B.x, A.y - B.y, A.z - B.z}
		v1 <- data.table(x = A.x - B.x, y = A.y - B.y, z = A.z - B.z)
	
		#Similarly the vector BC (call it v2) is:
		#v2 = {C.x - B.x, C.y - B.y, C.z - B.z}
		v2 <- data.table(x = C.x - B.x, y = C.y - B.y, z = C.z - B.z)
		
		#The dot product of v1 and v2 is a function of the cosine of the angle between them (it's scaled by the product of their magnitudes). 
		#So first normalize v1 and v2:
		v1mag	<- sqrt(v1$x * v1$x + v1$y * v1$y + v1$z * v1$z)
		v1norm	<- data.table(x = v1$x / v1mag, y = v1$y / v1mag, z = v1$z / v1mag)
	
		v2mag	<- sqrt(v2$x * v2$x + v2$y * v2$y + v2$z * v2$z)
		v2norm	<- data.table(x = v2$x / v2mag, y = v2$y / v2mag, z = v2$z / v2mag)
	
		#Then calculate the dot product:
		res		<- v1norm$x * v2norm$x + v1norm$y * v2norm$y + v1norm$z * v2norm$z
		#And finally, recover the angle:
		angle	<- acos(res)
	
		return(angle)
	}
	
	
	#Given indices:
	body <- fromJSON(readLines(labelfile))
	
	#Get the following angles:
	# Head, neck, pelvis [Center]
	# Head, neck, shoulder [(L)eft, (R)ight]
	# Neck, shoulder, elbow [(L)eft, (R)ight]
	# Shoulder, elbow, wrist [L, R]
	# Pelvis, hip, knee [L, R]
	# Hip, knee, ankle [L, R]
	angles <- fromJSON('[
		{"A":"left_ear",			"B":"neck",				"C":"left_shoulder"	},
		{"A":"right_ear",			"B":"neck",				"C":"left_shoulder"	},
		{"A":"left_ear",			"B":"neck",				"C":"right_shoulder"},
		{"A":"right_ear",			"B":"neck",				"C":"right_shoulder"},
		{"A":"neck",			"B":"left_shoulder",	"C":"left_elbow"	},
		{"A":"neck",			"B":"right_shoulder",	"C":"right_elbow"	},
		{"A":"left_shoulder",	"B":"left_elbow",		"C":"left_wrist"	},
		{"A":"right_shoulder",	"B":"right_elbow",		"C":"right_wrist"	},
		{"A":"neck",			"B":"left_hip",			"C":"left_knee"		},
		{"A":"neck",			"B":"right_hip",		"C":"right_knee"	},
		{"A":"left_hip",		"B":"left_knee",		"C":"left_ankle"	},
		{"A":"right_hip",		"B":"right_knee",		"C":"right_ankle"	}
	]
	')
	
	coordString <- c('x_', 'y_', 'z_')
	
	for(i in 1:(dim(angles)[1])){
		iA <- body[angles[i, "A"]]
		iB <- body[angles[i, "B"]]
		iC <- body[angles[i, "C"]]
		
	
		cA <- paste0(coordString, iA)
		cB <- paste0(coordString, iB)
		cC <- paste0(coordString, iC)
	
		x_traject[,(paste0('angle_', i)) := (
			eval(
				parse(
					text = paste0(
						'getAngleABC(', 
						paste(
							c(cA, cB, cC), 
							collapse = ','
						), 
						')'
					)
				)
			)
		)]
	
	}
	
	
	
	#Use methods from https://cran.r-project.org/web/packages/ecp/vignettes/ecp.pdf
	#get pearson covariance	matrix
	#xCov <- cov(y, method = 'pearson')
	#e.divisive(xCov, R = 499, alpha = 1)$estimates
	
	#Okay, my machine cannot handle more than like 1000 rows at a time of this;
	#that is fine, because we have a limit to our segment length anyway---
	#we do not yet want gestures longer than a few seconds
	
	#so what we will want to do is start each chunk from the beginning of the preceding

	getCols <- 	names(x_traject)[
			grepl(
				'angle_', 
				names(x_traject), 
				fixed = T
			)
		]
			
	for(ang in getCols){
		firstTheta <- eval(parse(text=paste0('x_traject[frame == 1, ', ang, ']')))
		
		if(is.nan(firstTheta)){
			firstTheta <- eval(parse(text=paste0('x_traject[!is.nan(', ang, ')][1,',ang,']')))
			eval(parse(text=paste0('x_traject[frame == 1, ', ang, ' := ', firstTheta, ']')))
		}
		
		for(i in 2:(dim(x_traject)[1])){
			thisTheta <- eval(parse(text=paste0('x_traject[frame == ', i, ',', ang, ']')))
			#print(thisTheta)
			if(is.nan(thisTheta)){
				#print('replacing')
				eval(parse(text=paste0('x_traject[frame == ', i, ',', ang, ' := x_traject[frame == ', i-1, ',', ang, ']]')))
			}
		}
		
		eval(parse(text=paste0('x_traject[is.nan(', ang, '), ', ang, ' := 0]')))
	}
	
	y <- as.matrix(x_traject[,
		getCols,
		with = F
	])
	
	print(head(y))
	print(tail(y))
	
	# This could work, if e.divisive weren't so inefficient
	#seg <- e.divisive(y, R = 499, alpha = 1)#$estimates
	
	#The lines below are only useful for many long videos; not needed for single vid
	seg <- list()
	nextStart <- 1;
	thresh <- 1
	for(k in 1:(dim(y)[1]/chunkSize)){
		print(paste('This start:', nextStart))
		#best segment estimates
		seg[[k]] <- (nextStart - 1) + e.divisive(
			y[nextStart:(chunkSize * k), ],
			R = 499, 
			alpha = 1
		)$estimates
	
		#set next starting row; should be second-to-last
		#print(toJSON(seg))
		if(length(seg[[k]]) == 2){
			nextStart <- seg[[k]][2]
		} else {
			nextStart <- seg[[k]][(length(seg[[k]])-1)]
		}
		kCheck <- (k - thresh) * chunkSize
		#print(seg[[k]])
		#print(paste('Next start check:', nextStart, '| (k - thresh) * chunkSize:', kCheck))
	}

	if (dim(y)[1] %% chunkSize > 0){
		print(paste('This start:', nextStart))

		seg[[1+(dim(y)[1]/chunkSize)]] <- (nextStart - 1) + e.divisive(
			y[nextStart:dim(y)[1], ],
			R = 499, 
			alpha = 1
		)$estimates
	}
	
	segCleanedUp <- list()

	for(i in 1:(length(seg)-1)){
		segCleanedUp[[i]] <- list()
		for(j in 1:(length(seg[[i]])-2)){
			segCleanedUp[[i]][[j]] <- seg[[i]][j]
		}
	}

	segCleanedUp[[length(seg)]] <- list()

	for(i in 1:length(seg[[length(seg)]])){
		segCleanedUp[[length(seg)]][[i]] <- seg[[length(seg)]][i]
	}

	segfnl <- unlist(segCleanedUp)

	writeLines(toJSON(segfnl), outfile)

	cmdstr <- paste("Rscript models/postseg_action_pred_only.R", infilerel, outfile, labelfile)

	system(cmdstr)

	#vidpath <- gsub('/all_pose_estimates.csv', '.mp4', infile, fixed = T) 
	#shortvidpath <- gsub(tolower(getwd()), '', tolower(vidpath), fixed = T) 
	#
	##actionBatchFile <- paste0(getwd(), "/assets/action_ienn_batch.sh")
	##file.create(actionBatchFile)
#
	#for(i in 2:length(segfnl)){
	#	cmdstr <- paste0("python models/kinetics-i3d/evaluate.py --vid ", shortvidpath, " --startframe ", segfnl[(i-1)], " --endframe ", segfnl[i])
	#	#write(cmdstr, file=actionBatchFile, append=TRUE)
	#	system(cmdstr)
	#}
#
	##system("./assets/action_ienn_batch.sh")

}

main()

