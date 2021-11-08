library(raster)
library(magick)
library(data.table)
library(bit64)
library(ecp)
#library(TSclust)
#library(linkcomm)
#library(RecordLinkage)
library(fastcluster)
library(stringdist)

#sensitivity testing BEFORE taking SAX approach? 
#Want to avoid pushing spurious results through whole framework

#Choose your microaction window size--suggest one
winSize <- 1


#remember to setwd() and set fps and max size of chunks
setwd('[dir]')
fps <- 29.97
x_traject <- readRDS('x_traject.rds')

#Chosen for semi-nonparametric results...
kval <- round(dim(x_traject)[1]/(fps^2), 0)

fpaths <- unique(substr(x_traject[,file],3,15))

z <- unlist(lapply(fpaths, function(fp){
#Now, we use a variation on the SAX method
	seg <- readRDS(paste0(
		gsub(
			'/', 
			'_', 
			fp, 
		fixed = T), 
		'_seg.rds'
	))
	

	segfnl <- unlist(lapply(seg, function(s) {return(s[1:(length(s)-1)])}))
	
	y <- as.data.table(readRDS(paste0(
		gsub(
			'/', 
			'_', 
			fp, 
		fixed = T), 
		'_chunk.rds'
	)))

	y[,file_key := x_traject[substr(file, 3, 15) == fp][,file]]
	
	zi <- lapply((1+winSize):(length(segfnl)-(1+winSize)), function(i) {
		st1 <- segfnl[i-winSize]+1
		st2 <- segfnl[(i+1+winSize)]-1

		return(y[st1:st2,])
	})
		
	return(zi)
	
	}), recursive = F)

bump <- 0

segfnl <-unlist(lapply(fpaths, function(fp){
#Now, we use a variation on the SAX method
	seg <- readRDS(paste0(
		gsub(
			'/', 
			'_', 
			fp, 
		fixed = T), 
		'_seg.rds'
	))
	

	segfnl <- unlist(lapply(seg, function(s) {return(s[1:(length(s)-1)] + bump)}))
	
	bump <<- segfnl[[length(segfnl)]]
	
	return(segfnl)
	
}))


#normalization and PAA functions from https://jmotif.github.io/sax-vsm_site/morea/algorithm/SAX.html
znorm <- function(ts){
  ts.mean <- mean(ts)
  ts.dev <- sd(ts)
  return((ts - ts.mean)/ts.dev)
}

paa <- function(ts, paa_size){
  len = length(ts)
  if (len == paa_size) {
    ts
  }
  else {
    if (len %% paa_size == 0) {
      colMeans(matrix(ts, nrow=len %/% paa_size, byrow=F))
    }
    else {
      res = rep.int(0, paa_size)
      for (i in c(0:(len * paa_size - 1))) {
        idx = i %/% len + 1# the spot
        pos = i %/% paa_size + 1 # the col spot
        res[idx] = res[idx] + ts[pos]
      }
      for (i in c(1:paa_size)) {
        res[i] = res[i] / len
      }
      res
    }
  }
}

y_indexed <- rbindlist(lapply(fpaths, function(fp){
	yi <- as.data.table(readRDS(paste0(
		gsub(
			'/', 
			'_', 
			fp, 
		fixed = T), 
		'_chunk.rds'
	)))

	yi[, file_key := x_traject[,file_key := x_traject[substr(file, 3, 15) == fp][,file]]]
	return()
}), use.names = T)

y <- as.matrix(rbindlist(lapply(fpaths, function(fp){
	yi <- readRDS(paste0(
		gsub(
			'/', 
			'_', 
			fp, 
		fixed = T), 
		'_chunk.rds'
	))
	return(as.data.table(yi))
}), use.names = T))



zN <- lapply((1+winSize):(length(segfnl)-(1+winSize)), function(i) {
	st1 <- segfnl[i-winSize]+1
	st2 <- segfnl[(i+1+winSize)]-1
	v <- sapply(1:dim(y)[2], function(j){
			znt <- znorm(y[st1:st2,j])
			#Actually, just skip this since we are looking at spatially-normalized angles
			#znt <- y[st1:st2,j]
			if(all(is.nan(znt))){
				znt <- rep(0, length(znt))
			}
			return(znt)
	})
	return(v)
})

z_indexed <- lapply((1+winSize):(length(segfnl)-(1+winSize)), function(i) {
	st1 <- segfnl[i]
	st2 <- segfnl[(i+1)]-1
	return(x_traject[st1:st2,.(file)])
})





##### RECONSIDER PAA
#Get mean framecount of WINDOWED seg
mf <- Reduce("+", lapply((1+winSize):(length(segfnl)-(1+winSize)), function(i) {
	st1 <- segfnl[i-winSize]+1
	st2 <- segfnl[(i+1+winSize)]-1
	return(st2 - st1)
}))/length(segfnl)


paa_size <- round(mf/(1+2*winSize))


zPAA <- lapply(zN, function(m) {
	if(!all(is.na(m))){
		o <- sapply(1:(dim(m)[2]), function(j){
			n <- m[,j]
			return(paa(n, paa_size))
		})
		return(o)
	} else {
		return(NA)
	}
})

missCheck <- unlist(lapply(zPAA, function(p){return(any(is.na(p)))}))

ltrsUL <- c(letters, LETTERS)

wordConv <- function(znt){
	if(is.null(dim(znt)[2])){return('0')}
	wordz <- sapply(1:dim(znt)[2], function(j){
		#quant <- as.numeric(quantile(znt[,j]))
		quant <- as.numeric(quantile(znt[,j],  prob = seq(0, 1, length = 11)))
		names(quant) <- ltrsUL[1:length(quant)]
		#maxlet <- ltrsUL[(j-1)*length(quant) + 1]
		#print(list(z = znt[,j], q = quant))
		word <- lapply(znt[,j], function(ltr){
			if(is.na(ltr)) {
				return('-1')
			}

			if(ltr <= quant[names(quant)[1]]) {
				return(names(quant)[1])
			}

			for(k in 1:(length(quant)-1)){
				if(ltr <= quant[names(quant)[(k+1)]] & ltr > quant[names(quant)[k]]){
					return(names(quant)[(k+1)])
				}
			}

			if(ltr > quant[names(quant)[length(quant)]]){
				return(ltrsUL[length(quant)+1])
			}
		})
		#print(word)
		return(paste(word, collapse = ''))
		
	})
	return(wordz)
	#return(paste(wordz, collapse = ';'))
}


wordsN <- sapply(1:length(zPAA), function(i){
	zt <- zPAA[[i]]
	wordConv(zt)
})

sentences <- sapply(1:dim(wordsN)[2], function(j){return(paste(wordsN[,j], collapse = ' '))})

saveRDS(sentences, 'sentences.rds')
#sentences <- readRDS('sentences.rds')
#dists <- adist(sentences)

dists <- stringdistmatrix(sentences, method = 'lv')
#rownames(dists) <- sentences


saveRDS(dists, 'levDistances.rds')
#dists <- readRDS('levDistances.rds')
dst <- data.matrix(dists)

dim <- ncol(dst)
#img <- image(1:dim, 1:dim, dst, axes = FALSE, xlab="", ylab="", useRaster = T)
#image_write(img,   'dist_matrix.png', format = 'png')

img <- raster(dst)
writeRaster(img, 'dist_matrix.tiff')


hc <- fastcluster::hclust(as.dist(dists))

saveRDS(hc, 'distCluster.rds')
#hc <- readRDS('distCluster.rds')

plot(hc)
rect.hclust(hc,k=kval)

#dists <- readRDS('levDistances.rds')
#sentences <- rownames(dists)
#hc <- readRDS('distCluster.rds')


df <- data.frame(sentences,cutree(hc,k=kval))

names(df) <- c('sentence', 'gesture')

taggedFrames <- rbindlist(
	lapply(1:length(z_indexed), function(i){
		zi <- z_indexed[[i]]
		zi[, gesture := paste0('G', df$gesture[i])]
		return(zi)
	}), 
	use.names = T)

	taggedFrames[,file:=gsub('/pose', '_pose', file, fixed  = T)]

write.table(taggedFrames, 'taggedFrames.csv', row.names = F, sep = ',')


	#OLD CODE
#splitwords <- list()
#
#
#silent <- lapply(1:dim(wordsN)[1], function(i){
#	splitwords[[i]] <<- unlist(lapply(
#		1:dim(wordsN)[2],
#		function(j){
#			return(wordsN[i,j])
#		}
#	))
#	return('')
#})
#
#
##ought to have some means of handing anatomical symmetry here...
#levScorer <- function(word, colNum){
#	return(levenshteinSim(word, splitwords[[colNum]]))
#}
#
#
#lscoreN <- lapply(wordsN, function(wordvec){ 
#	lev <- lapply(1:length(wordvec), function(i){
#		word <- wordvec[i]
#		return(levScorer(word, i))
#	})
#})
#
#saveRDS(lscoreN, 'levenshteinSimilarities.rds')

#zeroNA <- function(x) {
#	x[is.na(x)] <- 0
#	return(x)
#}
#
#add <- function(x) Reduce("+", x)
#
#lsz <- (1:length(lscoreN))
#
##Get best Levenshtein edit similarities for each var?
## Get levenshtein similarities over some arbitrary value?
#simM <- do.call(rbind, lapply(lsz, function(i){	
#	simVec <- add(lapply(lscoreN[[i]], zeroNA))
#	return(simVec)
#}))
#
##na
#minDist <- rbindlist(lapply(lsz, function(i){	
#	simVec <- add(lapply(lscoreN[[i]], zeroNA))
#	#return zero for all segments with only one observation
#	try({
#		if(is.null(dim(zN[[i]]))) {
#			return(data.table(idx = 0, best = 0))
#		}
#	
#		if(dim(zN[[i]])[1] == 1) {
#			return(data.table(idx = 0, best = 0))
#		}
#		#max sum
#	 	nn <- (1:length(lscoreN))[
#				simVec == max(simVec[1:length(lscoreN) != i])
#			]
#		#	return(guys)
#		#}))
#		if(length(nn) >= 1){
#			b <- sort(unique(c(nn)))
#			a <- rep(i, length(b))		
#			return(data.table(idx = a, best = b))
#		
#		} else {
#			return(data.table(idx = 0, best = 0))
#		}
#	})
#	
#	return(data.table(idx = 0, best = 0))
#	
#}), use.names = T)[idx != 0]
#
#justOne <- minDist[,.(.N), by = best][N == 1]
#gtOne <- minDist[,.(.N), by = best][N > 1]
#
#
#
# lc <- getLinkCommunities(as.matrix(minDist), hcmethod = "complete")
#
#
#	# plot(lc, type = "graph")
#	nnClust <- unique(lapply(
#		unique(minDist$best), 
#		function(b){
#			return(
#				sort(
#					unique(
#						c(
#							b, 
#							minDist[best == b, idx]
#						)
#					)
#				)
#			)
#		}
#	))
#	
#
##now we label our start-stop
#sfDT <- rbindlist(lapply(1:(length(segfnl)-1), function(i) {
#	st1 <- segfnl[i]
#	st2 <- segfnl[(i+1)]
#	return(data.table(start = st1, stop = st2))
#	}), 
#	use.names = T
#)
#
#
#shh <- lapply(1:length(lc$clusters), function(cl){
##	shh <- lapply(1:length(nnClust), function(cl){
#	clNo <- cl
#	thisc <- lc$clusters[[cl]]
#	#thisc <- nnClust[[cl]]
#	sfDT[thisc, cluster := paste0('G', clNo)]
#	return('')
#})
#
#
#taggedFrames <- rbindlist(lapply(
#	1:(dim(sfDT)[1]), 
#	function(i){
#		st1 <- sfDT[i, start] 
#		st2 <- sfDT[i, stop]
#		ges <- sfDT[i, cluster]
#		
#		a <- as.data.table(x_traject[st1:st2,.(file)])
#		a[, gesture := ges]
#		return(a)
#	}
#), use.names = T)
#
#taggedFrames[,file := gsub('/pose', '_pose', file, fixed = T)]
#write.table(taggedFrames, 'taggedFrames.csv', row.names = F, sep = ',')
#
##write.table(taggedFrames, paste0(
##	gsub(
##		'/', 
##		'_', 
##		fp, 
##	fixed = T), 
##	'_tagged_frames.csv'
##), row.names = F, sep = ',')

