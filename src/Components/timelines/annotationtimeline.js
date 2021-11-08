var uxvideo = document.getElementById("video_html5_api");

d3.select('body').append('div')
.attr('id', 'annotimetooltip')
.style('opacity', 0)

var annosvg = d3.select("#AnnotationTimeline")
.append("svg")
.attr("width", "100%")

function createAnnotationsTimeline(loopagain=false){
    refreshuxSDimVars();
    // we want to aggregate our count to some level of timespan for plotting, and then we want the user to be able to drill down into it
    annosvg
        .attr("height", (height + margin.top + margin.bottom))
        .attr("viewBox", "0 0 " + (width + margin.left + margin.right) + " " + (height + margin.top + margin.bottom))
        .attr("preserveAspectRatio", "none")
        .style('display', 'inline-block')
        .style('position', 'relative')

//    .attr("width", width + margin.left + margin.right)
//    .attr("height", height + margin.top + margin.bottom)
    var annotationsColors = {"Action1":'#ffa31a', "Emotion":'#e3f43a', "pitch":'steelblue', "vidtimelineholder": '#009900', "speech-rate":'#ff3300', "AnnotationTimeline":'purple'}
    var annotationsTimelineLabs = {"Action1":'Actions', "Emotion":'Emotions', "pitch":'Pitch', "vidtimelineholder": 'Thumbnails', "speech-rate":'Speech Rate', "AnnotationTimeline":'Annotations'}

    d3.json('userAnnotations/data.json', function(rawdata){
        refreshuxSDimVars();
        if(!rawdata){
            console.log("annotations not loaded")
            return(null)
        }
        //wipe the slate first
        try{
            annosvg.selectAll('g:not(.add-annotation-button)').remove();
        } catch(err){
            console.log(err)
        }

        annograph = annosvg.append("g")
            .attr('id', 'annotationsgraphicmain')
            .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

        if(rawdata.length>0){
            var filtannotations = _.filter(rawdata, {'videoname':uxSenseVideoPath})
            var data=_.sortBy(filtannotations, function(o){
                if(o.annotationtype == "interval" & o.focusbrushed=="true"){
                    return parseFloat(o.annotatedintervalmin)
                } else {
                    return parseFloat(o.timestamp)
                }
            })
    
        } else {
            var data=rawdata
        }
         
        //console.log(data)

        annograph.data(data);
        annograph.attr('origdata', JSON.stringify(data));

        //Check to see if there is a selected time interval focus
        var focussvg = d3.select('#focussvg').select('g')
        var focusBrush = focussvg.select('g.x.brush')
        var selrect = focusBrush.select('rect.selection')

        var selwid = parseFloat(selrect.attr('width'))
        var selX = parseFloat(selrect.attr('x'))

        var isFocused=true;
        var minTime = uxvideo.duration * selX/width
        var maxTime = uxvideo.duration * ( selX + selwid )/width

        if(isNaN(minTime)|isNaN(maxTime)){
            selwid = width;
            selX = 1;
            minTime =  0
            maxTime =  uxvideo.duration 
            isFocused=false;
        }


        var crectwidthmin = 7.5

        if(data.length>0){
            var annotHeight = (height/data.length)

           for(i=0; i<data.length; i++){
                if(data[i].annotationtype == "interval" & data[i].focusbrushed=="true"){
                    if((parseFloat(data[i].annotatedintervalmin) <= maxTime) & (parseFloat(data[i].annotatedintervalmax) >= minTime)){

                        var startTime = parseFloat(data[i].annotatedintervalmin)
                        var endTime = parseFloat(data[i].annotatedintervalmax)
                        var timeGap = endTime-startTime;
                        var selRangeRatio = uxvideo.duration/(maxTime-minTime);
                        var wid2dur = width/uxvideo.duration
                        var trueX = ((startTime-minTime)*selRangeRatio*wid2dur) - crectwidthmin/2
                        var returnedX = trueX

                        if(returnedX<0){
                            returnedX=0
                        }

                        var trueWidth = crectwidthmin + timeGap*selRangeRatio*wid2dur;
                        var returnedWidth = trueWidth;

                        if((returnedX + returnedWidth)>width){
                            returnedWidth = width - returnedX  
                            if(returnedX > trueX){
                                returnedWidth = width
                            }
                        }

                        if(startTime < minTime){
                            returnedWidth = crectwidthmin + (timeGap-(minTime-startTime))*selRangeRatio*wid2dur
                            if((returnedX + returnedWidth)>width){
                                returnedWidth = width
                            }
                        }

                        annograph.append('rect')
                        .datum(data[i])
                        .attr('fill', annotationsColors[data[i].timeline])
                        .attr('class', 'annottimelinerect')
                        .attr('id', 'rect_annot_'+data[i]._id)
                        .attr('width', returnedWidth)
                        .attr('height', annotHeight)
                        .attr('x', returnedX)
                        .attr('rx', 2)
                        .attr('ry', 2)
                        .attr('y', i*annotHeight)
                        .on('mouseover', function(d){
                            d3.select('#annotimetooltip')
                            .transition().duration(100)
                            .style("left", (d3.event.pageX) + "px")
                            .style("top", (d3.event.pageY - 28) + "px")   
                            .style('opacity', 1)
                
                            d3.select('#annotimetooltip')
                            .html("Interval note: " + d.annotation + "<br/>Timeline: "+annotationsTimelineLabs[d.timeline])
                
                            //Track event
                            interactiontracking(d, 'annottimelinerects', 'rect_annot_'+d._id, 'mouseover'    )
                    
                        })
                        .on('mouseout', function(d){
                            d3.select('#annotimetooltip')
                            .transition().duration(100)
                            .style('opacity', 0)
                
                            //Track event
                            interactiontracking(d, 'annottimelinerects', 'rect_annot_'+d._id, 'mouseout'    )
                
                        })
                        .on('click', function(d){
                            var fps = maxTime/uxvideo.duration;
                            var uxvidPrevTime = uxvideo.currentTime
                            uxvideo.currentTime = parseFloat(d.annotatedintervalmin)/fps;
                
                            //track event
                            interactiontracking(d, 'annottimelinerects', 'rect_annot_'+d._id, 'click', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])
                        })
                    }

                } else {

                    if((parseFloat(data[i].timestamp) <= maxTime) & (parseFloat(data[i].timestamp) >= minTime)){

                        var startTime = parseFloat(data[i].timestamp)
                        var selRangeRatio = uxvideo.duration/(maxTime-minTime);
                        var wid2dur = width/uxvideo.duration
                        var returnedX = ((startTime-minTime)*selRangeRatio*wid2dur) - crectwidthmin/2

                        if(returnedX<0){
                            returnedX=0
                        }

                        annograph.append('rect')
                        .datum(data[i])
                        .attr('fill', annotationsColors[data[i].timeline])
                        .attr('class', 'annottimelinerects')
                        .attr('id', 'rect_annot_'+data[i]._id)
                        .attr('width', crectwidthmin)
                        .attr('height', annotHeight)
                        .attr('x', returnedX)
                        .attr('rx', 2)
                        .attr('ry', 2)
                        .attr('y', i*annotHeight)
                        .on('mouseover', function(d){
                            d3.select('#annotimetooltip')
                            .transition().duration(100)
                            .style("left", (d3.event.pageX) + "px")
                            .style("top", (d3.event.pageY - 28) + "px")   
                            .style('opacity', 1)
                
                            d3.select('#annotimetooltip')
                            .html("Point note: " + d.annotation + "<br/>Timeline: "+annotationsTimelineLabs[d.timeline])
                
                            //Track event
                            interactiontracking(d, 'annottimelinerects', 'rect_annot_'+d._id, 'mouseover'    )
                    
                        })
                        .on('mouseout', function(d){
                            d3.select('#annotimetooltip')
                            .transition().duration(100)
                            .style('opacity', 0)
                
                            //Track event
                            interactiontracking(d, 'annottimelinerects', 'rect_annot_'+d._id, 'mouseout'    )
                
                        })
                        .on('click', function(d){
                            var fps = maxTime/uxvideo.duration;
                            var uxvidPrevTime = uxvideo.currentTime
                            uxvideo.currentTime = parseFloat(d.timestamp)/fps;
                
                            //track event
                            interactiontracking(d, 'annottimelinerects', 'rect_annot_'+d._id, 'click', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])
                        })
                    }

                }               
           } 
          
           //just handle this directly
        /*
        //and we're going to add rects as background to our filter sliders (and also to block out edges on focus)
        annosvg.append('g').append('rect')
        .attr('fill', 'white')
        .attr('height', height + margin.top + margin.bottom)
        .attr('width', margin.left)
        .attr('x', -margin.left)
        .attr('y', 0)

        annosvg.append('g').append('rect')
        .attr('fill', 'white')
        .attr('height', height + margin.top + margin.bottom)
        .attr('width', margin.right)
        .attr('x', width)
        .attr('y', 0)
        */

        }
    })    

    if(loopagain){
        setTimeout('createAnnotationsTimeline(true)', 3000)
    }
}

uxvideo.addEventListener('loadeddata', function(){
    createAnnotationsTimeline(true)
})
