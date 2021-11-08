import refreshuxSDimVars from "../refreshuxSDimVars";
import * as d3 from 'd3';
//This is where we handle both the focus timeline and the timeline marker that hits all the table svgs
var uxvideo = document.getElementById('video_html5_api');
refreshuxSDimVars();

var focusHeight = 20;
var hmargin = 10;



var focussvg = d3.select('#premierefocus').append('svg')
    .attr('id', 'focussvg')
    .attr('width', "100%")
    .attr('height',focusHeight + hmargin)
    .attr("viewBox", "0 0 " + (width + margin.left + margin.right) + " " + (focusHeight + hmargin))
    .attr("preserveAspectRatio", "none")
    .style('display', 'inline-block')
    .style('position', 'relative')
    .append('g')
    .attr("transform",
        "translate(" + .95*margin.left + ",0)");
    //.attr('transform', 'translate(0,' + hmargin + ')');

var bgslide = focussvg.append('rect')
.attr('id', 'focusrectback')
.attr('fill', "lightgrey")
.attr('width', width)
.attr('height', focusHeight)
.attr('y', hmargin)

var xScaleTop = d3.scaleTime().range([0, width]);
var xScaleBottom = d3.scaleTime().range([0, width]);
var xAxisTop = d3.axisBottom(xScaleTop);

var brush = d3.brushX()
    .extent([[0, 0], [width, focusHeight]])
    .on("brush end", function () {
        var s = d3.event.selection || xScaleBottom.range();
        xScaleTop.domain(s.map(xScaleBottom.invert, xScaleBottom));
        //d3.selectAll('.x.axis').call(xAxisTop);
        var uxvidPrevTime = uxvideo.currentTime;
        uxvideo.currentTime = video.duration * (focussvg.select('rect.selection').attr('x') / width)
        
        var selrect = focussvg.select('rect.selection');

        //focus.select(".line").attr("d", lineTop);
        if(focussvg.select('rect.selection').attr('width') > 0 | selrect.attr('width') == null){
            interactiontracking(xScaleTop.domain(), 'premierefocus', 'rect.selection', 'brush end', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])

            rescaleTimelines();

            //Add panning dragging to all nodes again
            addPanningToSVGs();            

        }
    })

//Leave space for cursor
brushg = focussvg.append('g').attr('class', 'context').attr('transform', 'translate(0,'+ hmargin + ')')

brushg.append('g')
.attr('class', 'x brush')
.call(brush)
.selectAll('rect');

function rescaleTimelines(){
    refreshuxSDimVars();
        rescaleEmotions();
        rescaleActions();
        rescaleSpeechrate();
        rescalePitch();
        rescaleFrames();
        rescaleAnnotations();    
}

function rescaleEmotions(){
    var emotions = d3.select("#emotionrects")
    if(!emotions){
        return("")
    }
    var maxEnd = parseFloat(emotions.attr('maxEnd'))
    var fps = maxEnd/uxvideo.duration

    var selrect = d3.select('#focussvg').select('rect.selection');

    var selwid = parseFloat(selrect.attr('width'))
    var selX = parseFloat(selrect.attr('x'))

    if(isNaN(selwid)){
        selwid = width;
        selX = 1;
    }

    var minTime = uxvideo.duration * selX/width
    var maxTime = uxvideo.duration * ( selX + selwid )/width  

    var widMult = width/selwid

    var newMinFrame = minTime * fps
    var newMaxFrame = maxTime * fps

    var x = d3.scaleLinear()
    .domain([newMinFrame, newMaxFrame])
    .range([0, width]);

    function rectWidth(lowerVal, upperVal){
        gap = upperVal-lowerVal;
        rangeMult = widMult * (width/maxEnd)
        return (gap * rangeMult)
    }

    d3.select('#emoxaxis')
    .call(d3.axisBottom(x)
        .tickFormat(function(d) {
            return d3.timeFormat('%M:%S')( new Date(0).setSeconds(d/fps) )            
        })
    )


    emotions.selectAll('.emotionrect')
    .transition().duration(1)
    .attr('width', function(d){
        return(rectWidth(d.start, d.end))
    })
    .attr('x', function(d){
        return(x(d.start))
    })

}

function rescaleActions(){
    var actions = d3.select("#action1rects")
    if(!actions){
        return("")
    }
    var maxEnd = parseFloat(actions.attr('maxEnd'))
    var fps = maxEnd/uxvideo.duration

    var selrect = d3.select('#focussvg').select('rect.selection');

    var selwid = parseFloat(selrect.attr('width'))
    var selX = parseFloat(selrect.attr('x'))

    if(isNaN(selwid)){
        selwid = width;
        selX = 1;
    }


    var minTime = uxvideo.duration * selX/width
    var maxTime = uxvideo.duration * ( selX + selwid )/width  

    var widMult = width/selwid

    var newMinFrame = minTime * fps
    var newMaxFrame = maxTime * fps

    var x = d3.scaleLinear()
    .domain([newMinFrame, newMaxFrame])
    .range([0, width]);

    function rectWidth(lowerVal, upperVal){
        gap = upperVal-lowerVal;
        rangeMult = widMult * (width/maxEnd)
        return (gap * rangeMult)
    }


    d3.select('#actxaxis')
    .call(d3.axisBottom(x)
        .tickFormat(function(d) {
            return d3.timeFormat('%M:%S')( new Date(0).setSeconds(d/fps) )            
        })
    )



    actions.selectAll('.action1rect')
    .transition().duration(1)
    .attr('width', function(d){
        return(rectWidth(d.start, d.end))
    })
    .attr('x', function(d){
        return(x(d.start))
    })

}

function rescaleSpeechrate(){
    
    var line = d3.select("#speechratelinepath")
    if(!line){
        return("")
    }
    var maxEnd = parseFloat(line.attr("maxEnd"));

    var data = JSON.parse(line.attr("origdata"));

    var fps = maxEnd/video.duration

    var selrect = d3.select('#focussvg').select('rect.selection');

    var selwid = parseFloat(selrect.attr('width'))
    var selX = parseFloat(selrect.attr('x'))

    if(isNaN(selwid)){
        selwid = width;
        selX = 1;
    }


    var minTime = uxvideo.duration * selX/width
    var maxTime = uxvideo.duration * ( selX + selwid )/width  

    var newMinFrame = minTime * fps
    var newMaxFrame = maxTime * fps
    var filtdata = _.filter(data, function(o){return parseFloat(o.Start) >= newMinFrame & parseFloat(o.Start) <= newMaxFrame })

    var fakestart = {};
        fakestart.End = minTime.toString()
        fakestart.Rate = "0.0"
        fakestart.Start = minTime.toString()
        
    var fakeend = {};
        fakeend.End = maxTime.toString()
        fakeend.Rate = "0.0"
        fakeend.Start = maxTime.toString()

    filtdata = [fakestart].concat(filtdata)

    filtdata.push(fakeend)

    line.datum(filtdata)

    var x = d3.scaleLinear()
        .domain([newMinFrame, newMaxFrame])
        .range([0, width]);

    var y = d3.scaleLinear()
        .domain([0,5.5])
        .range([height, 0]);

    line.transition().duration(1).attr("d", function(d){    
        return(d3.line()
        .curve(d3.curveStepAfter)
        .x(function (d) { return x(d.Start) })
        .y(function (d) { return y(d.Rate) })
        )(d)
    })

    /**
     * need to update our mouse activity
     * */
     // Create the circle that travels along the curve of chart
     var focus = d3.select('#speechratefocuscircle')
     // Create bg for the text that travels along the curve of chart
    var focusTextRect = d3.select('#speechratefocustextbg')
     // Create the text that travels along the curve of chart
     var focusText = d3.select('#speechratefocustext')
    
   d3.select("#speechratemouserect")
    .on('mousemove', mousemove)
    .on('click', mouseclick);
        
    // This allows to find the closest X index of the mouse:
    var bisect = d3.bisector(function (d) { return d.Start; }).left;


    function mousemove() {
        // recover coordinate we need
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]
        var minutes = Math.floor((uxvideo.duration * selectedData.Start/maxEnd)/60)
        var seconds = Math.round(60 * (((uxvideo.duration * selectedData.Start/maxEnd)/60) - minutes))
        var secStr = seconds < 10 ? "0" + seconds.toString() : seconds.toString()
        focus
            .attr("cx", x(selectedData.Start))
            .attr("cy", y(selectedData.Rate))
        focusTextRect
            .attr("x", x(selectedData.Start) + 10)
            .attr("y", y(selectedData.Rate) - 15)
        focusText
        .html("Time:" + minutes.toString() + ":" + secStr + "  -  " + "Rate:" + selectedData.Rate)
        .attr("x", x(selectedData.Start) + 15)
            .attr("y", y(selectedData.Rate))
    }

    function mouseclick(){
        //get a new maxEnd--paths handle focus differently, so we need to do this.
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]

        uxvideo.currentTime = uxvideo.duration * selectedData.Start/maxEnd 

    }
}


function rescalePitch(){
    
    var line = d3.select("#pitchlinepath")
    if(!line){
        return("")
    }
    var maxEnd = parseFloat(line.attr("maxEnd"));

    var data = JSON.parse(line.attr("origdata"));

    var fps = maxEnd/uxvideo.duration

    var selrect = d3.select('#focussvg').select('rect.selection');

    var selwid = parseFloat(selrect.attr('width'))
    var selX = parseFloat(selrect.attr('x'))

    if(isNaN(selwid)){
        selwid = width;
        selX = 1;
    }

    var minTime = uxvideo.duration * selX/width
    var maxTime = uxvideo.duration * ( selX + selwid )/width  

    var newMinFrame = minTime * fps
    var newMaxFrame = maxTime * fps
    var filtdata = _.filter(data, function(o){return parseFloat(o.x) >= newMinFrame & parseFloat(o.x) <= newMaxFrame })
    line.datum(filtdata)

    var x = d3.scaleLinear()
        .domain([newMinFrame, newMaxFrame])
        .range([0, width]);


    var y = d3.scaleLinear()
        .domain([0, 200])
        .range([height, 0]);
        
    d3.select('#pitch').select('#pitchxaxis')
        .call(d3.axisBottom(x)
            .tickFormat(function(d) {
                return d3.timeFormat('%M:%S')( new Date(0).setSeconds(d) )            
            })    
        );


    line.transition().duration(1).attr("d", function(d){ 
        return(d3.area()
        .x(function (d) { return x(d.x) })
        .y0(function (d) { return y(1) })
        .y1(function (d) { return y(d.y) })
        )(d)
    })

        /**
     * need to update our mouse activity
     * */
     // Create the circle that travels along the curve of chart
     var focus = d3.select('#pitchfocuscircle')
     // Create bg for the text that travels along the curve of chart
    var focusTextRect = d3.select('#pitchfocustextbg')
     // Create the text that travels along the curve of chart
     var focusText = d3.select('#pitchfocustext')
    
     //Different from line
    /*
   d3.select("#pitchmouserect")
    .on('mousemove', mousemove)
    .on('click', mouseclick);
    */

    line
    .on('mousemove', mousemove)
    .on('click', mouseclick);


    // This allows to find the closest X index of the mouse:
    var bisect = d3.bisector(function (d) { return d.x; }).left;


    function mousemove() {
        // recover coordinate we need
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]
        var minutes = Math.floor((uxvideo.duration * selectedData.x/maxEnd)/60)
        var seconds = Math.round(60 * (((uxvideo.duration * selectedData.x/maxEnd)/60) - minutes))
        var secStr = seconds < 10 ? "0" + seconds.toString() : seconds.toString()
        focus
            .attr("cx", x(selectedData.x))
            .attr("cy", y(selectedData.y))
        focusTextRect
            .attr("x", x(selectedData.x) + 10)
            .attr("y", y(selectedData.y) - 15)
        focusText
        .html("Time:" + minutes.toString() + ":" + secStr + "  -  " + "Pitch:" + selectedData.y)
        .attr("x", x(selectedData.x) + 15)
            .attr("y", y(selectedData.y))
    }

    function mouseclick(){
        //get a new maxEnd--paths handle focus differently, so we need to do this.
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]

        uxvideo.currentTime = uxvideo.duration * selectedData.x/maxEnd

    }


}


function rescaleFrames(){
    var thumbs = d3.select("#thumbframes")
    if(!thumbs){
        return("")
    }
    var maxEnd = parseFloat(thumbs.attr('maxEnd'))
    var fps = maxEnd/uxvideo.duration

    var data = JSON.parse(thumbs.attr("origdata"));

    var selrect = d3.select('#focussvg').select('rect.selection');

    var selwid = parseFloat(selrect.attr('width'))
    var selX = parseFloat(selrect.attr('x'))

    if(isNaN(selwid)){
        selwid = width;
        selX = 1;
    }


    var minTime = uxvideo.duration * selX/width
    var maxTime = uxvideo.duration * ( selX + selwid )/width

    var widMult = width/selwid

    var newMinFrame = Math.floor(minTime * fps)
    var newMaxFrame = Math.ceil(maxTime * fps)

    var filtdata = _.filter(data, function(o){return parseFloat(o.vidnum) >= newMinFrame & parseFloat(o.vidnum) <= newMaxFrame })

    
    var x = d3.scaleLinear()
    .domain([newMinFrame, newMaxFrame])
    .range([0, width]);

    function rectWidth(lowerVal, upperVal){
        gap = upperVal-lowerVal;
        rangeMult = widMult * (width/maxEnd)
        return (gap * rangeMult)
    }

    //var sliderImgCount = Math.ceil(120*selwid/width);
    var sliderImgCount = 120;

    var bunchMult = sliderImgCount/width;
    var frameSkip = Math.ceil(newMaxFrame/sliderImgCount);
    

    d3.select("#vidthumbnailsvg")
    .on("mouseover", function(){
      var fisheye = d3.fisheye.circular()
      .radius(width/5)
      .distortion(7*width/sliderImgCount);

      fisheye.focus(d3.mouse(this));

        var thumbs = d3.selectAll('.thumbframe')
        thumbs.each(function(d) { 
          d.fisheye = fisheye(d); 
        })
        .attr("x", function(d) {
          var distortX = Math.min(Math.max(d.fisheye.x, 0), (width-(margin.right+margin.left)))
          var xChk = fisheye(imgPaths[(imgPaths.length-1)]).x
          if((xChk < 1000) || (d3.select("#vidthumbnailsvg").attr('isdragging') == "true")){
            return d.x
          } else {
            return distortX;   
          }
        })

        //Track event
        interactiontracking(JSON.stringify(d3.mouse(this)), 'vidtimelineholder', 'vidtimelineholder', 'mouseover')

        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(filtdata, x0, 1);
        var focalThumbnail;
        if(i < filtdata.length){
            focalThumbnail = filtdata[i].vidnum

        if(focalThumbnail <= maxTime & focalThumbnail > 0 & d3.event.x < (width-margin.right)){

            var timelinesvgElem = document.getElementById('vidtimelineholder')
            var bodyRect = document.body.getBoundingClientRect(),
            elemRect = timelinesvgElem.getBoundingClientRect(),
            offset   = elemRect.top - bodyRect.top;
                
          //populate that tooltip
          d3.select('#thumbtooltip')
          .style("left", Math.min((d3.event.pageX - 100), .85*width) + "px")
          //.style("top", (d3.event.pageY - 100) + "px")   
          .style("top", offset + "px")
          .style('opacity', 1)
          .on('click', function(){
            try{
              uxvideo.currentTime=focalThumbnail
            } catch(err){
              console.log(err)
            }
          })
          
          try{
            d3.select('#thumbtooltip')
          .html('"<img src="frames/frame'+focalThumbnail+'.png" style="min-width:10vw;"></img>')
          } catch(err){
            console.log(err)
          }
        }
        
    }


    })
    .on("mouseleave", function(){
        var thumbs = d3.selectAll('.thumbframe')
 
        thumbs
          .attr("x", function(d) { return(d.x)})

        interactiontracking(JSON.stringify(d3.mouse(this)), 'vidtimelineholder', 'vidtimelineholder', 'mouseleave')

        d3.select('#thumbtooltip')
        .transition().duration(100)
        .style('opacity', 0)
  

    })
    .on('click', mouseclick)

    //inelegant
    d3.selectAll(".thumbframeg").remove()

    for(i = 0; i<filtdata.length; i+=frameSkip){
        var thumbX = widMult * ( (i/frameSkip)*(width/sliderImgCount) - i*bunchMult )
        var thumbWid = width/(bunchMult*sliderImgCount)

        filtdata[i].x = thumbX
        filtdata[i].y = 1

        if(thumbX + thumbWid <= width){
            thumbs.append('g').datum(filtdata[i]).attr('class', 'thumbframeg').append("svg:image")
            .attr("xlink:href",  'frames/frame'+filtdata[i].vidnum.toString()+'.png')
            .attr("class",  'thumbframe')
            .attr("x", thumbX)
            .attr("y",  0)
            .attr("height", height)
            .attr("width", thumbWid)
        }
    }

    thumbs    
    .data(filtdata)

    // This allows to find the closest X index of the mouse:
    var bisect = d3.bisector(function (d) { return d.vidnum; }).left;

    function mouseclick(){
        //get a new maxEnd--paths handle focus differently, so we need to do this.
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(filtdata, x0, 1);
        selectedData = filtdata[i]

        var uxvidPrevTime =  uxvideo.currentTime;
        uxvideo.currentTime = uxvideo.duration * selectedData.vidnum/maxEnd

        interactiontracking(JSON.stringify(d3.mouse(this)), 'vidtimelineholder', 'vidtimelineholder', 'click', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])

    }


    /*
    thumbs.selectAll('.thumbframe')
    .transition().duration(1)
    .attr('width', function(d){
        return(rectWidth(d.vidnum, d.vidnum+width/(bunchMult*sliderImgCount)))
    })
    .attr('x', function(d){
        return(x(d.start))
    })
    */
}

function rescaleAnnotations(){
    //We could just redraw the damned thing, but that's a really ugly fix
    //...except we're already doing it, kind of, whenever we update the annotations.
    //So we'll need to make sure this works anyway! 
    createAnnotationsTimeline(false)
}

export {rescaleActions, rescaleAnnotations, rescaleEmotions, rescaleFrames, rescalePitch, rescaleSpeechrate, rescaleTimelines};