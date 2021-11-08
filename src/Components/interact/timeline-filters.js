var uxvideo = document.getElementById('video_html5_api');

refreshuxSDimVars();

//document.addEventListener("DOMContentLoaded", function (event) {
$(document).ready(function () {
    setTimeout("drawPathSlider('speech-rate', 'speechratelinepath');", 1500)
    setTimeout("drawPathSlider('pitch', 'pitchlinepath');", 1500)
    setTimeout("drawBarSlider('Action1');", 1500)
    setTimeout("drawBarSlider('Emotion');", 1500)

});


function drawPathSlider(containerID, pathID){
    refreshuxSDimVars();

    var svg = d3.select("#" + containerID).select('svg')

    var path = svg.select("#" + pathID)

    try{
        var data = JSON.parse(path.attr('origdata'));
    } catch(err){
        setTimeout('drawPathSlider("' + containerID + '","' + pathID + '")', 100);
        return 'Done';
    }

    var yScaleBottom = d3.scaleLinear().range([ 0 , filterheight]);

    yScaleBottom.domain(d3.extent(data, function (d) { 
        if(containerID == 'speech-rate'){ 
            return d.Rate
        } 
        if(containerID == 'pitch'){
            return d.y
        }
    }));

    var yScaleTop = d3.scaleLinear().range([0, filterheight]);
    var yAxisTop = d3.axisBottom(yScaleTop);

    var filterbrush = d3.brushY()    
    .extent([[0, 0], [sliderwidth, filterheight]])
    .on("brush end", function () {
        var s = d3.event.selection || yScaleBottom.range();
        yScaleTop.domain(s.map(yScaleBottom.invert, yScaleBottom));
        

        interactiontracking(yScaleTop.domain(), containerID, pathID, 'brush end')

        //d3.selectAll('.x.axis').call(xAxisTop);
        //uxvideo.currentTime = uxvideo.duration * (focussvg.select('rect.selection').attr('x') / width)
        //focus.select(".line").attr("d", lineTop);
        //if(focussvg.select('rect.selection').attr('width') > 0 | focussvg.select('rect.selection').attr('width') == null){rescaleTimelines();}
    })
         
    var slider = svg
    .append('g')
    .attr('height', filterheight)
    .attr('width', 3*sliderwidth)
    .attr('id', containerID + '-filt-slider')
    .attr('class', 'filter-slider')
    .attr('transform', 'translate(0,'+margin.top+')')

    var brushr = slider.append('rect')
    .attr('height', filterheight)
    .attr('x', 0)
    .attr('rx', 3)
    .attr('width', sliderwidth)
    .attr('fill', 'plum')
    //#862d86
    var brushg = slider.append('g')
    .attr('height', filterheight)
    .attr('width', 3*sliderwidth)
    .attr("class", "y brush")
    .call(filterbrush);
//.call(d3.brushY().extent([[0, 0], [sliderwidth, filterheight]]).on("brush", filterbrush));


}


function drawBarSlider(containerID){

    refreshuxSDimVars();

    var svg = d3.select("#" + containerID).select('svg')

    var g = svg.select("g")

    var rectclass; 

    if(containerID == 'Emotion'){
        g = svg.select('#emotionrects')
        rectclass = 'emotionrect'
    }

    if(containerID == 'Action1'){
        g = svg.select('#action1rects')
        rectclass = 'action1rect'
    }

    try{
        var data = JSON.parse(g.attr('origdata'));
    } catch(err){
        setTimeout('drawBarSlider("' + containerID + '")', 300);
        return 'Done';
    }

    var yScaleBottom = d3.scaleLinear().range([ filterheight, 0]);

    yScaleBottom.domain(d3.extent(data, function (d) { 
        return 1*d.prob
    }));

    var yScaleTop = d3.scaleLinear().range([0, filterheight]);
    var yAxisTop = d3.axisBottom(yScaleTop);
    
    var filterbrush = d3.brushY()    
    .extent([[0, 0], [sliderwidth, filterheight]])
    .on("brush end", function () {
        var s = d3.event.selection || yScaleBottom.range();
        yScaleTop.domain(s.map(yScaleBottom.invert, yScaleBottom));

        interactiontracking(yScaleTop.domain(), containerID, containerID, 'brush end')

        g.selectAll('.'+rectclass).each(function(d){
            if(1*d.prob >= yScaleTop.domain()[1] & 1*d.prob <= yScaleTop.domain()[0]){
                d3.select(this).attr('opacity', 1)
            } else {
                d3.select(this).attr('opacity', 0.3)
            }
        })

    })
         
    var slider = svg
    .append('g')
    .attr('height', filterheight)
    .attr('width', 3*sliderwidth)
    .attr('id', containerID + '-filt-slider')
    .attr('class', 'filter-slider')
    .attr('transform', 'translate(0,'+margin.top+')')

    var brushr = slider.append('rect')
    .attr('height', filterheight)
    .attr('x', 0)
    .attr('rx', 3)
    .attr('width', sliderwidth)
    .attr('fill', 'plum')

    var brushg = slider.append('g')
    .attr('height', filterheight)
    .attr('width', 3*sliderwidth)
    .attr("class", "y brush")
    .call(filterbrush)
    //.call(d3.brushY().extent([[0, 0], [15, filterheight]]).on("brush", filterbrush));


}

