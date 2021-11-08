/**
 * I should really start commenting my code better. Create timeline "cursor" (current time indicator)--should be draggable
 */
var uxvideo = document.getElementById('video_html5_api');

var focussvg = d3.select('#focussvg').select('g')
var markersize = 100;

var symbolGenerator = d3.symbol()
  .type(d3.symbolTriangle)
  .size(markersize);

var pathData = symbolGenerator();

//create draggable marker
focussvg.append('path')
  .attr('d', pathData)
  .attr('id', 'markertriangle')
  .attr('transform', 'rotate(180)')
  .attr('style', 'fill:red; pointer-events:all;')
  .call(d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended));

function dragstarted() {
  d3.select(this).raise().classed("active", true);
  interactiontracking(JSON.stringify(d3.event), "premierefocus", "markertriangle", 'drag start')

}

function dragged() {
  uxvideo.currentTime = uxvideo.duration * d3.event.x/width
  d3.select(this)
  .attr('transform', 'rotate(180) translate('+(-d3.event.x)+',0)')

  interactiontracking(JSON.stringify(d3.event), "premierefocus", "markertriangle", 'dragged')

}

function dragended() {
  uxvideo.currentTime = uxvideo.duration * d3.event.x/width
  d3.select(this).classed("active", false);
  interactiontracking(JSON.stringify(d3.event), "premierefocus", "markertriangle", 'drag end')
}

function moveMarker() {
  var minTime = 0;
  var maxTime = uxvideo.duration;
  var focusselectbox = focussvg.select('rect.selection');
  
  if(focusselectbox.attr('style') == ""){
    minTime = uxvideo.duration * focusselectbox.attr('x')/width
    maxTime = uxvideo.duration * (parseFloat(focusselectbox.attr('x')) + parseFloat(focusselectbox.attr('width')))/width  
  } 

  var cursorlineX = width * (uxvideo.currentTime - minTime)/(maxTime-minTime)

  try{
    //try to remove marker g.rects for all timelines
    d3.selectAll('.timeline-marker').remove();
  }catch(err){
    console.log(err)
  }

  var markerX = width * (uxvideo.currentTime/uxvideo.duration)

  if(cursorlineX > -Infinity)
  //add marker g.rects for all timelines
  d3.select('.timelines-box').selectAll('svg')
  .append('g')
  .attr('class', 'timeline-marker')
  .attr('transform', 'translate('+margin.left+','+margin.top+')')
  .append('rect')
  .attr('fill', 'red')
  .attr('width', '2')
  .attr('height', height)
  .attr('x', cursorlineX)

  var marker = d3.select("#markertriangle");

  if(marker.attr('class') == 'active'){
    setTimeout("moveMarker()", 50)
  } else {
    marker.transition().duration(50).attr('transform', 'rotate(180) translate('+(-markerX)+',0)')
    .on("end",function(){
      moveMarker()
    })
  }
}

uxvideo.addEventListener('loadeddata', function(){
  setTimeout("moveMarker();", 1550)
})
