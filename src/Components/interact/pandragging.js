/**
 * Add pan dragging to all timelines. 
 */
var video = document.getElementById('video_html5_api');
refreshuxSDimVars();

var focussvg = d3.select('#focussvg').select('g')

var mouseStartX;
var mouseStartExtMin;
var mouseStartExtMax;

//Add panning dragging to all nodes
function addPanningToSVGs(){
  d3.select('.timelines-box').selectAll('svg')
  .call(d3.drag()
    .on("start", panstarted)
    .on("drag", panned)
    .on("end", panended))
    .on("click", goToTimelineTime);
}

function redrawBrush() {
    var focusBrush = focussvg.select('g.x.brush')
    var selRect = focusBrush.select('rect.selection')

    // redefine our brush extent; we subtract event from start bc we want to drag the same dir as mouse moves
    var moveDistance = (mouseStartX - d3.event.x) * (parseFloat(selRect.attr('width'))/width);

    var newextmin = mouseStartExtMin + moveDistance
    var newextmax = mouseStartExtMax + moveDistance

    var newext = [newextmin, newextmax]

    if(newextmin < 0){
        newext = [0, parseFloat(selRect.attr('width'))];
    }

    if(newextmax > width){
        newext = [width - parseFloat(selRect.attr('width')), width];
    }

    //we are kind of depending on the brush var from our focustimelines.js script to not be redefined elsewhere; not a great practice, but time limits!
    brush.move(focusBrush, newext)

}

    
function panstarted() {
  var focusBrush = focussvg.select('g.x.brush')
  var selRect = focusBrush.select('rect.selection')
  
  d3.select(this).attr('isdragging', true);
  mouseStartX = 1*d3.event.x
  mouseStartExtMin = parseFloat(selRect.attr('x'))
  mouseStartExtMax = mouseStartExtMin + parseFloat(selRect.attr('width')) 
  
  interactiontracking(JSON.stringify(d3.event), this.parentNode.getAttribute("id"), this.parentNode.getAttribute("id"), 'drag start')
}

function panned() {
    var focusselectbox = focussvg.select('rect.selection');

    if(focusselectbox.attr('style') == ""){
      var uxvidPrevTime =  uxvideo.currentTime;
      redrawBrush()  
      interactiontracking(JSON.stringify(d3.event), this.getAttribute("id"), this.getAttribute("id"), 'drag', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])
    }
}

function panended() {
  d3.select(this).attr('isdragging', false);
  interactiontracking(JSON.stringify(d3.event), this.getAttribute("id"), this.getAttribute("id"), 'drag end')
    //do something if needed, but I think we are good
}


function goToTimelineTime(){
  refreshuxSDimVars();
  var focusBrush = focussvg.select('g.x.brush')
  var selRect = focusBrush.select('rect.selection')

  mouseStartExtMin = parseFloat(selRect.attr('x'))
  mouseStartExtMax = mouseStartExtMin + parseFloat(selRect.attr('width')) 

  if(isNaN(mouseStartExtMax)){
    mouseStartExtMin = 0;
    mouseStartExtMax = width;
  }

  var minTime = uxvideo.duration * mouseStartExtMin/width
  var maxTime = uxvideo.duration * mouseStartExtMax/width  

  var invertx = d3.scaleLinear()
  //.domain([margin.left, width-(margin.left+margin.right)])
  .domain([margin.left, width-margin.right])
  .range([minTime, maxTime]);

  if(d3.event.x < (width-margin.right) & d3.event.target.class != "annbtnfill"){
    var uxvidPrevTime =  uxvideo.currentTime;
    //uxvideo.currentTime = maxTime * (d3.event.x - margin.left)/(width-(margin.left+margin.right)) + minTime;
    uxvideo.currentTime = invertx(d3.event.x);
  
    interactiontracking(JSON.stringify(d3.event), this.getAttribute("id"), this.getAttribute("id"), 'click', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])
  }
}

uxvideo.addEventListener('loadeddata', function(){
  addPanningToSVGs()
  setTimeout('addPanningToSVGs()', 1600)
})
