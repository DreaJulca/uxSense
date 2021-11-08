var uxvideo = document.getElementById('video_html5_api');


//At some point, we may not have exactly 701 frames... but for now we have frames 0-700 and we are going to hard code instead of relying on server data
var maxThumb = 700
//var sliderImgCount = 10 * width/maxThumb;
var sliderImgCount = 120;

var imgPaths = []; 

for(i = 0; i<=maxThumb; i++){
    imgPaths.push({'vidnum': i, 'vidpath':'frames/frame'+i.toString()+'.png'});
}

var focalThumbnail = 0;

d3.select('body').append('div')
.attr('id', 'thumbtooltip')
.style('opacity', 0)


// append the svg object to the body of the page
var thumbsvg = d3.select("#vidtimelineholder")
    .append("svg")
    .attr("id", "vidthumbnailsvg")
    .attr("width", "100%")
    .attr("height", (height + margin.top + margin.bottom))
    .attr("viewBox", "0 0 " + (width + margin.left + margin.right) + " " + (height + margin.top + margin.bottom))
    .attr("preserveAspectRatio", "none")
    .attr("isdragging", false)
    .style('display', 'inline-block')
    .style('position', 'relative')
    .on("mouseover", function(){
      
// deal with fisheye effect


      var fisheye = d3.fisheye.circular()
      .radius(width/5)
      .distortion(7*width/sliderImgCount);

      //xFisheye = d3.fisheye.scale(d3.scaleIdentity).focus(0)
      
      fisheye.focus(d3.mouse(this));

        var thumbs = d3.selectAll('.thumbframe')
        thumbs.each(function(d) { 
          //d.fisheye = xFisheye;
          d.fisheye = fisheye(d); 
        })
        .attr("x", function(d) {
          var distortX = Math.min(Math.max(d.fisheye.x, 0), (width-(margin.right+margin.left)))
          //var adjX = Math.max(d3.mouse(this).x, distortX)
          var xChk = fisheye(imgPaths[(imgPaths.length-1)]).x
          //if(xChk-(margin.left+margin.right) < width-2*(margin.right+margin.left)){
          if((xChk < 1000) || (d3.select(this).attr('isdragging') == "true")){
            return d.x
          } else {
            return distortX;   
          }
        })

        //Track event
        interactiontracking(JSON.stringify(d3.mouse(this)), 'vidtimelineholder', 'vidtimelineholder', 'mouseover')

        focalThumbnail = Math.round(uxvideo.duration * (d3.event.pageX-margin.left)/(width-(margin.left+margin.right)))

        if(focalThumbnail <= maxThumb & focalThumbnail > 0 & d3.event.x < (width-margin.right)){

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
    .on('click', function(){
      var uxvidPrevTime = uxvideo.currentTime;

      uxvideo.currentTime = uxvideo.duration*(d3.mouse(this)[0]-margin.left)/width;
      interactiontracking(JSON.stringify(d3.mouse(this)), 'vidtimelineholder', 'vidtimelineholder', 'click', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])

    })
//    .attr("width", width + margin.left + margin.right)
//    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")")

var thumbs = thumbsvg.append('g')
    .attr('id', 'thumbframes')
    .attr('origdata', JSON.stringify(imgPaths))
    .attr('maxEnd', maxThumb)


//little bit of bad programming here
var bunchMult = sliderImgCount/width;
var frameSkip = Math.ceil(maxThumb/sliderImgCount);

  // Append images
//  imgPaths.forEach(function(img){
//      if(img.vidnum % sliderImgCount == 0){
    for(i = 0; i<=maxThumb; i+=frameSkip){
      var thumbX =  (i/frameSkip)*(width/sliderImgCount) - i*bunchMult
      var thumbWid = width/(bunchMult*sliderImgCount)

      imgPaths[i].x = thumbX
      imgPaths[i].y = 1

      if(thumbX + thumbWid <= width){
        thumbs.append('g').datum(imgPaths[i]).attr('class', 'thumbframeg').append("svg:image")
        .attr("xlink:href",  'frames/frame'+i.toString()+'.png')
        .attr("class",  'thumbframe')
        .attr("x", thumbX)
        .attr("y",  0)
        .attr("height", height)
        .attr("width", thumbWid)
      }
    }
    
    thumbs    
    .data(imgPaths)


  //and we're going to add rects as background to our filter sliders (and also to block out edges on focus)
  thumbsvg.append('rect')
  .attr('fill', 'white')
  .attr('height', height + margin.top + margin.bottom)
  .attr('width', margin.left)
  .attr('x', -margin.left)
  .attr('y', 0)

  thumbsvg.append('rect')
  .attr('fill', 'white')
  .attr('height', height + margin.top + margin.bottom)
  .attr('width', margin.right)
  .attr('x', width)
  .attr('y', 0)


//      }
//  })


//Use Cartesian distortion on mouseover; borrow from https://bost.ocks.org/mike/fisheye/
/*
//(function() {
    d3.fisheye = {
      scale: function(scaleType) {
        return d3_fisheye_scale(scaleType(), 3, 0);
      },
      circular: function() {
        var radius = 200,
            distortion = 2,
            k0,
            k1,
            focus = [0, 0];
  
        function fisheye(d) {
          var dx = d.x - focus[0],
              dy = d.y - focus[1],
              dd = Math.sqrt(dx * dx + dy * dy);
          if (!dd || dd >= radius) return {x: d.x, y: d.y, z: dd >= radius ? 1 : 10};
          var k = k0 * (1 - Math.exp(-dd * k1)) / dd * .75 + .25;
          return {x: focus[0] + dx * k, y: focus[1] + dy * k, z: Math.min(k, 10)};
        }
  
        function rescale() {
          k0 = Math.exp(distortion);
          k0 = k0 / (k0 - 1) * radius;
          k1 = distortion / radius;
          return fisheye;
        }
  
        fisheye.radius = function(_) {
          if (!arguments.length) return radius;
          radius = +_;
          return rescale();
        };
  
        fisheye.distortion = function(_) {
          if (!arguments.length) return distortion;
          distortion = +_;
          return rescale();
        };
  
        fisheye.focus = function(_) {
          if (!arguments.length) return focus;
          focus = _;
          return fisheye;
        };
  
        return rescale();
      }
    };
  
    function d3_fisheye_scale(scale, d, a) {
  
      function fisheye(_) {
        var x = scale(_),
            left = x < a,
            range = d3.extent(scale.range()),
            min = range[0],
            max = range[1],
            m = left ? a - min : max - a;
        if (m == 0) m = max - min;
        return (left ? -1 : 1) * m * (d + 1) / (d + (m / Math.abs(x - a))) + a;
      }
  
      fisheye.distortion = function(_) {
        if (!arguments.length) return d;
        d = +_;
        return fisheye;
      };
  
      fisheye.focus = function(_) {
        if (!arguments.length) return a;
        a = +_;
        return fisheye;
      };
  
      fisheye.copy = function() {
        return d3_fisheye_scale(scale.copy(), d, a);
      };
  
      fisheye.nice = scale.nice;
      fisheye.ticks = scale.ticks;
      fisheye.tickFormat = scale.tickFormat;
      return fisheye //d3.rebind(fisheye, scale, "domain", "range");
    }
//  })();  
*/