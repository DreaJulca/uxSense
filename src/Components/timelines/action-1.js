var uxvideo = document.getElementById("video_html5_api");

function createActionsTimeline(){
    refreshuxSDimVars();

    // append the svg object to the body of the page
    var actsvg = d3.select("#Action1")
        .append("svg")
        .attr("width", "100%")
        .attr("height", (height + margin.top + margin.bottom))
        .attr("viewBox", "0 0 " + (width + margin.left + margin.right) + " " + (height + margin.top + margin.bottom))
        .attr("preserveAspectRatio", "none")
        .style('display', 'inline-block')
        .style('position', 'relative')
    //    .attr("width", width + margin.left + margin.right)
    //    .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    $.ajaxSetup({
        timeout: 5000 //Time in milliseconds
    });

            
    d3.csv('modeloutput/'+uxSenseVideoPath+'/actions_best.csv', function(data){
    
    //d3.csv('modeloutput/'+uxSenseVideoPath+'/actions_best.csv', function(data){
    d3.csv('arbitrary_action_relabel.csv', function(renamedata){
        refreshuxSDimVars();
        var colorScale = d3.scaleOrdinal(d3.schemeCategory20c)
        .domain(_.uniq(_.map(renamedata, 'arbitrary_action')));
        
        data.forEach(function(d){
            d.old_action = d.action;
            try{
                d.action = _.filter(renamedata, function(o){return o.action == d.old_action})[0].arbitrary_action        
            } catch(err){
                console.log(err);

            }
        })

        var maxEnd = _.max(_.map(data, function(dp){return(1*dp['end'])}));
        var fps = maxEnd/uxvideo.duration

        var x = d3.scaleLinear()
            .domain([1, maxEnd])
            .range([0, width]);

        var maxProb = _.max(_.map(data, function(dp){return(1*dp['prob'])}));
        var y = d3.scaleLinear()
            .domain([0, maxProb])
            .range([height, 0]);

        function rectWidth(lowerVal, upperVal){
            gap = upperVal-lowerVal;
            rangeMult = (width/maxEnd)
            return (gap * rangeMult)
        }

        function rectHeight(prob){
            rangeMult = (height/maxProb)
            return (prob * rangeMult)
        }

        actsvg.append('g')
        .attr('id', 'actxaxis')
        .attr('class', 'timelinexaxis')
        .attr("transform", "translate(0," + 50 + ")")
        .call(d3.axisBottom(x)
            .tickFormat(function(d) {
                return d3.timeFormat('%M:%S')( new Date(0).setSeconds(d/fps) )            
            })
        )

        var actg = actsvg.append('g')
            .data(data)
            .attr('id', 'action1rects')
            .attr('origdata', JSON.stringify(data))
            .attr('maxEnd', maxEnd)

        d3.select('body').append('div')
        .attr('id', 'acttooltip')
        .style('opacity', 0)


        var allStarts = _.map(data, 'start')
        var startFrames = _.uniq(allStarts)


        startFrames.forEach(function(startframe){
            //var thisData = _.filter(data, {'start':startframe})
            //var bestprob = _.max(_.map(thisData,  function(dp){return(1*dp['prob'])}));
            //var d = _.filter(thisData, function(dp){return(1*dp['prob']==bestprob)})[0]
            var d = _.filter(data, {'start':startframe})[0]

            actg.append('rect')
            .datum(d)
            .attr('width', rectWidth(parseFloat(d.start), parseFloat(d.end)))
            .attr('id', 'rect_action1_'+d.start)
            .attr('class', 'action1rect')
            .attr('x', x(d.start))
            .attr('height', rectHeight(1*d.prob))
            .attr('y', y(parseFloat(d.prob)))
            .attr('fill', colorScale(d.action))
            .on('mouseover', function(){
                d3.select('#acttooltip')
                .transition().duration(100)
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY - 28) + "px")   
                .style('opacity', 1)

                d3.select('#acttooltip')
                .html("action: " + d.action + "<br/>probability: " + (Math.round(10000*(d.prob))/100) + "%")

                //Track event
                interactiontracking(d, 'action1rects', 'rect_action1_'+d.start, 'mouseover'    )
        
            })
            .on('mouseout', function(){
                d3.select('#acttooltip')
                .transition().duration(100)
                .style('opacity', 0)

                //Track event
                interactiontracking(d, 'action1rects', 'rect_action1_'+d.start, 'mouseout')

            })
            .on('click', function(){
                var fps = maxEnd/uxvideo.duration;
                var uxvidPrevTime = uxvideo.currentTime
                uxvideo.currentTime = d.start/fps;
                //track event
                interactiontracking(d, 'action1rects', 'rect_action1_'+d.start, 'click', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])
            })

        })
        //and we're going to add rects as background to our filter sliders (and also to block out edges on focus)
        actsvg.append('rect')
        .attr('fill', 'white')
        .attr('height', height + margin.top + margin.bottom)
        .attr('width', margin.left)
        .attr('x', -margin.left)
        .attr('y', 0)

        actsvg.append('rect')
        .attr('fill', 'white')
        .attr('height', height + margin.top + margin.bottom)
        .attr('width', margin.right)
        .attr('x', width)
        .attr('y', 0)

        //Add axis labels and ticks after
        actsvg.append("g")
            .call(d3.axisLeft(y).ticks(5))


        rescaleActions();
    })
    })
}

uxvideo.addEventListener('loadeddata', function(){
    createActionsTimeline()
})
