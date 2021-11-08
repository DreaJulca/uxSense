var uxvideo = document.getElementById("video_html5_api");


// append the svg object to the body of the page
function createEmotionsTimeline(){

    refreshuxSDimVars();

    var emosvg = d3.select("#Emotion")
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

            /*
    $.getJSON('modeloutput/face_all_emotions_poses_gender.json', function(){
        //console.log( "success" );
    })
    .fail(function() {
        //console.log( "error" );
    })
    .always(function() {
        //console.log( "complete" );
    })
    .done(function(rawdata) {
*/
        //console.log( "second success" );
    d3.json('modeloutput/'+uxSenseVideoPath+'/face_all_emotions_poses_gender.json', function(rawdata){
        refreshuxSDimVars();
        //console.log(rawdata)
        var detaildata = [];
        var data = [];
        var framecnt = 0;

        for(i = 0; i < rawdata.length; i++){
            var d = rawdata[i];
            d.forEach(function(df){
                detaildata.push({"frame": framecnt, "emotion": df[2]})
                if(typeof (detaildata[framecnt].emotion) == "undefined"){
                    detaildata[framecnt].emotion = "N/A"
                }
                framecnt++;
            })
        }
        //console.log(data)
        for(i = emotChunkWid; (i+emotChunkWid-1) < detaildata.length; i+=emotChunkWid){
            var thisChunk = _.filter(detaildata, function(d){return ((d.frame < i) & (d.frame >= i-emotChunkWid))})
            
            var groupCnt = _.countBy(thisChunk, function(d){return(d.emotion)})
            var groupCntNoNA = _.omit(groupCnt, 'N/A');
            //console.log([groupCntNoNA, groupCnt])
            var winner = 'N/A';

            if(groupCnt.length == 1 ){
                winner = Object.keys(groupCnt)[0];
                try{
                } catch(err){
                        console.log(err)
                    }
            } else {
                if(Object.keys(groupCnt)[0].length > 0){
                    try{
                        winner = _.reduce(groupCntNoNA, function(max, current, key) {
                            return max && max.value > current ? max : {
                                value: current,
                                key: key
                            };
                        }).key
                    } catch(err){
                        console.log(err)
                        winner = Object.keys(groupCnt)[0];
                    }
                }
            }

            var obs = {'start':(i-emotChunkWid),'end':i, 'emotion':winner, 'prob':groupCnt[winner]/emotChunkWid}
            if(typeof obs.emotion == 'undefined'){
                obs = {'start':(i-emotChunkWid),'end':i, 'emotion':'N/A', 'prob':0}
            }

            data.push(obs) 
        }

        //var unqemot = _.uniq(_.map(detaildata, function(d){return(d.emotion)}))
        
        var colorScale = d3.scaleOrdinal(['#ececec', '#ff0101', '#68c500', '#e33af4', '#e3f43a', '#3a86f4', '#ffc900', '#b2b2b2'])
        .domain(['N/A', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']);

        var maxEnd = _.max(_.map(data, function(dp){return(1*dp['end'])}));
        var fps = maxEnd/uxvideo.duration

        var x = d3.scaleLinear()
            .domain([1, maxEnd])
            .range([0, width]);

        var maxProb = _.max(_.map(_.reject(data,{'emotion':"N/A"}), function(dp){return(1*dp['prob'])}));
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


        emosvg.append('g')
        .attr('id', 'emoxaxis')
        .attr('class', 'timelinexaxis')
        .attr("transform", "translate(0," + 50 + ")")
        .call(d3.axisBottom(x)
            .tickFormat(function(d) {
                return d3.timeFormat('%M:%S')( new Date(0).setSeconds(d/fps) )            
            })
        )

        for(i=0; i<data.length; i++){
            if(data[i].emotion == "N/A"){
                data[i].prob = 0;
            }
        }
        var emog = emosvg.append('g')
            .data(data)
            .attr('id', 'emotionrects')
            .attr('origdata', JSON.stringify(data))
            .attr('maxEnd', maxEnd)

        d3.select('body').append('div')
        .attr('id', 'emotooltip')
        .style('opacity', 0)


        var allStarts = _.map(data, 'start')
        var startFrames = _.uniq(allStarts)

        startFrames.forEach(function(startframe){
            var thisData = _.filter(data, {'start':startframe})
            var bestprob = _.max(_.map(thisData,  function(dp){return(1*dp['prob'])}));

            var d = _.filter(thisData, function(dp){return(1*dp['prob']==bestprob)})[0]
            
            emog.append('rect')
            .datum(d)
            .attr('width', rectWidth(parseFloat(d.start), parseFloat(d.end)))
            .attr('id', 'rect_emotion_'+d.start)
            .attr('class', 'emotionrect')
            .attr('x', x(d.start))
            .attr('height', rectHeight(1*d.prob))
            .attr('y', y(parseFloat(d.prob)))
            .attr('fill', colorScale(d.emotion))
            .on('mouseover', function(){
                d3.select('#emotooltip')
                .transition().duration(100)
                .style("left", (d3.event.pageX) + "px")
                .style("top", (d3.event.pageY - 28) + "px")   
                .style('opacity', 1)

                d3.select('#emotooltip')
                .html("emotion: " + d.emotion + "<br/>rel. likelihood: " + (Math.round(10000*(d.prob))/100) + "%")

                //Track event
                interactiontracking(d, 'emotionrects', 'rect_emotion_'+d.start, 'mouseover')
        
            })
            .on('mouseout', function(){
                d3.select('#emotooltip')
                .transition().duration(100)
                .style('opacity', 0)

                //Track event
                interactiontracking(d, 'emotionrects', 'rect_emotion_'+d.start, 'mouseout')

            })
            .on('click', function(){
                var fps = maxEnd/uxvideo.duration;
                var uxvidPrevTime = uxvideo.currentTime;
                uxvideo.currentTime = d.start/fps;
                interactiontracking(d, 'emotionrects', 'rect_emotion_'+d.start, 'click', [{oldtime: uxvidPrevTime}, {newtime: uxvideo.currentTime}])

            })

        })

        //and we're going to add rects as background to our filter sliders (and also to block out edges on focus)
        emosvg.append('rect')
        .attr('fill', 'white')
        .attr('height', height + margin.top + margin.bottom)
        .attr('width', margin.left)
        .attr('x', -margin.left)
        .attr('y', 0)

        emosvg.append('rect')
        .attr('fill', 'white')
        .attr('height', height + margin.top + margin.bottom)
        .attr('width', margin.right)
        .attr('x', width)
        .attr('y', 0)


        //Add axis labels and ticks after
        emosvg.append("g")
        .call(d3.axisLeft(y).ticks(5))

        rescaleEmotions();

    })

}

uxvideo.addEventListener('loadeddata', function(){
    createEmotionsTimeline()
})
