var width = 560
var height = 100

var headerspace = 100
var transcriptspace = 450

var margin = {left:40, top:40, right:40, bottom:40}

var emotChunkWid = 240;
var emoColorScale = d3.scaleOrdinal(['#ececec', '#ff0101', '#68c500', '#e33af4', '#e3f43a', '#3a86f4', '#ffc900', '#b2b2b2'])
    .domain(['N/A', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']);

var actColorScale = d3.scaleOrdinal(d3.schemeCategory10);
d3.csv('arbitrary_action_relabel.csv', {credentials: 'same-origin'})
.then(function(renamedata){
//console.log(d3.schemeCategory20c)
actColorScale
    .domain(_.uniq(_.map(renamedata, 'arbitrary_action')));

//load video.durations in seconds--this is a last-minute hack job
d3.json("userAnnotations/videotimes.json",{credentials: 'same-origin'})
.then(function(videoTimeLengths){
    //load user annotations
    d3.json("userAnnotations/data.json",{credentials: 'same-origin'})
    .then(function(allUserAnnotations){
        //console.log(allUserAnnotations)
        var videoNames = _.uniq(_.map(allUserAnnotations, 'videoname'))
        //console.log(videoNames)
        //loop through each video's notes 
        videoNames.forEach(function(vid){
            var videoAnnotations = _.filter(allUserAnnotations,{"videoname":vid});
            //console.log(videoAnnotations)
            var videoDuration = _.filter(videoTimeLengths, {"video":vid})//.duration
            var captionDataFile = "captions/" + vid + "_captions.vtt"
            var actionDataFile = "modeloutput/" + vid + "/actions_best.csv"
            var emotionDataFile = "modeloutput/" + vid + "/face_all_emotions_poses_gender.json" 

            $.get(captionDataFile, function(captiondata) {
    
                // Read all captions into an array
                var captions = captiondata.split('\n\r\n');
    
                //console.log(captions);
                var transcript = [];
                //Loop through all captions
                $.each(captions, function( index, value ) {
                
                    var caption = captions[index].split('\n');
                    if(caption.length == 3){
                        //console.log(caption);    
                        var captiontext = caption[2]
                        var times = caption[1].split(" --> ")

                        for(i = 0; i<times.length; i++){
                            times[i] = times[i].replace('\n','').replace('\r', '')
                        }

                        //just use unix epoch as date for all timestamps; don't anticipate evaluating a video longer than 24 hours.
                        var startTimestamp = new Date("1970-01-01 " + times[0])
                        var startSeconds =  (60**2)*startTimestamp.getHours() + 60*startTimestamp.getMinutes() + startTimestamp.getSeconds() + startTimestamp.getMilliseconds()/1000
                        //console.log("1970-01-01 " + times[0] )
                        //console.log(startTimestamp )
                        //console.log(startSeconds )


                        var endTimestamp = new Date("1970-01-01 " + times[1])
                        var endSeconds =  (60**2)*endTimestamp.getHours() + 60*endTimestamp.getMinutes() + endTimestamp.getSeconds() + endTimestamp.getMilliseconds()/1000

                        var captionarrval = {'start':startSeconds, 'end':endSeconds, 'text':captiontext.replace("Andrea Carmen Batch", "\nResearcher").replace("Participant ", "\nParticipant ")}

                        transcript.push(captionarrval);

                    }

                });
                //console.log(transcript);
                d3.csv(actionDataFile,{credentials: 'same-origin'})
                .then(function(actiondata){
                    d3.json(emotionDataFile,{credentials: 'same-origin'})
                    .then(function(emotiondataraw){
                        //now we have our video timestamps, our transcripts, our action data, and our emotion data for this specific video loaded; 
                        //we want to append each annotation with a view of data for the relevant range 
                        if(typeof(videoDuration[0]) == "undefined"){
                            console.log(vid)
                            return 
                        } 

                        videoDuration = videoDuration[0].duration
                        console.log(videoDuration)
                        
                        //clean up emotion data
                        var emotiondetaildata = [];
                        var emotiondata = [];
                        var framecnt = 0;

                        for(i = 0; i < emotiondataraw.length; i++){
                            var d = emotiondataraw[i];
                            d.forEach(function(df){
                                emotiondetaildata.push({"frame": framecnt, "emotion": df[2]})
                                if(typeof (emotiondetaildata[framecnt].emotion) == "undefined"){
                                    emotiondetaildata[framecnt].emotion = "N/A"
                                }
                                framecnt++;
                            })
                        }
                        //console.log(data)
                        for(i = emotChunkWid; (i+emotChunkWid-1) < emotiondetaildata.length; i+=emotChunkWid){
                            var thisChunk = _.filter(emotiondetaildata, function(d){return ((d.frame < i) & (d.frame >= i-emotChunkWid))})
                            
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
                                        //console.log(err)
                                        winner = Object.keys(groupCnt)[0];
                                    }
                                }
                            }

                            var obs = {'start':(i-emotChunkWid),'end':i, 'emotion':winner, 'prob':groupCnt[winner]/emotChunkWid}
                            if(typeof obs.emotion == 'undefined'){
                                obs = {'start':(i-emotChunkWid),'end':i, 'emotion':'N/A', 'prob':0}
                            }

                            emotiondata.push(obs) 
                        }

                        //console.log(emotiondata)
                        var emotionMaxEnd = _.max(_.map(emotiondata, function(dp){return(1*dp['end'])}));
                        var emotionfps = emotionMaxEnd/videoDuration

                        
                        //process action data        
                        actiondata.forEach(function(d){
                            d.old_action = d.action;
                            try{
                                d.action = _.filter(renamedata, function(o){return o.action == d.old_action})[0].arbitrary_action        
                            } catch(err){
                                console.log(err);

                            }
                        })
                        //console.log(actiondata)
                        var actionMaxEnd = _.max(_.map(actiondata, function(dp){return(1*dp['end'])}));
                        var actionfps = actionMaxEnd/videoDuration



                        videoAnnotations.forEach(function(d){

                            //this stuff is just for labeling                                
                            var timeNum = parseFloat(d.timestamp)

                            if(d.annotationtype == "interval" & d.focusbrushed == "true"){
                                timeNum = parseFloat(d.annotatedintervalmin)
                            }

                            var minute = Math.floor(timeNum/60)
                            var second = Math.round(timeNum - 60*minute)
                            var minstr = minute < 10 ? "0" + minute.toString() : minute.toString();
                            var secstr = second < 10 ? "0" + second.toString() : second.toString();

                            var timestr = minstr + ":" + secstr


                            if(d.annotationtype == "interval" & d.focusbrushed == "true"){
                                timeNum2 = parseFloat(d.annotatedintervalmax)
                                var minute2 = Math.floor(timeNum2/60)
                                var second2 = Math.round(timeNum2 - 60*minute2)
                                var minstr2 = minute2 < 10 ? "0" + minute2.toString() : minute2.toString();
                                var secstr2 = second2 < 10 ? "0" + second2.toString() : second2.toString();

                                var timestr2 = minstr2 + ":" + secstr2
                                var timestr1 = timestr;
                                timestr = timestr1 + "-" + "\n" + timestr2
                            }


                            //now we start changing our view
                            var div = d3.select("#main")
                            .append("svg")
                            .attr("id", d._id)
                            .attr("class", "annotlette")


                            var startTime = parseFloat(d.timestamp)
                            var endTime = parseFloat(d.timestamp)

                            //we want to grab all rows of data with max vals greater than the min
                            if(d.annotationtype == "interval" & d.focusbrushed == "true"){
                                startTime = parseFloat(d.annotatedintervalmin)
                                endTime = parseFloat(d.annotatedintervalmax)
                            }
                            //console.log([startTime, endTime])

                            var includeQuotes = _.filter(
                                transcript, 
                                function(o){
                                    var condition = o.start <= endTime & o.end >= startTime
                                    //if(condition){
                                    //    console.log([o.start, o.end])
                                    //}
                                    return(condition)
                                }
                            )
                            /* actually not the way to do this; maybe later. its late rn
                            for(i=0; i<includeQuotes.length; i++){
                                if(includeQuotes[i].replace("Researcher:","").length < includeQuotes[i]){
                                    if(includeQuotes[i-1].replace("Researcher:", "").length < includeQuotes[i-1].length){
                                        includeQuotes[i-1] = includeQuotes[i-1] + "... " + includeQuotes[i]
                                        includeQuotes[i] = null 
                                    }
                                }
                            }
                            */
                            for(i=0; i<includeQuotes.length; i++){
                                if(includeQuotes[i].text.replace("\nResearcher:","").length < includeQuotes[i].text.length){
                                    includeQuotes[i].text = "";
                                }
                            }
                            //console.log(includeQuotes)
                            div.append('g').append('text').attr('class', 'transcript-holder').append("tspan")
                            .style("white-space", "pre-line")
                            .attr("class", "transcript-text")
                            .text(_.map(includeQuotes,"text").join("... ").replace("... ... ", "... ").replace("... ... ", "... ").replace("... ... ", "... "))

                            //div.append("br")
                            var svg = div.append("g").attr("class", "svg-holder").append("g")
                            .append("g")
                            .attr("transform", "translate("+margin.left+","+margin.top+")")
                           //div.append("br")

                           div.append("g").append("text")
                           .attr("x", 0)
                           .attr("y", 0)
                            //.attr("height", "100px")
                            //.append("tspan")
                            .attr("class", "ann-head")
                            .text("[" + vid.replace("private/P","Participant ") + ", " + d.timeline.replace("1", "").replace("Timeline", "") + " Timeline]: " + timestr)

                            div.append("g").append("text")
                            .attr("x", 0)
                            .attr("y", 100)
                            .append("tspan")
                            .attr("class", "ann-text")
                            .text(d.annotation)
                            //console.log(d)



                            if(d.timeline == "Emotion"){
                                var data = _.reject(_.filter(emotiondata, function(o){
                                    return( (parseFloat(o.start)/emotionfps) <= endTime & (parseFloat(o.end)/emotionfps) >= startTime)
                                }), {'emotion':"N/A"})

                                //just take a look for now; suppress this in prod
                                //div.append("span")
                                //.text(JSON.stringify(data))

                                var maxEnd = _.max(_.map(data, function(dp){
                                    return(1*dp['end'])
                                }));
                                
                                var rectStart = _.min(_.map(data, function(dp){
                                    return(1*dp['start'])
                                }));



                                var bands = _.map(data, function(o){return((1*o.start)/emotionfps)})

                                bands.push.apply(bands, _.map(data, function(o){return((1*o.end)/emotionfps)}));

                                bands = _.uniq(bands).sort()

                                bands.push(1+bands[(bands.length-1)])
                                bands.unshift(bands[0]-1)

                                var x = d3.scaleBand()
                                    .range([0, width])
                                    .padding(0.1);

                                var maxProb = _.max(_.map(_.reject(data,{'emotion':"N/A"}), function(dp){return(1*dp['prob'])}));

                                var y = d3.scaleLinear()
                                    .domain([0, maxProb])
                                    .range([height, 0]);


                                function rectWidth(lowerVal, upperVal){
                                        gap = upperVal-lowerVal;
                                        rangeMult = (width/maxEnd)
                                        return (gap * rangeMult)
                                }

                                var allStarts = _.map(data, 'start')
                                var startFrames = _.uniq(allStarts)

                                x.domain(bands)



                                var tickvals = [];

                                _.map(x.domain(), function(d,i){

                                    if(!(i % 3)){
                                        var minute = Math.floor((1*d)/60)
                                    var second = Math.round((1*d) - 60*minute)
                                    var minstr = minute < 10 ? "0" + minute.toString() : minute.toString();
                                    var secstr = second < 10 ? "0" + second.toString() : second.toString();

                                    var timestr = minstr + ":" + secstr
                                    tickvals.push(timestr) 

                                    } else {
                                        tickvals.push("")
                                    }

                                })

                                var xAxis = svg.append("g")
                                .attr("transform", "translate(0," + height + ")")
                                .call(d3.axisBottom(x)
                                .tickValues(tickvals)

                                )
                                .attr("class","xaxis")
                                .attr("fill", "#000")
                                //.attr("transform", "rotate(90)")
                                .attr("x", 6)
                                .attr("dx", "0.71em")
                                .attr("text-anchor", "end")

                                svg.append("g")
                                .call(d3.axisLeft(y))
                                .append("text")
                                .attr("fill", "#000")
                                .attr("transform", "rotate(-90)")
                                .attr("y", 6)
                                .attr("dy", "0.71em")
                                .attr("text-anchor", "end")
                                .text("Probability");

                                svg.selectAll(".bar")
                                .data(data)
                                .enter().append("rect")
                                .attr("class", "bar")
                                .attr('fill', function(c){return(emoColorScale(c.emotion))})
                                .attr("x", function (c) {
                                    return x((1*c.start)/emotionfps);
                                })
                                .attr("y", function (c) {
                                    return y(1*c.prob);
                                })
                                .attr("width", x.bandwidth())
                                .attr("height", function (c) {
                                    return (height - y(1*c.prob));
                                })

                                xAxis.selectAll(".tick").transition().duration(40)
                                //console.log([d,i])
                                .attr("transform", function(d,i){
                                    return "translate(" + (i*width/data.length) + ",0)"
                                })

                            }

                            if(d.timeline == "Action1"){
                                var data = _.reject(_.filter(actiondata, function(o){
                                    return( (parseFloat(o.start)/actionfps) <= endTime & (parseFloat(o.end)/actionfps) >= startTime)
                                }), {'emotion':"N/A"})

                                //just take a look for now; supprses in prod
                                //div.append("span")
                                //.text(JSON.stringify(data))

                                var maxEnd = _.max(_.map(data, function(dp){
                                    return(1*dp['end'])
                                }));
                                
                                var rectStart = _.min(_.map(data, function(dp){
                                    return(1*dp['start'])
                                }));



                                var bands = _.map(data, function(o){return((1*o.start)/actionfps)})

                                bands.push.apply(bands, _.map(data, function(o){return((1*o.end)/actionfps)}));

                                bands = _.uniq(bands).sort()

                                bands.push(1+bands[(bands.length-1)])
                                bands.unshift(bands[0]-1)

                                var x = d3.scaleBand()
                                    .range([0, width])
                                    .padding(0.1);

                                var maxProb = _.max(_.map(_.reject(data,{'action':"N/A"}), function(dp){return(1*dp['prob'])}));

                                var y = d3.scaleLinear()
                                    .domain([0, maxProb])
                                    .range([height, 0]);


                                function rectWidth(lowerVal, upperVal){
                                        gap = upperVal-lowerVal;
                                        rangeMult = (width/maxEnd)
                                        return (gap * rangeMult)
                                }

                                var allStarts = _.map(data, 'start')
                                var startFrames = _.uniq(allStarts)

                                x.domain(bands)



                                var tickvals = [];

                                _.map(x.domain(), function(d,i){

                                    if(!(i % 3)){
                                        var minute = Math.floor((1*d)/60)
                                    var second = Math.round((1*d) - 60*minute)
                                    var minstr = minute < 10 ? "0" + minute.toString() : minute.toString();
                                    var secstr = second < 10 ? "0" + second.toString() : second.toString();

                                    var timestr = minstr + ":" + secstr
                                    tickvals.push(timestr) 

                                    } else {
                                        tickvals.push("")
                                    }

                                })

                                var xAxis = svg.append("g")
                                .attr("transform", "translate(0," + height + ")")
                                .call(d3.axisBottom(x)
                                .tickValues(tickvals)

                                )
                                .attr("class","xaxis")
                                .attr("fill", "#000")
                                //.attr("transform", "rotate(90)")
                                .attr("x", 6)
                                .attr("dx", "0.71em")
                                .attr("text-anchor", "end")

                                svg.append("g")
                                .call(d3.axisLeft(y))
                                .append("text")
                                .attr("fill", "#000")
                                .attr("transform", "rotate(-90)")
                                .attr("y", 6)
                                .attr("dy", "0.71em")
                                .attr("text-anchor", "end")
                                .text("Probability");

                                svg.selectAll(".bar")
                                .data(data)
                                .enter().append("rect")
                                .attr("class", "bar")
                                .attr('fill', function(c){return(actColorScale(c.action))})
                                .attr("x", function (c) {
                                    return x((1*c.start)/actionfps);
                                })
                                .attr("y", function (c) {
                                    return y(1*c.prob);
                                })
                                .attr("width", x.bandwidth())
                                .attr("height", function (c) {
                                    return (height - y(1*c.prob));
                                })

                                xAxis.selectAll(".tick").transition().duration(40)
                                //console.log([d,i])
                                .attr("transform", function(d,i){
                                    return "translate(" + (i*width/data.length) + ",0)"
                                })

                                
                            }
                        })

                    })
                    //.error(function(error) { console.log(error); });

                })
                //.error(function(error) { console.log(error); });
            }); 
        });

    })
    //.error(function(error) { console.log(error); });
})
})



const svgToPdfExample = (svg) => {
const doc = new window.PDFDocument();
const chunks = [];
const stream = doc.pipe({
// writable stream implementation
write: (chunk) => chunks.push(chunk),
end: () => {
  const pdfBlob = new Blob(chunks, {
    type: 'application/octet-stream'
  });
  var blobUrl = URL.createObjectURL(pdfBlob);
  window.open(blobUrl);
},
// readable streaaam stub iplementation
on: (event, action) => {},
once: (...args) => {},
emit: (...args) => {},
});

window.SVGtoPDF(doc, svg, 0, 0);

doc.end();
};

function printToPDF(){
//pdfkit approach fails
const pageContents = document.documentElement.outerHTML;
//svgToPdfExample(pageContents);

//var pdf = new jsPDF('p', 'pt', 'letter');
var pdf = new jsPDF();
pdf.fromHTML($('#main'))
pdf.save('Test.pdf');



}

//setTimeout("printToPDF()", 10000)

