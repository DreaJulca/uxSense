var video = document.getElementById('video_html5_api');

// set the dimensions and margins of the graph
var margin = { top: 10, right: 50, bottom: 10, left: 50 },
    // width = 460 - margin.left - margin.right,
    // height = 400 - margin.top - margin.bottom;
    height = 100 - margin.top - margin.bottom;

//windowsize fix
var margin = { top: 10, right: 50, bottom: 10, left: 50 }




// append the svg object to the body of the page
var svg2 = d3.select("#pitch")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

//Read the data
d3.csv('modeloutput/TableauUser_Pitch_Preprocessed.csv', function (data) {

    var maxEnd = _.max(_.map(data, function(dp){return(1*dp['x'])}));

    // Add X axis --> it is a date format
    var x = d3.scaleLinear()
        .domain([1, maxEnd])
        .range([0, width]);
    svg2.append("g")
        .attr("transform", "translate(0," + height + ")");
    // .call(d3.axisBottom(x));

    // Add Y axis
    var y = d3.scaleLinear()
        .domain([0, 200])
        .range([height, 0]);
    // svg.append("g")
    //     .call(d3.axisLeft(y));

    // This allows to find the closest X index of the mouse:
    var bisect = d3.bisector(function (d) { return d.x; }).left;

    // Create a rect on top of the svg area: this rectangle recovers mouse position
    svg2
        .append('rect')
        .attr('id', 'pitchmouserect')
        .style("fill", "none")
        .style("pointer-events", "all")
        .attr('width', width)
        .attr('height', height)
        .on('mouseover', mouseover)
        .on('mousemove', mousemove)
        .on('mouseout', mouseout)
        .on('click', mouseclick)

    // Add the line
    svg2
        .append("path")
        .datum(data)
        .attr("id", "pitchlinepath")
        .attr("origdata", JSON.stringify(data))
        .attr("maxEnd", maxEnd)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("d", d3.line()
            .x(function (d) { return x(d.x) })
            .y(function (d) { return y(d.y) })
        )


    // do these after creating the chart path
    // Create the circle that travels along the curve of chart
    var focus = svg2
        .append('g')
        .append('circle')
        .attr('id', 'pitchfocuscircle')
        .style("fill", "none")
        .attr("stroke", "black")
        .attr('r', 8.5)
        .style("opacity", 0)

    // Create bg for the text that travels along the curve of chart
    var focusTextRect = svg2
        .append('g')
        .append('rect')
        .attr('id', 'pitchfocustextbg')
        .style("opacity", 0)
        .attr("width", "110px")
        .attr("height", "25px")
        .attr("fill", "lightgrey")

    var focusText = svg2
        .append('g')
        .append('text')
        .attr('id', 'pitchfocustext')
        .style('text-shadow', '1px 1px 5px white') 
        .style("opacity", 0)
        .attr("text-anchor", "left")
        .attr("alignment-baseline", "middle")


    // What happens when the mouse move -> show the annotations at the right positions.
    function mouseover() {
        focus.style("opacity", 1)
        focusTextRect.style("opacity", 0.75)
        focusText.style("opacity", 1)
    }

    function mousemove() {
        // recover coordinate we need
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]
        var minutes = Math.floor((video.duration * selectedData.x/maxEnd)/60)
        var seconds = Math.round(60 * (((video.duration * selectedData.x/maxEnd)/60) - minutes))
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
    function mouseout() {
        focus.style("opacity", 0)
        focusTextRect.style("opacity", 0)
        focusText.style("opacity", 0)
    }

    function mouseclick(){
        //get a new maxEnd--paths handle focus differently, so we need to do this.
        var x0 = x.invert(d3.mouse(this)[0]);
        var i = bisect(data, x0, 1);
        selectedData = data[i]

        video.currentTime = video.duration * selectedData.x/maxEnd

    }


})