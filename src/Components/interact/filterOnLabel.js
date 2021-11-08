function filterOnLabel(containerID){

    var svg = d3.select("#" + containerID).select('svg')

    var g = svg.select("g")

    var rectclass; 
    var container = document.getElementById(containerID).parentElement.parentElement;
    var collapseheader;
    var filtercontainer;
    var allfilters =[];
    var checkedfilters =[];
    var attrName;


    if(containerID == 'Emotion'){
        g = svg.select('#emotionrects')
        rectclass = 'emotionrect'
        attrName = 'emotion'
    }

    if(containerID == 'Action1'){
        g = svg.select('#action1rects')
        rectclass = 'action1rect'
        attrName = 'action'
    }


    var containerChecks = d3.select(container).selectAll('.filters')//.selectAll('input');
    var containerInputs = containerChecks.selectAll('input');
    containerInputs.each(function(){
        var thisLabel = {value:d3.select(this.parentElement).select('span').text()}
        allfilters.push(thisLabel)
        if(this.checked){
            checkedfilters.push(thisLabel)
        }
    })
    //var containerSpans = containerChecks.selectAll('span')//.selectAll('input');
    


    /*
    for (var i = 0; i < container.childNodes.length; i++) {
        if (container.childNodes[i].className == "collapsible-header") {
            collapseheader = container.childNodes[i];
            //console.log(collapseheader)
            //for(var j = 0; j < collapseheader.childNodes.length; j++){
            //var filtercontainer = d3.select(collapseheader).selectAll('.filters')._groups[0];
            var filtercontainer = d3.select(collapseheader).selectAll('.filters');
            //console.log(filtercontainer)


                //if(collapseheader.childNodes[j].className == "filters"){
                    //filtercontainer = collapseheader.childNodes[j];
                    //var filters = filtercontainer.getElementsByTagName('label');
                    var filters = filtercontainer.selectAll('label')._groups[0];
                    //console.log(filters)
                    for(var k = 0; k < filters.length; k++){
                        //console.log(container)
                        var inputfield = filters[k].childNodes[1];
                        var spanfield = filters[k].childNodes[3].firstChild.nodeValue;
                        allfilters.push({[attrName]:spanfield});
                        if (inputfield.checked) {
                            checkedfilters.push({[attrName]:spanfield})
                        }
                    }
                    break;
                //}
            //}
            break;
        }        
    }
    */

    if(checkedfilters.length>0){
        showFilters(checkedfilters, rectclass, containerID, g)
    } else {
        showFilters(allfilters, rectclass, containerID, g)
    }
}

function showFilters(checkedfilters, rectclass, containerID, g){
    var dataIndex = {Action1:'action', Emotion:'emotion'}

    g.selectAll('.'+rectclass).each(function(d){
        if(_.filter(checkedfilters, {value:d[dataIndex[containerID]]}).length > 0){
            d3.select(this).attr('opacity', 1)
        } else {
            d3.select(this).attr('opacity', 0.3)
        }
    })    
    
    interactiontracking(checkedfilters, containerID, rectclass, 'click', [{hasquerytype:"filterOnLabel"}])
    
}    


//setTimeout('filterOnLabel("Emotion")', 1000)
//setTimeout('filterOnLabel("Action1")', 1000)

d3.selectAll('.collapsible-header').selectAll('.filters').selectAll('span').on('click', function(){
    var thisElem = d3.select(this.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement);
    console.log(thisElem)
    var thiscontainerID = thisElem.selectAll('.timelineholder').attr('id');
    console.log(thiscontainerID)

    //console.log(thisElem);
    //console.log(thiscontainerID);
    setTimeout("filterOnLabel('"+thiscontainerID+"')", 50);
})