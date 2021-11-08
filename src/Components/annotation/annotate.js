function addPointAnnotation(timelineID){
    var annotation = document.getElementById('annotation-text');

    interactiontracking(annotation.value, timelineID, 'annotate-submit', 'click', [{annotationtype: "point"}])


    var focussvg = d3.select('#focussvg').select('g')
    var focusBrush = focussvg.select('g.x.brush')
    var selrect = focusBrush.select('rect.selection')

    var selwid = parseFloat(selrect.attr('width'))
    var selX = parseFloat(selrect.attr('x'))

    var isFocused=true;

    if(isNaN(selwid)){
        selwid = width;
        selX = 1;
        isFocused=false;
    }
    
    var minTime = uxvideo.duration * selX/width
    var maxTime = uxvideo.duration * ( selX + selwid )/width

    $.ajaxSetup({
        timeout: 3000 //Time in milliseconds
    });
    //send annotation to server
    $.ajax({ 
        url: '/annotate'
        , type: 'POST'
        , cache: false
        , data: { annotation: annotation.value, timestamp:document.getElementById("video_html5_api").currentTime, posttime:(new Date().getTime()), timeline:timelineID, annotationtype:"point", annotatedintervalmin:minTime, annotatedintervalmax:maxTime, focusbrushed:isFocused, videoname:uxSenseVideoPath}
        , callback: function(response){
            console.log(response)
        } 
        , success: function(data){
            document.getElementById("annotation-text").value = null
            annotationQueryRefresh();
            d3.select('#AnnotationTooltip').style('display', 'none')
        }
        , error: function(jqXHR, textStatus, err){
            console.log('text status '+textStatus+', err '+err)
        }
    }).fail(function (jqXHR, textStatus, errorThrown) {    
        console.log(jqXHR);
        console.log(textStatus);
        console.log(errorThrown);
    });
}

function addIntervalAnnotation(timelineID){
    var annotation = document.getElementById('annotation-text');

    interactiontracking(annotation.value, timelineID, 'annotate-submit', 'click', [{annotationtype: "interval"}])


    var focussvg = d3.select('#focussvg').select('g')
    var focusBrush = focussvg.select('g.x.brush')
    var selrect = focusBrush.select('rect.selection')

    var selwid = parseFloat(selrect.attr('width'))
    var selX = parseFloat(selrect.attr('x'))

    var isFocused=true;

    if(isNaN(selwid)){
        selwid = width;
        selX = 1;
        isFocused=false;
    }

    var minTime = uxvideo.duration * selX/width
    var maxTime = uxvideo.duration * ( selX + selwid )/width
    d3.select('#AnnotationTooltip').style('display', 'none')

    $.ajaxSetup({
        timeout: 3000 //Time in milliseconds
    });
    //send annotation to server
    $.ajax({ 
        url: '/annotate'
        , type: 'POST'
        , cache: false
        , data: { annotation: annotation.value, timestamp:document.getElementById("video_html5_api").currentTime, posttime:(new Date().getTime()), timeline:timelineID, annotationtype:"interval", annotatedintervalmin:minTime, annotatedintervalmax:maxTime, focusbrushed:isFocused, videoname:uxSenseVideoPath}
        , callback: function(response){
            console.log(response)
        } 
        , success: function(data){
            document.getElementById("annotation-text").value = null
            annotationQueryRefresh();
        }
        , error: function(jqXHR, textStatus, err){
            console.log('text status '+textStatus+', err '+err)
        }
    }).fail(function (jqXHR, textStatus, errorThrown) {    
        console.log(jqXHR);
        console.log(textStatus);
       console.log(errorThrown);
    });
}

