var uxvideo = document.getElementById("video_html5_api");

// The annotations box looks good and should be what we use when the user opts to add annotation to timeline
function appendAnnotations() {
    refreshuxSDimVars()

    var annotDiv = d3.select('body').append('div')
    .attr('id', 'AnnotationTooltip')
    .attr('class', 'annotations ui-widget-content')
    .style('max-width', '20vw')
    .style('max-height', '20vh')
    .style('width', 'auto')
    .style('height', 'auto')
    .style('display', 'none'); //we will change this when called
    
    //make it draggable
    $( function() {
        $( "#AnnotationTooltip" ).draggable();
      } );
      
    var annotBox = annotDiv.append('div')
        .attr('class', 'annotations-box')
        .style('width', 'auto')
        .style('height', 'auto')

    var annotForm = annotBox.append('form')
    .attr('accept-charset', 'UTF-8')
    .attr('autocomplete', 'off')
    .attr('display', 'flex')
    .style('max-width', '20vw')

    var annotFieldSet = annotForm.append('fieldset')
    .style('max-width', '20vw')
    .style('min-width', '20vw')

    var annotLegend = annotFieldSet.append('legend')
        .text('Annotation');

    annotFieldSet.append('label')
        .attr('for', 'name')
        .text('Notes');

    annotFieldSet.append('textarea')
        .attr('id', 'annotation-text')
        .style('max-width', '18vw')

    annotFieldSet.append('br');

    var annotSubmitButton = annotFieldSet.append('input')
        .attr('id', 'annotate-submit')
        .attr('class', 'btn')
        .attr('type', 'button')
        .attr('value', 'Submit');
        //.on('click', addAnnotation) //we will add and change this when called

    //also add cancel button
    annotFieldSet.append('input')
        .attr('id', 'annotate-cancel')
        .attr('class', 'btn')
        .attr('type', 'button')
        .style('margin-left', '20px')
        .style('background-color', 'lightslategrey')
        .attr('value', 'Cancel')
        .on('click', function(){
            annotDiv.style('display', 'none')

            interactiontracking('cancel annotate', 'annotationTooltip', 'annotate-cancel', 'click')

        }) //we will add and change this when called

    //Now, for each timeline, add icons
    var addannbtn = d3.select('.timelines-box').selectAll('svg')
    .append('g')
    .attr('class', 'add-annotation-button')
    .attr('transform', 'translate('+(width + margin.left + annotbtnXPad)+','+(margin.top + annotbtnYPad)+')');


    /*
    var annPointBtn = addannbtn.append('circle')
    .attr('r', '10')
    .attr('cx', 0)
    .attr('fill', 'slategrey')
  
    var annIntrvBtn = addannbtn.append('circle')
    .attr('r', '10')
    .attr('cy', 25)
    .attr('fill', 'slategrey')
    */

    var annPointBtn = addannbtn.append('g').attr('class', 'add-annotation-button')
    
    annPointBtn
    .append('rect')
    .attr('class', 'annbtnfill')
    .attr('width', 20)
    .attr('height', 20)
    .attr('rx', 3)
    .attr('x', -1)
    .attr('y', -20)
    .attr('fill', 'slategrey');

    annPointBtn
    .append('text')
    .attr('style', 'font:bold 30px sans-serif; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none; fill:white')
    .text('+')
  

    annPointBtn.on('mouseover', annMouseOver)
    annPointBtn.on('mouseout', annMouseOut)
    annPointBtn.on('click', function(){
        //annMouseClick('Point')
        annMouseClick(this.parentNode.parentNode.parentNode.getAttribute("id"),'Point')

    })


    var annIntrvlBtn = addannbtn.append('g').attr('class', 'add-annotation-button')
    
    annIntrvlBtn
    .append('rect')
    .attr('class', 'annbtnfill')
    .attr('width', 20)
    .attr('height', 20)
    .attr('rx', 3)
    .attr('x', -1)
    .attr('y', 15)
    .attr('fill', 'slategrey');

    annIntrvlBtn
    .append('text')
    .attr('x', 1)
    .attr('y', 28)
    .attr('style', 'font:bold 12px sans-serif; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none; fill:white')
    .text('[..]')
  

    annIntrvlBtn.on('mouseover', annMouseOver)
    annIntrvlBtn.on('mouseout', annMouseOut)
    annIntrvlBtn.on('click', function(){
        annMouseClick(this.parentNode.parentNode.parentNode.getAttribute("id"),'Interval')
    })


    function annMouseOver(){
        d3.select(this).select('.annbtnfill')
        .attr('fill', 'gold')

        d3.select(this.nextSibling)
        .select('text')
        .attr('x', -35)
        .text('Point Note')

        d3.select(this.nextSibling)
        .select('rect')
        .attr('width', 80)
        .attr('height', 35)
        .attr('x', -45)
        .attr('y', 5)

        d3.select(this.previousSibling)
        .select('text')
        .attr('style', 'font:bold 12px sans-serif; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none; fill:white')
        .attr('x', -50)
        .text('Interval Note')

        d3.select(this.previousSibling)
        .select('rect')
        .attr('width', 100)
        .attr('height', 35)
        .attr('x', -60)

        interactiontracking('call annotation tooltip button label show', this.parentNode.parentNode.parentNode.getAttribute("id"), 'add-annotation-button', 'mouseover')

    }

    function annMouseOut(){
        d3.select(this).select('.annbtnfill')
        .attr('fill', 'slategrey')

        d3.select(this.nextSibling).
        select('text')
        .attr('x', 1)
        .text('[..]')

        d3.select(this.nextSibling)
        .select('rect')
        .attr('x', -1)
        .attr('width', 20)
        .attr('height', 20)
        .attr('y', 15)

        d3.select(this.previousSibling)
        .select('text')
        .attr('x', 0)
        .attr('style', 'font:bold 30px sans-serif; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none; fill:white')
        .text('+')

        d3.select(this.previousSibling)
        .select('rect')
        .attr('x', -1)
        .attr('width', 20)
        .attr('height', 20)

        interactiontracking('call annotation tooltip button label hide', this.parentNode.parentNode.parentNode.getAttribute("id"), 'add-annotation-button', 'mouseout')

    }

    function annMouseClick(timelineID, annotType){
        d3.select('#AnnotationTooltip')
        .style("transform", "translate(70vw," + (100*(d3.event.pageY - (50 + window.innerHeight))/window.innerHeight) + "vh)")   
        .style('display', 'flex')

        //focus on the text entry field
        document.getElementById("annotation-text").focus();

        annotLegend.text(annotType + " Annotation")

        if(annotType == 'Interval'){
            annotSubmitButton.on('click', function(){addIntervalAnnotation(timelineID)})
        } else {
            annotSubmitButton.on('click', function(){addPointAnnotation(timelineID)})
        }

        
    }

}

/**
 * .transition().duration(100)
.style("left", (d3.event.pageX) + "px")
.style("top", (d3.event.pageY - 28) + "px")   
.style('opacity', 1)


 */

uxvideo.addEventListener('loadeddata', function(){
    setTimeout('appendAnnotations()', 1600)
})

