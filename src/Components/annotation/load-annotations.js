 /**
  * Need to continuously check if annotations table has been updated.
  */

  function annotationQueryRefresh(){
    $.ajax({ 
        url: '/annotationQuery'
        , type: 'GET'
        , cache: true
        , data: { }
        , callback: function(response){
            console.log(response)
        } 
        , success: function(data){
            annotationTabPop(false);
        }
        , error: function(jqXHR, textStatus, err){
            console.log('text status '+textStatus+', err '+err)
        }
    })

  }


  function annotationTabPop(redo=true){
    try{
    $.ajaxSetup({
        timeout: 2400 //Time in milliseconds
    });
    d3.json('userAnnotations/data.json', function(rawdata){
        //TODO: CHANGE THIS LINE BELOW SO THAT WE'RE NOT JUST PULLING ALL DATA; WANT TO MAKE SERVER-SIDE QUERY HERE
        var filtannotations = _.filter(rawdata, {'videoname':uxSenseVideoPath})
        var tdata = _.sortBy(filtannotations, [function(o) { return parseFloat(o.timestamp); }])
        var tbody = d3.select('#annotation-table')
        .select('tbody')
        .data(tdata)

        try{
            tbody.selectAll('.annotrow').remove();
        } catch(err){
            console.log(err);
        }
        
        tdata.forEach(function(d){
            var row = tbody.append('tr').datum(d).attr('class', 'annotrow')
            
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

            row.append('td')
            .attr('style', 'width:20%')
            .text(timestr)
    
            row.append('td')
            .attr('style', 'width:80%')
            .text(d.annotation)        
    
        })  
        //have to wrap this in a try because it may not be defined yet.
        try{
            if(JSON.parse(d3.select('#annotationsgraphicmain').attr('origdata')).length == rawdata.length){
                //console.log('data unchanged')
            } else {
                //console.log(JSON.parse(d3.select('#annotationsgraphicmain').attr('origdata')))
                //console.log(rawdata)
                createAnnotationsTimeline();    
            }
        } catch(err){
            //Do nothing.
            //console.log(err)
        }

    })

    } catch(err){
        console.log(err)
    }
    if(redo){
    //Sometimes success times out but the db is actually updated. This is a band-aid for that--we make it recursively update every four seconds.
        setTimeout("annotationTabPop()", 3000)

    }

}

annotationQueryRefresh();