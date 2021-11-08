    var filtersliderstates = [];
    var filterchecks = []

    //Track state for relevant dom objects
function statelog(){//, d3Event){
    var filtersliders = document.getElementsByClassName("filter-slider")
    var filters = document.getElementsByClassName("filters")
    filtersliderstates=[]
    filterchecks=[]

    for(i=0;i<filtersliders.length;i++){
        var thissliderrect = filtersliders[i].getElementsByClassName("selection")[0]
        filtersliderstates.push(JSON.stringify(thissliderrect.innerHTML))
    } 

    for(i=0;i<filters.length;i++){
        filterchecks.push(JSON.stringify(filters[i].innerHTML))
    }

    setTimeout("poststate()", 100)
    
    setTimeout("statelog()", 5000)
}

function poststate(){
    $.ajax({ 
        url: '/statelog',
        type: 'POST',
        cache: false, 
        data: {
            vidtime:document.getElementById('video_html5_api').currentTime, 
            vidbrushstate:JSON.stringify(document.getElementById("focussvg").getElementsByClassName("selection")[0].innerHTML),
            checkfilters:filterchecks,
            brushfilters:filtersliderstates
        }, 
        success: function(){
            //alert('Success!')
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


statelog();