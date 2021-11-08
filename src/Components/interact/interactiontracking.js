import {$,jQuery} from 'jquery';
import * as d3 from 'd3';

export default function interactiontracking(d, eventUIElement, eventObject, eventTrigger, eventSupplemental = ''){//, d3Event){
    var uxvideo = document.getElementById('video_html5_api');

    //Track event
    $.ajax({ 
        url: '/log',
        type: 'POST',
        cache: false, 
        data: { 
            data: d, 
            event: {
                main:JSON.stringify(d3.event),
                meta: {
                    uielement: eventUIElement, 
                    object: eventObject, 
                    trigger: eventTrigger, 
                    timestamp:(new Date().getTime()), 
                    videotime:uxvideo.currentTime, 
                    supplemental:eventSupplemental,
                    videoname:uxSenseVideoPath
                }
            }
        }, 
        success: function(){
            //alert('Success!')
        }
        , error: function(jqXHR, textStatus, err){
            console.log('text status '+textStatus+', err '+err)
        }
    })

}
