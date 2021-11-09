// The timelines don't actually exist as components yet; may need to start from scratch for some of this stuff.
//import ActionTimeline from './actions';
import React from 'react';
import TimelineTemplate from './TimelineTemplate';

export default function Timelines( props ) {
    var rows = [];
    //probably need to make it so that there's more to this timeline array than just types
    // but for now it's fine
    //TODO: We're going to add a selector for the specific video somewhere; 
    // right now, we're just choosing the first videoname in the list here
    for (var i = 0; i < props.timelineArray.length; i++) {
        rows.push(<TimelineTemplate 
            key={i} 
            videoName={props.videoArray[0]} 
            Type={props.timelineArray[i].Type}/>);
    }

    return(<div>{rows}</div>
    )
}