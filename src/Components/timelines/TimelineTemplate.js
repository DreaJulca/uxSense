import React from "react";
import ActionsTimeline from "./ActionsTimeline";
import EmotionsTimeline from "./EmotionsTimeline";

export default function TimelineTemplate(props){
    let timeline = <></>
    switch(props.Type){
        case 'action':
            timeline = <ActionsTimeline videoName = {props.videoName} />
            break;
        case 'emotion':
            timeline = <EmotionsTimeline videoName = {props.videoName} />
            break;
        default:
            timeline = <div>{props.Type+" timeline not available."}</div>
    }

    return <div className="timeline">
        <div className="timeline-label">{props.Type}</div>
        {timeline}
    </div>
}