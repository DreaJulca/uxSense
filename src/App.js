import './App.css';
import React, {useRef} from 'react';
import VideoPlayer from './Components/interact/VideoPlayer';
import Timelines from './Components/timelines/Timelines';
import SearchBar from './Components/interact/SearchBar';
import FilterBox from './Components/interact/FilterBox';
import AnnotationsBox from './Components/annotation/AnnotationsBox';
import TagsTable from './Components/annotation/TagsTable';

/*
var video = document.getElementById("video_html5_api");

var vidTimelineInterval = 30;

function vidTimelineRecur(i, duration){

    if(i < duration){
        video.currentTime = i;
        generateThumbnail(i);
        setTimeout(vidTimelineRecur(i+vidTimelineInterval, duration), 500)
    }
    else return;
}
function generateThumbnail(i) {
        var canvas = document.getElementById('myCanvas');
        var context = canvas.getContext('2d');
        // context.drawImage(video, (i / vidTimelineInterval) * 10, 0, 100, 50);
        context.drawImage(video, i, 0, 100, 50);
        var dataURL = canvas.toDataURL();

        var img = document.createElement('img');
        img.setAttribute('src', dataURL);

        document.getElementById('thumbnails').appendChild(img);
}
video.addEventListener('loadeddata', function () {
        var duration = video.duration;
        vidTimelineRecur(0, duration)
        video.currentTime = 1;
});
*/

//for now, manually define video to show; update this later--will need to
// turn into list and change the video renderer s.t. it loops through each.
// Right now, we're passing this to everything EXCEPT video as an array
// and in each of those components, it is just selecting the first one.
// TODO: change this once you add interface features to select vids
var VideoArray = ['TableauUser.mp4']

//for now, manually define list of timelines; update this later
var TimelineArray = [
  {Type: "action"},
  {Type: "emotion"},
  {Type: "speech"},
  {Type: "pitch"},
  {Type: "frames"},
  {Type: "annotations"}
]

function App() {
  return (<div className="App">
      <div className="grid-container">
            <div className="banner">
                <div className="menu-box">
                    <nav style={{'backgroundColor': 'darkolivegreen'}}>
                        <div className="nav-wrapper">
                            <a href="#" className="brand-logo center" style={{color:'white'}}>
                                uxSense
                            </a>
                        </div>
                    </nav>
                    <nav style={{'backgroundColor': 'black'}}>
                        <div className="panel-select">
                            <a href="#" className="analysis-view" style={{color:'white'}}>
                                Analysis
                            </a>
                            <a href="#" className="report-view" style={{color:'white'}}>
                                Report
                            </a>
                        </div>
                    </nav>
                </div>
            </div>
            <div className="video">
                <VideoPlayer videoPath={VideoArray[0]} />
            </div>
            <div className="transcript">
            </div>
            <div className="tagsTab">
              <TagsTable />
            </div>
            <div className="timelines">
              <Timelines videoArray={VideoArray} timelineArray={TimelineArray}/>
            </div>
        </div>
        <div className="search">
          <SearchBar />
        </div>
        <div className="filter">
          <FilterBox />
        </div>
        <div className="annotations">
          <AnnotationsBox />
        </div>
      </div>
  );
}

export default App;
