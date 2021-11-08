import './App.css';
import React from 'react';
import VideoPlayer from './Components/interact/VideoPlayer';
import Timelines from './Components/timelines/Timelines';
import SearchBar from './Components/interact/SearchBar';
import FilterBox from './Components/interact/FilterBox';
import AnnotationsBox from './Components/annotation/AnnotationsBox';
import TagsTable from './Components/annotation/TagsTable';

const playerRef = React.useRef(null);

const videoJsOptions = { // lookup the options in the docs for more options
  autoplay: true,
  controls: true,
  responsive: true,
  fluid: true,
  sources: [{
    src: '1-p6-coffeemachine.mp4',
    type: 'video/mp4'
  }]
}

const handlePlayerReady = (player) => {
  playerRef.current = player;

  // you can handle player events here
  player.on('waiting', () => {
    console.log('player is waiting');
  });

  player.on('dispose', () => {
    console.log('player will dispose');
  });
};

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

function App() {
  return (<div className="App">
      <div class="grid-container">
        <div class="left">
            <div class="menu">
                <div class="menu-box">
                    <nav style="background-color: darkolivegreen">
                        <div class="nav-wrapper">
                            <a href="#" class="brand-logo center" style="color:white">
                                <bold>uxSense</bold>
                            </a>
                        </div>
                    </nav>
                </div>
            </div>
            <div class="video">
                <div class="video-box">
                <VideoPlayer options={videoJsOptions} onReady={handlePlayerReady} />
                </div>
            </div>
            <div class="timelines">
                <div class="timelines-box">
                  <Timelines />
                </div>
            </div>
        </div>
        <div class="right">
            <div class="search">
              <SearchBar />
            </div>
            <div class="filter">
              <FilterBox />
            </div>
            <div class="annotations">
              <AnnotationsBox />
            </div>
            <div class="tags">
              <TagsTable />
            </div>
        </div>
    </div>
    </div>
  );
}

export default App;
