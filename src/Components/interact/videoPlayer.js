import React from "react";
import VideoManager from './VideoManager' 

export default function VideoPlayer(props){
  const playerRef = React.useRef(null);

  const videoJsOptions = { 
    autoplay: true,
    controls: true,
    responsive: true,
    fluid: true,
    sources: [{
      src: props.videoPath,
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

  // const changePlayerOptions = () => {
  //   // you can update the player through the Video.js player instance
  //   if (!playerRef.current) {
  //     return;
  //   }
  //   // [update player through instance's api]
  //   playerRef.current.src([{src: 'http://ex.com/video.mp4', type: 'video/mp4'}]);
  //   playerRef.current.autoplay(false);
  // };

  return (
    <>
      <VideoManager options={videoJsOptions} onReady={handlePlayerReady} />
    </>
  );
}

