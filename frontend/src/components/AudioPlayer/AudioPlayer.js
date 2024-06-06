import React, { useEffect, useRef, useState } from "react";

// AudioPlayer component to play audio files with controls and display the current audio name being played or paused
const AudioPlayer = ({
  audioFile,
  onAudioEnd,
  onTimeUpdate,
  audioRef,
  currentAudioName,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);

  // Add event listeners to the audio element to handle audio end, time update, play, and pause events
  useEffect(() => {
    // Add event listeners to the audio element
    if (audioRef.current) {
      audioRef.current.addEventListener("ended", onAudioEnd);
      audioRef.current.addEventListener("timeupdate", () => {
        onTimeUpdate(audioRef.current.currentTime);
      });
      // Add event listeners for play and pause events
      audioRef.current.addEventListener("play", () => {
        setIsPlaying(true);
      });
      // Add event listeners for play and pause events
      audioRef.current.addEventListener("pause", () => {
        setIsPlaying(false);
      });
      // Remove event listeners when the component is unmounted
      return () => {
        audioRef.current.removeEventListener("ended", onAudioEnd);
        audioRef.current.removeEventListener("timeupdate", () => {
          onTimeUpdate(audioRef.current.currentTime);
        });
      };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioRef, onAudioEnd, onTimeUpdate]);

  // Load the audio file and play it when it's available
  useEffect(() => {
    const loadAudio = async () => {
      // Load the audio file and play it when it's available
      if (audioFile && audioRef.current) {
        const blob = new Blob([audioFile], { type: audioFile.type });
        audioRef.current.src = URL.createObjectURL(blob);
        audioRef.current.load(); // Load the audio to avoid potential issues
        // Play the audio when the metadata is loaded
        audioRef.current.addEventListener("loadedmetadata", () => {
          if (audioRef.current) {
            audioRef.current.play();
          }
        });
      }
    };
    // Load the audio file and play it when it's available
    loadAudio();
  }, [audioFile, audioRef]);
  return (
    <div className="mt-8">
      <div className="bg-white text-black shadow-lg rounded cursor-pointer hover:shadow-2xl transition duration-300 ease-in-out">
        <audio
          ref={audioRef}
          controls
          className="w-full focus:outline-none transition-transform duration-300 transform hover:scale-105"
        />
      </div>
      {currentAudioName && (
        <p
          className={`text-center mt-4 text-gray-700 hover:text-indigo-400 transition duration-300 ease-in-out cursor-pointer font-semibold	 ${
            isPlaying ? "playing-animation" : "paused-animation"
          }`}
        >
          Now {isPlaying ? "Playing" : "Paused"}: {currentAudioName}
        </p>
      )}
    </div>
  );
};

export default AudioPlayer;
