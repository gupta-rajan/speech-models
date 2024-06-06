import React from "react";

// Playlist component to display the list of audio files in the playlist and trigger the callback to play, delete, or delete all audio files in the playlist
// The Playlist component is a simple list of audio files with play and delete buttons for each file
// The Playlist component is a stateless functional component that receives the playlist, onPlaylistItemClick, onDelete, onAllDelete, and onPlay callbacks as props and returns the list of audio files with play and delete buttons
// The Playlist component is used in the App component to display the list of audio files in the playlist and trigger the callbacks to play, delete, or delete all audio files in the playlist
// The Playlist component is a reusable component that can be used in other applications to display a list of items and trigger callbacks for each item
// The Playlist component receives the playlist, onPlaylistItemClick, onDelete, onAllDelete, and onPlay callbacks as props and returns the list of audio files with play and delete buttons
// The playlist prop is an array of audio files in the playlist
// The onPlaylistItemClick callback is triggered when an audio file in the playlist is clicked
// The onDelete callback is triggered when the delete button for an audio file is clicked
// The onAllDelete callback is triggered when the delete all button is clicked
// The onPlay callback is triggered when the play button for an audio file is clicked
// The Playlist component returns a list of audio files with play and delete buttons for each file
const Playlist = ({
  // The playlist prop is an array of audio files in the playlist
  playlist,
  onPlaylistItemClick,
  onDelete,
  onAllDelete,
  onPlay,
  currentAudioIndex,
}) => {
  return (
    <div className="bg-white shadow-lg p-4 rounded-lg">
      <h2 className="text-xl font-bold mb-4">Playlist</h2>
      <ul className="list-disc pl-4">
        {/* Map the available data  */}
        {playlist.map((item, index) => (
          // The onPlaylistItemClick callback is triggered when an audio file in the playlist is clicked
          <li
            key={index}
            className={`cursor-pointer mb-2 hover:bg-gray-100 hover:text-indigo-400 transition duration-300 ease-in-out p-2 px-3 list-none rounded-sm ${
              index === currentAudioIndex
                ? "font-bold bg-gray-200 px-3 animate-pulse"
                : "font-normal"
            }`}
            onClick={() => onPlaylistItemClick(index)}
          >
            {item.name}
            <button
              onClick={() => onPlay(index)}
              className="ml-2 px-2 py-1 bg-white text-black rounded shadow-md border transition duration-300 ease-in-out hover:shadow-2xl"
            >
              Play
            </button>
            <button
              onClick={() => onDelete(index)}
              className="ml-2 px-2 py-1 bg-red-500 text-white rounded transition duration-300 ease-in-out hover:bg-red-600 hover:text-gray-100"
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
      {playlist.length > 0 && (
        <div className="mt-4">
          <button
            onClick={() => onAllDelete()}
            className="px-4 py-2 bg-rose-500 text-white rounded hover:bg-rose-800 hover:shadow-2xl transition duration-300 ease-in-out"
          >
            Delete All
          </button>
        </div>
      )}
    </div>
  );
};

export default Playlist;
