import React from "react";
// This component is responsible for uploading audio files to the playlist and triggering the callback
//to add the files to the playlist state in the App component when a file is selected.
//The FileUploader component is a simple form with a file input field and a label.
//The label is styled to look like a button, and when clicked, it triggers the file input field to open the file picker dialog.
// When a file is selected, the handleFileChange function is called, which triggers the onFileUpload callback with the selected files as an argument.
//The onFileUpload callback is passed as a prop from the App component, and it adds the selected files to the playlist state.
//The FileUploader component is a stateless functional component that receives the onFileUpload callback as a prop and returns the form with the file input field and label.
// The FileUploader component is used in the App component to allow users to upload audio files to the playlist.
// The FileUploader component is a reusable component that can be used in other applications to allow users to upload files.
const FileUploader = ({ onFileUpload }) => {
  // Handle file change event

  const handleFileChange = (e) => {
    // Get the selected files
    const selectedFiles = e.target.files;
    // Trigger the callback to add files to the playlist
    onFileUpload(selectedFiles); // Trigger the callback to add files to the playlist
  };

  return (
    <div className="bg-gray-100 p-4 rounded-lg">
      <label className="block text-lg font-bold mb-2">File Uploader</label>
      <div className="flex flex-col sm:flex-row items-center">
        <label className="flex items-center px-4 py-2 bg-white text-black shadow-lg rounded cursor-pointer hover:shadow-2xl transition duration-300 ease-in-out mb-2 sm:mb-0">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 mr-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            />
          </svg>
          Upload File
          {/* Change the state of handleFileChange by props */}
          <input
            type="file"
            accept=".mp3"
            className="hidden"
            onChange={handleFileChange}
          />
        </label>
        <span className="ml-4 text-gray-700">Supported format: .mp3</span>
      </div>
    </div>
  );
};

export default FileUploader;
