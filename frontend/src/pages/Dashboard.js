// HomePage.js
import React from 'react';

function HomePage() {
  return (
    <div className="bg-gradient-to-r from-blue-500 to-purple-600 min-h-screen flex items-center justify-center">
      <div className="text-center text-white">
        <h1 className="text-6xl font-extrabold mb-4">Fake Speech Detection Demo</h1>
        <p className="text-lg mb-8">Let's Identify the fake people !!</p>
        <button className="bg-white text-blue-500 py-3 px-6 rounded-full shadow-md hover:bg-blue-400 hover:text-white transition duration-300 ease-in-out transform hover:scale-105">Get Started</button>
        <div className="mt-4 text-gray-200 text-sm">
          
        </div>
      </div>
    </div>
  );
}

export default HomePage;
