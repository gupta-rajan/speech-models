import React from 'react';

export default function HeroSection() {
  return (
    <section id="heroSection" className="grid grid-cols-2 gap-8 p-16 items-center justify-between bg-gray-100">
      <div className="flex flex-col gap-8">
        <div className="flex flex-col gap-5">
          <p className="text-3xl font-semibold">Hey all , welcome to </p>
          <h1 className="text-7xl font-bold">
            <span className="text-primary">Fake Speech</span>{" "}
            <br />
            Detection
          </h1>
          <p className="text-xl text-darkblue">
           Voice is a Data, Data is Everything
            <br /> Wanna Engage With FSD?
          </p>
        </div>
        <button className="btn btn-primary mt-5">Get In Touch</button>
      </div>
      <div className="flex">
        <img src="../images/5.jpeg" alt="Hero Section" className="w-full h-full" />
      </div>
    </section>
  );
}
