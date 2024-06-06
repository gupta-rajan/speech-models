import React from 'react';
import { AiOutlineSearch } from 'react-icons/ai'; // Importing search icon from react-icons library

const Settings = () => {
    return (
        <div className='settings-container'>
            <h1 className='text-3xl font-bold text-black mt-6 p-4'>Settings</h1>
            <h2 className='text-1xl font-bold text-black mt-2'>Diabetic Retinopathy</h2>
            <div className='flex justify-between mt-6 gap-1'>
                <Card title='Microaneurysms' description='Find small red dots. These are the earliest sign of DR' />
                <Card title='Hemorrhages' description='Find red spots. These are a sign of advanced DR' />
                <Card title='Exudates' description='Find yellow or white spots. These are a sign of advanced DR' />
                <Card title='Cotton Wool Spots' description='Find white patches. These are a sign of advanced DR' />
            </div>
            <div className="preferences mt-8">
                <h2 className='text-2xl font-bold text-black'>Preferences</h2>
                <Preference title="Minimum Microaneurysms Size" description="The minimum size of a microaneurysms in pixels" value="10px" />
                <Preference title="Minimum Hemorrhage Size" description="The minimum size of a hemorrhage in pixels" value="50px" />
                <Preference title="Minimum Exudates Size" description="The minimum size of a exudates in pixels" value="20px" />
                <Preference title="Minimum Cotton Wool Spots Size" description="The minimum size of a cotton wool spots in pixels" value="30px" />
            </div>
            <div className="flex justify-end">
                <button className="save-button bg-blue-800 text-white rounded-lg px-4 py-2 mt-8">Save Changes</button>
            </div>
        </div>
    );
}

const Card = ({ title, description }) => {
    return (
        <div className='card p-4 w-1/4 bg-white rounded-lg border border-black-200 relative'>
            <AiOutlineSearch className='text-black w-6 h-6 absolute left-2 top-2' />
            <h3 className='text-xl font-bold mb-2 mt-8'>{title}</h3>
            <p className='text-sm'>{description}</p>
        </div>
    );
}

const Preference = ({ title, description, value }) => {
    return (
        <div className='preference flex justify-between items-center mt-2'>
            <div>
                <h3 className='text-lg font-bold'>{title}</h3>
                <p className='text-sm'>{description}</p>
            </div>
            <span className="text-sm">{value}</span>
        </div>
    );
}

export default Settings;
