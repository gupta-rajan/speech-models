import React from 'react'
import Sidebar from './Sidebar'

const Layout = ({ children }) => {
    return (
        <>
            <div className='flex flex-auto h-screen'>
                <Sidebar />
                <div className='grow'>
                    <div className='m-5'>{children}</div>
                </div>
            </div>
        </>
    )
}

export default Layout
