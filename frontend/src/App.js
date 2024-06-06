
import Layout from './components/Layout';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import PatientRecords from './pages/PatientRecords';
import Login from './pages/Login';
import AImodel from './pages/AImodel';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import Payment from './pages/Payment';

function App() {
    return (
        <BrowserRouter>
            <Layout>
                <Routes>
                    <Route path='/' element={<Home />} />
                    <Route path='/dashboard' element={<Dashboard />} />
                    <Route path='/patientrecords' element={<PatientRecords />} />
                    <Route path='/model' element={<AImodel />} />
                    <Route path='/reports' element={<Reports />} />
                    <Route path='/settings' element={<Settings />} />
                    <Route path='/payments' element={<Payment />} />
                    <Route path='/login' element={<Login />} />
                </Routes>
            </Layout>
        </BrowserRouter>
    );
}

export default App;

