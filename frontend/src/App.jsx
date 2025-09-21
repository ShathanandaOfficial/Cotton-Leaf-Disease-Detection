import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Detection from './pages/Detection'
import History from './pages/History'
import About from './pages/About'
function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/detect" element={<Detection />} />
          <Route path="/history" element={<History />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App