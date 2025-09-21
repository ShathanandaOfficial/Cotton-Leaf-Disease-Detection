import { useState, useEffect } from 'react'
import { Calendar, Clock, Image, BarChart3, Download } from 'lucide-react'
import axios from 'axios'
import React from 'react'

const History = () => {
  const [predictions, setPredictions] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchPredictions()
    fetchStats()
  }, [])

  const fetchPredictions = async () => {
    try {
      const response = await axios.get('http://localhost:8000/predictions?limit=50')
      setPredictions(response.data)
    } catch (err) {
      setError('Failed to fetch prediction history')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await axios.get('http://localhost:8000/stats')
      setStats(response.data)
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    }
  }

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleDateString()
  }

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const getStatusColor = (confidence) => {
    if (confidence > 0.8) return 'text-green-600'
    if (confidence > 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const exportToCSV = () => {
    const headers = ['Date', 'Time', 'Filename', 'Prediction', 'Confidence']
    const csvContent = [
      headers.join(','),
      ...predictions.map(p => [
        formatDate(p.timestamp),
        formatTime(p.timestamp),
        p.filename,
        p.prediction,
        p.confidence
      ].join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'cotton_disease_predictions.csv'
    a.click()
    window.URL.revokeObjectURL(url)
  }

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading prediction history...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Prediction History</h1>
        <p className="text-gray-600">View your past cotton leaf disease predictions and analysis results</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Stats Overview */}
      {stats && (
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <BarChart3 className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Total Predictions</p>
                <p className="text-2xl font-bold text-gray-800">{stats.total_predictions}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <Clock className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Recent Accuracy</p>
                <p className="text-2xl font-bold text-gray-800">
                  {Math.round(stats.recent_accuracy * 100)}%
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Image className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm text-gray-600">Diseases Detected</p>
                <p className="text-2xl font-bold text-gray-800">
                  {Object.keys(stats.disease_distribution || {}).length}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Export Button */}
      <div className="flex justify-end mb-6">
        <button
          onClick={exportToCSV}
          className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
        >
          <Download className="h-4 w-4 mr-2" />
          Export to CSV
        </button>
      </div>

      {/* Predictions Table */}
      <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Date & Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Image
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Prediction
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {predictions.length === 0 ? (
                <tr>
                  <td colSpan="4" className="px-6 py-8 text-center text-gray-500">
                    No predictions found. Start by analyzing some cotton leaf images!
                  </td>
                </tr>
              ) : (
                predictions.map((prediction, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Calendar className="h-4 w-4 text-gray-400 mr-2" />
                        <span className="text-sm text-gray-900">
                          {formatDate(prediction.timestamp)}
                        </span>
                      </div>
                      <div className="flex items-center mt-1">
                        <Clock className="h-4 w-4 text-gray-400 mr-2" />
                        <span className="text-xs text-gray-500">
                          {formatTime(prediction.timestamp)}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Image className="h-4 w-4 text-gray-400 mr-2" />
                        <span className="text-sm text-gray-900 truncate max-w-xs">
                          {prediction.filename}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded-full capitalize">
                        {prediction.prediction}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`text-sm font-medium ${getStatusColor(prediction.confidence)}`}>
                        {Math.round(prediction.confidence * 100)}%
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default History