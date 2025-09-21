import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Loader2, AlertCircle, CheckCircle } from 'lucide-react'
import axios from 'axios'
import React from 'react'


const Detection = () => {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const onDrop = (acceptedFiles) => {
    setError(null)
    setResult(null)
    const selectedFile = acceptedFiles[0]
    
    if (selectedFile) {
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please upload an image file')
        return
      }
      
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    maxFiles: 1
  })

  const handleSubmit = async () => {
    if (!file) {
      setError('Please select an image first')
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred during prediction')
    } finally {
      setIsLoading(false)
    }
  }

  const resetForm = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Detect Cotton Leaf Disease</h1>
        <p className="text-gray-600">Upload an image of a cotton leaf to detect potential diseases</p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-green-500'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-green-600">Drop the image here</p>
            ) : (
              <div>
                <p className="text-gray-600 mb-2">Drag & drop an image here, or click to select</p>
                <p className="text-sm text-gray-500">Supports JPG, JPEG, PNG</p>
              </div>
            )}
          </div>

          {preview && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-2">Preview</h3>
              <img
                src={preview}
                alt="Preview"
                className="w-full h-48 object-contain rounded-lg border"
              />
              <p className="text-sm text-gray-600 mt-2 truncate">{file.name}</p>
            </div>
          )}

          <div className="mt-6 flex gap-4">
            <button
              onClick={handleSubmit}
              disabled={isLoading || !file}
              className="flex-1 bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin mr-2 h-4 w-4" />
                  Analyzing...
                </>
              ) : (
                'Analyze Image'
              )}
            </button>
            <button
              onClick={resetForm}
              className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Reset
            </button>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md flex items-center">
              <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
              <span className="text-red-700">{error}</span>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
          
          {isLoading ? (
            <div className="text-center py-12">
              <Loader2 className="animate-spin mx-auto h-8 w-8 text-green-600 mb-4" />
              <p className="text-gray-600">Analyzing your image...</p>
            </div>
          ) : result ? (
            <div className="animate-fade-in">
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6 flex items-center">
                <CheckCircle className="h-6 w-6 text-green-600 mr-3" />
                <div>
                  <h3 className="font-semibold text-green-800">Analysis Complete</h3>
                  <p className="text-green-600 text-sm">Image processed successfully</p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <span className="text-gray-600">Detected Condition:</span>
                  <p className="text-xl font-semibold text-gray-800 capitalize">{result.class}</p>
                </div>
                
                <div>
                  <span className="text-gray-600">Confidence Level:</span>
                  <div className="mt-1">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div
                        className="bg-green-600 h-2.5 rounded-full transition-all duration-1000"
                        style={{ width: `${result.confidence * 100}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      {Math.round(result.confidence * 100)}% confident
                    </p>
                  </div>
                </div>

                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h4 className="font-semibold text-blue-800 mb-2">Recommendation</h4>
                  <p className="text-blue-700 text-sm">
                    {result.class === 'Healthy' 
                      ? 'Your plant appears healthy. Continue with regular maintenance.'
                      : 'Consider consulting with an agricultural expert for treatment options.'}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <Upload className="mx-auto h-12 w-12 mb-4 opacity-50" />
              <p>Upload an image to see analysis results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Detection