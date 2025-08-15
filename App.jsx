import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Upload, Play, Pause, Square, Music, Brain, Sparkles } from 'lucide-react'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [predictions, setPredictions] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [topK, setTopK] = useState(4)
  const [error, setError] = useState(null)
  
  const fileInputRef = useRef(null)
  const audioRef = useRef(null)

  const composers = [
    { name: 'Bach', color: 'bg-blue-500', description: 'Baroque master of counterpoint' },
    { name: 'Beethoven', color: 'bg-purple-500', description: 'Classical to Romantic bridge' },
    { name: 'Chopin', color: 'bg-green-500', description: 'Romantic piano virtuoso' },
    { name: 'Mozart', color: 'bg-yellow-500', description: 'Classical period genius' }
  ]

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file && (file.name.endsWith('.mid') || file.name.endsWith('.midi'))) {
      setSelectedFile(file)
      setPredictions(null)
      setError(null)
      setCurrentTime(0)
      setDuration(0)
      setIsPlaying(false)
    } else {
      setError('Please select a valid MIDI file (.mid or .midi)')
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handlePredict = async () => {
    if (!selectedFile) return
    
    setIsAnalyzing(true)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('top_k', topK.toString())
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to analyze file')
      }
      
      const data = await response.json()
      setPredictions(data.predictions)
      
    } catch (err) {
      setError(err.message)
      console.error('Prediction error:', err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleSampleLoad = async (composer) => {
    try {
      setError(null)
      const response = await fetch(`/api/sample/${composer.toLowerCase()}`)
      
      if (!response.ok) {
        throw new Error('Sample file not available')
      }
      
      const blob = await response.blob()
      const file = new File([blob], `${composer.toLowerCase()}_sample.mid`, { type: 'audio/midi' })
      
      setSelectedFile(file)
      setPredictions(null)
      setCurrentTime(0)
      setDuration(0)
      setIsPlaying(false)
      
    } catch (err) {
      setError(`Could not load ${composer} sample: ${err.message}`)
    }
  }

  const handlePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      setCurrentTime(0)
      setIsPlaying(false)
    }
  }

  const formatTime = (time) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white">Composer Classifier</h1>
          </div>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Upload a MIDI file and let AI identify the composer. Our deep learning model analyzes 
            musical patterns to predict among Bach, Beethoven, Chopin, and Mozart.
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-8">
          {/* Error Display */}
          {error && (
            <Card className="bg-red-900/50 border-red-700 backdrop-blur-sm">
              <CardContent className="pt-6">
                <p className="text-red-300">{error}</p>
              </CardContent>
            </Card>
          )}

          {/* Upload Section */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Upload className="w-5 h-5" />
                Upload MIDI File
              </CardTitle>
              <CardDescription className="text-gray-400">
                Upload a .mid/.midi file. The server will preprocess it, run the BiLSTM, and return the top matches.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mid,.midi"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <Button 
                  onClick={handleUploadClick}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  Choose File
                </Button>
                {selectedFile && (
                  <span className="text-gray-300">{selectedFile.name}</span>
                )}
              </div>
              
              <div className="flex items-center gap-4">
                <span className="text-gray-400">Top K</span>
                <select 
                  value={topK} 
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="bg-slate-700 text-white border border-slate-600 rounded px-3 py-1"
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                  <option value={4}>4</option>
                </select>
                <Button 
                  onClick={handlePredict}
                  disabled={!selectedFile || isAnalyzing}
                  className="bg-green-600 hover:bg-green-700 disabled:opacity-50"
                >
                  {isAnalyzing ? (
                    <>
                      <Sparkles className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    'Predict'
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Audio Player */}
          {selectedFile && (
            <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Music className="w-5 h-5" />
                  Audio Player
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4">
                  <Button
                    onClick={handlePlay}
                    size="sm"
                    className="bg-green-600 hover:bg-green-700"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>
                  <Button
                    onClick={handleStop}
                    size="sm"
                    variant="outline"
                    className="border-slate-600 text-gray-300 hover:bg-slate-700"
                  >
                    <Square className="w-4 h-4" />
                  </Button>
                  <div className="flex-1">
                    <Progress value={(currentTime / duration) * 100 || 0} className="h-2" />
                  </div>
                  <span className="text-gray-400 text-sm">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </span>
                </div>
                
                {/* Note about MIDI playback */}
                <p className="text-sm text-gray-500">
                  Tip: pieces shorter than 200 notes are skipped.
                </p>
              </CardContent>
            </Card>
          )}

          {/* Predictions */}
          {predictions && (
            <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white">Prediction Results</CardTitle>
                <CardDescription className="text-gray-400">
                  AI analysis of musical patterns and composer characteristics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {predictions.slice(0, topK).map((prediction, index) => {
                    const composer = composers.find(c => c.name.toLowerCase() === prediction.composer.toLowerCase())
                    const percentage = (prediction.confidence * 100).toFixed(1)
                    
                    return (
                      <div key={prediction.composer} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <Badge 
                              className={`${composer?.color || 'bg-gray-500'} text-white`}
                            >
                              #{index + 1}
                            </Badge>
                            <div>
                              <span className="text-white font-medium">{prediction.composer}</span>
                              <p className="text-sm text-gray-400">{composer?.description}</p>
                            </div>
                          </div>
                          <span className="text-white font-bold">{percentage}%</span>
                        </div>
                        <Progress 
                          value={prediction.confidence * 100} 
                          className="h-2"
                        />
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Sample Files */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white">Sample Files</CardTitle>
              <CardDescription className="text-gray-400">
                Try these sample MIDI files to test the classifier
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {composers.map((composer) => (
                  <Button
                    key={composer.name}
                    variant="outline"
                    className="border-slate-600 text-gray-300 hover:bg-slate-700 h-auto p-4 flex flex-col items-center gap-2"
                    onClick={() => handleSampleLoad(composer.name)}
                  >
                    <div className={`w-8 h-8 rounded-full ${composer.color}`}></div>
                    <span className="font-medium">{composer.name}</span>
                    <span className="text-xs text-gray-500">{composer.description}</span>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default App

