import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Header from './components/Header'
import InputCard from './components/InputCard'
import ResultCard from './components/ResultCard'
import ErrorCard from './components/ErrorCard'
import ExamplesSection from './components/ExamplesSection'
import Footer from './components/Footer'
import BackgroundAnimation from './components/BackgroundAnimation'
import { analyzeSentiment, checkHealth } from './utils/api'

function App() {
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    checkApiHealth()
    const interval = setInterval(checkApiHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  const checkApiHealth = async () => {
    try {
      const status = await checkHealth()
      setApiStatus(status ? 'online' : 'offline')
    } catch {
      setApiStatus('offline')
    }
  }

  const handleAnalyze = async (text) => {
    if (!text.trim()) {
      setError('Please enter some text to analyze')
      setResult(null)
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await analyzeSentiment(text)
      setResult(data)
    } catch (err) {
      setError(err.message || 'An error occurred while analyzing sentiment')
      setResult(null)
    } finally {
      setIsLoading(false)
    }
  }

  const handleExampleClick = (text) => {
    const event = new CustomEvent('fillExample', { detail: text })
    window.dispatchEvent(event)
  }

  return (
    <div className="app">
      <BackgroundAnimation />
      <div className="container">
        <Header />
        
        <main className="main-content">
          <InputCard 
            onAnalyze={handleAnalyze} 
            isLoading={isLoading}
          />

          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                transition={{ duration: 0.3 }}
              >
                <ResultCard 
                  result={result} 
                  onClose={() => setResult(null)}
                />
              </motion.div>
            )}

            {error && (
              <motion.div
                key="error"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
              >
                <ErrorCard 
                  message={error} 
                  onClose={() => setError(null)}
                />
              </motion.div>
            )}
          </AnimatePresence>

          <ExamplesSection onExampleClick={handleExampleClick} />
        </main>

        <Footer apiStatus={apiStatus} />
      </div>
    </div>
  )
}

export default App

