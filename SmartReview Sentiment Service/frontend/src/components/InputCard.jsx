import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import './InputCard.css'

const InputCard = ({ onAnalyze, isLoading }) => {
  const [text, setText] = useState('')
  const [charCount, setCharCount] = useState(0)

  useEffect(() => {
    const handleFillExample = (e) => {
      setText(e.detail)
      setCharCount(e.detail.length)
    }
    
    window.addEventListener('fillExample', handleFillExample)
    return () => window.removeEventListener('fillExample', handleFillExample)
  }, [])

  const handleChange = (e) => {
    const value = e.target.value
    if (value.length <= 1000) {
      setText(value)
      setCharCount(value.length)
    }
  }

  const handleSubmit = () => {
    onAnalyze(text)
  }

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleSubmit()
    }
  }

  return (
    <motion.div
      className="card input-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.1 }}
      whileHover={{ y: -4 }}
    >
      <div className="card-header">
        <h2>Analyze Review Sentiment</h2>
        <p className="card-subtitle">Enter any customer review text below</p>
      </div>

      <div className="input-wrapper">
        <div className="textarea-container">
          <textarea
            value={text}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Type or paste your review here...&#10;&#10;Example: The delivery was slow but customer support was excellent and resolved my issue quickly."
            rows="8"
            maxLength="1000"
            disabled={isLoading}
          />
          <div className="char-counter">
            <span className={charCount > 900 ? 'warning' : charCount > 700 ? 'caution' : ''}>
              {charCount}
            </span>
            /1000
          </div>
        </div>

        <motion.button
          className="analyze-button"
          onClick={handleSubmit}
          disabled={isLoading || !text.trim()}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {isLoading ? (
            <span className="button-loader">
              <span className="spinner"></span>
              <span>Analyzing...</span>
            </span>
          ) : (
            <span className="button-content">
              <span className="button-icon">🔍</span>
              <span className="button-text">Analyze Sentiment</span>
            </span>
          )}
        </motion.button>
      </div>
    </motion.div>
  )
}

export default InputCard

