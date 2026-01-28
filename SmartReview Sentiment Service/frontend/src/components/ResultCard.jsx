import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import './ResultCard.css'

const ResultCard = ({ result, onClose }) => {
  const [animatedWidth, setAnimatedWidth] = useState(0)

  useEffect(() => {
    if (result) {
      setAnimatedWidth(0)
      setTimeout(() => {
        setAnimatedWidth(result.score * 100)
      }, 100)
    }
  }, [result])

  if (!result) return null

  const isPositive = result.label === 'positive'
  const confidencePercent = (result.score * 100).toFixed(1)

  return (
    <motion.div
      className="card result-card"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
    >
      <div className="result-header">
        <h3>Analysis Result</h3>
        <button className="close-btn" onClick={onClose}>
          ×
        </button>
      </div>

      <div className="result-body">
        <div className="sentiment-display">
          <motion.div
            className="sentiment-icon-wrapper"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 15 }}
          >
            <div className={`sentiment-icon ${isPositive ? 'positive' : 'negative'}`}>
              <span>{isPositive ? '😊' : '😞'}</span>
            </div>
          </motion.div>

          <div className="sentiment-info">
            <div className="sentiment-label-wrapper">
              <motion.span
                className="sentiment-label"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                {result.label.charAt(0).toUpperCase() + result.label.slice(1)}
              </motion.span>
              <motion.span
                className={`sentiment-badge ${isPositive ? 'positive' : 'negative'}`}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3, type: "spring" }}
              >
                {isPositive ? 'POS' : 'NEG'}
              </motion.span>
            </div>

            <div className="confidence-display">
              <div className="confidence-header">
                <span>Confidence Level</span>
                <motion.span
                  className="confidence-percentage"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  {confidencePercent}%
                </motion.span>
              </div>
              <div className="confidence-bar-container">
                <div className="confidence-bar">
                  <motion.div
                    className={`confidence-fill ${isPositive ? 'positive' : 'negative'}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${animatedWidth}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  >
                    <div className="confidence-shine"></div>
                    <span className="confidence-text">{confidencePercent}%</span>
                  </motion.div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default ResultCard

