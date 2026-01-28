import { motion } from 'framer-motion'
import './ErrorCard.css'

const ErrorCard = ({ message, onClose }) => {
  return (
    <motion.div
      className="card error-card"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
    >
      <div className="error-content">
        <div className="error-icon">⚠️</div>
        <div className="error-text">
          <h4>Error</h4>
          <p>{message}</p>
        </div>
        <button className="close-btn" onClick={onClose}>×</button>
      </div>
    </motion.div>
  )
}

export default ErrorCard

