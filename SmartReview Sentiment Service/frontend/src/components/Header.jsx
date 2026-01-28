import { motion } from 'framer-motion'
import './Header.css'

const Header = () => {
  return (
    <motion.header
      className="header"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="header-content">
        <div className="logo">
          <motion.div
            className="logo-icon"
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
          >
            ✨
          </motion.div>
          <h1>SmartReview</h1>
        </div>
        <p className="tagline">AI-Powered Sentiment Analysis</p>
      </div>
    </motion.header>
  )
}

export default Header

