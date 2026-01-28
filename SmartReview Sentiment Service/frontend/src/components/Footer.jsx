import { motion } from 'framer-motion'
import './Footer.css'

const Footer = ({ apiStatus }) => {
  return (
    <motion.footer
      className="footer"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.4 }}
    >
      <p>Powered by <strong>PyTorch</strong> + <strong>FastAPI</strong> + <strong>React</strong></p>
      <div className="footer-links">
        <span className={`status-indicator ${apiStatus === 'offline' ? 'offline' : ''}`}>
          <span className="status-dot"></span>
          <span>API {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}</span>
        </span>
      </div>
    </motion.footer>
  )
}

export default Footer

