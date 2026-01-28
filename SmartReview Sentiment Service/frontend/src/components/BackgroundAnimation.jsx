import { motion } from 'framer-motion'
import './BackgroundAnimation.css'

const BackgroundAnimation = () => {
  return (
    <div className="background-animation">
      <motion.div
        className="gradient-orb orb-1"
        animate={{
          x: [0, 50, -50, 0],
          y: [0, -50, 50, 0],
          scale: [1, 1.1, 0.9, 1],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      <motion.div
        className="gradient-orb orb-2"
        animate={{
          x: [0, -50, 50, 0],
          y: [0, 50, -50, 0],
          scale: [1, 0.9, 1.1, 1],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 7
        }}
      />
      <motion.div
        className="gradient-orb orb-3"
        animate={{
          x: [0, 30, -30, 0],
          y: [0, -30, 30, 0],
          scale: [1, 1.05, 0.95, 1],
        }}
        transition={{
          duration: 18,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 14
        }}
      />
    </div>
  )
}

export default BackgroundAnimation

