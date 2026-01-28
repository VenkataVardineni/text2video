import { motion } from 'framer-motion'
import './ExamplesSection.css'

const examples = [
  {
    text: 'The delivery was slow but support was great',
    icon: '📦',
    id: 1
  },
  {
    text: 'This product is absolutely terrible and broke after one day',
    icon: '😞',
    id: 2
  },
  {
    text: 'I love this service, it exceeded all my expectations!',
    icon: '❤️',
    id: 3
  },
  {
    text: 'The quality is okay, nothing special but gets the job done',
    icon: '🤷',
    id: 4
  }
]

const ExamplesSection = ({ onExampleClick }) => {
  return (
    <motion.section
      className="examples-section"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2 }}
    >
      <div className="section-header">
        <h3>Try Example Reviews</h3>
        <p>Click any example to analyze</p>
      </div>
      <div className="examples-grid">
        {examples.map((example, index) => (
          <motion.button
            key={example.id}
            className="example-card"
            onClick={() => {
              window.dispatchEvent(new CustomEvent('fillExample', { detail: example.text }))
              onExampleClick(example.text)
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 + index * 0.1 }}
            whileHover={{ y: -4, scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="example-icon">{example.icon}</div>
            <div className="example-text">"{example.text}"</div>
          </motion.button>
        ))}
      </div>
    </motion.section>
  )
}

export default ExamplesSection

