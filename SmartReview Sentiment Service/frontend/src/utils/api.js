const API_URL = import.meta.env.VITE_API_URL || window.location.origin

export const analyzeSentiment = async (text) => {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text })
  })

  if (!response.ok) {
    const errorData = await response.json()
    throw new Error(errorData.detail || 'Failed to analyze sentiment')
  }

  return await response.json()
}

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_URL}/health`)
    if (response.ok) {
      const data = await response.json()
      return data.status === 'healthy'
    }
    return false
  } catch {
    return false
  }
}

