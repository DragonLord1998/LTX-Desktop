import { useState, useCallback, useRef } from 'react'
import { getBackendUrl } from '../lib/electron-shim'

interface ImageEditState {
  isEditing: boolean
  progress: number
  statusMessage: string
  editResult: string | null  // file URL of the edited image
  error: string | null
}

interface UseImageEditReturn extends ImageEditState {
  editImage: (
    imagePath: string,
    instruction: string,
    loraId: string,
    loraStrength: number,
    numSteps: number,
  ) => Promise<void>
  cancelEdit: () => void
  resetEdit: () => void
}

export function useImageEdit(): UseImageEditReturn {
  const [state, setState] = useState<ImageEditState>({
    isEditing: false,
    progress: 0,
    statusMessage: '',
    editResult: null,
    error: null,
  })

  const abortControllerRef = useRef<AbortController | null>(null)

  const editImage = useCallback(async (
    imagePath: string,
    instruction: string,
    loraId: string,
    loraStrength: number,
    numSteps: number,
  ) => {
    setState({
      isEditing: true,
      progress: 0,
      statusMessage: 'Starting edit...',
      editResult: null,
      error: null,
    })

    abortControllerRef.current = new AbortController()
    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldApplyPollingUpdates = true

    try {
      const backendUrl = await getBackendUrl()

      const pollProgress = async () => {
        if (!shouldApplyPollingUpdates) return
        try {
          const res = await fetch(`${backendUrl}/api/generation/progress`)
          if (res.ok) {
            const data = await res.json()
            if (!shouldApplyPollingUpdates) return
            setState(prev => ({
              ...prev,
              progress: data.progress ?? prev.progress,
              statusMessage: data.phase === 'complete' ? 'Finalizing...' : 'Editing image...',
            }))
          }
        } catch {
          // Ignore polling errors
        }
      }

      progressInterval = setInterval(pollProgress, 500)

      const response = await fetch(`${backendUrl}/api/qwen-edit/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: imagePath,
          instruction,
          lora_id: loraId === 'none' ? null : loraId,
          lora_strength: loraStrength,
          num_steps: numSteps,
        }),
        signal: abortControllerRef.current.signal,
      })

      shouldApplyPollingUpdates = false

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Image edit failed')
      }

      const result = await response.json()

      if (result.status === 'complete' && result.image_path) {
        const { filePathToUrl } = await import('../lib/electron-shim')
        const fileUrl = filePathToUrl(result.image_path)
        setState({
          isEditing: false,
          progress: 100,
          statusMessage: 'Complete!',
          editResult: fileUrl,
          error: null,
        })
      } else if (result.status === 'cancelled') {
        setState(prev => ({ ...prev, isEditing: false, statusMessage: 'Cancelled' }))
      } else if (result.error) {
        throw new Error(result.error)
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        setState(prev => ({ ...prev, isEditing: false, statusMessage: 'Cancelled' }))
      } else {
        setState(prev => ({
          ...prev,
          isEditing: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        }))
      }
    } finally {
      shouldApplyPollingUpdates = false
      if (progressInterval) {
        clearInterval(progressInterval)
      }
    }
  }, [])

  const cancelEdit = useCallback(async () => {
    abortControllerRef.current?.abort()
    try {
      const backendUrl = await getBackendUrl()
      await fetch(`${backendUrl}/api/generate/cancel`, { method: 'POST' })
    } catch {
      // Ignore errors from cancel request
    }
    setState(prev => ({ ...prev, isEditing: false, statusMessage: 'Cancelled' }))
  }, [])

  const resetEdit = useCallback(() => {
    setState({
      isEditing: false,
      progress: 0,
      statusMessage: '',
      editResult: null,
      error: null,
    })
  }, [])

  return {
    ...state,
    editImage,
    cancelEdit,
    resetEdit,
  }
}
