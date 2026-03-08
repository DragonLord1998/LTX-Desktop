/**
 * electron-shim.ts
 *
 * Provides web-compatible fallbacks for window.electronAPI so the app works
 * in a plain browser (e.g. served from RunPod) without breaking the existing
 * Electron desktop build.
 *
 * Strategy: feature-detect window.electronAPI. If it exists, use it as-is.
 * Otherwise, return shims that talk directly to the same-origin FastAPI server
 * (or the URL in VITE_BACKEND_URL).
 */

// ---------------------------------------------------------------------------
// Backend URL resolution
// ---------------------------------------------------------------------------

export function getBackendBaseUrl(): string {
  // 1. Build-time override (set in .env or Vite config)
  if (import.meta.env.VITE_BACKEND_URL) {
    return import.meta.env.VITE_BACKEND_URL as string
  }
  // 2. Same-origin: the FastAPI backend serves the frontend at the same host
  return window.location.origin
}

/**
 * Returns the backend base URL. In Electron this delegates to the native IPC
 * call; in a browser it resolves from the environment / same-origin.
 */
export async function getBackendUrl(): Promise<string> {
  if (window.electronAPI) {
    return window.electronAPI.getBackendUrl()
  }
  return getBackendBaseUrl()
}

// ---------------------------------------------------------------------------
// Health-status polling (replaces onBackendHealthStatus IPC in web mode)
// ---------------------------------------------------------------------------

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface BackendHealthStatusPayload {
  status: BackendProcessStatus
  exitCode?: number | null
}

type HealthStatusCallback = (data: BackendHealthStatusPayload) => void

let _pollIntervalId: ReturnType<typeof setInterval> | null = null
const _healthListeners: Set<HealthStatusCallback> = new Set()

function _notifyListeners(payload: BackendHealthStatusPayload) {
  _healthListeners.forEach(cb => cb(payload))
}

function _startPollingIfNeeded() {
  if (_pollIntervalId !== null) return

  const poll = async () => {
    const url = `${getBackendBaseUrl()}/health`
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(5000) })
      if (res.ok) {
        _notifyListeners({ status: 'alive' })
      } else {
        _notifyListeners({ status: 'dead' })
      }
    } catch {
      // Network error → treat as dead for now; caller will retry
      _notifyListeners({ status: 'dead' })
    }
  }

  // Fire immediately, then every 3 s
  void poll()
  _pollIntervalId = setInterval(poll, 3000)
}

function _stopPollingIfEmpty() {
  if (_healthListeners.size === 0 && _pollIntervalId !== null) {
    clearInterval(_pollIntervalId)
    _pollIntervalId = null
  }
}

/**
 * Subscribe to backend health status events.
 * In Electron: delegates to native IPC.
 * In browser: polls /health on an interval.
 * Returns an unsubscribe function (matching Electron API contract).
 */
export function onBackendHealthStatus(cb: HealthStatusCallback): () => void {
  if (window.electronAPI) {
    return window.electronAPI.onBackendHealthStatus(cb)
  }

  _healthListeners.add(cb)
  _startPollingIfNeeded()

  return () => {
    _healthListeners.delete(cb)
    _stopPollingIfEmpty()
  }
}

/**
 * Get the current backend health status (snapshot).
 * In Electron: delegates to native IPC.
 * In browser: does a single /health fetch.
 */
export async function getBackendHealthStatus(): Promise<BackendHealthStatusPayload | null> {
  if (window.electronAPI) {
    return window.electronAPI.getBackendHealthStatus()
  }

  try {
    const res = await fetch(`${getBackendBaseUrl()}/health`, { signal: AbortSignal.timeout(5000) })
    return { status: res.ok ? 'alive' : 'dead' }
  } catch {
    return { status: 'dead' }
  }
}

// ---------------------------------------------------------------------------
// File-system / dialog stubs (no-ops in web mode)
// ---------------------------------------------------------------------------

/** In Electron: extracts a video frame via IPC. In browser: not supported. */
export async function extractVideoFrame(
  _videoUrl: string,
  _seekTime: number,
  _width?: number,
  _quality?: number,
): Promise<{ path: string; url: string }> {
  if (window.electronAPI) {
    return window.electronAPI.extractVideoFrame(_videoUrl, _seekTime, _width, _quality)
  }
  // In web mode, frame extraction is not available
  return { path: '', url: '' }
}

/** In Electron: returns the models directory path. In browser: returns empty string. */
export async function getModelsPath(): Promise<string> {
  if (window.electronAPI) {
    return window.electronAPI.getModelsPath()
  }
  return ''
}

/** In Electron: saves a file. In browser: triggers a download. */
export async function saveFile(
  filePath: string,
  data: string,
  encoding?: string,
): Promise<{ success: boolean; path?: string; error?: string }> {
  if (window.electronAPI) {
    return window.electronAPI.saveFile(filePath, data, encoding)
  }
  // Trigger browser download
  try {
    let blob: Blob
    if (encoding === 'base64') {
      const binaryStr = atob(data)
      const bytes = new Uint8Array(binaryStr.length)
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i)
      }
      blob = new Blob([bytes], { type: 'application/octet-stream' })
    } else {
      blob = new Blob([data], { type: 'text/plain' })
    }
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filePath.split(/[/\\]/).pop() || 'download'
    a.click()
    URL.revokeObjectURL(url)
    return { success: true }
  } catch (e) {
    return { success: false, error: String(e) }
  }
}

/** In Electron: shows a save dialog. In browser: returns null (no native dialog). */
export async function showSaveDialog(
  _options: { title?: string; defaultPath?: string; filters?: { name: string; extensions: string[] }[] },
): Promise<string | null> {
  if (window.electronAPI) {
    return window.electronAPI.showSaveDialog(_options)
  }
  // No native dialog in browser — return null so callers use the blob-download fallback
  return null
}

/** In Electron: opens a directory picker. In browser: returns null. */
export async function showOpenDirectoryDialog(
  _options: { title?: string },
): Promise<string | null> {
  if (window.electronAPI) {
    return window.electronAPI.showOpenDirectoryDialog(_options)
  }
  return null
}

/** In Electron: opens a file picker. In browser: returns null. */
export async function showOpenFileDialog(
  _options: { title?: string; filters?: { name: string; extensions: string[] }[]; properties?: string[] },
): Promise<string[] | null> {
  if (window.electronAPI) {
    return window.electronAPI.showOpenFileDialog(_options)
  }
  return null
}

/** In Electron: checks Python readiness. In browser: always returns ready=true. */
export async function checkPythonReady(): Promise<{ ready: boolean }> {
  if (window.electronAPI) {
    return window.electronAPI.checkPythonReady()
  }
  return { ready: true }
}

/** In Electron: starts the Python backend. In browser: no-op. */
export async function startPythonBackend(): Promise<void> {
  if (window.electronAPI) {
    return window.electronAPI.startPythonBackend()
  }
}

/** In Electron: checks first-run state. In browser: no setup needed. */
export async function checkFirstRun(): Promise<{ needsSetup: boolean; needsLicense: boolean }> {
  if (window.electronAPI) {
    return window.electronAPI.checkFirstRun()
  }
  return { needsSetup: false, needsLicense: false }
}

/** In Electron: completes first-run setup. In browser: no-op returning true. */
export async function completeSetup(): Promise<boolean> {
  if (window.electronAPI) {
    return window.electronAPI.completeSetup()
  }
  return true
}

/** In Electron: accepts the license. In browser: no-op returning true. */
export async function acceptLicense(): Promise<boolean> {
  if (window.electronAPI) {
    return window.electronAPI.acceptLicense()
  }
  return true
}

/** In Electron: fetches license text. In browser: returns empty string. */
export async function fetchLicenseText(): Promise<string> {
  if (window.electronAPI) {
    return window.electronAPI.fetchLicenseText()
  }
  return ''
}

/** In Electron: returns downloads path. In browser: returns empty string. */
export async function getDownloadsPath(): Promise<string> {
  if (window.electronAPI) {
    return window.electronAPI.getDownloadsPath()
  }
  return ''
}

/** In Electron: returns app info. In browser: returns minimal stub. */
export async function getAppInfo(): Promise<{ version: string; isPackaged: boolean; modelsPath: string; userDataPath: string }> {
  if (window.electronAPI) {
    return window.electronAPI.getAppInfo()
  }
  return { version: 'web', isPackaged: false, modelsPath: '', userDataPath: '' }
}

/** In Electron: opens the LTX API key page in the system browser. In browser: opens a new tab. */
export async function openLtxApiKeyPage(): Promise<boolean> {
  if (window.electronAPI) {
    return window.electronAPI.openLtxApiKeyPage()
  }
  window.open('https://app.ltx.studio', '_blank')
  return true
}

/** In Electron: opens the FAL API key page in the system browser. In browser: opens a new tab. */
export async function openFalApiKeyPage(): Promise<boolean> {
  if (window.electronAPI) {
    return window.electronAPI.openFalApiKeyPage()
  }
  window.open('https://fal.ai/dashboard/keys', '_blank')
  return true
}

/** In Electron: writes a log entry via IPC. In browser: no-op. */
export async function writeLog(_level: string, _message: string): Promise<void> {
  if (window.electronAPI) {
    return window.electronAPI.writeLog(_level, _message)
  }
}

/** In Electron: gets logs from the log file. In browser: returns empty result. */
export async function getLogs(): Promise<{ logPath: string; lines: string[]; error?: string }> {
  if (window.electronAPI) {
    return window.electronAPI.getLogs()
  }
  return { logPath: '', lines: [] }
}

/** In Electron: returns analytics state. In browser: returns disabled state. */
export async function getAnalyticsState(): Promise<{ analyticsEnabled: boolean; installationId: string }> {
  if (window.electronAPI) {
    return window.electronAPI.getAnalyticsState()
  }
  return { analyticsEnabled: false, installationId: '' }
}

/** In Electron: sets analytics enabled state. In browser: no-op. */
export async function setAnalyticsEnabled(_enabled: boolean): Promise<void> {
  if (window.electronAPI) {
    return window.electronAPI.setAnalyticsEnabled(_enabled)
  }
}

/** In Electron: returns the NOTICES/third-party licenses text. In browser: returns empty string. */
export async function getNoticesText(): Promise<string> {
  if (window.electronAPI) {
    return window.electronAPI.getNoticesText()
  }
  return ''
}
