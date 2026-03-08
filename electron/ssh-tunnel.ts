/**
 * ssh-tunnel.ts
 *
 * Manages an SSH tunnel to a remote backend (e.g. RunPod) so the Electron app
 * connects to localhost:8000 which forwards to the remote GPU server.
 *
 * Config: place a file at ~/.ltx-remote.json:
 * {
 *   "host": "b73i02x9mv48kf-6441128f@ssh.runpod.io",
 *   "keyPath": "~/.ssh/id_ed25519"
 * }
 *
 * When this file exists, the app starts an SSH tunnel instead of a local Python
 * backend. Remove the file to switch back to local mode.
 */

import { ChildProcess, spawn } from 'child_process'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { PYTHON_PORT } from './config'
import { logger } from './logger'
import { getMainWindow } from './window'

interface RemoteConfig {
  host: string
  keyPath?: string
  remotePort?: number
}

let tunnelProcess: ChildProcess | null = null
let isIntentionalStop = false
let restartTimer: ReturnType<typeof setTimeout> | null = null

const CONFIG_FILENAME = '.ltx-remote.json'
const RESTART_DELAY_MS = 3000

function expandHome(p: string): string {
  if (p.startsWith('~/') || p === '~') {
    return path.join(os.homedir(), p.slice(1))
  }
  return p
}

export function loadRemoteConfig(): RemoteConfig | null {
  const configPath = path.join(os.homedir(), CONFIG_FILENAME)
  try {
    if (!fs.existsSync(configPath)) {
      return null
    }
    const raw = fs.readFileSync(configPath, 'utf-8')
    const config = JSON.parse(raw) as RemoteConfig
    if (!config.host) {
      return null
    }
    return config
  } catch (err) {
    logger.error(`Failed to read ${configPath}: ${err}`)
    return null
  }
}

export function isRemoteBackend(): boolean {
  return loadRemoteConfig() !== null
}

function publishTunnelStatus(status: 'alive' | 'restarting' | 'dead', exitCode?: number | null): void {
  getMainWindow()?.webContents.send('backend-health-status', { status, exitCode })
}

export function startSSHTunnel(): void {
  const config = loadRemoteConfig()
  if (!config) {
    logger.error('Cannot start SSH tunnel: no remote config found')
    return
  }

  if (tunnelProcess) {
    logger.info('SSH tunnel already running')
    return
  }

  isIntentionalStop = false

  const keyPath = expandHome(config.keyPath || '~/.ssh/id_ed25519')
  const remotePort = config.remotePort || PYTHON_PORT
  const localPort = PYTHON_PORT

  const sshArgs = [
    '-N',                              // No remote command
    '-L', `${localPort}:localhost:${remotePort}`,  // Port forwarding
    '-o', 'StrictHostKeyChecking=no',
    '-o', 'ServerAliveInterval=30',    // Keep alive every 30s
    '-o', 'ServerAliveCountMax=3',     // Disconnect after 3 missed keepalives
    '-o', 'ExitOnForwardFailure=yes',  // Fail if port forwarding fails
    '-i', keyPath,
    config.host,
  ]

  logger.info(`Starting SSH tunnel: ssh ${sshArgs.join(' ')}`)

  tunnelProcess = spawn('ssh', sshArgs, {
    stdio: ['ignore', 'pipe', 'pipe'],
  })

  tunnelProcess.stdout?.on('data', (data: Buffer) => {
    logger.info(`[SSH Tunnel] ${data.toString().trim()}`)
  })

  tunnelProcess.stderr?.on('data', (data: Buffer) => {
    const output = data.toString().trim()
    if (output) {
      logger.info(`[SSH Tunnel] ${output}`)
    }
  })

  tunnelProcess.on('error', (error) => {
    logger.error(`SSH tunnel process error: ${error}`)
    tunnelProcess = null
    publishTunnelStatus('dead')
  })

  tunnelProcess.on('exit', (code) => {
    logger.info(`SSH tunnel exited with code ${code}`)
    tunnelProcess = null

    if (isIntentionalStop) {
      isIntentionalStop = false
      return
    }

    publishTunnelStatus('restarting', code)
    logger.info(`SSH tunnel will restart in ${RESTART_DELAY_MS}ms...`)
    restartTimer = setTimeout(() => {
      restartTimer = null
      startSSHTunnel()
    }, RESTART_DELAY_MS)
  })

  // Wait briefly for the tunnel to establish, then probe health
  setTimeout(async () => {
    if (!tunnelProcess) return
    try {
      const response = await fetch(`http://localhost:${localPort}/health`, {
        signal: AbortSignal.timeout(5000),
      })
      if (response.ok) {
        logger.info('SSH tunnel established — remote backend is reachable')
        publishTunnelStatus('alive')
      } else {
        logger.warn('SSH tunnel connected but backend health check failed')
        publishTunnelStatus('alive') // tunnel works, backend may still be loading
      }
    } catch {
      logger.warn('SSH tunnel health probe failed — backend may still be starting')
      // Publish alive anyway — the tunnel process is running, backend may need warmup
      publishTunnelStatus('alive')
    }
  }, 3000)
}

export function stopSSHTunnel(): void {
  isIntentionalStop = true

  if (restartTimer) {
    clearTimeout(restartTimer)
    restartTimer = null
  }

  if (tunnelProcess) {
    logger.info('Stopping SSH tunnel...')
    const pid = tunnelProcess.pid
    tunnelProcess.kill('SIGTERM')
    tunnelProcess = null

    if (pid) {
      setTimeout(() => {
        try {
          process.kill(pid, 0)
          process.kill(pid, 'SIGKILL')
        } catch {
          // Already dead
        }
      }, 3000)
    }
  }
}
