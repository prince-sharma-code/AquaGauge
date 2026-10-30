
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent, ReactNode } from 'react'
import {
  Activity, AlertTriangle, Bell, CheckCircle2,
  CloudRain, Droplets, FileImage, LayoutDashboard, MapPin,
  Menu, RefreshCw, Settings, ShieldCheck, Upload, Waves, X, Zap,
  Satellite, Database, Users, ImagePlus, Info, 
  Thermometer, Wind, Cloud, Gauge, Wifi, WifiOff, Send, Filter,
  Search, Trash2, LocateFixed, Server, CloudUpload, ClipboardList,
} from 'lucide-react'
import {
  Area, AreaChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { MapContainer, Marker, Popup, TileLayer } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const C = {
  bg: '#F0EBF0',
  navy: '#244A69',
  peach: '#F2A58B',
  lavender: '#B9A8E8',
  turquoise: '#6DBDB4',
  muted: '#71808A',
  border: '#E7E0E9',
  high: '#E87979',
  medium: '#E9B45F',
  low: '#6DBDB4',
  white: '#FFFFFF',
}

const cardStyle =
  'rounded-2xl border border-[#E7E0E9] bg-white shadow-[0_4px_20px_rgba(36,74,105,0.045)]'

const MAX_IMAGE_SIZE = 10 * 1024 * 1024
const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const STORAGE_KEY = 'aquagauge-community-reports-v1'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

const API = {
  incidents: '/api/incidents',
  communityReports: '/api/community-reports',
  analyzeImage: '/api/detect',
  weather: '/api/weather',
}

type Risk = 'High' | 'Medium' | 'Low'
type SyncStatus = 'Pending sync' | 'Synced' | 'Local only'
type VerificationStatus = 'Unverified' | 'Verified' | 'Rejected'

type CommunityReport = {
  id: string
  location: string
  description: string
  severity: Risk
  photoName: string
  latitude: number | null
  longitude: number | null
  submittedAt: string
  status: VerificationStatus
  syncStatus: SyncStatus
  source: 'Community'
}

type Incident = {
  id: string
  location: string
  description: string
  risk: Risk
  depth: string | null
  time: string
  source: string
  verificationStatus: string
  latitude: number | null
  longitude: number | null
  syncStatus: string
}

const sourceOptions = [
  {
    name: 'Weather',
    label: 'Weather Data',
    detail: 'Rainfall observations and forecasts',
    description: 'Rainfall and forecasts can indicate potential flood conditions, not confirm flooding.',
    icon: CloudRain,
    color: 'turquoise',
  },
  {
    name: 'Satellite',
    label: 'Satellite Data',
    detail: 'Remote flood observations',
    description: 'Satellite observations may identify inundated areas, subject to availability and resolution.',
    icon: Satellite,
    color: 'lavender',
  },
  {
    name: 'Water Sensors',
    label: 'Water Sensors',
    detail: 'Measured water levels',
    description: 'Connected water-level sensors can provide measured local readings.',
    icon: Database,
    color: 'peach',
  },
  {
    name: 'Community',
    label: 'Community Reports',
    detail: 'Local observations',
    description: 'Community observations are unverified until checked.',
    icon: Users,
    color: 'blue',
  },
]

const depthData = [
  { time: '08 AM', depth: 12 }, { time: '09 AM', depth: 18 },
  { time: '10 AM', depth: 15 }, { time: '11 AM', depth: 28 },
  { time: '12 PM', depth: 35 }, { time: '01 PM', depth: 31 },
  { time: '02 PM', depth: 48 }, { time: '03 PM', depth: 42 },
]

const riskData = [
  { name: 'Low', value: 45, color: C.low },
  { name: 'Medium', value: 30, color: C.medium },
  { name: 'High', value: 25, color: C.high },
]

const rainfallData = [
  { time: '8 AM', rainfall: 1 }, { time: '10 AM', rainfall: 3 },
  { time: '12 PM', rainfall: 7 }, { time: '2 PM', rainfall: 12 },
  { time: '4 PM', rainfall: 9 }, { time: '6 PM', rainfall: 5 },
  { time: '8 PM', rainfall: 2 },
]

const forecastData = [
  { time: 'Now', condition: 'Cloudy', temperature: '28°', rain: '30%' },
  { time: '4 PM', condition: 'Light rain', temperature: '27°', rain: '45%' },
  { time: '6 PM', condition: 'Rain', temperature: '25°', rain: '65%' },
  { time: '8 PM', condition: 'Cloudy', temperature: '24°', rain: '25%' },
  { time: '10 PM', condition: 'Cloudy', temperature: '23°', rain: '15%' },
]

const demoIncidents: Incident[] = [
  {
    id: 'DEMO-001',
    location: 'Sector 12 Demo Zone',
    description: 'Illustrative sample water-level observation.',
    risk: 'High',
    depth: '48 cm',
    time: '10 minutes ago',
    source: 'Demo sensor',
    verificationStatus: 'Demo record',
    latitude: 28.6139,
    longitude: 77.209,
    syncStatus: 'Demo data',
  },
  {
    id: 'DEMO-002',
    location: 'Riverside Road Demo Zone',
    description: 'Illustrative sample observation.',
    risk: 'Medium',
    depth: '32 cm',
    time: '25 minutes ago',
    source: 'Demo sensor',
    verificationStatus: 'Demo record',
    latitude: 28.625,
    longitude: 77.215,
    syncStatus: 'Demo data',
  },
  {
    id: 'DEMO-003',
    location: 'Central Avenue Demo Zone',
    description: 'Illustrative sample observation.',
    risk: 'Low',
    depth: '12 cm',
    time: '42 minutes ago',
    source: 'Demo sensor',
    verificationStatus: 'Demo record',
    latitude: 28.605,
    longitude: 77.195,
    syncStatus: 'Demo data',
  },
]

const makeRiskIcon = (color: string) =>
  L.divIcon({
    className: '',
    html: `<div style="width:20px;height:20px;background:${color};border:3px solid white;border-radius:50%;box-shadow:0 2px 10px rgba(36,74,105,.25)"></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  })

function readSavedReports(): CommunityReport[] {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? JSON.parse(saved) as CommunityReport[] : []
  } catch {
    return []
  }
}

function saveReports(reports: CommunityReport[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(reports))
    return true
  } catch {
    return false
  }
}

function getRiskColor(risk: string) {
  if (risk === 'High') return C.high
  if (risk === 'Medium') return C.medium
  return C.low
}

function formatTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [imageName, setImageName] = useState('')
  const [imageSize, setImageSize] = useState(0)
  const [imageType, setImageType] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<unknown>(null)
  const [uploadError, setUploadError] = useState('')
  const parsedAnalysis = analysisResult as {
  analysis_source?: string
  model_inference?: boolean
  result?: {
    status?: string
    detections?: Array<{ confidence?: number; [key: string]: unknown }>
    water_region?: {
      detected?: boolean
      confidence?: number
      area_percentage?: number
    }
  }
} | null
  const [activeSection, setActiveSection] = useState('overview')
  const [currentTime, setCurrentTime] = useState(new Date())
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [selectedSources, setSelectedSources] = useState([
    'Weather', 'Satellite', 'Water Sensors', 'Community',
  ])

  const [communityReports, setCommunityReports] = useState<CommunityReport[]>(readSavedReports)
  const [reportLocation, setReportLocation] = useState('')
  const [reportDescription, setReportDescription] = useState('')
  const [reportSeverity, setReportSeverity] = useState<Risk>('Medium')
  const [reportPhoto, setReportPhoto] = useState<File | null>(null)
  const [reportLatitude, setReportLatitude] = useState('')
  const [reportLongitude, setReportLongitude] = useState('')
  const [reportMessage, setReportMessage] = useState('')
  const [reportError, setReportError] = useState('')
  const [submittingReport, setSubmittingReport] = useState(false)

  const [online, setOnline] = useState(navigator.onLine)
  const [syncing, setSyncing] = useState(false)
  const [syncMessage, setSyncMessage] = useState('')
  const [apiStatus, setApiStatus] = useState<'not-configured' | 'connected' | 'error'>(
    API_BASE_URL ? 'not-configured' : 'not-configured',
  )
  const [apiMessage, setApiMessage] = useState(
    API_BASE_URL ? 'API configured; connection not checked yet.' : 'API base URL is not configured.',
  )

  const [apiIncidents, setApiIncidents] = useState<Incident[]>([])
  const [incidentFilter, setIncidentFilter] = useState('All')
  const [incidentSearch, setIncidentSearch] = useState('')
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null)
  const [weatherStatus, setWeatherStatus] = useState('Sample data')
  const [weatherSummary, setWeatherSummary] = useState<Record<string, unknown> | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const reportPhotoRef = useRef<HTMLInputElement>(null)

  const pendingCount = communityReports.filter(r => r.syncStatus !== 'Synced').length

  useEffect(() => {
    const timer = window.setInterval(() => setCurrentTime(new Date()), 1000)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    const handleOnline = () => setOnline(true)
    const handleOffline = () => setOnline(false)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  useEffect(() => {
    saveReports(communityReports)
  }, [communityReports])

  useEffect(() => {
    return () => {
      if (selectedImage) URL.revokeObjectURL(selectedImage)
    }
  }, [selectedImage])

  const navigateTo = (section: string) => {
    setActiveSection(section)
    setMobileMenuOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const toggleSource = (sourceName: string) => {
    setSelectedSources(previous =>
      previous.includes(sourceName)
        ? previous.filter(item => item !== sourceName)
        : [...previous, sourceName],
    )
  }

  const apiRequest = useCallback(async (path: string, options: RequestInit = {}) => {
    if (!API_BASE_URL) throw new Error('API base URL is not configured.')
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: {
        ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...options.headers,
      },
    })
    if (!response.ok) {
      const message = await response.text().catch(() => '')
      throw new Error(message || `Server returned HTTP ${response.status}`)
    }
    if (response.status === 204) return null
    return response.json()
  }, [])

  const checkBackend = async () => {
    if (!API_BASE_URL) {
      setApiStatus('not-configured')
      setApiMessage('Set VITE_API_BASE_URL in frontend/.env to your backend address.')
      return
    }
    try {
      await apiRequest(API.incidents)
      setApiStatus('connected')
      setApiMessage('Incident endpoint responded successfully.')
    } catch (error) {
      setApiStatus('error')
      setApiMessage(error instanceof Error ? error.message : 'Could not reach the backend.')
    }
  }

  const loadIncidents = useCallback(async () => {
    if (!API_BASE_URL || !online) return
    try {
      const data = await apiRequest(API.incidents)
      const rows = Array.isArray(data) ? data : data?.incidents
      if (!Array.isArray(rows)) throw new Error('Unexpected incidents response format.')
      const parsed: Incident[] = rows.map((item: any, index: number) => ({
        id: String(item.incident_id ?? item.id ?? `API-${index + 1}`),
        location: String(item.location?.name ?? item.location ?? 'Unknown location'),
        description: String(item.description ?? item.observation ?? 'No description supplied.'),
        risk: (['High', 'Medium', 'Low'].includes(item.risk_level ?? item.risk)
          ? item.risk_level ?? item.risk : 'Medium') as Risk,
        depth: item.water_depth_m != null
          ? `${item.water_depth_m} m`
          : item.water_depth_cm != null ? `${item.water_depth_cm} cm` : null,
        time: String(item.timestamp ?? item.created_at ?? 'Time unavailable'),
        source: String(item.source ?? 'Unknown source'),
        verificationStatus: String(item.verification_status ?? 'Unverified'),
        latitude: Number.isFinite(Number(item.location?.latitude ?? item.latitude))
          ? Number(item.location?.latitude ?? item.latitude) : null,
        longitude: Number.isFinite(Number(item.location?.longitude ?? item.longitude))
          ? Number(item.location?.longitude ?? item.longitude) : null,
        syncStatus: 'Server record',
      }))
      setApiIncidents(parsed)
      setApiStatus('connected')
    } catch (error) {
      setApiStatus('error')
      setApiMessage(error instanceof Error ? error.message : 'Could not load incidents.')
    }
  }, [apiRequest, online])

  const loadWeather = useCallback(async () => {
    if (!API_BASE_URL || !online) return
    try {
      const data = await apiRequest(API.weather)
      setWeatherSummary(data)
      setWeatherStatus('Connected API data')
    } catch {
      setWeatherStatus('Sample data · weather endpoint unavailable')
    }
  }, [apiRequest, online])

  useEffect(() => {
    void loadIncidents()
    void loadWeather()
  }, [loadIncidents, loadWeather])

  const syncPendingReports = async () => {
    if (!online) {
      setSyncMessage('You are offline. Reports remain saved on this device.')
      return
    }
    if (!API_BASE_URL) {
      setSyncMessage('Backend is not configured. Reports remain saved locally.')
      return
    }
    const pending = communityReports.filter(r => r.syncStatus === 'Pending sync')
    if (pending.length === 0) {
      setSyncMessage('There are no pending reports to sync.')
      return
    }

    setSyncing(true)
    setSyncMessage('')
    let synced = 0

    for (const report of pending) {
      try {
        await apiRequest(API.communityReports, {
          method: 'POST',
          body: JSON.stringify({
            report_id: report.id,
            location: {
              name: report.location,
              latitude: report.latitude,
              longitude: report.longitude,
            },
            description: report.description,
            severity: report.severity.toLowerCase(),
            source: 'community',
            timestamp: report.submittedAt,
            verification_status: 'unverified',
          }),
        })
        setCommunityReports(previous => previous.map(item =>
          item.id === report.id ? { ...item, syncStatus: 'Synced' } : item,
        ))
        synced += 1
      } catch {
        // Keep this report pending so the user can retry later.
      }
    }

    setSyncing(false)
    setSyncMessage(
      synced > 0
        ? `${synced} report(s) synced. Any unsuccessful reports remain pending.`
        : 'Sync failed. Reports remain saved locally for another attempt.',
    )
    void loadIncidents()
  }

  useEffect(() => {
    if (online && API_BASE_URL && pendingCount > 0) {
      // Deliberately do not auto-upload without the user pressing Sync.
    }
  }, [online, pendingCount])

  const handleCommunitySubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setReportError('')
    setReportMessage('')

    if (!reportLocation.trim() || !reportDescription.trim()) {
      setReportError('Please enter a location and describe what you observed.')
      return
    }

    const lat = reportLatitude.trim() ? Number(reportLatitude) : null
    const lng = reportLongitude.trim() ? Number(reportLongitude) : null
    if ((reportLatitude.trim() && !Number.isFinite(lat)) ||
        (reportLongitude.trim() && !Number.isFinite(lng))) {
      setReportError('Latitude and longitude must be valid numbers.')
      return
    }
    if ((lat !== null && (lat < -90 || lat > 90)) ||
        (lng !== null && (lng < -180 || lng > 180))) {
      setReportError('Latitude must be between -90 and 90; longitude between -180 and 180.')
      return
    }

    setSubmittingReport(true)
    const report: CommunityReport = {
      id: `LOCAL-${Date.now()}`,
      location: reportLocation.trim(),
      description: reportDescription.trim(),
      severity: reportSeverity,
      photoName: reportPhoto?.name ?? '',
      latitude: lat,
      longitude: lng,
      submittedAt: new Date().toISOString(),
      status: 'Unverified',
      syncStatus: 'Pending sync',
      source: 'Community',
    }

    // Save locally first. This ensures reports survive refreshes and offline use.
    setCommunityReports(previous => [report, ...previous])
    const persisted = saveReports([report, ...communityReports])

    setReportLocation('')
    setReportDescription('')
    setReportSeverity('Medium')
    setReportPhoto(null)
    setReportLatitude('')
    setReportLongitude('')
    if (reportPhotoRef.current) reportPhotoRef.current.value = ''
    setSubmittingReport(false)

    setReportMessage(
      persisted
        ? online
          ? 'Report saved on this device. It is pending backend sync.'
          : 'Report saved on this device while offline. It will remain pending until synced.'
        : 'Report was added for this session, but browser storage could not save it.',
    )
  }

  const handleReportPhoto = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    if (!file) {
      setReportPhoto(null)
      return
    }
    if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      setReportError('Report photo must be JPG, PNG or WebP.')
      event.target.value = ''
      return
    }
    if (file.size > MAX_IMAGE_SIZE) {
      setReportError('Report photo must be smaller than 10 MB.')
      event.target.value = ''
      return
    }
    setReportPhoto(file)
    setReportError('')
  }

  const removeReport = (id: string) => {
    setCommunityReports(previous => previous.filter(report => report.id !== id))
  }

  const handleUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setUploadError('')

    if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      setUploadError('Unsupported file type. Choose a JPG, PNG or WebP image.')
      event.target.value = ''
      return
    }
    if (file.size > MAX_IMAGE_SIZE) {
      setUploadError('This image is larger than 10 MB.')
      event.target.value = ''
      return
    }
    if (file.size === 0) {
      setUploadError('This image file appears to be empty.')
      event.target.value = ''
      return
    }
    if (selectedImage) URL.revokeObjectURL(selectedImage)
    setSelectedImage(URL.createObjectURL(file))
    setImageName(file.name)
    setImageSize(file.size)
    setImageType(file.type)
  }

  const handleAnalyze = async () => {
    if (!selectedImage || isAnalyzing) return
    const file = fileInputRef.current?.files?.[0]
    if (!file) {
      setUploadError('Please select the image again before analysis.')
      return
    }

    setIsAnalyzing(true)
    setUploadError('')

    if (!API_BASE_URL) {
      window.setTimeout(() => {
        setIsAnalyzing(false)
      }, 900)
      return
    }

    try {
      const data = await apiRequest(API.analyzeImage, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: file.name }),
    })
      setAnalysisResult({
        analysis_source: data.source === 'PRECOMPUTED_DEMO'
          ? 'PRECOMPUTED_DEMO_FIXTURE'
          : data.source,
        model_inference: data.model_info?.fallback_active !== true,
        result: {
          status: data.source ?? 'UNKNOWN',
          image_dimensions: data.image_dimensions,
          detections: data.detections ?? [],
          water_boundary: data.water_boundary,
          water_region: {
            detected: Boolean(data.water_boundary),
          },
        },
      })
    } catch (error) {
      setUploadError(
        `Image API unavailable: ${error instanceof Error ? error.message : 'Request failed'}. The image has not been analysed.`,
      )
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    if (selectedImage) URL.revokeObjectURL(selectedImage)
    setSelectedImage(null)
    setImageName('')
    setImageSize(0)
    setImageType('')
    setIsAnalyzing(false)
    setUploadError('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }
  const allIncidents = useMemo<Incident[]>(() => {
    const local: Incident[] = communityReports.map(report => ({
      id: report.id,
      location: report.location,
      description: report.description,
      risk: report.severity,
      depth: null,
      time: report.submittedAt,
      source: 'Community report',
      verificationStatus: report.status,
      latitude: report.latitude,
      longitude: report.longitude,
      syncStatus: report.syncStatus,
    }))
    return [...local, ...apiIncidents, ...demoIncidents]
  }, [communityReports, apiIncidents])

  const filteredIncidents = useMemo(() => {
    return allIncidents.filter(item => {
      const matchesRisk = incidentFilter === 'All' || item.risk === incidentFilter
      const search = incidentSearch.toLowerCase()
      const matchesSearch = !search ||
        item.location.toLowerCase().includes(search) ||
        item.description.toLowerCase().includes(search) ||
        item.source.toLowerCase().includes(search)
      return matchesRisk && matchesSearch
    })
  }, [allIncidents, incidentFilter, incidentSearch])

  const mapIncidents = allIncidents.filter(item =>
    item.latitude !== null && item.longitude !== null &&
    Number.isFinite(item.latitude) && Number.isFinite(item.longitude),
  )

  const formattedTime = currentTime.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
  const formattedDate = currentTime.toLocaleDateString([], {
    weekday: 'long', day: 'numeric', month: 'long', year: 'numeric',
  })

  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'sources', label: 'Data Sources', icon: Database },
    { id: 'weather', label: 'Weather Monitoring', icon: CloudRain },
    { id: 'community', label: 'Community Reports', icon: Users },
    { id: 'map', label: 'Flood Map & Incidents', icon: MapPin },
    { id: 'analysis', label: 'Image Analysis', icon: FileImage },
    { id: 'settings', label: 'Settings', icon: Settings },
  ]
  const activePage = navItems.find(item => item.id === activeSection) ?? navItems[0]

  return (
    <div className="min-h-screen bg-[#F0EBF0] text-[#244A69]">
      <div className="flex min-h-screen">
        {mobileMenuOpen && (
          <button
            aria-label="Close navigation menu"
            onClick={() => setMobileMenuOpen(false)}
            className="fixed inset-0 z-40 bg-[#172F43]/40 lg:hidden"
          />
        )}

        <aside className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col bg-[#244A69] text-white shadow-xl transition-transform lg:static lg:translate-x-0 ${
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        }`}>
          <div className="flex h-20 items-center justify-between border-b border-white/15 px-6">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-[#6DBDB4] p-2.5"><Waves className="h-7 w-7 text-[#244A69]" /></div>
              <div>
                <h1 className="text-xl font-bold tracking-wide">AquaGauge</h1>
                <p className="text-xs text-[#C5D9E5]">Flood Intelligence</p>
              </div>
            </div>
            <button onClick={() => setMobileMenuOpen(false)} className="text-white/70 lg:hidden" aria-label="Close menu"><X /></button>
          </div>

          <div className="px-4 py-7">
            <p className="mb-4 px-3 text-[10px] font-bold uppercase tracking-[0.2em] text-[#B9A8E8]">Workspace</p>
            <nav className="space-y-2">
              {navItems.map(item => {
                const Icon = item.icon
                const active = activeSection === item.id
                return (
                  <button key={item.id} onClick={() => navigateTo(item.id)} aria-current={active ? 'page' : undefined}
                    className={`flex w-full items-center gap-3 rounded-xl px-3 py-3.5 text-left text-sm transition ${
                      active ? 'bg-[#F2A58B] font-bold text-[#244A69] shadow-lg' : 'text-[#E1EAF0] hover:bg-white/10'
                    }`}>
                    <Icon size={19} />{item.label}
                  </button>
                )
              })}
            </nav>
          </div>

          <div className="mt-auto border-t border-white/15 p-5">
            <div className="rounded-2xl border border-white/15 bg-white/10 p-4">
              <div className="mb-3 flex items-center gap-2 text-[#B9A8E8]">
                <ShieldCheck size={18} /><span className="text-sm font-semibold text-white">Connection Status</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-[#DCE8ED]">
                {online ? <Wifi size={15} className="text-[#6DBDB4]" /> : <WifiOff size={15} className="text-[#F2A58B]" />}
                {online ? 'Device online' : 'Offline mode'}
              </div>
              <p className="mt-2 text-xs text-[#DCE8ED]">{pendingCount} report(s) pending sync</p>
            </div>
            <p className="mt-5 text-center text-xs text-[#C5D9E5]">AquaGauge • Hackathon Demo</p>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-30 flex min-h-20 items-center justify-between border-b border-[#E7E0E9] bg-[#F0EBF0]/95 px-4 backdrop-blur-md sm:px-8">
            <div className="flex items-center gap-3">
              <button onClick={() => setMobileMenuOpen(true)} className="text-[#244A69] lg:hidden" aria-label="Open menu"><Menu /></button>
              <div>
                <h2 className="text-lg font-bold text-[#244A69]">{activePage.label}</h2>
                <p className="hidden text-xs text-[#71808A] sm:block">Multi-source flood monitoring and risk intelligence</p>
              </div>
            </div>
            <div className="flex items-center gap-3 sm:gap-5">
              <div className="hidden text-right sm:block">
                <p className="font-mono text-sm font-bold text-[#287E83]">{formattedTime}</p>
                <p className="text-xs text-[#71808A]">{formattedDate}</p>
              </div>
              <div className={`flex items-center gap-2 rounded-full border px-3 py-2 text-xs font-bold ${
                online ? 'border-[#B8DED8] bg-[#DCEFEB] text-[#287E83]' : 'border-[#F0CFC7] bg-[#FDF0EC] text-[#B34E43]'
              }`}>
                {online ? <Wifi size={15} /> : <WifiOff size={15} />}
                <span className="hidden sm:inline">{online ? 'Online' : 'Offline'}</span>
              </div>
              <div className="rounded-full border border-[#E7E0E9] bg-white p-2.5"><Bell size={18} /></div>
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#B9A8E8] text-sm font-bold">AG</div>
            </div>
          </header>

          <div className="space-y-9 p-4 sm:p-8">
            {/* OVERVIEW */}
            {activeSection === 'overview' && (
            <section className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold text-[#287E83]">AQUAGAUGE / COMMAND CENTER</p>
                  <h2 className="mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl">Flood Intelligence Overview</h2>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-[#71808A]">
                    Monitor flood-related observations, review potential risk indicators and track information from multiple sources.
                  </p>
                </div>
                <div className="rounded-2xl border border-[#B8DED8] bg-[#DCEFEB] px-4 py-3">
                  <div className="flex items-center gap-2 text-sm font-bold text-[#287E83]"><span className="h-2.5 w-2.5 rounded-full bg-[#6DBDB4]" />Dashboard active</div>
                  <p className="mt-1 text-xs text-[#71808A]">{API_BASE_URL ? 'Backend configured' : 'Demo mode · API not configured'}</p>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <StatCard title="Total Incidents" value={String(allIncidents.length)} detail="Demo, API and local reports" icon={<MapPin />} color="turquoise" />
                <StatCard title="High-Risk Records" value={String(allIncidents.filter(i => i.risk === 'High').length)} detail="Classification, not emergency confirmation" icon={<AlertTriangle />} color="peach" />
                <StatCard title="Community Reports" value={String(communityReports.length)} detail="Reports saved in this browser" icon={<Users />} color="lavender" />
                <StatCard title="Pending Sync" value={String(pendingCount)} detail="Waiting for backend sync" icon={<CloudUpload />} color="blue" />
              </div>

              <div className="grid gap-6 xl:grid-cols-3">
                <div className={`${cardStyle} p-5 xl:col-span-2`}>
                  <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                    <div><h3 className="text-base font-bold">Water Level Trend</h3><p className="mt-1 text-xs text-[#71808A]">Illustrative readings across the day</p></div>
                    <span className="rounded-full bg-[#DCEFEB] px-3 py-2 text-xs font-bold text-[#287E83]">Sample data · cm</span>
                  </div>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={depthData}>
                        <defs><linearGradient id="depthGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#6DBDB4" stopOpacity={0.55} /><stop offset="100%" stopColor="#6DBDB4" stopOpacity={0.02} /></linearGradient></defs>
                        <CartesianGrid stroke="#EAE4EC" strokeDasharray="4 4" />
                        <XAxis dataKey="time" stroke="#71808A" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                        <YAxis stroke="#71808A" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} unit=" cm" />
                        <Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #E7E0E9', borderRadius: 12 }} formatter={(value) => [`${value} cm`, 'Sample depth']} />
                        <Area type="monotone" dataKey="depth" stroke="#287E83" strokeWidth={3} fill="url(#depthGradient)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="mt-2 text-xs text-[#71808A]">Illustrative readings only; not connected to a live sensor.</p>
                </div>
                <div className={`${cardStyle} p-5`}>
                  <h3 className="text-base font-bold">Risk Distribution</h3>
                  <p className="mt-1 text-xs text-[#71808A]">Illustrative demo dataset</p>
                  <div className="relative mt-3 h-52">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={riskData} dataKey="value" nameKey="name" innerRadius={58} outerRadius={82} paddingAngle={5}>
                          {riskData.map(entry => <Cell key={entry.name} fill={entry.color} />)}
                        </Pie>
                        <Tooltip formatter={(value) => [`${value}%`, 'Demo zones']} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                      <span className="text-3xl font-extrabold">12</span><span className="text-xs text-[#71808A]">Demo zones</span>
                    </div>
                  </div>
                  <div className="space-y-3">
                    {riskData.map(risk => <div key={risk.name} className="flex items-center justify-between text-sm"><div className="flex items-center gap-2 text-[#71808A]"><span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: risk.color }} />{risk.name} risk</div><span className="font-bold">{risk.value}%</span></div>)}
                  </div>
                </div>
              </div>

              <div className={`${cardStyle} p-5`}>
                <div className="mb-5 flex flex-wrap items-center justify-between gap-2">
                  <div><h3 className="text-base font-bold">Monitoring Sources</h3><p className="mt-1 text-xs text-[#71808A]">Select sources to configure monitoring preferences</p></div>
                  <span className="rounded-full bg-[#E9E2F5] px-3 py-1.5 text-xs font-bold text-[#7664A8]">{selectedSources.length} of 4 selected</span>
                </div>
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  {sourceOptions.map(source => {
                    const Icon = source.icon
                    return <button key={source.name} onClick={() => { toggleSource(source.name); navigateTo('sources') }} className="text-left"><SourceCard icon={<Icon />} label={source.label} detail={source.detail} color={source.color} selected={selectedSources.includes(source.name)} /></button>
                  })}
                </div>
                <p className="mt-4 rounded-xl border border-[#F0DEBC] bg-[#FFF6E7] p-3 text-xs leading-5 text-[#8A682F]">Selecting a category does not itself activate an external integration.</p>
              </div>

              <div className={`${cardStyle} overflow-hidden`}>
                <div className="border-b border-[#E7E0E9] p-5"><h3 className="font-bold">Recent Flood Observations</h3><p className="mt-1 text-xs text-[#71808A]">Latest records available in this dashboard</p></div>
                {allIncidents.slice(0, 5).map((incident, index) => (
                  <div key={incident.id} className={`flex flex-wrap items-center justify-between gap-3 p-4 ${index < Math.min(allIncidents.length, 5) - 1 ? 'border-b border-[#E7E0E9]' : ''}`}>
                    <div className="flex items-center gap-3"><div className="rounded-xl bg-[#F9E5DF] p-3"><Waves className="text-[#D77F69]" size={19} /></div><div><p className="text-sm font-bold">{incident.location}</p><p className="mt-1 text-xs text-[#71808A]">{incident.source} · {incident.verificationStatus}</p></div></div>
                    <div className="flex items-center gap-3">{incident.depth && <span className="text-sm text-[#71808A]">{incident.depth}</span>}<RiskBadge risk={incident.risk} /></div>
                  </div>
                ))}
                <div className="border-t border-[#E7E0E9] bg-[#F8F5F8] px-5 py-3 text-xs text-[#71808A]">Demo records are illustrative. Community reports remain unverified until checked.</div>
              </div>
            </section>
            )}

            {/* DATA SOURCES */}
            {activeSection === 'sources' && (
            <section className="space-y-5">
              <div><p className="text-sm font-semibold text-[#287E83]">MONITORING CONFIGURATION</p><h2 className="mt-2 text-2xl font-extrabold">Data Source Selection</h2><p className="mt-2 text-sm text-[#71808A]">Choose which information sources should be included in your monitoring workflow.</p></div>
              <div className="grid gap-4 sm:grid-cols-2">
                {sourceOptions.map(source => {
                  const Icon = source.icon
                  const selected = selectedSources.includes(source.name)
                  return <button key={source.name} onClick={() => toggleSource(source.name)} aria-pressed={selected} className={`rounded-2xl border p-5 text-left transition hover:-translate-y-1 hover:shadow-md ${selected ? 'border-[#6DBDB4] bg-[#F0F8F6]' : 'border-[#E7E0E9] bg-white'}`}>
                    <div className="flex items-start justify-between gap-3"><div className={`rounded-xl p-3 ${sourceColor(source.color)}`}><Icon size={24} /></div><span className={`rounded-full px-3 py-1 text-xs font-bold ${selected ? 'bg-[#DCEFEB] text-[#287E83]' : 'bg-[#F0EBF0] text-[#71808A]'}`}>{selected ? 'Selected' : 'Not selected'}</span></div>
                    <h3 className="mt-4 font-bold">{source.label}</h3><p className="mt-2 text-sm leading-6 text-[#71808A]">{source.description}</p>
                  </button>
                })}
              </div>
              <div className={`${cardStyle} p-5`}><h3 className="font-bold">Your monitoring preferences</h3><p className="mt-1 text-sm text-[#71808A]">{selectedSources.length} of 4 source categories selected · Stored for this session</p><div className="mt-4 flex flex-wrap gap-2">{selectedSources.map(name => <span key={name} className="rounded-full border border-[#B8DED8] bg-[#F0F8F6] px-3 py-1.5 text-xs font-semibold text-[#287E83]">{name}</span>)}</div></div>
            </section>
            )}

            {/* WEATHER */}
            {activeSection === 'weather' && (
            <section className="space-y-5">
              <div className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-sm font-semibold text-[#287E83]">WEATHER INTELLIGENCE</p><h2 className="mt-2 text-2xl font-extrabold">Weather Monitoring</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-[#71808A]">Review rainfall patterns and weather conditions that may contribute to flood risk.</p></div><span className="rounded-full border border-[#F0DEBC] bg-[#FFF6E7] px-3 py-2 text-xs font-bold text-[#8A682F]">{weatherStatus}</span></div>

              {weatherSummary ? (
                <div className="rounded-2xl border border-[#B8DED8] bg-[#F0F8F6] p-4 text-sm text-[#287E83]"><CheckCircle2 className="mr-2 inline" size={18} />Weather API data received. Review the returned response with your backend teammate to confirm its fields and units.</div>
              ) : (
                <div className="flex items-start gap-3 rounded-2xl border border-[#F0DEBC] bg-[#FFF6E7] p-4"><Info size={20} className="mt-0.5 shrink-0 text-[#A87323]" /><div><p className="text-sm font-bold text-[#8A682F]">Sample weather values</p><p className="mt-1 text-xs leading-5 text-[#8A682F]">No weather API data has been loaded. Values below are illustrative, not current measurements or forecasts.</p></div></div>
              )}

              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <WeatherCard label="Temperature" value={String(weatherSummary?.temperature ?? '28°C')} detail="Sample unless supplied by API" icon={<Thermometer />} />
                <WeatherCard label="Rainfall" value={String(weatherSummary?.rainfall ?? '12 mm')} detail="Sample hourly rainfall" icon={<CloudRain />} />
                <WeatherCard label="Humidity" value={String(weatherSummary?.humidity ?? '76%')} detail="Sample relative humidity" icon={<Droplets />} />
                <WeatherCard label="Wind speed" value={String(weatherSummary?.wind_speed ?? '14 km/h')} detail="Sample wind speed" icon={<Wind />} />
              </div>

              <div className="grid gap-6 xl:grid-cols-3">
                <div className={`${cardStyle} p-5 xl:col-span-2`}><div className="mb-5"><h3 className="font-bold">Rainfall Trend</h3><p className="mt-1 text-xs text-[#71808A]">Illustrative hourly rainfall pattern · mm</p></div><div className="h-64"><ResponsiveContainer width="100%" height="100%"><AreaChart data={rainfallData}><defs><linearGradient id="rainGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#6DBDB4" stopOpacity={0.5} /><stop offset="100%" stopColor="#6DBDB4" stopOpacity={0.03} /></linearGradient></defs><CartesianGrid stroke="#EAE4EC" strokeDasharray="4 4" /><XAxis dataKey="time" stroke="#71808A" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis stroke="#71808A" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} unit=" mm" /><Tooltip formatter={(value) => [`${value} mm`, 'Sample rainfall']} /><Area type="monotone" dataKey="rainfall" stroke="#287E83" strokeWidth={3} fill="url(#rainGradient)" /></AreaChart></ResponsiveContainer></div><p className="mt-2 text-xs text-[#71808A]">This chart is illustrative and does not represent observed rainfall.</p></div>
                <div className={`${cardStyle} p-5`}><div className="flex items-center gap-2"><div className="rounded-xl bg-[#FBE8E0] p-2.5 text-[#C8755E]"><Gauge size={20} /></div><h3 className="font-bold">Weather Risk Indicator</h3></div><div className="mt-5 rounded-xl border border-[#F0DEBC] bg-[#FFF6E7] p-4"><p className="text-xs font-semibold uppercase tracking-wide text-[#A87323]">Illustrative assessment</p><p className="mt-2 text-2xl font-extrabold text-[#A87323]">Moderate</p><p className="mt-2 text-xs leading-5 text-[#8A682F]">Rainfall can contribute to potential flood risk but does not confirm flooding.</p></div><p className="mt-4 text-xs leading-5 text-[#71808A]">A specific flooded road or water depth requires supporting local evidence.</p></div>
              </div>

              <div className={`${cardStyle} overflow-hidden`}><div className="border-b border-[#E7E0E9] p-5"><h3 className="font-bold">Sample Weather Forecast</h3><p className="mt-1 text-xs text-[#71808A]">Illustrative upcoming conditions</p></div><div className="grid grid-cols-2 gap-3 p-4 sm:grid-cols-3 lg:grid-cols-5">{forecastData.map(item => <div key={item.time} className="rounded-xl border border-[#E7E0E9] bg-[#FAF8FB] p-4 text-center"><p className="text-xs font-bold text-[#71808A]">{item.time}</p><div className="my-3 flex justify-center text-[#287E83]">{item.condition.includes('rain') || item.condition === 'Rain' ? <CloudRain size={26} /> : <Cloud size={26} />}</div><p className="text-xl font-extrabold">{item.temperature}</p><p className="mt-1 text-xs text-[#71808A]">{item.condition}</p><p className="mt-3 text-xs font-bold text-[#287E83]"><Droplets size={12} className="mr-1 inline" />{item.rain}</p></div>)}</div><div className="border-t border-[#E7E0E9] bg-[#F8F5F8] px-5 py-3 text-xs text-[#71808A]">Forecast entries are sample interface data.</div></div>
            </section>
            )}

            {/* COMMUNITY REPORTING */}
            {activeSection === 'community' && (
            <section className="space-y-5">
              <div><p className="text-sm font-semibold text-[#287E83]">LOCAL OBSERVATIONS</p><h2 className="mt-2 text-2xl font-extrabold">Community Flood Reporting</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-[#71808A]">Record observations from residents and volunteers. New reports are marked unverified until checked.</p></div>

              <div className="grid gap-4 sm:grid-cols-3">
                <StatCard title="Reports received" value={String(communityReports.length)} detail="Stored in this browser" icon={<ClipboardList />} color="turquoise" />
                <StatCard title="Awaiting sync" value={String(pendingCount)} detail="Not confirmed on server" icon={<CloudUpload />} color="peach" />
                <StatCard title="Verified reports" value={String(communityReports.filter(r => r.status === 'Verified').length)} detail="Verification requires backend workflow" icon={<ShieldCheck />} color="lavender" />
              </div>

              <div className="grid gap-6 xl:grid-cols-2">
                <form onSubmit={handleCommunitySubmit} className={`${cardStyle} space-y-4 p-5`}>
                  <div><h3 className="font-bold">Submit a flood observation</h3><p className="mt-1 text-xs text-[#71808A]">Fields marked required must be completed.</p></div>
                  <div><label htmlFor="report-location" className="mb-1.5 block text-sm font-semibold">Location *</label><input id="report-location" value={reportLocation} onChange={e => setReportLocation(e.target.value)} required maxLength={160} placeholder="Street, area, landmark or village" className="w-full rounded-xl border border-[#DCD3E4] bg-[#FCFAFC] px-4 py-3 text-sm outline-none focus:border-[#287E83]" /></div>
                  <div><label htmlFor="report-description" className="mb-1.5 block text-sm font-semibold">What did you observe? *</label><textarea id="report-description" value={reportDescription} onChange={e => setReportDescription(e.target.value)} required maxLength={1200} rows={4} placeholder="Describe the water level, affected road, time observed or other relevant details..." className="w-full resize-y rounded-xl border border-[#DCD3E4] bg-[#FCFAFC] px-4 py-3 text-sm outline-none focus:border-[#287E83]" /><p className="mt-1 text-right text-xs text-[#71808A]">{reportDescription.length}/1200</p></div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div><label htmlFor="report-severity" className="mb-1.5 block text-sm font-semibold">Observed severity</label><select id="report-severity" value={reportSeverity} onChange={e => setReportSeverity(e.target.value as Risk)} className="w-full rounded-xl border border-[#DCD3E4] bg-white px-4 py-3 text-sm"><option>Low</option><option>Medium</option><option>High</option></select><p className="mt-1 text-xs text-[#71808A]">Reporter-selected, not an automated risk score.</p></div>
                    <div><label htmlFor="report-photo" className="mb-1.5 block text-sm font-semibold">Optional photo</label><input ref={reportPhotoRef} id="report-photo" type="file" accept="image/jpeg,image/png,image/webp" onChange={handleReportPhoto} className="block w-full text-xs text-[#71808A] file:mr-2 file:rounded-lg file:border-0 file:bg-[#E9E2F5] file:px-3 file:py-2 file:font-semibold file:text-[#51447C]" /><p className="mt-1 text-xs text-[#71808A]">{reportPhoto ? reportPhoto.name : 'JPG, PNG or WebP · max 10 MB'}</p></div>
                  </div>
                  <div><p className="mb-2 text-sm font-semibold">Optional map coordinates</p><div className="grid grid-cols-2 gap-3"><input aria-label="Latitude" value={reportLatitude} onChange={e => setReportLatitude(e.target.value)} placeholder="Latitude" className="min-w-0 rounded-xl border border-[#DCD3E4] px-3 py-2.5 text-sm" /><input aria-label="Longitude" value={reportLongitude} onChange={e => setReportLongitude(e.target.value)} placeholder="Longitude" className="min-w-0 rounded-xl border border-[#DCD3E4] px-3 py-2.5 text-sm" /></div><p className="mt-1 text-xs text-[#71808A]">Coordinates are optional. Without them, the report appears in the list but not as a map marker.</p></div>

                  {reportError && <div role="alert" className="rounded-xl border border-[#F0CFC7] bg-[#FDF0EC] p-3 text-sm text-[#B34E43]">{reportError}</div>}
                  {reportMessage && <div role="status" className="rounded-xl border border-[#B8DED8] bg-[#F0F8F6] p-3 text-sm text-[#287E83]">{reportMessage}</div>}
                  <button disabled={submittingReport} type="submit" className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#287E83] px-5 py-3 text-sm font-bold text-white transition hover:bg-[#206B70] disabled:opacity-50"><Send size={17} />{submittingReport ? 'Saving...' : 'Save Community Report'}</button>
                  <p className="text-xs leading-5 text-[#71808A]">Reports are saved locally first. Optional photos are not uploaded to the backend by this form. The report is unverified and should not be treated as an emergency alert.</p>
                </form>

                <div className={`${cardStyle} p-5`}>
                  <div className="flex items-center justify-between gap-3"><div><h3 className="font-bold">Offline reporting status</h3><p className="mt-1 text-xs text-[#71808A]">Local save and manual sync controls</p></div><div className={`rounded-xl p-3 ${online ? 'bg-[#DCEFEB] text-[#287E83]' : 'bg-[#FDF0EC] text-[#B34E43]'}`}>{online ? <Wifi size={22} /> : <WifiOff size={22} />}</div></div>
                  <div className={`mt-5 rounded-xl border p-4 ${online ? 'border-[#B8DED8] bg-[#F0F8F6]' : 'border-[#F0CFC7] bg-[#FDF0EC]'}`}><p className="text-sm font-bold">{online ? 'Device is online' : 'Device is offline'}</p><p className="mt-1 text-xs leading-5 text-[#71808A]">{online ? 'Reports remain stored locally until you choose to sync them.' : 'You can still create reports. They are stored in this browser and can be synced when connectivity returns.'}</p></div>
                  <div className="mt-4 grid grid-cols-2 gap-3"><div className="rounded-xl bg-[#FAF8FB] p-4"><p className="text-xs text-[#71808A]">Pending reports</p><p className="mt-1 text-2xl font-extrabold">{pendingCount}</p></div><div className="rounded-xl bg-[#FAF8FB] p-4"><p className="text-xs text-[#71808A]">Synced reports</p><p className="mt-1 text-2xl font-extrabold">{communityReports.filter(r => r.syncStatus === 'Synced').length}</p></div></div>
                  <button onClick={() => void syncPendingReports()} disabled={syncing || !online || pendingCount === 0 || !API_BASE_URL} className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-[#DCD3E4] px-4 py-3 text-sm font-bold text-[#244A69] hover:bg-[#F6F2FA] disabled:cursor-not-allowed disabled:opacity-40"><RefreshCw size={17} className={syncing ? 'animate-spin' : ''} />{syncing ? 'Syncing reports...' : 'Sync pending reports'}</button>
                  {!API_BASE_URL && <p className="mt-2 text-xs text-[#A87323]">Configure the backend URL in your frontend .env file to enable syncing.</p>}
                  {syncMessage && <p role="status" className="mt-3 rounded-lg bg-[#F6F2FA] p-3 text-xs leading-5 text-[#51447C]">{syncMessage}</p>}
                </div>
              </div>

              <div className={`${cardStyle} overflow-hidden`}>
                <div className="border-b border-[#E7E0E9] p-5"><h3 className="font-bold">Submitted community reports</h3><p className="mt-1 text-xs text-[#71808A]">Reports saved in this browser session or loaded from its saved storage</p></div>
                {communityReports.length === 0 ? <div className="p-8 text-center"><Users className="mx-auto text-[#B9A8E8]" size={30} /><p className="mt-3 text-sm font-semibold">No community reports yet</p><p className="mt-1 text-xs text-[#71808A]">Submit an observation using the form above.</p></div> : communityReports.map(report => <div key={report.id} className="border-b border-[#E7E0E9] p-4 last:border-0"><div className="flex flex-wrap items-start justify-between gap-3"><div className="min-w-0"><p className="font-bold">{report.location}</p><p className="mt-1 text-xs text-[#71808A]">{formatTime(report.submittedAt)} · {report.id}</p></div><div className="flex flex-wrap gap-2"><RiskBadge risk={report.severity} /><span className={`rounded-full px-3 py-1.5 text-xs font-bold ${report.syncStatus === 'Synced' ? 'bg-[#DCEFEB] text-[#287E83]' : 'bg-[#FFF6E7] text-[#A87323]'}`}>{report.syncStatus}</span></div></div><p className="mt-3 text-sm leading-5 text-[#71808A]">{report.description}</p>{report.photoName && <p className="mt-2 text-xs text-[#71808A]"><FileImage className="mr-1 inline" size={13} />{report.photoName} · filename only; photo not uploaded</p>}<div className="mt-3 flex items-center justify-between gap-3"><span className="text-xs text-[#71808A]">Verification: {report.status}</span><button onClick={() => removeReport(report.id)} className="flex items-center gap-1 text-xs font-semibold text-[#B34E43] hover:underline"><Trash2 size={13} />Remove local record</button></div></div>)}
              </div>
            </section>
            )}

            {/* MAP AND INCIDENTS */}
            {activeSection === 'map' && (
            <section className="space-y-5">
              <div><p className="text-sm font-semibold text-[#287E83]">GEOGRAPHIC INTELLIGENCE</p><h2 className="mt-2 text-2xl font-extrabold">Flood Map & Incident Records</h2><p className="mt-1 text-sm text-[#71808A]">Explore available geolocated records and inspect their source and verification status.</p></div>

              <div className="grid gap-4 sm:grid-cols-3">
                <StatCard title="Incident records" value={String(allIncidents.length)} detail="Includes illustrative demo records" icon={<ClipboardList />} color="turquoise" />
                <StatCard title="Geolocated records" value={String(mapIncidents.length)} detail="Records with usable coordinates" icon={<LocateFixed />} color="lavender" />
                <StatCard title="Unverified reports" value={String(allIncidents.filter(i => i.verificationStatus.toLowerCase().includes('unverified')).length + communityReports.length)} detail="Check individual source/status" icon={<AlertTriangle />} color="peach" />
              </div>

              <div className="overflow-hidden rounded-2xl border border-[#E7E0E9] bg-white shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#E7E0E9] p-4"><div className="flex items-center gap-2"><MapPin className="text-[#287E83]" size={18} /><span className="text-sm font-bold">Available geolocated observations</span></div><div className="flex flex-wrap gap-3 text-xs"><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#E87979]" />High</span><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#E9B45F]" />Medium</span><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#6DBDB4]" />Low</span></div></div>
                <div className="h-[380px] w-full">
                  <MapContainer center={[28.6139, 77.209]} zoom={13} scrollWheelZoom className="h-full w-full">
                    <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                    {mapIncidents.map(item => <Marker key={item.id} position={[item.latitude!, item.longitude!]} icon={makeRiskIcon(getRiskColor(item.risk))}><Popup><div style={{ minWidth: 170, color: '#244A69' }}><strong>{item.location}</strong><p>Risk: {item.risk}</p><p>Source: {item.source}</p><p>Status: {item.verificationStatus}</p>{item.depth && <p>Depth: {item.depth}</p>}<small>{item.id}</small></div></Popup></Marker>)}
                  </MapContainer>
                </div>
                <div className="border-t border-[#E7E0E9] bg-[#F8F5F8] p-3 text-xs text-[#71808A]">Map tiles © OpenStreetMap contributors. Demo locations are illustrative. Reports without coordinates are not shown as map markers.</div>
              </div>

              <div className={`${cardStyle} overflow-hidden`}>
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#E7E0E9] p-5"><div><h3 className="font-bold">Incident register</h3><p className="mt-1 text-xs text-[#71808A]">{filteredIncidents.length} records match your filters</p></div><button onClick={() => { void loadIncidents(); void loadWeather() }} className="flex items-center gap-2 rounded-xl border border-[#DCD3E4] px-3 py-2 text-xs font-bold hover:bg-[#F6F2FA]"><RefreshCw size={14} />Refresh API data</button></div>
                <div className="grid gap-3 border-b border-[#E7E0E9] p-4 sm:grid-cols-2"><div className="relative"><Search className="absolute left-3 top-3 text-[#71808A]" size={16} /><input value={incidentSearch} onChange={e => setIncidentSearch(e.target.value)} placeholder="Search location, source or description" className="w-full rounded-xl border border-[#DCD3E4] py-2.5 pl-9 pr-3 text-sm outline-none focus:border-[#287E83]" /></div><div className="relative"><Filter className="absolute left-3 top-3 text-[#71808A]" size={16} /><select value={incidentFilter} onChange={e => setIncidentFilter(e.target.value)} className="w-full rounded-xl border border-[#DCD3E4] bg-white py-2.5 pl-9 pr-3 text-sm"><option>All</option><option>High</option><option>Medium</option><option>Low</option></select></div></div>
                {filteredIncidents.length === 0 ? <p className="p-8 text-center text-sm text-[#71808A]">No incidents match these filters.</p> : filteredIncidents.map(item => <button key={item.id} onClick={() => setSelectedIncident(item)} className="flex w-full flex-wrap items-center justify-between gap-3 border-b border-[#E7E0E9] p-4 text-left last:border-0 hover:bg-[#FAF8FB]"><div className="min-w-0"><p className="font-bold">{item.location}</p><p className="mt-1 text-xs text-[#71808A]">{item.source} · {item.verificationStatus} · {item.id}</p><p className="mt-1 truncate text-xs text-[#71808A]">{item.description}</p></div><div className="flex items-center gap-2">{item.depth && <span className="text-xs text-[#71808A]">{item.depth}</span>}<RiskBadge risk={item.risk} /></div></button>)}
              </div>

              {selectedIncident && <div className={`${cardStyle} p-5`}><div className="flex items-center justify-between gap-3"><h3 className="font-bold">Incident details</h3><button onClick={() => setSelectedIncident(null)} aria-label="Close incident details"><X size={18} /></button></div><div className="mt-4 grid gap-3 sm:grid-cols-2"><Detail label="Incident ID" value={selectedIncident.id} /><Detail label="Location" value={selectedIncident.location} /><Detail label="Source" value={selectedIncident.source} /><Detail label="Risk category" value={selectedIncident.risk} /><Detail label="Verification" value={selectedIncident.verificationStatus} /><Detail label="Sync status" value={selectedIncident.syncStatus} /><Detail label="Depth" value={selectedIncident.depth ?? 'Unavailable'} /><Detail label="Time" value={formatTime(selectedIncident.time)} /></div><p className="mt-4 text-sm leading-6 text-[#71808A]">{selectedIncident.description}</p></div>}
            </section>
            )}

            {/* IMAGE ANALYSIS */}
            {activeSection === 'analysis' && (
            <section className="space-y-4">
              <div><p className="text-sm font-semibold text-[#287E83]">IMAGE-BASED OBSERVATION</p><h2 className="text-2xl font-extrabold">Flood Image Analysis</h2><p className="mt-1 text-sm text-[#71808A]">Upload a flood-related image for preview and, if configured, submit it to the analysis API.</p></div>
              <div className="grid gap-6 xl:grid-cols-2">
                <div className={`${cardStyle} p-5`}>
                  <div className="mb-4 flex items-center gap-2"><div className="rounded-lg bg-[#DCEFEB] p-2"><Upload className="text-[#287E83]" size={19} /></div><h3 className="font-bold">Upload Image</h3></div>
                  <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleUpload} className="hidden" />
                  {!selectedImage ? <button onClick={() => fileInputRef.current?.click()} className="flex min-h-64 w-full flex-col items-center justify-center rounded-2xl border-2 border-dashed border-[#B9A8E8] bg-[#F6F2FA] p-6 text-center transition hover:border-[#287E83]"><div className="mb-4 rounded-2xl bg-[#E9E2F5] p-4"><ImagePlus className="h-8 w-8 text-[#7664A8]" /></div><p className="font-bold">Click to upload an image</p><p className="mt-2 text-xs text-[#71808A]">JPG, PNG or WebP · Maximum 10 MB</p></button> : <div className="space-y-3"><div className="relative overflow-hidden rounded-xl border border-[#E7E0E9] bg-[#F0EBF0]"><img src={selectedImage} alt="Selected flood observation" className="max-h-72 w-full object-contain" /><button onClick={handleReset} className="absolute right-3 top-3 rounded-lg bg-[#244A69] p-2 text-white" aria-label="Remove image"><X size={17} /></button></div><div className="rounded-xl border border-[#E7E0E9] bg-[#FAF8FB] p-3"><p className="truncate text-sm font-semibold">{imageName}</p><p className="mt-2 text-xs text-[#71808A]">{(imageSize / (1024 * 1024)).toFixed(2)} MB · {imageType.replace('image/', '').toUpperCase()}</p></div></div>}
                  {uploadError && <div role="alert" className="mt-4 rounded-xl border border-[#F0CFC7] bg-[#FDF0EC] p-3 text-sm text-[#B34E43]">{uploadError}</div>}
                  <div className="mt-5 flex flex-wrap gap-3"><button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 rounded-xl border border-[#DCD3E4] px-4 py-2.5 text-sm font-semibold"><Upload size={16} />{selectedImage ? 'Change Image' : 'Choose Image'}</button><button onClick={() => void handleAnalyze()} disabled={!selectedImage || isAnalyzing} className="flex items-center gap-2 rounded-xl bg-[#287E83] px-5 py-2.5 text-sm font-bold text-white disabled:opacity-40">{isAnalyzing ? <RefreshCw size={16} className="animate-spin" /> : <Zap size={16} />}{isAnalyzing ? 'Processing...' : 'Analyze Image'}</button></div>
                  <p className="mt-4 text-xs leading-5 text-[#71808A]">Images are submitted to the configured analysis API. Results may use precomputed demo fixtures; live AI inference is not currently enabled.</p>
                </div>

                <div className={`${cardStyle} p-5`}>
                  <div className="rounded-xl border border-[#F0DEBC] bg-[#FFF6E7] p-4">
                    <p className="text-sm font-bold text-[#8A682F]">
                      {parsedAnalysis?.analysis_source === 'PRECOMPUTED_DEMO_FIXTURE'
                        ? 'Precomputed Demo Result'
                        : 'Analysis Status'}
                    </p>
                    <p className="mt-1 text-xs leading-5 text-[#8A682F]">
                      {parsedAnalysis?.model_inference === false
                        ? 'This result uses a precomputed fixture. Live AI model inference was not performed.'
                        : `Pipeline status: ${parsedAnalysis?.result?.status ?? 'Unavailable'}`}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <ResultCard label="Water boundary" value={
                      parsedAnalysis?.result?.water_region?.detected === true
                        ? 'Proposed — review required'
                        : 'Unavailable'
                    } />
                    <ResultCard label="Estimated depth" value="Not estimated" />
                    <ResultCard label="Flood risk" value="Not classified" />
                    <ResultCard label="Water confidence" value={
                    parsedAnalysis?.result?.water_region?.confidence != null
                    ? `${(parsedAnalysis.result.water_region.confidence * 100).toFixed(0)}%`
                    : 'Unavailable'
                    } />
                  </div>
              <div className="mt-4 rounded-xl border border-[#E7E0E9] bg-[#FAF8FB] p-4">
                <p className="text-sm font-bold text-[#244A69]">
                Vehicle Detections
                </p>
                {parsedAnalysis?.result?.detections?.length ? (
                  <div className="mt-3 space-y-2">
                    {parsedAnalysis.result.detections.map((vehicle, index) => (
                      <div key={index} className="flex justify-between text-sm">
                        <span>
                          {String(vehicle.class_name ?? 'Vehicle')}
                        </span>
                        <span className="font-semibold text-[#287E83]">
                          {vehicle.confidence != null
                            ? `${(vehicle.confidence * 100).toFixed(0)}% confidence`
                            : 'Confidence unavailable'}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-[#71808A]">
                    No vehicle detections available.
                  </p>
                )}
              </div>
              <div className="rounded-2xl border border-[#B9A8E8] bg-[#F6F2FA] p-4"><div className="flex items-start gap-3"><ShieldCheck className="mt-0.5 shrink-0 text-[#7664A8]" size={20} /><div><p className="text-sm font-bold text-[#51447C]">Responsible reporting</p><p className="mt-1 text-xs leading-5 text-[#71808A]">An uploaded image is an observation, not proof of a confirmed flood. Depth and risk require a validated analysis model.</p></div></div></div>
              </div>
            </div>
            </section>
            )}

            {/* SETTINGS */}
            {activeSection === 'settings' && (
            <section className="space-y-4">
              <div><h2 className="text-2xl font-extrabold">System Settings</h2><p className="mt-1 text-sm text-[#71808A]">Connection, storage and dashboard environment information</p></div>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className={`${cardStyle} p-5`}><div className="flex items-center gap-3"><div className="rounded-xl bg-[#DCEFEB] p-3 text-[#287E83]"><Activity /></div><div><p className="font-bold">Frontend Status</p><p className="mt-1 text-xs font-medium text-[#287E83]">Running</p></div></div></div>
                <div className={`${cardStyle} p-5`}><div className="flex items-center gap-3"><div className="rounded-xl bg-[#E9E2F5] p-3 text-[#7664A8]"><Database /></div><div><p className="font-bold">Local Storage</p><p className="mt-1 text-xs font-medium text-[#7664A8]">Community reports stored in browser</p></div></div></div>
              </div>

              <div className={`${cardStyle} p-5`}>
                <div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="font-bold">Backend API Connection</h3><p className="mt-1 text-xs text-[#71808A]">Configure the API base URL through your Vite environment file.</p></div><div className={`rounded-full px-3 py-1.5 text-xs font-bold ${apiStatus === 'connected' ? 'bg-[#DCEFEB] text-[#287E83]' : apiStatus === 'error' ? 'bg-[#FDF0EC] text-[#B34E43]' : 'bg-[#FFF6E7] text-[#A87323]'}`}>{apiStatus === 'connected' ? 'Connected' : apiStatus === 'error' ? 'Connection error' : 'Not verified'}</div></div>
                <div className="mt-4 rounded-xl bg-[#FAF8FB] p-4"><div className="flex items-center gap-2 text-sm font-semibold"><Server size={17} />{API_BASE_URL || 'VITE_API_BASE_URL is not set'}</div><p className="mt-2 text-xs leading-5 text-[#71808A]">{apiMessage}</p></div>
                <button onClick={() => void checkBackend()} disabled={!API_BASE_URL} className="mt-4 flex items-center gap-2 rounded-xl bg-[#287E83] px-4 py-2.5 text-sm font-bold text-white disabled:opacity-40"><RefreshCw size={15} />Test backend connection</button>
                <div className="mt-5 border-t border-[#E7E0E9] pt-4"><p className="text-sm font-bold">Expected API routes in this frontend</p><div className="mt-3 space-y-2 text-xs text-[#71808A]"><p><code>GET {API.incidents}</code> — load incident records</p><p><code>POST {API.communityReports}</code> — submit community reports</p><p><code>POST {API.analyzeImage}</code> — submit an image for analysis</p><p><code>GET {API.weather}</code> — load weather information</p></div><p className="mt-3 text-xs leading-5 text-[#A87323]">These are frontend expectations, not confirmation that your backend already implements these routes. Confirm them with your backend teammate.</p></div>
              </div>

              <div className={`${cardStyle} p-5`}>
                <h3 className="font-bold">Offline data management</h3>
                <p className="mt-1 text-xs text-[#71808A]">Local reports are stored in this browser using localStorage.</p>
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3"><div><p className="text-sm font-semibold">{communityReports.length} locally stored report(s)</p><p className="mt-1 text-xs text-[#71808A]">{pendingCount} waiting for sync</p></div><button onClick={() => { if (window.confirm('Delete all locally stored community reports from this browser?')) setCommunityReports([]) }} className="flex items-center gap-2 rounded-xl border border-[#F0CFC7] px-3 py-2 text-xs font-bold text-[#B34E43] hover:bg-[#FDF0EC]"><Trash2 size={14} />Clear local reports</button></div>
              </div>
            </section>
            )}

            <footer className="border-t border-[#DCD3E4] py-6 text-center text-xs leading-5 text-[#71808A]">AquaGauge Flood Intelligence Dashboard<br />Hackathon demonstration • Sample readings and map locations are illustrative, not live emergency information.</footer>
          </div>
        </main>
      </div>
    </div>
  )
}

/* REUSABLE COMPONENTS */

function sourceColor(color: string) {
  const colors: Record<string, string> = {
    turquoise: 'bg-[#DCEFEB] text-[#287E83]',
    lavender: 'bg-[#E9E2F5] text-[#7664A8]',
    peach: 'bg-[#FBE8E0] text-[#C8755E]',
    blue: 'bg-[#E3EAF3] text-[#416C91]',
  }
  return colors[color] ?? 'bg-[#F0EBF0] text-[#71808A]'
}

function StatCard({ title, value, detail, icon, color }: {
  title: string; value: string; detail: string; icon: ReactNode; color: string
}) {
  const colors: Record<string, string> = {
    turquoise: 'bg-[#DCEFEB] text-[#287E83]',
    peach: 'bg-[#FBE8E0] text-[#C8755E]',
    lavender: 'bg-[#E9E2F5] text-[#7664A8]',
    blue: 'bg-[#E3EAF3] text-[#416C91]',
  }
  return <div className={`${cardStyle} p-5 transition duration-200 hover:-translate-y-1 hover:shadow-lg`}><div className="flex items-start justify-between"><div><p className="text-sm font-medium text-[#71808A]">{title}</p><p className="mt-3 text-2xl font-extrabold">{value}</p></div><div className={`rounded-xl p-3 ${colors[color]}`}>{icon}</div></div><p className="mt-3 text-xs text-[#71808A]">{detail}</p></div>
}

function SourceCard({ icon, label, detail, color, selected }: {
  icon: ReactNode; label: string; detail: string; color: string; selected: boolean
}) {
  return <div className={`flex items-center gap-3 rounded-xl border p-3 transition ${selected ? 'border-[#6DBDB4] bg-[#F0F8F6]' : 'border-[#E7E0E9] bg-[#FCFAFC]'}`}><div className={`rounded-xl p-2.5 ${sourceColor(color)}`}>{icon}</div><div className="min-w-0 flex-1"><p className="text-sm font-bold">{label}</p><p className="mt-0.5 text-xs text-[#71808A]">{detail}</p></div><span className={`h-3 w-3 shrink-0 rounded-full ${selected ? 'bg-[#6DBDB4]' : 'border border-[#B9A8E8]'}`} /></div>
}

function ResultCard({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-[#E7E0E9] bg-[#FAF8FB] p-4"><p className="text-xs font-medium text-[#71808A]">{label}</p><p className="mt-2 text-lg font-extrabold text-[#287E83]">{value}</p></div>
}

function RiskBadge({ risk }: { risk: string }) {
  const styles: Record<string, string> = {
    High: 'border-[#F0CFC7] bg-[#FDF0EC] text-[#B34E43]',
    Medium: 'border-[#F0DEBC] bg-[#FFF6E7] text-[#A87323]',
    Low: 'border-[#B8DED8] bg-[#DCEFEB] text-[#287E83]',
  }
  return <span className={`rounded-full border px-3 py-1.5 text-xs font-bold ${styles[risk] ?? 'border-gray-200 bg-gray-100 text-gray-600'}`}>{risk}</span>
}

function WeatherCard({ label, value, detail, icon }: {
  label: string; value: string; detail: string; icon: ReactNode
}) {
  return <div className={`${cardStyle} p-5`}><div className="flex items-center justify-between"><p className="text-sm font-medium text-[#71808A]">{label}</p><div className="rounded-xl bg-[#DCEFEB] p-2.5 text-[#287E83]">{icon}</div></div><p className="mt-4 text-3xl font-extrabold">{value}</p><p className="mt-2 text-xs text-[#71808A]">{detail}</p></div>
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl bg-[#FAF8FB] p-3"><p className="text-xs text-[#71808A]">{label}</p><p className="mt-1 break-words text-sm font-semibold">{value}</p></div>
}

export default App
