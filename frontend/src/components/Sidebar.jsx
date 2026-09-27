import { NavLink } from 'react-router-dom'
import ModelStatus from './ModelStatus'
import ThemeToggle from './ThemeToggle'

// Navigation model. Icons are small inline SVGs kept in this file so the
// sidebar has no external icon dependency.
const SECTIONS = [
  {
    items: [{ to: '/', label: 'Overview', icon: GridIcon, end: true }],
  },
  {
    title: 'Workspace',
    items: [
      { to: '/playground', label: 'NER Playground', icon: SparkIcon },
      { to: '/extract', label: 'Information Extraction', icon: LayersIcon },
      { to: '/search', label: 'Search', icon: SearchIcon },
      { to: '/assistant', label: 'AI Assistant', icon: ChatIcon },
    ],
  },
  {
    title: 'Resources',
    items: [
      { to: '/model', label: 'Model & Dataset', icon: DatabaseIcon },
      { to: '/about', label: 'About', icon: InfoIcon },
    ],
  },
]

export default function Sidebar({ onNavigate }) {
  return (
    <div className="flex h-full flex-col gap-6 p-4">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-1">
        <span
          className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-sm font-semibold text-accent-fg"
          aria-hidden="true"
        >
          ने
        </span>
        <div className="leading-tight">
          <div className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Marathi NLP Studio
          </div>
          <div className="text-[11px] text-gray-500 dark:text-gray-400">
            NER-powered text tools
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-5 overflow-y-auto" aria-label="Primary">
        {SECTIONS.map((section, i) => (
          <div key={section.title || i}>
            {section.title && (
              <div className="mb-1.5 px-3 text-[11px] font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500">
                {section.title}
              </div>
            )}
            <ul className="space-y-0.5">
              {section.items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      `flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors ${
                        isActive
                          ? 'bg-accent-soft font-medium text-accent dark:bg-accent/15 dark:text-indigo-300'
                          : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                      }`
                    }
                  >
                    <item.icon />
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer: status + theme */}
      <div className="space-y-3">
        <ModelStatus />
        <div className="flex items-center justify-between px-1">
          <span className="text-[11px] text-gray-400 dark:text-gray-500">Theme</span>
          <ThemeToggle />
        </div>
      </div>
    </div>
  )
}

/* --- inline icons (16px, currentColor) --- */
const base = {
  width: 16,
  height: 16,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
}
function GridIcon() {
  return (
    <svg {...base}>
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  )
}
function SparkIcon() {
  return (
    <svg {...base}>
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8" />
    </svg>
  )
}
function LayersIcon() {
  return (
    <svg {...base}>
      <path d="M12 2 2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5M2 12l10 5 10-5" />
    </svg>
  )
}
function SearchIcon() {
  return (
    <svg {...base}>
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  )
}
function ChatIcon() {
  return (
    <svg {...base}>
      <path d="M21 11.5a8.5 8.5 0 0 1-12.5 7.5L3 21l2-5.5A8.5 8.5 0 1 1 21 11.5z" />
    </svg>
  )
}
function DatabaseIcon() {
  return (
    <svg {...base}>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
    </svg>
  )
}
function InfoIcon() {
  return (
    <svg {...base}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  )
}
