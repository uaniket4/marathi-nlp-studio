import { useState } from 'react'
import { Routes, Route, Outlet, useLocation } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Overview from './pages/Overview'
import NerPlayground from './pages/NerPlayground'
import InformationExtraction from './pages/InformationExtraction'
import Search from './pages/Search'
import Assistant from './pages/Assistant'
import ModelDataset from './pages/ModelDataset'
import About from './pages/About'
import NotFound from './pages/NotFound'

function Shell() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen lg:flex">
      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 border-r border-gray-200 bg-white lg:block dark:border-gray-800 dark:bg-gray-900">
        <div className="sticky top-0 h-screen">
          <Sidebar />
        </div>
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            className="absolute inset-0 bg-black/40"
            onClick={() => setMobileOpen(false)}
            aria-hidden="true"
          />
          <div className="absolute left-0 top-0 h-full w-64 border-r border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
            <Sidebar onNavigate={() => setMobileOpen(false)} />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Mobile top bar */}
        <header className="flex items-center gap-3 border-b border-gray-200 bg-white px-4 py-3 lg:hidden dark:border-gray-800 dark:bg-gray-900">
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="btn-secondary !p-2"
            aria-label="Open navigation menu"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M3 6h18M3 12h18M3 18h18" />
            </svg>
          </button>
          <span className="font-semibold text-gray-900 dark:text-gray-100">
            Marathi NLP Studio
          </span>
        </header>

        <main
          key={location.pathname}
          className="mx-auto w-full max-w-5xl flex-1 px-4 py-6 sm:px-6 sm:py-8 motion-safe:animate-[fadein_.2s_ease]"
        >
          <Outlet />
        </main>

        <footer className="border-t border-gray-200 py-4 dark:border-gray-800">
          <p className="mx-auto max-w-5xl px-4 text-xs text-gray-400 dark:text-gray-500 sm:px-6">
            Marathi NLP Studio · Built on the L3Cube-MahaNER dataset. For research
            and educational use.
          </p>
        </footer>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route index element={<Overview />} />
        <Route path="playground" element={<NerPlayground />} />
        <Route path="extract" element={<InformationExtraction />} />
        <Route path="search" element={<Search />} />
        <Route path="assistant" element={<Assistant />} />
        <Route path="model" element={<ModelDataset />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
